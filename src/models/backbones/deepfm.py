from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.models.backbones.layers import FeatureEmbedding, MLP


class DeepFMBackbone(nn.Module):
    """
    DeepFM backbone that outputs shared representation h (no task heads).
    """

    def __init__(
        self,
        feature_meta: Dict[str, Dict[str, Any]],
        deep_hidden_dims: List[int],
        deep_dropout: float,
        deep_activation: str,
        deep_use_bn: bool,
        fm_enabled: bool = True,
        fm_projection_dim: Optional[int] = None,
        out_dim: int = 128,
    ):
        super().__init__()
        self.feature_meta = feature_meta
        self.feat_emb = FeatureEmbedding(feature_meta)
        self.field_names_sorted: List[str] = sorted(feature_meta.keys())

        # Deep component
        self.deep_mlp = MLP(
            input_dim=self.feat_emb.concat_dim,
            hidden_dims=deep_hidden_dims,
            activation=deep_activation,
            dropout=deep_dropout,
            use_bn=deep_use_bn,
        )

        # FM component config
        self.fm_enabled = bool(fm_enabled)
        self.use_projection = False
        self.fm_dim = 0
        self.fm_proj: Optional[nn.ModuleDict] = None

        emb_dims = {base: int(meta["embedding_dim"]) for base, meta in feature_meta.items()}
        dim_set = set(emb_dims.values())
        projection_dim = fm_projection_dim

        if self.fm_enabled:
            if projection_dim is not None:
                self.use_projection = True
                self.fm_dim = int(projection_dim)
                self.fm_proj = nn.ModuleDict(
                    {
                        base: nn.Linear(dim, self.fm_dim, bias=False)
                        for base, dim in emb_dims.items()
                    }
                )
            elif len(dim_set) == 1:
                self.fm_dim = int(next(iter(dim_set)))
            else:
                self.fm_enabled = False
                self.fm_dim = 0
                print(
                    "DeepFMBackbone: FM disabled because embedding dims differ and fm_projection_dim is None."
                )

        final_in_dim = self.deep_mlp.output_dim + (self.fm_dim if self.fm_enabled else 0)
        self.out_proj = nn.Linear(final_in_dim, out_dim)
        self.out_dim = out_dim

        # Linear (first-order) component: true sparse linear via EmbeddingBag
        self.linear_bags = nn.ModuleDict(
            {
                base: nn.EmbeddingBag(
                    num_embeddings=int(meta["num_embeddings"]),
                    embedding_dim=1,
                    # Sum + manual length normalization (to keep per_sample_weights supported).
                    mode="sum",
                    include_last_offset=False,
                )
                for base, meta in feature_meta.items()
            }
        )
        # Tighter init to keep initial logits in a learnable range.
        for emb in self.linear_bags.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def _compute_fm(self, embedding_out: Dict[str, Any]) -> torch.Tensor:
        field_names = embedding_out["field_names"]
        emb_dict = embedding_out["emb_dict"]

        if self.use_projection and self.fm_proj is not None:
            proj_list = [self.fm_proj[base](emb_dict[base]) for base in field_names]
            fm_stack = torch.stack(proj_list, dim=1)
        else:
            fm_stack = embedding_out.get("emb_stack")
            if fm_stack is None:
                raise RuntimeError("FM stack is None while FM is enabled without projection.")

        sum_emb = fm_stack.sum(dim=1)
        sum_square = sum_emb * sum_emb
        square_sum = (fm_stack * fm_stack).sum(dim=1)
        fm_vec = 0.5 * (sum_square - square_sum)
        return fm_vec

    def _compute_linear(self, features: Dict[str, Any]) -> torch.Tensor:
        field_names: List[str] = features["field_names"]
        fields = features["fields"]

        # Basic presence/dtype assertions per field to avoid silent misuse.
        for base in field_names:
            if base not in fields:
                raise KeyError(f"Feature {base} missing in features['fields'].")
            if base not in self.linear_bags:
                raise KeyError(f"Linear embedding for {base} not initialized.")

        # Batch size inferred from offsets (guaranteed include_last_offset=False so len == B)
        first_base = field_names[0]
        B = int(fields[first_base]["offsets"].shape[0])

        device = self.linear_bags[first_base].weight.device
        logit = torch.zeros((B, 1), device=device, dtype=self.linear_bags[first_base].weight.dtype)

        for base in field_names:
            fd = fields[base]
            idx = fd["indices"]
            offsets = fd["offsets"]
            wts = fd.get("weights")

            assert idx.dtype == torch.int64, f"{base}: indices dtype must be int64"
            assert offsets.dtype == torch.int64, f"{base}: offsets dtype must be int64"
            if wts is not None:
                assert wts.dtype == torch.float32, f"{base}: weights dtype must be float32"
                wts = wts.to(device)

            if int(offsets.shape[0]) != B:
                raise ValueError(f"{base}: offsets length {offsets.shape[0]} != batch size {B}")

            idx = idx.to(device)
            offsets = offsets.to(device)

            bag_out = self.linear_bags[base](idx, offsets, per_sample_weights=wts)
            # Normalize by bag length to avoid length-driven magnitude explosion while keeping mode='sum' for weights support.
            if offsets.numel() > 0:
                lengths = torch.empty_like(offsets, dtype=bag_out.dtype, device=device)
                if offsets.numel() > 1:
                    lengths[:-1] = offsets[1:] - offsets[:-1]
                lengths[-1] = idx.numel() - offsets[-1]
                lengths = lengths.clamp_min(1)
                bag_out = bag_out / lengths.unsqueeze(-1)
            logit = logit + bag_out

        return logit

    def forward(
        self,
        features: Dict[str, Any],
        dense_x: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if dense_x is not None:
            raise ValueError("dense_x is not supported in DeepFMBackbone backbone.")

        emb_out = self.feat_emb(features)
        deep_h = self.deep_mlp(emb_out["emb_concat"])

        parts = [deep_h]
        if self.fm_enabled:
            fm_vec = self._compute_fm(emb_out)
            parts.append(fm_vec)

        h = torch.cat(parts, dim=1)
        h_out = self.out_proj(h)

        if not return_aux:
            return h_out

        fm_vec = parts[1] if self.fm_enabled else None
        linear_logit = self._compute_linear(features)
        out = {"h": h_out, "logit_linear": linear_logit}
        # Optional extras for debugging/ablation
        out["deep_h"] = deep_h
        if fm_vec is not None:
            out["fm_vec"] = fm_vec
        return out


__all__ = ["DeepFMBackbone"]


def _smoke_test():
    torch.manual_seed(0)
    feature_meta = {
        "f0": {
            "num_embeddings": 10,
            "embedding_dim": 4,
            "mode": "sum",
            "is_multi_hot": False,
            "use_value": False,
        },
        "f1": {
            "num_embeddings": 12,
            "embedding_dim": 4,
            "mode": "sum",
            "is_multi_hot": True,
            "use_value": True,
        },
    }

    B = 3
    features = {
        "field_names": ["f0", "f1"],
        "fields": {
            "f0": {
                "indices": torch.tensor([1, 2, 3], dtype=torch.int64),
                "offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
                "weights": None,
            },
            "f1": {
                "indices": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                "offsets": torch.tensor([0, 2, 4], dtype=torch.int64),
                "weights": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            },
        },
    }

    model = DeepFMBackbone(
        feature_meta=feature_meta,
        deep_hidden_dims=[8],
        deep_dropout=0.0,
        deep_activation="relu",
        deep_use_bn=False,
        fm_enabled=True,
        fm_projection_dim=None,
        out_dim=16,
    )

    out = model(features, return_aux=True)
    assert out["h"].shape == (B, 16)
    assert out["logit_linear"].shape == (B, 1)
    assert torch.isfinite(out["h"]).all()
    assert torch.isfinite(out["logit_linear"]).all()
    print("Smoke test passed: h", out["h"].shape, "logit_linear", out["logit_linear"].shape)


if __name__ == "__main__":  # pragma: no cover
    # Allow running as a script: add repo root to sys.path when __package__ is None.
    if __package__ is None:  # type: ignore
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        sys.path.append(str(repo_root))
    _smoke_test()
