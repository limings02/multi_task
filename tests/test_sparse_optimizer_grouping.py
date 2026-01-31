import torch
import torch.nn as nn
import pytest

from src.train.optim import build_optimizer_bundle


class TinySparseModel(nn.Module):
    def __init__(self, sparse: bool):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=4, mode="sum", include_last_offset=False, sparse=sparse)
        self.lin = nn.Linear(4, 1)

    def forward(self, idx, offsets):
        h = self.emb(idx, offsets)
        return self.lin(h)


def _make_inputs():
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    offsets = torch.tensor([0, 2], dtype=torch.int64)
    return idx, offsets


def test_sparse_grouping_collects_sparse_params():
    model = TinySparseModel(sparse=True)
    cfg = {
        "optim": {
            "type": "dual_sparse_dense",
            "dense": {"lr": 1e-3},
            "sparse": {"enabled": True, "lr": 2e-3},
        }
    }
    bundle = build_optimizer_bundle(cfg, model, scaler=None)
    assert bundle.sparse_opt is not None
    assert bundle.sparse_params is not None
    assert len(list(bundle.sparse_params)) > 0
    sparse_ids = {id(p) for p in bundle.sparse_params}
    dense_ids = {id(p) for p in bundle.dense_params}
    assert sparse_ids.isdisjoint(dense_ids)
    # embedding weight must be in sparse params
    assert id(model.emb.weight) in sparse_ids


def test_sparse_enabled_without_sparse_grad_raises():
    model = TinySparseModel(sparse=False)
    cfg = {
        "optim": {
            "type": "dual_sparse_dense",
            "sparse": {"enabled": True},
        }
    }
    with pytest.raises(ValueError):
        build_optimizer_bundle(cfg, model, scaler=None)


def test_sparse_backward_step_smoke():
    model = TinySparseModel(sparse=True)
    cfg = {
        "optim": {
            "type": "dual_sparse_dense",
            "dense": {"lr": 1e-3},
            "sparse": {"enabled": True, "lr": 1e-2},
        }
    }
    bundle = build_optimizer_bundle(cfg, model, scaler=None)
    idx, offsets = _make_inputs()
    out = model(idx, offsets)
    loss = out.sum()
    loss.backward()
    bundle.step()
