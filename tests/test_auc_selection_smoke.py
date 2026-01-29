import logging
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

from src.loss.bce import MultiTaskBCELoss
from src.train.loops import validate


class _DummyDataset(IterableDataset):
    def __iter__(self) -> Iterator[Tuple[dict, dict, dict]]:
        # Two samples: perfectly separable for CTR/CVR to yield AUC=1.0
        labels = {
            "y_ctr": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "y_cvr": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "y_ctcvr": torch.tensor([0.0, 1.0], dtype=torch.float32),
            # Mark both rows as clicked so CVR mask contains pos/neg.
            "click_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "row_id": torch.tensor([0, 1], dtype=torch.int64),
        }
        # Minimal feature payload; dummy model ignores content.
        features = {
            "fields": {
                "f0": {
                    "indices": torch.tensor([0, 1], dtype=torch.int64),
                    "offsets": torch.tensor([0, 1], dtype=torch.int64),
                    "weights": None,
                }
            },
            "field_names": ["f0"],
        }
        meta = {}
        yield labels, features, meta


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: dict) -> dict:
        # Perfect separation: low logit for negative, high for positive.
        return {
            "ctr": torch.tensor([ -5.0, 5.0 ], dtype=torch.float32),
            "cvr": torch.tensor([ -5.0, 5.0 ], dtype=torch.float32),
        }


def test_validate_returns_auc_primary():
    loader = DataLoader(_DummyDataset(), batch_size=None)
    model = _DummyModel()
    loss_fn = MultiTaskBCELoss(enabled_heads=["ctr", "cvr"])
    logger = logging.getLogger("dummy")

    metrics = validate(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        logger=logger,
        epoch=1,
        calc_auc=True,
    )

    assert metrics["auc_primary"] is not None
    assert metrics["auc_primary"] > 0.99
    assert metrics["auc_ctr"] > 0.99
    assert metrics["auc_cvr"] > 0.99
