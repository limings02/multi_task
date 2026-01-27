import torch
from pathlib import Path

from src.data.dataloader import make_dataloader
from src.utils.config import load_yaml
from src.utils.feature_meta import build_model_feature_meta


def test_dataloader_spawn_single_batch():
    cfg = load_yaml("configs/experiments/deepfm_sharedbottom_train.yaml")
    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg["embedding"])
    loader = make_dataloader(
        split="train",
        batch_size=4,
        num_workers=2,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        seed=123,
        feature_meta=feature_meta,
        debug=False,
    )
    labels, features, meta = next(iter(loader))
    assert labels["y_ctr"].shape[0] == 4
    assert "fields" in features and "field_names" in features

