import torch
import torch.nn as nn
import pytest

from src.train.optim import build_optimizer_bundle


def test_optimizer_bundle_dual_without_sparse_loads_legacy_and_new():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    cfg = {
        "optim": {
            "type": "dual_sparse_dense",
            "dense": {"lr": 0.01},
            "sparse": {"enabled": False},
        }
    }

    bundle = build_optimizer_bundle(cfg, model, scaler=None)

    x = torch.randn(5, 4)
    y = torch.randn(5)
    bundle.zero_grad()
    pred = model(x).squeeze(-1)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    bundle.step()

    # new-format state dict roundtrip
    sd = bundle.state_dict()
    bundle2 = build_optimizer_bundle(cfg, model, scaler=None)
    bundle2.load_state_dict(sd)

    # legacy-format load (only dense optimizer)
    legacy = {"optimizer": bundle.dense_opt.state_dict()}
    bundle2.load_state_dict(legacy)

    # ensure step still works after loading legacy
    bundle2.zero_grad()
    pred2 = model(x).squeeze(-1)
    loss2 = ((pred2 - y) ** 2).mean()
    loss2.backward()
    bundle2.step()


def test_optimizer_bundle_sparse_enabled_but_missing_params_raises():
    model = nn.Linear(4, 1)  # no sparse params
    cfg = {"optim": {"type": "dual_sparse_dense", "sparse": {"enabled": True}}}
    with pytest.raises(ValueError):
        build_optimizer_bundle(cfg, model, scaler=None)


def test_optimizer_bundle_sparse_enabled_fallback_warns_and_degrades(caplog):
    import logging

    model = nn.Linear(4, 1)  # no sparse params
    cfg = {
        "optim": {
            "type": "dual_sparse_dense",
            "dense": {"lr": 0.01},
            "sparse": {"enabled": True, "allow_fallback_if_empty": True},
        }
    }
    caplog.set_level(logging.WARNING)
    bundle = build_optimizer_bundle(cfg, model, scaler=None)
    assert bundle.sparse_opt is None  # downgraded
    assert any("falling back to dense-only" in rec.message for rec in caplog.records)


def test_optimizer_bundle_load_new_and_legacy_schema():
    model = nn.Linear(4, 1)
    cfg = {"optim": {"type": "single"}}
    bundle = build_optimizer_bundle(cfg, model, scaler=None)
    sd = bundle.state_dict()
    bundle2 = build_optimizer_bundle(cfg, model, scaler=None)
    bundle2.load_state_dict(sd)  # new schema

    legacy = {"optimizer": bundle.dense_opt.state_dict()}
    bundle2.load_state_dict(legacy)  # legacy schema accepted
