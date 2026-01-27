import tempfile
from pathlib import Path

import torch
from torch import nn, optim

from src.core.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_linear():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        model = nn.Linear(4, 2)
        opt = optim.SGD(model.parameters(), lr=0.1)

        # make deterministic weights
        torch.manual_seed(0)
        for p in model.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

        save_checkpoint(path, model, opt, cfg={"foo": "bar"}, step=5, best_metric=0.42, extra={"note": "test"})

        # new model/opt to load into
        model2 = nn.Linear(4, 2)
        opt2 = optim.SGD(model2.parameters(), lr=0.1)
        info = load_checkpoint(path, model2, opt2, map_location="cpu", strict=True)

        # forward compare
        x = torch.randn(3, 4)
        torch.manual_seed(1)
        out1 = model(x)
        out2 = model2(x)
        assert torch.allclose(out1, out2)
        assert info["step"] == 5
        assert abs(info["best_metric"] - 0.42) < 1e-8
        assert info["cfg"]["foo"] == "bar"
        assert info["extra"]["note"] == "test"

