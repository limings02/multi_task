import torch

from src.loss.bce import MultiTaskBCELoss
from src.train.loops import train_one_epoch


class _DummyLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class _DummyCvrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single parameter so optimizer has something to step when enabled.
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        # Infer batch size from dummy field indices.
        fields = features.get("fields", {})
        first_field = next(iter(fields.values()), None)
        batch_size = int(first_field["indices"].shape[0]) if first_field else 1
        logits = torch.ones(batch_size, device=self.bias.device) * self.bias
        return {"cvr": logits}


def _make_cvr_only_batch(batch_size: int = 4):
    labels = {
        "y_ctr": torch.zeros(batch_size),
        "y_cvr": torch.zeros(batch_size),
        "click_mask": torch.zeros(batch_size),
    }
    features = {
        "fields": {
            "dummy": {
                "indices": torch.arange(batch_size, dtype=torch.long),
                "offsets": torch.tensor([0, batch_size], dtype=torch.long),
                "weights": None,
            }
        },
        "field_names": ["dummy"],
    }
    meta = {}
    return labels, features, meta


def test_train_loop_skips_cvr_only_empty_mask():
    device = torch.device("cpu")
    model = _DummyCvrModel().to(device)
    loss_fn = MultiTaskBCELoss(enabled_heads=["cvr"])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = _DummyLogger()

    labels, _, _ = _make_cvr_only_batch(batch_size=3)
    # Reproduce the zero-mask loss that would lack a grad_fn.
    loss, _ = loss_fn.compute({"cvr": torch.zeros(3, requires_grad=True)}, {"labels": labels})
    assert loss.requires_grad is False

    loader = [_make_cvr_only_batch(batch_size=3) for _ in range(2)]
    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        logger=logger,
        epoch=0,
        log_every=1,
        max_steps=None,
        amp_enabled=False,
        scaler=None,
    )

    assert metrics["steps"] == 2
    assert metrics["mask_cvr_sum"] == 0.0
    # No backward should have run, so grads stay None.
    assert model.bias.grad is None

