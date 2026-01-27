import torch

from src.loss.bce import MultiTaskBCELoss


def test_bce_loss_masking_and_backward():
    loss_fn = MultiTaskBCELoss(w_ctr=1.0, w_cvr=1.0, eps=1e-6)
    outputs = {
        "ctr": torch.tensor([0.2, -0.4, 0.0, 0.8], requires_grad=True),
        "cvr": torch.tensor([0.1, 0.2, -0.3, 0.4], requires_grad=True),
    }
    labels = {
        "y_ctr": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "y_cvr": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "click_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    loss, log = loss_fn.compute(outputs, labels)
    assert loss.ndim == 0
    loss.backward()
    # cvr mask should consider only positions 0 and 2
    loss_fn2 = MultiTaskBCELoss()
    labels_zero_mask = {**labels, "click_mask": torch.zeros_like(labels["click_mask"])}
    _, log_zero = loss_fn2.compute(outputs, labels_zero_mask)
    assert log_zero["loss_cvr"] == 0.0  # when no clicks, cvr loss should be zero not NaN


