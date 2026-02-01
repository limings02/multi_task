"""
Smoke test for aux_focal implementation (方案1: ESMM 主链路 BCE + CTCVR Aux-Focal).

Tests:
1. enabled=false: behavior identical to baseline (no focal)
2. enabled=true, step < warmup_steps: only BCE (no focal)
3. enabled=true, step >= warmup_steps: BCE + focal
4. No NaN under AMP
5. Backward compatibility: missing aux_focal config doesn't crash
"""
import torch
import pytest
from src.loss.bce import MultiTaskBCELoss, focal_on_logits_aux


def test_focal_on_logits_aux_basic():
    """Test focal loss basic computation."""
    B = 10
    logits = torch.randn(B, requires_grad=True)
    targets = torch.randint(0, 2, (B,)).float()
    
    # gamma=0 should be close to standard BCE
    loss_focal_g0 = focal_on_logits_aux(logits, targets, gamma=0.0, alpha=None)
    loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    )
    assert torch.allclose(loss_focal_g0, loss_bce, atol=1e-5), "gamma=0 should equal BCE"
    
    # gamma=2 should downweight easy examples
    loss_focal_g2 = focal_on_logits_aux(logits, targets, gamma=2.0, alpha=None)
    assert loss_focal_g2.item() >= 0, "Focal loss should be non-negative"
    
    # Should be differentiable
    loss_focal_g2.backward()
    assert logits.grad is not None, "Focal loss should backpropagate"
    
    print(f"✓ focal_on_logits_aux basic test passed (BCE={loss_bce.item():.4f}, Focal(g=2)={loss_focal_g2.item():.4f})")


def test_aux_focal_warmup():
    """Test warmup behavior: focal only activates after warmup_steps."""
    B = 16
    ctr_logit = torch.randn(B, requires_grad=True)
    cvr_logit = torch.randn(B, requires_grad=True)
    y_ctr = torch.randint(0, 2, (B,)).float()
    y_ctcvr = torch.randint(0, 2, (B,)).float()
    
    outputs = {"ctr": ctr_logit, "cvr": cvr_logit}
    labels = {"y_ctr": y_ctr, "y_ctcvr": y_ctcvr, "y_cvr": torch.zeros(B)}
    batch = {"labels": labels}
    
    # Case 1: enabled=false (baseline)
    loss_fn_disabled = MultiTaskBCELoss(
        use_esmm=True,
        esmm_version="v2",
        static_pos_weight_ctr=1.0,
        static_pos_weight_ctcvr=1.0,
        aux_focal_enabled=False,
        aux_focal_warmup_steps=1000,
        aux_focal_lambda=0.1,
        aux_focal_gamma=1.0,
        global_step=2000,  # > warmup, but disabled
    )
    loss_baseline, dict_baseline = loss_fn_disabled.compute(outputs, batch)
    
    # Case 2: enabled=true, but step < warmup (should equal baseline)
    loss_fn_warmup = MultiTaskBCELoss(
        use_esmm=True,
        esmm_version="v2",
        static_pos_weight_ctr=1.0,
        static_pos_weight_ctcvr=1.0,
        aux_focal_enabled=True,
        aux_focal_warmup_steps=1000,
        aux_focal_lambda=0.1,
        aux_focal_gamma=1.0,
        global_step=500,  # < warmup
    )
    loss_warmup, dict_warmup = loss_fn_warmup.compute(outputs, batch)
    
    # Should be identical (within numerical tolerance)
    assert torch.allclose(loss_warmup, loss_baseline, atol=1e-6), "Warmup phase should match baseline"
    assert dict_warmup.get("aux_focal_on", False) == False, "Focal should not be active during warmup"
    
    # Case 3: enabled=true, step >= warmup (should include focal)
    loss_fn_active = MultiTaskBCELoss(
        use_esmm=True,
        esmm_version="v2",
        static_pos_weight_ctr=1.0,
        static_pos_weight_ctcvr=1.0,
        aux_focal_enabled=True,
        aux_focal_warmup_steps=1000,
        aux_focal_lambda=0.1,
        aux_focal_gamma=1.0,
        aux_focal_log_components=True,
        global_step=2000,  # >= warmup
    )
    loss_active, dict_active = loss_fn_active.compute(outputs, batch)
    
    assert dict_active.get("aux_focal_on", False) == True, "Focal should be active after warmup"
    assert "loss_ctcvr_bce" in dict_active, "Should log BCE component"
    assert "loss_ctcvr_focal" in dict_active, "Should log focal component"
    assert dict_active["loss_ctcvr_focal"] > 0, "Focal loss should be positive"
    
    # Total CTCVR loss should be different from baseline (focal adds extra term)
    loss_ctcvr_baseline = dict_baseline["loss_ctcvr"]
    loss_ctcvr_active = dict_active["loss_ctcvr"]
    assert loss_ctcvr_active != loss_ctcvr_baseline, "Active focal should change CTCVR loss"
    
    print(f"✓ Warmup test passed:")
    print(f"  - Baseline (disabled): loss_ctcvr={loss_ctcvr_baseline:.4f}")
    print(f"  - Warmup phase: loss_ctcvr={dict_warmup['loss_ctcvr']:.4f} (should equal baseline)")
    print(f"  - Active phase: loss_ctcvr={loss_ctcvr_active:.4f} (BCE={dict_active['loss_ctcvr_bce']:.4f} + Focal={dict_active['loss_ctcvr_focal']:.4f})")


def test_aux_focal_amp_stable():
    """Test numerical stability under AMP."""
    B = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ctr_logit = torch.randn(B, device=device, dtype=torch.float16, requires_grad=True)
    cvr_logit = torch.randn(B, device=device, dtype=torch.float16, requires_grad=True)
    y_ctr = torch.randint(0, 2, (B,), device=device).float()
    y_ctcvr = torch.randint(0, 2, (B,), device=device).float()
    
    outputs = {"ctr": ctr_logit, "cvr": cvr_logit}
    labels = {"y_ctr": y_ctr, "y_ctcvr": y_ctcvr, "y_cvr": torch.zeros(B, device=device)}
    batch = {"labels": labels}
    
    loss_fn = MultiTaskBCELoss(
        use_esmm=True,
        esmm_version="v2",
        static_pos_weight_ctr=1.0,
        static_pos_weight_ctcvr=1.0,
        aux_focal_enabled=True,
        aux_focal_warmup_steps=0,  # immediately active
        aux_focal_lambda=0.1,
        aux_focal_gamma=2.0,
        aux_focal_compute_fp32=True,  # critical for AMP stability
        global_step=100,
    )
    
    loss, loss_dict = loss_fn.compute(outputs, batch)
    
    assert not torch.isnan(loss), "Loss should not be NaN under AMP"
    assert not torch.isinf(loss), "Loss should not be Inf under AMP"
    
    # Backward should work
    loss.backward()
    assert ctr_logit.grad is not None, "Should have gradients"
    
    print(f"✓ AMP stability test passed (device={device}, loss={loss.item():.4f})")


def test_backward_compatibility():
    """Test that missing aux_focal config doesn't crash."""
    B = 16
    ctr_logit = torch.randn(B, requires_grad=True)
    cvr_logit = torch.randn(B, requires_grad=True)
    y_ctr = torch.randint(0, 2, (B,)).float()
    y_ctcvr = torch.randint(0, 2, (B,)).float()
    
    outputs = {"ctr": ctr_logit, "cvr": cvr_logit}
    labels = {"y_ctr": y_ctr, "y_ctcvr": y_ctcvr, "y_cvr": torch.zeros(B)}
    batch = {"labels": labels}
    
    # Old-style initialization without aux_focal params (should default to disabled)
    loss_fn_legacy = MultiTaskBCELoss(
        use_esmm=True,
        esmm_version="v2",
        static_pos_weight_ctr=1.0,
        static_pos_weight_ctcvr=1.0,
        # No aux_focal params provided
    )
    
    loss, loss_dict = loss_fn_legacy.compute(outputs, batch)
    assert not torch.isnan(loss), "Legacy initialization should not crash"
    assert loss_dict.get("aux_focal_on", False) == False, "Should default to disabled"
    
    print(f"✓ Backward compatibility test passed (loss={loss.item():.4f})")


if __name__ == "__main__":
    print("=== Running Aux Focal Smoke Tests ===\n")
    
    test_focal_on_logits_aux_basic()
    test_aux_focal_warmup()
    test_aux_focal_amp_stable()
    test_backward_compatibility()
    
    print("\n=== All tests passed! ===")
