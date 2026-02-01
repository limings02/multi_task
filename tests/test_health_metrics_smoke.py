"""
Quick smoke test for health metrics functionality.
Validates that the health metrics computation works correctly without full training.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.utils.health_metrics import (
    compute_label_counts,
    compute_logit_stats,
    compute_prob_stats,
    aggregate_gate_metrics,
)

def test_label_counts():
    """Test label counts and reliability flags."""
    print("Testing compute_label_counts...")
    
    # Create mock data
    n = 1000
    y_ctr = torch.zeros(n)
    y_ctr[:70] = 1.0  # 7% CTR
    
    y_cvr = torch.zeros(n)
    y_cvr[:20] = 1.0  # 20 CVR positives in clicked subset
    
    click_mask = torch.zeros(n)
    click_mask[:70] = 1.0  # 70 clicks
    
    y_ctcvr = torch.zeros(n)
    y_ctcvr[:5] = 1.0  # 5 CTCVR positives
    
    result = compute_label_counts(y_ctr, y_cvr, click_mask, y_ctcvr)
    
    print(f"  ctr_pos: {result['ctr_pos']} (expected: 70)")
    print(f"  ctr_pos_rate: {result['ctr_pos_rate']:.4f} (expected: 0.07)")
    print(f"  cvr_pos_masked: {result['cvr_pos_masked']} (expected: 20)")
    print(f"  cvr_pos_rate_masked: {result['cvr_pos_rate_masked']:.4f}")
    print(f"  ctcvr_pos: {result['ctcvr_pos']} (expected: 5)")
    print(f"  ctcvr_pos_rate: {result['ctcvr_pos_rate']:.6f}")
    print(f"  masked_n: {result['masked_n']} (expected: 70)")
    print(f"  cvr_metric_reliable: {result['cvr_metric_reliable']} (expected: False, needs >=100)")
    print(f"  ctcvr_metric_reliable: {result['ctcvr_metric_reliable']} (expected: False, needs >=100)")
    
    assert result['ctr_pos'] == 70
    assert result['masked_n'] == 70
    assert result['cvr_metric_reliable'] == False
    assert result['ctcvr_metric_reliable'] == False
    print("  ✓ PASSED\n")


def test_logit_stats():
    """Test logit distribution statistics."""
    print("Testing compute_logit_stats...")
    
    # Create logits with known distribution
    logits = torch.randn(10000) * 2 + 1  # mean ~1, std ~2
    
    result = compute_logit_stats(logits, "logit_ctr")
    
    print(f"  logit_ctr_mean: {result['logit_ctr_mean']:.4f}")
    print(f"  logit_ctr_std: {result['logit_ctr_std']:.4f}")
    print(f"  logit_ctr_min: {result['logit_ctr_min']:.4f}")
    print(f"  logit_ctr_max: {result['logit_ctr_max']:.4f}")
    print(f"  logit_ctr_p01: {result['logit_ctr_p01']:.4f}")
    print(f"  logit_ctr_p99: {result['logit_ctr_p99']:.4f}")
    
    assert result['logit_ctr_mean'] is not None
    assert result['logit_ctr_std'] is not None
    assert result['logit_ctr_p01'] < result['logit_ctr_p99']
    print("  ✓ PASSED\n")


def test_prob_stats():
    """Test probability distribution statistics."""
    print("Testing compute_prob_stats...")
    
    # Create probabilities
    probs = torch.sigmoid(torch.randn(5000))
    
    result = compute_prob_stats(probs, "prob_ctcvr")
    
    print(f"  prob_ctcvr_mean: {result['prob_ctcvr_mean']:.4f}")
    print(f"  prob_ctcvr_p01: {result['prob_ctcvr_p01']:.4f}")
    print(f"  prob_ctcvr_p99: {result['prob_ctcvr_p99']:.4f}")
    
    assert result['prob_ctcvr_mean'] is not None
    assert 0 <= result['prob_ctcvr_mean'] <= 1
    assert result['prob_ctcvr_p01'] < result['prob_ctcvr_p99']
    print("  ✓ PASSED\n")


def test_gate_metrics():
    """Test MMoE gate health metrics."""
    print("Testing aggregate_gate_metrics...")
    
    num_experts = 4
    batch_size = 256
    num_batches = 10
    
    # Create mock gate probabilities (softmax outputs)
    gate_probs_list = []
    for _ in range(num_batches):
        logits = torch.randn(batch_size, num_experts)
        probs = torch.softmax(logits, dim=-1)
        gate_probs_list.append(probs)
    
    result = aggregate_gate_metrics(gate_probs_list, "ctr")
    
    print(f"  gate_ctr_mean: {result['gate_ctr_mean']}")
    print(f"  gate_ctr_entropy_mean: {result['gate_ctr_entropy_mean']:.4f}")
    print(f"  gate_ctr_top1_share: {result['gate_ctr_top1_share']}")
    
    assert result['gate_ctr_mean'] is not None
    assert len(result['gate_ctr_mean']) == num_experts
    assert result['gate_ctr_entropy_mean'] is not None
    assert result['gate_ctr_top1_share'] is not None
    assert len(result['gate_ctr_top1_share']) == num_experts
    
    # Verify top1_share sums to ~1.0
    top1_sum = sum(result['gate_ctr_top1_share'])
    print(f"  top1_share sum: {top1_sum:.4f} (expected: ~1.0)")
    assert 0.99 <= top1_sum <= 1.01
    
    # Verify mean gate weights sum to ~1.0 (softmax property)
    mean_sum = sum(result['gate_ctr_mean'])
    print(f"  gate_mean sum: {mean_sum:.4f} (expected: ~1.0)")
    assert 0.99 <= mean_sum <= 1.01
    
    print("  ✓ PASSED\n")


def test_mmoe_log_gates():
    """Test that MMoE model returns gate weights when log_gates=True."""
    print("Testing MMoE log_gates functionality...")
    
    from src.utils.config import load_yaml
    from src.models.build import build_model
    
    cfg = load_yaml("configs/experiments/smoke_test_health_metrics.yaml")
    model = build_model(cfg)
    
    # Check log_gates is enabled
    assert hasattr(model, 'log_gates')
    assert model.log_gates == True
    print(f"  model.log_gates: {model.log_gates}")
    
    # Create dummy input
    from src.utils.feature_meta import build_model_feature_meta
    from pathlib import Path
    
    feature_meta = build_model_feature_meta(
        Path(cfg["data"]["metadata_path"]),
        cfg.get("embedding", {})
    )
    
    # Create minimal features dict
    batch_size = 4
    fields = {}
    for fname, fmeta in feature_meta.items():
        # Simple: one token per sample
        indices = torch.randint(0, fmeta.get("vocab_size", 100), (batch_size,))
        offsets = torch.arange(batch_size + 1)
        fields[fname] = {
            "indices": indices,
            "offsets": offsets,
            "weights": None,
        }
    
    features = {"fields": fields, "field_names": list(feature_meta.keys())}
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    
    # Check gate weights are in outputs
    assert "aux" in outputs, "Expected 'aux' in model outputs"
    assert "gates" in outputs["aux"], "Expected 'gates' in aux outputs"
    
    gates = outputs["aux"]["gates"]
    print(f"  Gate tasks returned: {list(gates.keys())}")
    
    assert "ctr" in gates, "Expected 'ctr' gate weights"
    assert "cvr" in gates, "Expected 'cvr' gate weights"
    
    ctr_gate = gates["ctr"]
    print(f"  CTR gate shape: {ctr_gate.shape} (expected: [batch_size, num_experts])")
    assert ctr_gate.shape[0] == batch_size
    assert ctr_gate.shape[1] == cfg["model"]["mmoe"]["num_experts"]
    
    # Verify gate weights are valid softmax (sum to 1)
    gate_sum = ctr_gate.sum(dim=-1)
    print(f"  CTR gate row sums: {gate_sum.tolist()}")
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5)
    
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Health Metrics Smoke Test")
    print("=" * 60 + "\n")
    
    test_label_counts()
    test_logit_stats()
    test_prob_stats()
    test_gate_metrics()
    # test_mmoe_log_gates() - requires proper data loader
    
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
