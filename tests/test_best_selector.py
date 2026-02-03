"""
Unit test for BestSelector

Tests various gate scenarios:
1. Primary improvement insufficient
2. Auxiliary metric degradation
3. Gate passes successfully
4. Confirmation requirement
5. Cooldown period
6. Moving average
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.train.best_selector import BestSelector
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_best_selector")


def test_auc_primary_strategy():
    """Test legacy auc_primary strategy."""
    print("\n" + "=" * 80)
    print("TEST 1: Legacy auc_primary strategy")
    print("=" * 80)
    
    selector = BestSelector(strategy="auc_primary", logger=logger, log_decision=True)
    
    test_cases = [
        ({"auc_primary": 0.75}, False, "Initial baseline"),
        ({"auc_primary": 0.76}, True, "Improvement -> update"),
        ({"auc_primary": 0.755}, False, "Worse than best -> no update"),
        ({"auc_primary": 0.77}, True, "Better than best -> update"),
    ]
    
    for i, (metrics, expected, description) in enumerate(test_cases):
        should_update, info = selector.should_update_best(metrics, step=i)
        status = "✓" if should_update == expected else "✗"
        print(f"{status} Step {i}: {description} | update={should_update} (expected={expected})")
        assert should_update == expected, f"Test failed at step {i}: {description}"
    
    print("✓ All auc_primary tests passed\n")


def test_gate_primary_insufficient():
    """Test gate strategy when primary improvement is insufficient."""
    print("=" * 80)
    print("TEST 2: Gate strategy - primary improvement insufficient")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.002,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        logger=logger,
        log_decision=True,
    )
    
    # Establish baseline
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    should_update, info = selector.should_update_best(metrics, step=0)
    print(f"  Baseline: {should_update} (first eval always updates)")
    
    # Primary improves by only 0.001 (< tol_primary=0.002)
    metrics = {"auc_ctcvr": 0.701, "auc_ctr": 0.65, "auc_cvr": 0.55}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  Small improve: {should_update} (expected False)")
    assert should_update is False, "Should not update when primary improvement < tol_primary"
    assert "insufficient" in info.get("reason", "").lower(), "Reason should mention insufficient improvement"
    
    print("✓ Primary insufficient test passed\n")


def test_gate_aux_degrades():
    """Test gate strategy when auxiliary metric degrades too much."""
    print("=" * 80)
    print("TEST 3: Gate strategy - auxiliary metric degrades")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.002,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        logger=logger,
        log_decision=True,
    )
    
    # Establish baseline
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)
    
    # Primary improves enough, but CTR drops by 0.01 (> tol_ctr=0.003)
    metrics = {"auc_ctcvr": 0.705, "auc_ctr": 0.640, "auc_cvr": 0.55}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  CTR degrades: {should_update} (expected False)")
    assert should_update is False, "Should not update when aux degrades beyond tolerance"
    assert "degraded" in info.get("reason", "").lower(), "Reason should mention degradation"
    
    print("✓ Aux degradation test passed\n")


def test_gate_passes():
    """Test gate strategy when all conditions pass."""
    print("=" * 80)
    print("TEST 4: Gate strategy - gate passes")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.002,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        logger=logger,
        log_decision=True,
    )
    
    # Establish baseline
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)
    
    # All metrics improve
    metrics = {"auc_ctcvr": 0.703, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  All good: {should_update} (expected True)")
    assert should_update is True, "Should update when gate passes"
    assert info.get("ok_primary") is True
    assert info.get("all_aux_ok") is True
    
    print("✓ Gate pass test passed\n")


def test_confirmation():
    """Test confirmation requirement."""
    print("=" * 80)
    print("TEST 5: Gate with confirmation requirement (confirm_times=2)")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.001,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        confirm_times=2,
        logger=logger,
        log_decision=True,
    )
    
    # Establish baseline
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)
    
    # First pass
    metrics = {"auc_ctcvr": 0.702, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  1st pass: {should_update} (expected False, confirm 1/2)")
    assert should_update is False, "Should not update on first confirmation"
    assert info.get("confirm_count") == 1
    
    # Second pass
    metrics = {"auc_ctcvr": 0.703, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=2)
    print(f"  2nd pass: {should_update} (expected True, confirmed)")
    assert should_update is True, "Should update after consecutive confirmations"
    
    print("✓ Confirmation test passed\n")


def test_cooldown():
    """Test cooldown period."""
    print("=" * 80)
    print("TEST 6: Gate with cooldown (cooldown_evals=2)")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.001,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        cooldown_evals=2,
        logger=logger,
        log_decision=True,
    )
    
    # Establish baseline and trigger update
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)
    metrics = {"auc_ctcvr": 0.705, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  Trigger update: {should_update} (expected True)")
    assert should_update is True
    
    # During cooldown
    metrics = {"auc_ctcvr": 0.710, "auc_ctr": 0.652, "auc_cvr": 0.552}
    should_update, info = selector.should_update_best(metrics, step=2)
    print(f"  Cooldown 1: {should_update} (expected False)")
    assert should_update is False, "Should not update during cooldown"
    assert "cooldown" in info.get("reason", "").lower()
    
    metrics = {"auc_ctcvr": 0.715, "auc_ctr": 0.653, "auc_cvr": 0.553}
    should_update, info = selector.should_update_best(metrics, step=3)
    print(f"  Cooldown 2: {should_update} (expected False)")
    assert should_update is False, "Should not update during cooldown"
    
    # After cooldown
    metrics = {"auc_ctcvr": 0.720, "auc_ctr": 0.654, "auc_cvr": 0.554}
    should_update, info = selector.should_update_best(metrics, step=4)
    print(f"  After cooldown: {should_update} (expected True)")
    assert should_update is True, "Should update after cooldown expires"
    
    print("✓ Cooldown test passed\n")


def test_moving_average():
    """Test moving average for primary metric."""
    print("=" * 80)
    print("TEST 7: Gate with moving average (ma_window=3)")
    print("=" * 80)
    
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        use_primary_ma=True,
        ma_window=3,
        tol_primary=0.001,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        logger=logger,
        log_decision=True,
    )
    
    # Build up history
    for i, val in enumerate([0.70, 0.705, 0.710]):
        metrics = {"auc_ctcvr": val, "auc_ctr": 0.65, "auc_cvr": 0.55}
        selector.should_update_best(metrics, step=i)
    
    # Current MA should be ~0.705
    # A spike to 0.720 raw will have MA = (0.705 + 0.710 + 0.720)/3 = 0.7117
    metrics = {"auc_ctcvr": 0.720, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=3)
    
    primary_ma = info.get("primary_ma")
    print(f"  Spike with MA: update={should_update}, raw=0.720, MA={primary_ma:.6f}")
    assert primary_ma is not None, "Should have primary_ma in info"
    assert 0.711 < primary_ma < 0.712, f"MA should be ~0.7117, got {primary_ma}"
    
    print("✓ Moving average test passed\n")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("BEST SELECTOR UNIT TESTS")
    print("=" * 80 + "\n")
    
    try:
        test_auc_primary_strategy()
        test_gate_primary_insufficient()
        test_gate_aux_degrades()
        test_gate_passes()
        test_confirmation()
        test_cooldown()
        test_moving_average()
        
        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 80 + "\n")
        return 0
    
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"✗✗✗ TEST FAILED: {e}")
        print("=" * 80 + "\n")
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗✗✗ UNEXPECTED ERROR: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
