"""
Unit tests for GradConflictSampler.

Covers:
- init_epoch interval computation
- should_sample stepping logic
- record_sample with dense + sparse metrics (ESMM and non-ESMM)
- record_sample skips when task2 gradient is absent
- record_error counting
- aggregate returns all documented fields
- EMA correctness
- Minimum target clamping (< 500 → 500)
- Empty epoch (no samples) produces all-None metrics
- grad_norm_shared_cvr_mean vs ctcvr_mean correct assignment
"""

from __future__ import annotations

import math
import pytest
from src.train.grad_conflict_sampler import GradConflictSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diag_metrics(
    *,
    cos_dense: float = 0.5,
    cos_sparse: float = 0.3,
    cos_all: float = 0.4,
    norm_ctr: float = 1.0,
    norm_task2: float | None = 0.8,
    task2_name: str = "cvr",
    dense_count: int = 100,
    sparse_count: int = 50,
) -> dict:
    """Build a dict matching GradientDiagnostics.compute_metrics() output."""
    d = {
        "grad_cosine_shared_dense": cos_dense,
        "grad_cosine_shared_sparse": cos_sparse,
        "grad_cosine_shared_all": cos_all,
        "grad_norm_shared_ctr": norm_ctr,
        "shared_dense_count": dense_count,
        "shared_sparse_count": sparse_count,
    }
    d[f"grad_norm_shared_{task2_name}"] = norm_task2
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitEpoch:
    """Test init_epoch and should_sample logic."""

    def test_interval_computation(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(expected_steps=39000)
        assert s.sample_interval == 39  # 39000 // 1000

    def test_interval_small_epoch(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(expected_steps=500)
        # 500 // 1000 = 0 → clamped to 1
        assert s.sample_interval == 1

    def test_interval_exact(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(expected_steps=1000)
        assert s.sample_interval == 1

    def test_should_sample_steps(self):
        s = GradConflictSampler(grad_samples_target=500)
        s.init_epoch(expected_steps=5000)
        # interval = 10
        assert s.should_sample(0)
        assert not s.should_sample(1)
        assert not s.should_sample(9)
        assert s.should_sample(10)
        assert s.should_sample(20)


class TestMinimumTarget:
    """Test that targets below 500 get clamped."""

    def test_clamp_to_500(self):
        s = GradConflictSampler(grad_samples_target=100)
        assert s.grad_samples_target == 500

    def test_exactly_500_not_clamped(self):
        s = GradConflictSampler(grad_samples_target=500)
        assert s.grad_samples_target == 500


class TestRecordSample:
    """Test record_sample accumulates correctly."""

    def test_valid_sample(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        metrics = _make_diag_metrics(cos_all=0.5, norm_ctr=1.0, norm_task2=0.8)
        s.record_sample(metrics, global_step=0, use_esmm=False)
        assert s._collected == 1
        assert s._skipped_no_secondary_grad == 0

    def test_skip_no_task2_grad(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        metrics = _make_diag_metrics(norm_task2=None)
        s.record_sample(metrics, global_step=0, use_esmm=False)
        assert s._collected == 0
        assert s._skipped_no_secondary_grad == 1

    def test_skip_zero_task2_grad(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        metrics = _make_diag_metrics(norm_task2=0.0)
        s.record_sample(metrics, global_step=0, use_esmm=False)
        assert s._collected == 0
        assert s._skipped_no_secondary_grad == 1

    def test_esmm_mode_uses_ctcvr(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        metrics = _make_diag_metrics(
            norm_task2=0.5, task2_name="ctcvr"
        )
        s.record_sample(metrics, global_step=0, use_esmm=True)
        assert s._task2_name == "ctcvr"
        assert s._collected == 1

    def test_conflict_detection(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        metrics = _make_diag_metrics(cos_dense=-0.1, cos_sparse=-0.2, cos_all=-0.15)
        s.record_sample(metrics, global_step=0, use_esmm=False)
        assert s._conflict_hits == 1
        assert s._conflict_hits_dense == 1
        assert s._conflict_hits_sparse == 1


class TestRecordError:
    """Test error counting."""

    def test_error_count(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_error(global_step=0)
        s.record_error(global_step=1)
        assert s._error_count == 2


class TestEMA:
    """Test EMA computation for conflict rate."""

    def test_ema_first_sample(self):
        s = GradConflictSampler(grad_samples_target=1000, conflict_ema_alpha=0.1)
        s.init_epoch(1000)
        # First sample: conflict (cos_all < 0)
        metrics = _make_diag_metrics(cos_all=-0.5)
        s.record_sample(metrics, global_step=0)
        assert s._conflict_ema == 1.0  # first sample initializes

    def test_ema_second_sample_no_conflict(self):
        s = GradConflictSampler(grad_samples_target=1000, conflict_ema_alpha=0.1)
        s.init_epoch(1000)
        # First: conflict
        s.record_sample(_make_diag_metrics(cos_all=-0.5), global_step=0)
        # Second: no conflict
        s.record_sample(_make_diag_metrics(cos_all=0.5), global_step=1)
        # EMA = 0.1 * 0.0 + 0.9 * 1.0 = 0.9
        assert abs(s._conflict_ema - 0.9) < 1e-9

    def test_ema_converges(self):
        s = GradConflictSampler(grad_samples_target=1000, conflict_ema_alpha=0.1)
        s.init_epoch(10000)
        # 100 non-conflict samples => EMA → 0
        for i in range(100):
            s.record_sample(_make_diag_metrics(cos_all=0.5), global_step=i)
        assert s._conflict_ema < 1e-5


class TestAggregate:
    """Test aggregate produces all required fields."""

    def test_empty_aggregate(self):
        """No samples → all metrics None except metadata."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        result = s.aggregate(epoch=0, global_step=0)
        assert result["grad_samples_collected"] == 0
        assert result["grad_cosine_shared_mean"] is None
        assert result["conflict_rate"] is None
        assert result["grad_norm_shared_ctr_mean"] is None
        assert result["global_step"] == 0

    def test_field_completeness(self):
        """All documented fields exist in the aggregate output."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(), global_step=0)
        result = s.aggregate(epoch=0, global_step=999)

        required_fields = [
            "global_step",
            "grad_samples_target",
            "grad_samples_collected",
            "grad_samples_skipped_no_cvr_grad",
            "grad_sample_interval_steps",
            "grad_conflict_error_count",
            "grad_samples",
            "grad_norm_shared_ctr_mean",
            "grad_norm_shared_cvr_mean",
            "grad_norm_shared_ctcvr_mean",
            "grad_norm_ratio_mean",
            "grad_norm_ratio_p50",
            "grad_norm_ratio_p90",
            "grad_cosine_shared_dense_mean",
            "grad_cosine_dense_p10",
            "grad_cosine_dense_p50",
            "grad_cosine_dense_p90",
            "grad_cosine_shared_sparse_mean",
            "grad_cosine_sparse_p10",
            "grad_cosine_sparse_p50",
            "grad_cosine_sparse_p90",
            "grad_cosine_shared_mean",
            "grad_cosine_p10",
            "grad_cosine_p50",
            "grad_cosine_p90",
            "conflict_rate",
            "conflict_rate_ema",
            "neg_conflict_strength_mean",
            "shared_dense_count",
            "shared_sparse_count",
            "cvr_has_grad",
            "cvr_valid_count",
        ]
        for f in required_fields:
            assert f in result, f"Missing field: {f}"

    def test_non_esmm_norm_fields(self):
        """In non-ESMM mode, cvr_mean is populated, ctcvr_mean is None."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(
            norm_ctr=1.0, norm_task2=0.8, task2_name="cvr"
        ), global_step=0, use_esmm=False)
        result = s.aggregate(epoch=0, global_step=999)
        assert result["grad_norm_shared_cvr_mean"] == pytest.approx(0.8)
        assert result["grad_norm_shared_ctcvr_mean"] is None
        assert result["grad_norm_shared_ctr_mean"] == pytest.approx(1.0)

    def test_esmm_norm_fields(self):
        """In ESMM mode, ctcvr_mean is populated, cvr_mean is None."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(
            norm_ctr=1.0, norm_task2=0.6, task2_name="ctcvr"
        ), global_step=0, use_esmm=True)
        result = s.aggregate(epoch=0, global_step=999)
        assert result["grad_norm_shared_cvr_mean"] is None
        assert result["grad_norm_shared_ctcvr_mean"] == pytest.approx(0.6)

    def test_neg_conflict_strength(self):
        """neg_conflict_strength_mean should be average of negative cosines."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(cos_all=-0.2), global_step=0)
        s.record_sample(_make_diag_metrics(cos_all=0.5), global_step=1)
        s.record_sample(_make_diag_metrics(cos_all=-0.4), global_step=2)
        result = s.aggregate(epoch=0, global_step=999)
        # neg cosines: [-0.2, -0.4], mean = -0.3
        assert result["neg_conflict_strength_mean"] == pytest.approx(-0.3)

    def test_conflict_rate(self):
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(cos_all=-0.1), global_step=0)
        s.record_sample(_make_diag_metrics(cos_all=0.3), global_step=1)
        s.record_sample(_make_diag_metrics(cos_all=-0.5), global_step=2)
        s.record_sample(_make_diag_metrics(cos_all=0.7), global_step=3)
        result = s.aggregate(epoch=0, global_step=999)
        # 2 conflicts / 4 samples = 0.5
        assert result["conflict_rate"] == pytest.approx(0.5)

    def test_norm_ratio_percentiles(self):
        """Norm ratio p50/p90 should be close to expected for uniform values."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        # All ratio = 0.8/1.0 = 0.8
        for i in range(100):
            s.record_sample(_make_diag_metrics(
                norm_ctr=1.0, norm_task2=0.8
            ), global_step=i)
        result = s.aggregate(epoch=0, global_step=999)
        assert result["grad_norm_ratio_p50"] == pytest.approx(0.8, abs=0.01)
        assert result["grad_norm_ratio_p90"] == pytest.approx(0.8, abs=0.01)

    def test_insufficient_samples_warning(self):
        """When collected < target, aggregate should include reason."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(100)  # only 100 steps → can't reach 1000
        for i in range(100):
            s.record_sample(_make_diag_metrics(), global_step=i)
        result = s.aggregate(epoch=0, global_step=99)
        # collected = 100 which is < target=1000
        assert "grad_samples_insufficient_reason" in result

    def test_init_epoch_resets_state(self):
        """Calling init_epoch again clears previous state."""
        s = GradConflictSampler(grad_samples_target=1000)
        s.init_epoch(1000)
        s.record_sample(_make_diag_metrics(), global_step=0)
        assert s._collected == 1

        s.init_epoch(2000)
        assert s._collected == 0
        assert s._conflict_ema is None
        assert s._error_count == 0


class TestSimulatedTraining:
    """End-to-end simulation: run a simulated epoch and verify metrics."""

    def test_simulated_epoch_reaches_target(self):
        target = 1000
        total_steps = 40000
        s = GradConflictSampler(grad_samples_target=target)
        s.init_epoch(total_steps)

        collected = 0
        for step in range(total_steps):
            if s.should_sample(step):
                s.record_sample(
                    _make_diag_metrics(
                        cos_all=0.1 * (step % 10 - 5),
                        task2_name="ctcvr",
                    ),
                    global_step=step,
                    use_esmm=True,
                )
                collected += 1

        result = s.aggregate(epoch=0, global_step=total_steps - 1)
        assert result["grad_samples_collected"] == collected
        # Should be close to target (within 5%)
        assert collected >= target * 0.95
        assert result["conflict_rate"] is not None
        assert result["conflict_rate_ema"] is not None
        assert result["grad_cosine_p50"] is not None
