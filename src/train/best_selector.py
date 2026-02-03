"""
BestSelector: Configurable best model selection strategy for multi-task learning.

Supports two strategies:
1. auc_primary (legacy): Simple weighted combination of metrics
2. gate: Primary metric must improve while auxiliary metrics don't degrade
"""
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import logging


class BestSelector:
    """
    Determines whether to update the best checkpoint based on configurable strategies.
    
    Strategy 'auc_primary' (default, backward-compatible):
        - Uses a single scalar metric (auc_primary) for comparison
        - Updates best if current > best
    
    Strategy 'gate':
        - Primary metric (e.g., auc_ctcvr) must improve by at least tol_primary
        - Auxiliary metrics (e.g., auc_ctr, auc_cvr) must not degrade beyond their tolerance
        - Optionally uses moving average for primary metric
        - Optionally requires consecutive confirmations
        - Optionally enforces cooldown period after each best update
    """
    
    def __init__(
        self,
        strategy: str = "auc_primary",
        primary_key: str = "auc_ctcvr",
        aux_keys: Optional[List[str]] = None,
        use_primary_ma: bool = False,
        ma_window: int = 5,
        tol_primary: float = 0.0,
        tol_aux: Optional[Dict[str, float]] = None,
        confirm_times: int = 1,
        cooldown_evals: int = 0,
        log_decision: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            strategy: "auc_primary" or "gate"
            primary_key: Key for primary metric (e.g., "auc_ctcvr")
            aux_keys: List of auxiliary metric keys (e.g., ["auc_ctr", "auc_cvr"])
            use_primary_ma: Whether to use moving average for primary metric
            ma_window: Window size for moving average
            tol_primary: Minimum improvement required for primary metric
            tol_aux: Dict mapping aux_key -> max allowed degradation (e.g., {"auc_ctr": 0.003})
            confirm_times: Number of consecutive passes required before updating best
            cooldown_evals: Number of evals to skip after updating best
            log_decision: Whether to log decision details
            logger: Logger instance for outputting decision info
        """
        self.strategy = strategy
        self.primary_key = primary_key
        self.aux_keys = aux_keys or []
        self.use_primary_ma = use_primary_ma
        self.ma_window = ma_window
        self.tol_primary = tol_primary
        self.tol_aux = tol_aux or {}
        self.confirm_times = confirm_times
        self.cooldown_evals = cooldown_evals
        self.log_decision = log_decision
        self.logger = logger
        
        # State tracking
        self.best_primary: float = float("-inf")
        self.best_aux: Dict[str, float] = {k: float("-inf") for k in self.aux_keys}
        self.primary_history: deque = deque(maxlen=ma_window)
        self.confirm_count: int = 0
        self.cooldown_remaining: int = 0
        self.last_update_step: Optional[int] = None
        
        if self.strategy not in ["auc_primary", "gate"]:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'auc_primary' or 'gate'.")
        
        if self.strategy == "gate" and not self.primary_key:
            raise ValueError("primary_key must be specified for gate strategy.")
    
    def should_update_best(
        self, metrics: Dict[str, Any], step: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine whether to update the best checkpoint.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            step: Current global step
        
        Returns:
            (should_update, decision_info):
                - should_update: bool, whether to save as best
                - decision_info: dict with details for logging
        """
        if self.strategy == "auc_primary":
            return self._check_auc_primary(metrics, step)
        elif self.strategy == "gate":
            return self._check_gate(metrics, step)
        else:
            return False, {"error": f"Unknown strategy: {self.strategy}"}
    
    def _check_auc_primary(
        self, metrics: Dict[str, Any], step: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Legacy strategy: simple scalar comparison."""
        auc_primary = metrics.get("auc_primary")
        if auc_primary is None:
            return False, {
                "strategy": "auc_primary",
                "reason": "auc_primary not found in metrics",
                "should_update": False,
            }
        
        current = float(auc_primary)
        should_update = current > self.best_primary
        
        if should_update:
            self.best_primary = current
            self.last_update_step = step
        
        decision_info = {
            "strategy": "auc_primary",
            "current_auc_primary": current,
            "best_auc_primary": self.best_primary,
            "should_update": should_update,
            "step": step,
        }
        
        if self.log_decision and self.logger:
            if should_update:
                self.logger.info(
                    f"[BestSelector] NEW BEST: auc_primary={current:.6f} "
                    f"(prev={self.best_primary:.6f}) at step {step}"
                )
            else:
                self.logger.debug(
                    f"[BestSelector] No update: auc_primary={current:.6f} "
                    f"<= best={self.best_primary:.6f}"
                )
        
        return should_update, decision_info
    
    def _check_gate(
        self, metrics: Dict[str, Any], step: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Gate strategy: primary must improve, aux must not degrade.
        """
        decision_info: Dict[str, Any] = {
            "strategy": "gate",
            "step": step,
            "should_update": False,
        }
        
        # 1. Check cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            decision_info["reason"] = "cooldown_active"
            decision_info["cooldown_remaining"] = self.cooldown_remaining
            if self.log_decision and self.logger:
                self.logger.info(
                    f"[BestSelector] COOLDOWN: {self.cooldown_remaining} evals remaining, skipping update"
                )
            return False, decision_info
        
        # 2. Extract primary metric
        primary_raw = metrics.get(self.primary_key)
        if primary_raw is None:
            decision_info["reason"] = f"primary_key '{self.primary_key}' not found"
            if self.log_decision and self.logger:
                self.logger.warning(
                    f"[BestSelector] Missing primary metric '{self.primary_key}', skipping update"
                )
            return False, decision_info
        
        primary_raw = float(primary_raw)
        self.primary_history.append(primary_raw)
        
        # 3. Compute primary value (raw or MA)
        if self.use_primary_ma:
            primary_value = sum(self.primary_history) / len(self.primary_history)
            decision_info["primary_raw"] = primary_raw
            decision_info["primary_ma"] = primary_value
        else:
            primary_value = primary_raw
            decision_info["primary_value"] = primary_value
        
        decision_info["best_primary"] = self.best_primary
        
        # 4. Check primary improvement
        primary_delta = primary_value - self.best_primary
        ok_primary = primary_delta >= self.tol_primary
        decision_info["primary_delta"] = primary_delta
        decision_info["tol_primary"] = self.tol_primary
        decision_info["ok_primary"] = ok_primary
        
        if not ok_primary:
            self.confirm_count = 0  # Reset confirm
            decision_info["reason"] = f"primary improvement insufficient: {primary_delta:.6f} < {self.tol_primary:.6f}"
            if self.log_decision and self.logger:
                self.logger.info(
                    f"[BestSelector] GATE FAILED: primary {self.primary_key}={primary_value:.6f} "
                    f"(delta={primary_delta:.6f} < tol={self.tol_primary:.6f})"
                )
            return False, decision_info
        
        # 5. Check auxiliary metrics
        aux_status = {}
        all_aux_ok = True
        failed_aux_reasons = []
        
        for aux_key in self.aux_keys:
            aux_value = metrics.get(aux_key)
            if aux_value is None:
                # Missing aux key: fail gate by default for safety
                all_aux_ok = False
                aux_status[aux_key] = {
                    "value": None,
                    "best": self.best_aux.get(aux_key, float("-inf")),
                    "ok": False,
                    "reason": "missing",
                }
                failed_aux_reasons.append(f"{aux_key}=missing")
                continue
            
            aux_value = float(aux_value)
            best_aux = self.best_aux.get(aux_key, float("-inf"))
            tol_aux = self.tol_aux.get(aux_key, 0.0)
            aux_delta = aux_value - best_aux
            ok_aux = aux_delta >= -tol_aux
            
            aux_status[aux_key] = {
                "value": aux_value,
                "best": best_aux,
                "delta": aux_delta,
                "tol": tol_aux,
                "ok": ok_aux,
            }
            
            if not ok_aux:
                all_aux_ok = False
                failed_aux_reasons.append(
                    f"{aux_key}={aux_value:.6f} (delta={aux_delta:.6f} < -{tol_aux:.6f})"
                )
        
        decision_info["aux_status"] = aux_status
        decision_info["all_aux_ok"] = all_aux_ok
        
        if not all_aux_ok:
            self.confirm_count = 0  # Reset confirm
            decision_info["reason"] = f"aux metrics degraded: {', '.join(failed_aux_reasons)}"
            if self.log_decision and self.logger:
                self.logger.info(
                    f"[BestSelector] GATE FAILED: aux metrics degraded - {'; '.join(failed_aux_reasons)}"
                )
            return False, decision_info
        
        # 6. Gate passed: check confirmation requirement
        self.confirm_count += 1
        decision_info["confirm_count"] = self.confirm_count
        decision_info["confirm_required"] = self.confirm_times
        
        if self.confirm_count < self.confirm_times:
            decision_info["reason"] = f"confirmation pending: {self.confirm_count}/{self.confirm_times}"
            if self.log_decision and self.logger:
                self.logger.info(
                    f"[BestSelector] GATE PASSED but awaiting confirmation: "
                    f"{self.confirm_count}/{self.confirm_times}"
                )
            return False, decision_info
        
        # 7. All checks passed: update best
        self.best_primary = primary_value
        for aux_key, status in aux_status.items():
            if status["value"] is not None:
                self.best_aux[aux_key] = status["value"]
        
        self.confirm_count = 0  # Reset confirm
        self.cooldown_remaining = self.cooldown_evals
        self.last_update_step = step
        
        decision_info["should_update"] = True
        decision_info["reason"] = "gate_passed_confirmed"
        
        if self.log_decision and self.logger:
            aux_summary = ", ".join(
                f"{k}={v['value']:.6f}" for k, v in aux_status.items() if v["value"] is not None
            )
            self.logger.info(
                f"[BestSelector] NEW BEST: {self.primary_key}={primary_value:.6f} "
                f"(delta={primary_delta:+.6f}), aux=[{aux_summary}] at step {step}"
            )
        
        return True, decision_info
    
    def get_state(self) -> Dict[str, Any]:
        """Get internal state for checkpoint saving."""
        return {
            "best_primary": self.best_primary,
            "best_aux": self.best_aux,
            "primary_history": list(self.primary_history),
            "confirm_count": self.confirm_count,
            "cooldown_remaining": self.cooldown_remaining,
            "last_update_step": self.last_update_step,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.best_primary = state.get("best_primary", float("-inf"))
        self.best_aux = state.get("best_aux", {k: float("-inf") for k in self.aux_keys})
        hist = state.get("primary_history", [])
        self.primary_history = deque(hist, maxlen=self.ma_window)
        self.confirm_count = state.get("confirm_count", 0)
        self.cooldown_remaining = state.get("cooldown_remaining", 0)
        self.last_update_step = state.get("last_update_step")


# Self-test / example usage (not run during import)
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    print("=" * 80)
    print("BestSelector Self-Test")
    print("=" * 80)
    
    # Test 1: Legacy auc_primary strategy
    print("\n[Test 1] Legacy auc_primary strategy")
    selector = BestSelector(strategy="auc_primary", logger=logger, log_decision=True)
    
    test_metrics = [
        {"auc_primary": 0.75},
        {"auc_primary": 0.76},  # Should update
        {"auc_primary": 0.755}, # Should not update
        {"auc_primary": 0.77},  # Should update
    ]
    
    for i, m in enumerate(test_metrics):
        should_update, info = selector.should_update_best(m, step=i)
        print(f"  Step {i}: auc_primary={m['auc_primary']:.3f} -> update={should_update}")
    
    # Test 2: Gate strategy - primary not enough
    print("\n[Test 2] Gate strategy - primary improvement insufficient")
    selector = BestSelector(
        strategy="gate",
        primary_key="auc_ctcvr",
        aux_keys=["auc_ctr", "auc_cvr"],
        tol_primary=0.002,
        tol_aux={"auc_ctr": 0.003, "auc_cvr": 0.008},
        logger=logger,
        log_decision=True,
    )
    
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    should_update, info = selector.should_update_best(metrics, step=0)
    print(f"  Initial: update={should_update} (establishes baseline)")
    
    metrics = {"auc_ctcvr": 0.701, "auc_ctr": 0.65, "auc_cvr": 0.55}  # Only +0.001 < 0.002
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  Small improve: update={should_update} (expected False)")
    
    # Test 3: Gate strategy - aux degrades too much
    print("\n[Test 3] Gate strategy - aux metric degrades")
    metrics = {"auc_ctcvr": 0.705, "auc_ctr": 0.640, "auc_cvr": 0.55}  # ctr drops 0.01 > tol 0.003
    should_update, info = selector.should_update_best(metrics, step=2)
    print(f"  CTR degrades: update={should_update} (expected False, reason: {info.get('reason')})")
    
    # Test 4: Gate strategy - pass
    print("\n[Test 4] Gate strategy - gate passes")
    metrics = {"auc_ctcvr": 0.703, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=3)
    print(f"  All good: update={should_update} (expected True)")
    
    # Test 5: Confirmation requirement
    print("\n[Test 5] Gate with confirmation requirement (confirm_times=2)")
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
    
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)  # Baseline
    
    metrics = {"auc_ctcvr": 0.702, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  1st pass: update={should_update} (expected False, confirm 1/2)")
    
    metrics = {"auc_ctcvr": 0.703, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=2)
    print(f"  2nd pass: update={should_update} (expected True, confirmed)")
    
    # Test 6: Cooldown
    print("\n[Test 6] Gate with cooldown (cooldown_evals=2)")
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
    
    metrics = {"auc_ctcvr": 0.70, "auc_ctr": 0.65, "auc_cvr": 0.55}
    selector.should_update_best(metrics, step=0)
    metrics = {"auc_ctcvr": 0.705, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  Update triggers cooldown: update={should_update}")
    
    metrics = {"auc_ctcvr": 0.710, "auc_ctr": 0.652, "auc_cvr": 0.552}
    should_update, info = selector.should_update_best(metrics, step=2)
    print(f"  During cooldown 1: update={should_update} (expected False)")
    
    metrics = {"auc_ctcvr": 0.715, "auc_ctr": 0.653, "auc_cvr": 0.553}
    should_update, info = selector.should_update_best(metrics, step=3)
    print(f"  During cooldown 2: update={should_update} (expected False)")
    
    metrics = {"auc_ctcvr": 0.720, "auc_ctr": 0.654, "auc_cvr": 0.554}
    should_update, info = selector.should_update_best(metrics, step=4)
    print(f"  After cooldown: update={should_update} (expected True)")
    
    # Test 7: Moving average
    print("\n[Test 7] Gate with moving average (ma_window=3)")
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
    
    # Establish baseline with 3 values
    for val in [0.70, 0.705, 0.710]:
        metrics = {"auc_ctcvr": val, "auc_ctr": 0.65, "auc_cvr": 0.55}
        selector.should_update_best(metrics, step=0)
    # MA = (0.70 + 0.705 + 0.710) / 3 = 0.705
    
    # Spike up (raw=0.720) but MA will be (0.705 + 0.710 + 0.720)/3 = 0.7117
    metrics = {"auc_ctcvr": 0.720, "auc_ctr": 0.651, "auc_cvr": 0.551}
    should_update, info = selector.should_update_best(metrics, step=1)
    print(f"  Spike with MA: update={should_update}, MA={info.get('primary_ma', 0):.6f}")
    
    print("\n" + "=" * 80)
    print("Self-test complete. Review logs above to verify gate logic.")
    print("=" * 80)
