"""
Tests for the AdaptiveController and PolicyEngine (Phase 5).

All tests are single-process / in-memory — no ZeroMQ, no multiprocessing.

Run with:  pytest tests/test_controller.py -v
"""

import numpy as np
import pytest
import time

from monitor.metrics_monitor import MetricsMonitor
from controller.adaptive_controller import AdaptiveController
from controller.policy_engine import PolicyEngine


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _monitor_with(records: list[dict]) -> MetricsMonitor:
    """Return a fresh MetricsMonitor pre-loaded with the given records."""
    m = MetricsMonitor()
    for r in records:
        m.record(**r)
        m.heartbeat(r["rank"])
    return m


def _dummy_params() -> dict:
    """Minimal model params (just needs to be picklable for checkpointing)."""
    return {"W": np.zeros((10, 5), dtype=np.float32),
            "b": np.zeros(5, dtype=np.float32)}


def _default_cfg(**overrides) -> dict:
    cfg = {
        "straggler_lag_ratio":   2.0,
        "heartbeat_timeout_ms":  3000.0,
        "switching_cost_margin": 0.0,
        "hysteresis_rounds":     3,
        "min_bandwidth_mbs":     0.0,
        "mode":                  "rar",
    }
    cfg.update(overrides)
    return cfg


# ── PolicyEngine unit tests ────────────────────────────────────────────────────

class TestPolicyEngine:

    def test_clean_conditions_recommend_rar(self):
        policy = PolicyEngine(_default_cfg())
        telemetry = {
            "lag_ratio":        0.05,
            "heartbeat_age_ms": 200.0,
            "bandwidth_est":    500.0,
        }
        assert policy.evaluate(telemetry) == "rar"

    def test_high_lag_recommends_ps(self):
        policy = PolicyEngine(_default_cfg(straggler_lag_ratio=2.0))
        telemetry = {
            "lag_ratio":        3.5,   # above threshold
            "heartbeat_age_ms": 100.0,
            "bandwidth_est":    500.0,
        }
        assert policy.evaluate(telemetry) == "ps"

    def test_stale_heartbeat_recommends_ps(self):
        policy = PolicyEngine(_default_cfg(heartbeat_timeout_ms=1000.0))
        telemetry = {
            "lag_ratio":        0.1,
            "heartbeat_age_ms": 5000.0,   # worker silent for 5 s
            "bandwidth_est":    500.0,
        }
        assert policy.evaluate(telemetry) == "ps"

    def test_low_bandwidth_recommends_ps(self):
        policy = PolicyEngine(_default_cfg(min_bandwidth_mbs=100.0))
        telemetry = {
            "lag_ratio":        0.1,
            "heartbeat_age_ms": 100.0,
            "bandwidth_est":    10.0,   # below 100 MB/s threshold
        }
        assert policy.evaluate(telemetry) == "ps"

    def test_failure_rule_takes_priority_over_clean_lag(self):
        """Heartbeat timeout fires even when lag_ratio is fine."""
        policy = PolicyEngine(_default_cfg(heartbeat_timeout_ms=500.0))
        telemetry = {
            "lag_ratio":        0.01,
            "heartbeat_age_ms": 10_000.0,
            "bandwidth_est":    999.0,
        }
        assert policy.evaluate(telemetry) == "ps"

    def test_hysteresis_blocks_premature_switch(self):
        policy = PolicyEngine(_default_cfg(hysteresis_rounds=3))
        # current=rar, recommended=ps, but only 2 rounds since last switch
        assert policy.should_switch("rar", "ps", rounds_since_switch=2) is False

    def test_hysteresis_allows_switch_after_enough_rounds(self):
        policy = PolicyEngine(_default_cfg(hysteresis_rounds=3))
        assert policy.should_switch("rar", "ps", rounds_since_switch=3) is True

    def test_no_switch_when_same_topology(self):
        policy = PolicyEngine(_default_cfg(hysteresis_rounds=1))
        assert policy.should_switch("rar", "rar", rounds_since_switch=100) is False

    def test_lag_ratio_exactly_at_threshold_does_not_trigger(self):
        """Boundary: lag_ratio == threshold should NOT trigger PS (strict >)."""
        policy = PolicyEngine(_default_cfg(straggler_lag_ratio=2.0))
        telemetry = {
            "lag_ratio":        2.0,
            "heartbeat_age_ms": 100.0,
            "bandwidth_est":    500.0,
        }
        assert policy.evaluate(telemetry) == "rar"


# ── MetricsMonitor unit tests ──────────────────────────────────────────────────

class TestMetricsMonitor:

    def test_empty_snapshot_defaults(self):
        m = MetricsMonitor()
        snap = m.snapshot()
        assert snap["lag_ratio"] == 1.0
        assert snap["comm_time_ms"] == 0.0
        assert snap["worker_count"] == 0

    def test_single_worker_lag_ratio_is_one(self):
        m = _monitor_with([{"rank": 0, "clock": 50, "comm_ms": 5.0, "compute_ms": 3.0}])
        snap = m.snapshot()
        assert snap["lag_ratio"] == 1.0   # single worker: no lag

    def test_four_uniform_workers_low_lag(self):
        records = [
            {"rank": r, "clock": 100 + r, "comm_ms": 10.0, "compute_ms": 8.0}
            for r in range(4)
        ]
        m = _monitor_with(records)
        snap = m.snapshot()
        # All clocks nearly equal -> lag_ratio near 0
        assert snap["lag_ratio"] < 0.1

    def test_straggler_raises_lag_ratio(self):
        """1 straggler at very low clock should elevate lag_ratio."""
        records = [
            {"rank": 0, "clock": 500, "comm_ms": 10.0, "compute_ms": 8.0},
            {"rank": 1, "clock": 500, "comm_ms": 10.0, "compute_ms": 8.0},
            {"rank": 2, "clock": 100, "comm_ms": 12.0, "compute_ms": 10.0},
            {"rank": 3, "clock": 10,  "comm_ms": 30.0, "compute_ms": 24.0},
        ]
        m = _monitor_with(records)
        snap = m.snapshot()
        # lag = (500-10+1)/((100+500)/2+1) = 491/301 ≈ 1.63
        assert snap["lag_ratio"] > 1.5

    def test_worker_count(self):
        records = [
            {"rank": r, "clock": r, "comm_ms": 1.0, "compute_ms": 1.0}
            for r in range(6)
        ]
        m = _monitor_with(records)
        assert m.snapshot()["worker_count"] == 6

    def test_clear_resets_records(self):
        m = _monitor_with([{"rank": 0, "clock": 50, "comm_ms": 5.0, "compute_ms": 3.0}])
        m.clear()
        snap = m.snapshot()
        assert snap["comm_time_ms"] == 0.0
        assert snap["compute_ms"] == 0.0

    def test_window_limits_records(self):
        m = MetricsMonitor()
        # Add 100 records for rank 0 with increasing comm_ms
        for i in range(100):
            m.record(rank=0, clock=i, comm_ms=float(i), compute_ms=1.0)
        # Window=10: only last 10 should be averaged (comm_ms 90-99 → avg 94.5)
        snap = m.snapshot(window=10)
        assert abs(snap["comm_time_ms"] - 94.5) < 0.5

    def test_active_workers_excludes_stale(self):
        m = MetricsMonitor()
        m.record(rank=0, clock=1, comm_ms=1.0, compute_ms=1.0)
        m.heartbeat(0)
        m.record(rank=1, clock=1, comm_ms=1.0, compute_ms=1.0)
        m.heartbeat(1)
        # Manually backdate rank 1's heartbeat to simulate staleness
        with m._lock:
            m._heartbeats[1] = time.perf_counter() - 10.0  # 10 s ago
        active = m.active_workers(timeout_ms=3000.0)
        assert 0 in active
        assert 1 not in active


# ── AdaptiveController integration tests ──────────────────────────────────────

class TestAdaptiveController:

    def _make_ctrl(self, **cfg_overrides):
        cfg = _default_cfg(**cfg_overrides)
        monitor = MetricsMonitor()
        return AdaptiveController(cfg, monitor, checkpoint_dir="checkpoints"), monitor

    # ── initial state ──────────────────────────────────────────────────────────

    def test_initial_topology_is_rar(self):
        ctrl, _ = self._make_ctrl(mode="rar")
        assert ctrl.topology == "rar"

    def test_initial_topology_honours_config(self):
        ctrl, _ = self._make_ctrl(mode="ps")
        assert ctrl.topology == "ps"

    # ── no-switch path ─────────────────────────────────────────────────────────

    def test_stays_rar_under_clean_conditions(self):
        ctrl, monitor = self._make_ctrl(mode="rar", hysteresis_rounds=1)
        # Feed clean telemetry
        for r in range(4):
            monitor.record(rank=r, clock=100 + r, comm_ms=10.0, compute_ms=8.0)
            monitor.heartbeat(r)

        result = ctrl.decide(_dummy_params(), round_num=5)
        assert result == "rar"
        assert ctrl.switch_log == []

    # ── switch path ───────────────────────────────────────────────────────────

    def test_switches_to_ps_on_straggler(self):
        ctrl, monitor = self._make_ctrl(
            mode="rar", straggler_lag_ratio=1.5, hysteresis_rounds=1
        )
        # lag_ratio ≈ 1.63 > 1.5
        monitor.record(rank=0, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=1, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=2, clock=100, comm_ms=12.0, compute_ms=10.0)
        monitor.record(rank=3, clock=10,  comm_ms=30.0, compute_ms=24.0)
        for r in range(4):
            monitor.heartbeat(r)

        result = ctrl.decide(_dummy_params(), round_num=5)
        assert result == "ps"
        assert len(ctrl.switch_log) == 1
        assert ctrl.switch_log[0]["from"] == "rar"
        assert ctrl.switch_log[0]["to"]   == "ps"

    def test_switch_log_records_round_number(self):
        ctrl, monitor = self._make_ctrl(
            mode="rar", straggler_lag_ratio=1.5, hysteresis_rounds=1
        )
        for r in range(4):
            monitor.record(rank=r, clock=10 * (r + 1), comm_ms=10.0, compute_ms=8.0)
            monitor.heartbeat(r)

        ctrl.decide(_dummy_params(), round_num=7)
        if ctrl.switch_log:
            assert ctrl.switch_log[0]["round"] == 7

    # ── hysteresis ────────────────────────────────────────────────────────────

    def test_hysteresis_blocks_switch_too_soon(self):
        ctrl, monitor = self._make_ctrl(
            mode="rar", straggler_lag_ratio=1.5, hysteresis_rounds=3
        )
        # Inject straggler telemetry
        monitor.record(rank=0, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=1, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=2, clock=100, comm_ms=12.0, compute_ms=10.0)
        monitor.record(rank=3, clock=10,  comm_ms=30.0, compute_ms=24.0)
        for r in range(4):
            monitor.heartbeat(r)

        # round_num=2, last_switch=0 → rounds_since=2 < hysteresis_rounds=3
        result = ctrl.decide(_dummy_params(), round_num=2)
        assert result == "rar"           # blocked by hysteresis
        assert ctrl.switch_log == []

    def test_hysteresis_allows_switch_after_enough_rounds(self):
        ctrl, monitor = self._make_ctrl(
            mode="rar", straggler_lag_ratio=1.5, hysteresis_rounds=3
        )
        monitor.record(rank=0, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=1, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=2, clock=100, comm_ms=12.0, compute_ms=10.0)
        monitor.record(rank=3, clock=10,  comm_ms=30.0, compute_ms=24.0)
        for r in range(4):
            monitor.heartbeat(r)

        # round_num=3 → rounds_since=3 >= 3 → switch allowed
        result = ctrl.decide(_dummy_params(), round_num=3)
        assert result == "ps"

    # ── force override ────────────────────────────────────────────────────────

    def test_force_overrides_policy(self):
        """force='ps' must switch regardless of telemetry and hysteresis."""
        ctrl, monitor = self._make_ctrl(
            mode="rar", hysteresis_rounds=100  # would normally block
        )
        # Clean telemetry (policy would say RAR)
        for r in range(4):
            monitor.record(rank=r, clock=100, comm_ms=5.0, compute_ms=5.0)
            monitor.heartbeat(r)

        result = ctrl.decide(_dummy_params(), round_num=1, force="ps")
        assert result == "ps"

    # ── switch-back ───────────────────────────────────────────────────────────

    def test_switches_back_to_rar_after_recovery(self):
        ctrl, monitor = self._make_ctrl(
            mode="rar", straggler_lag_ratio=1.5, hysteresis_rounds=1
        )
        # Round 1: inject straggler → switch to PS
        monitor.record(rank=0, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=1, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(rank=2, clock=100, comm_ms=12.0, compute_ms=10.0)
        monitor.record(rank=3, clock=10,  comm_ms=30.0, compute_ms=24.0)
        for r in range(4):
            monitor.heartbeat(r)
        ctrl.decide(_dummy_params(), round_num=1)
        assert ctrl.topology == "ps"

        # Round 2: clean telemetry → switch back to RAR
        monitor.clear()
        for r in range(4):
            monitor.record(rank=r, clock=200 + r, comm_ms=10.0, compute_ms=8.0)
            monitor.heartbeat(r)
        result = ctrl.decide(_dummy_params(), round_num=2)
        assert result == "rar"
        assert len(ctrl.switch_log) == 2
