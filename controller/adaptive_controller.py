"""
Phase 5 — Adaptive Controller.

Monitors telemetry, decides when to switch topology, and orchestrates safe
topology transitions:
    finish current round → checkpoint → update membership → activate new mode

Decision rules (from Deliverable 2 §9.2):
    failure detected       → checkpoint + fall back to PS
    lag_ratio > threshold  → PS (SSP)
    bandwidth healthy      → RAR
    hysteresis guard       → don't switch unless benefit > margin

Usage
-----
ctrl = AdaptiveController(config, monitor, checkpoint_dir)
topology = ctrl.decide(current_topology, params, round_num)
"""

from __future__ import annotations
import time

from controller.policy_engine import PolicyEngine
from monitor.metrics_monitor import MetricsMonitor
from checkpoint.store import save_checkpoint


class AdaptiveController:
    """
    Stateful controller that tracks topology history and enforces hysteresis.

    Parameters
    ----------
    config          : full training config dict
    monitor         : MetricsMonitor instance shared with workers
    checkpoint_dir  : directory for pre-switch checkpoints
    """

    def __init__(self, config: dict, monitor: MetricsMonitor,
                 checkpoint_dir: str = "checkpoints") -> None:
        self.policy          = PolicyEngine(config)
        self.monitor         = monitor
        self.checkpoint_dir  = checkpoint_dir
        self._topology       = config.get("mode", "ps")
        self._last_switch    = 0         # round number of last switch
        self._switch_log: list[dict] = []

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def topology(self) -> str:
        return self._topology

    @property
    def switch_log(self) -> list[dict]:
        return list(self._switch_log)

    # ── Main decision method ───────────────────────────────────────────────────

    def decide(
        self,
        params:    dict,
        round_num: int,
        force:     str | None = None,
    ) -> str:
        """
        Evaluate telemetry and return the topology to use for this round.

        Parameters
        ----------
        params    : current model params (for checkpointing on switch)
        round_num : current training round (used for hysteresis)
        force     : override — pass "ps" or "rar" to force a topology

        Returns the topology string ("ps" or "rar").
        """
        telemetry  = self.monitor.snapshot()
        recommended = force if force else self.policy.evaluate(telemetry)

        rounds_since = round_num - self._last_switch
        should_switch = self.policy.should_switch(
            self._topology, recommended, rounds_since
        )

        if should_switch or force:
            self._do_switch(recommended, params, round_num, telemetry)

        return self._topology

    # ── Switch protocol ────────────────────────────────────────────────────────

    def _do_switch(self, new_topology: str, params: dict,
                   round_num: int, telemetry: dict) -> None:
        old = self._topology

        # Checkpoint before switching
        meta = {
            "round":    round_num,
            "from":     old,
            "to":       new_topology,
            "ts":       time.time(),
            "telemetry": telemetry,
        }
        try:
            save_checkpoint(params, meta, self.checkpoint_dir, version=round_num)
        except Exception as exc:
            print(f"[controller] checkpoint failed: {exc}")

        self._topology    = new_topology
        self._last_switch = round_num
        self._switch_log.append({
            "round":   round_num,
            "from":    old,
            "to":      new_topology,
            "reason":  telemetry,
        })
        print(
            f"[controller] round {round_num}: switch {old} -> {new_topology} | "
            f"lag={telemetry.get('lag_ratio', '?'):.2f} "
            f"comm={telemetry.get('comm_time_ms', '?'):.1f}ms"
        )
