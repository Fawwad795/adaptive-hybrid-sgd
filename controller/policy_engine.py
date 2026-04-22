"""
Phase 5 — Policy Engine.

Encapsulates threshold rules for topology selection.
All thresholds come from config/default.yaml and can be overridden at runtime.

Decision matrix (in priority order):
    1. Failure detected (heartbeat_age > timeout) → PS
    2. lag_ratio > straggler_lag_ratio            → PS (SSP)
    3. bandwidth_est < bandwidth_threshold         → PS
    4. otherwise                                  → RAR
"""

from __future__ import annotations


class PolicyEngine:
    """
    Stateless threshold evaluator.

    Parameters
    ----------
    config : dict loaded from default.yaml (or overrides)
    """

    def __init__(self, config: dict) -> None:
        self.straggler_lag_ratio  = float(config.get("straggler_lag_ratio", 2.0))
        self.heartbeat_timeout_ms = float(config.get("heartbeat_timeout_ms", 3000.0))
        self.switching_cost_margin= float(config.get("switching_cost_margin", 0.05))
        self.hysteresis_rounds    = int(config.get("hysteresis_rounds", 3))
        # bandwidth below which PS is preferred (MB/s); 0 = no bandwidth rule
        self.min_bandwidth_mbs    = float(config.get("min_bandwidth_mbs", 0.0))

    def evaluate(self, telemetry: dict) -> str:
        """
        Return the recommended topology: "ps" or "rar".

        Parameters
        ----------
        telemetry : dict from MetricsMonitor.snapshot()
        """
        # Rule 1 — failure
        if telemetry.get("heartbeat_age_ms", 0.0) > self.heartbeat_timeout_ms:
            return "ps"

        # Rule 2 — straggler
        if telemetry.get("lag_ratio", 1.0) > self.straggler_lag_ratio:
            return "ps"

        # Rule 3 — low bandwidth
        if (self.min_bandwidth_mbs > 0 and
                telemetry.get("bandwidth_est", float("inf")) < self.min_bandwidth_mbs):
            return "ps"

        return "rar"

    def should_switch(self, current: str, recommended: str,
                      rounds_since_switch: int) -> bool:
        """
        Apply hysteresis: only switch if enough rounds have passed and the
        recommendation differs from the current topology.
        """
        if current == recommended:
            return False
        return rounds_since_switch >= self.hysteresis_rounds
