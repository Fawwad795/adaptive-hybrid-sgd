"""
Phase 5 — Policy Engine (threshold rules for topology selection).

NOT YET IMPLEMENTED — stub for skeleton completeness.

Thresholds (from config/default.yaml):
    bw_saturation_ratio  : fraction of link capacity; above → prefer PS
    straggler_lag_ratio  : worker_lag / median_lag; above → prefer PS
    heartbeat_timeout_ms : ms without heartbeat → worker considered dead
    switching_cost_margin: min expected gain to justify a switch
    hysteresis_rounds    : min rounds before another topology switch
"""

raise NotImplementedError("Phase 5: Policy Engine not yet implemented.")
