"""
Phase 4 — Metrics Monitor.

NOT YET IMPLEMENTED — stub for skeleton completeness.

Collects per-round telemetry:
    comm_time_ms   — time spent on gradient synchronization per iteration
    lag_ratio      — max worker lag / median lag  (PS mode)
    heartbeat_age  — ms since last heartbeat per worker
    bandwidth_est  — estimated effective bandwidth (GB/s)
    worker_count   — active membership count

Emits unified JSON-lines telemetry consumed by the Adaptive Controller.
"""

raise NotImplementedError("Phase 4: Metrics Monitor not yet implemented.")
