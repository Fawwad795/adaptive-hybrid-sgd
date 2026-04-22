"""
Phase 4 — Metrics Monitor.

Thread-safe per-round telemetry collector. Workers record comm/compute times
each step; the controller reads aggregated statistics to make switching decisions.

Telemetry fields (per round):
    comm_time_ms   — time spent on gradient synchronisation
    compute_ms     — time spent on local forward+backward
    lag_ratio      — max worker lag / median lag (PS mode)
    bandwidth_est  — effective bandwidth estimate (MB/s)
    worker_count   — number of active workers
    topology       — current mode ("ps" | "rar")
"""

from __future__ import annotations
import threading
import time
from collections import defaultdict


class MetricsMonitor:
    """
    Shared telemetry store.

    Workers call record() each step.
    The controller calls snapshot() to get current aggregated stats.
    """

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._rounds: list[dict] = []
        self._heartbeats: dict[int, float] = {}   # rank → last ping timestamp

    # ── Worker calls ───────────────────────────────────────────────────────────

    def record(self, rank: int, comm_ms: float, compute_ms: float,
               clock: int = 0, topology: str = "ps") -> None:
        with self._lock:
            self._rounds.append({
                "rank":       rank,
                "comm_ms":    comm_ms,
                "compute_ms": compute_ms,
                "clock":      clock,
                "topology":   topology,
                "ts":         time.perf_counter(),
            })
            self._heartbeats[rank] = time.perf_counter()

    def heartbeat(self, rank: int) -> None:
        with self._lock:
            self._heartbeats[rank] = time.perf_counter()

    # ── Controller calls ───────────────────────────────────────────────────────

    def snapshot(self, window: int = 50) -> dict:
        """
        Return aggregated telemetry over the last `window` records.

        Returns
        -------
        dict with keys: comm_time_ms, compute_ms, lag_ratio, bandwidth_est,
                        worker_count, heartbeat_age_ms
        """
        with self._lock:
            recent = self._rounds[-window:] if len(self._rounds) >= window \
                     else list(self._rounds)

            if not recent:
                return {
                    "comm_time_ms":  0.0,
                    "compute_ms":    0.0,
                    "lag_ratio":     1.0,
                    "bandwidth_est": float("inf"),
                    "worker_count":  len(self._heartbeats),
                    "heartbeat_age_ms": 0.0,
                }

            # Per-rank clock values (latest entry per rank)
            rank_clocks: dict[int, int] = {}
            for r in recent:
                rank_clocks[r["rank"]] = r["clock"]

            clocks = list(rank_clocks.values())
            if len(clocks) > 1:
                import statistics
                median_clock = statistics.median(clocks)
                max_clock    = max(clocks)
                lag_ratio    = (max_clock - min(clocks) + 1) / (median_clock + 1)
            else:
                lag_ratio = 1.0

            avg_comm    = sum(r["comm_ms"]    for r in recent) / len(recent)
            avg_compute = sum(r["compute_ms"] for r in recent) / len(recent)

            # Rough bandwidth estimate: assume 1 MB of gradient data per ms
            bandwidth_est = 1.0 / (avg_comm / 1000.0 + 1e-9) / 1e6  # MB/s

            now = time.perf_counter()
            heartbeat_ages = [
                (now - ts) * 1000 for ts in self._heartbeats.values()
            ]
            max_hb_age = max(heartbeat_ages) if heartbeat_ages else 0.0

        return {
            "comm_time_ms":     avg_comm,
            "compute_ms":       avg_compute,
            "lag_ratio":        lag_ratio,
            "bandwidth_est":    bandwidth_est,
            "worker_count":     len(self._heartbeats),
            "heartbeat_age_ms": max_hb_age,
        }

    def clear(self) -> None:
        with self._lock:
            self._rounds.clear()

    def active_workers(self, timeout_ms: float = 3000.0) -> list[int]:
        now = time.perf_counter()
        with self._lock:
            return [
                rank for rank, ts in self._heartbeats.items()
                if (now - ts) * 1000 < timeout_ms
            ]
