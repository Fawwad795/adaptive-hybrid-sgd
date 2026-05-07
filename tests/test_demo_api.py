from __future__ import annotations

from demo_api.comparisons import load_comparisons
from demo_api.runtime import _aggregate_epoch, _record_epoch_telemetry, _scenario_for_epoch
from demo_api.schemas import RunRequest, ScenarioConfig
from monitor.metrics_monitor import MetricsMonitor


def _request(mode: str = "hybrid") -> RunRequest:
    return RunRequest(
        mode=mode,
        epochs=5,
        scenario=ScenarioConfig(
            straggler_epochs=[3, 4],
            straggler_rank=3,
            straggler_factor=3.0,
            bandwidth_epochs=[2],
            throttle_ms=10.0,
        ),
    )


def test_scenario_schedule_activates_expected_epochs():
    request = _request()
    assert _scenario_for_epoch(request, 1)["straggler_active"] is False
    assert _scenario_for_epoch(request, 2)["bandwidth_active"] is True
    assert _scenario_for_epoch(request, 3)["straggler_active"] is True


def test_aggregate_epoch_sums_system_throughput():
    summaries = {
        0: {
            "train_loss": 0.4,
            "train_acc": 0.9,
            "val_loss": 0.3,
            "val_acc": 0.91,
            "avg_compute_ms": 8.0,
            "avg_comm_ms": 10.0,
            "elapsed_sec": 2.5,
            "throughput_samples_sec": 2000.0,
        },
        1: {
            "train_loss": 0.42,
            "train_acc": 0.88,
            "val_loss": 0.31,
            "val_acc": 0.9,
            "avg_compute_ms": 9.0,
            "avg_comm_ms": 11.0,
            "elapsed_sec": 2.6,
            "throughput_samples_sec": 1800.0,
        },
    }
    summary = _aggregate_epoch(2, "ps", summaries)
    assert summary["epoch"] == 2
    assert summary["topology"] == "ps"
    assert summary["throughput_samples_sec"] == 3800.0
    assert summary["worker_count"] == 2


def test_real_metric_projection_exposes_straggler_lag():
    monitor = MetricsMonitor()
    telemetry = _record_epoch_telemetry(
        monitor,
        "rar",
        {
            0: {"avg_compute_ms": 8.0, "avg_comm_ms": 9.0},
            1: {"avg_compute_ms": 8.2, "avg_comm_ms": 9.1},
            2: {"avg_compute_ms": 22.0, "avg_comm_ms": 18.0},
            3: {"avg_compute_ms": 8.1, "avg_comm_ms": 9.0},
        },
    )
    assert telemetry["lag_ratio"] > 1.5
    assert telemetry["worker_count"] == 4


def test_comparison_loader_returns_empty_when_tables_missing():
    response = load_comparisons()
    assert isinstance(response.datasets, list)
    assert isinstance(response.available_files, list)
