from __future__ import annotations

import csv
from pathlib import Path

from demo_api.schemas import ComparisonDataset, ComparisonResponse, ComparisonSeries


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_comparisons() -> ComparisonResponse:
    datasets: list[ComparisonDataset] = []
    available_files = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in TABLES_DIR.glob("*.csv"))

    e1_rows = _read_csv(TABLES_DIR / "e1_scalability.csv")
    if e1_rows:
        series_map: dict[str, list[dict]] = {}
        for row in e1_rows:
            series_map.setdefault(row["mode"], []).append(
                {
                    "workers": int(row["workers"]),
                    "throughput_mean": float(row["throughput_mean"]),
                    "speedup": float(row["speedup"]),
                    "efficiency": float(row["efficiency"]),
                }
            )
        datasets.append(
            ComparisonDataset(
                id="e1-scalability",
                title="Scalability",
                description="Worker scaling across topologies.",
                source="results/tables/e1_scalability.csv",
                x_key="workers",
                y_key="throughput_mean",
                series=[
                    ComparisonSeries(key=mode, label=mode.upper(), points=points)
                    for mode, points in sorted(series_map.items())
                ],
            )
        )

    e4_rows = _read_csv(TABLES_DIR / "e4_bandwidth.csv")
    if e4_rows:
        series_map = {}
        for row in e4_rows:
            series_map.setdefault(row["mode"], []).append(
                {
                    "bandwidth": row["bandwidth"],
                    "bandwidth_mbps": int(row["bandwidth_mbps"]),
                    "throughput_mean": float(row["throughput_mean"]),
                }
            )
        datasets.append(
            ComparisonDataset(
                id="e4-bandwidth",
                title="Bandwidth Crossover",
                description="Throughput changes under simulated communication pressure.",
                source="results/tables/e4_bandwidth.csv",
                x_key="bandwidth",
                y_key="throughput_mean",
                series=[
                    ComparisonSeries(key=mode, label=mode.upper(), points=points)
                    for mode, points in sorted(series_map.items())
                ],
            )
        )

    e6_rows = _read_csv(TABLES_DIR / "e6_adaptive.csv")
    if e6_rows:
        series_map = {}
        for row in e6_rows:
            mode = row["mode"]
            series_map.setdefault(mode, []).append(
                {
                    "epoch": int(row["epoch"]),
                    "topology": row["topology"],
                    "sys_tp": float(row["sys_tp"]),
                    "val_acc": float(row["val_acc"]),
                    "straggler": row["straggler"] == "True",
                }
            )
        datasets.append(
            ComparisonDataset(
                id="e6-adaptive",
                title="Adaptive Switching",
                description="Epoch-level topology decisions under a straggler schedule.",
                source="results/tables/e6_adaptive.csv",
                x_key="epoch",
                y_key="sys_tp",
                series=[
                    ComparisonSeries(key=mode, label=mode.upper(), points=points)
                    for mode, points in sorted(series_map.items())
                ],
            )
        )

    return ComparisonResponse(datasets=datasets, available_files=available_files)
