"""
Structured JSON-lines logger.

Usage:
    logger = RunLogger(run_id, log_dir)
    logger.log({"epoch": 1, "iter": 42, "loss": 0.31, "acc": 0.91, "compute_ms": 12.3})
    logger.close()

Each call appends one JSON line to results/raw/<run_id>.jsonl.
The file is flushed after every call so no data is lost on crash.
"""

import json
import os
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "results/raw"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(log_dir) / f"{run_id}.jsonl"
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1)
        self._start = time.perf_counter()

    def log(self, row: dict) -> None:
        row.setdefault("wall_sec", round(time.perf_counter() - self._start, 4))
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def path(self) -> str:
        return str(self._path)

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_log(path: str) -> list[dict]:
    """Load a .jsonl log file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
