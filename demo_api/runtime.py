from __future__ import annotations

import json
import multiprocessing as mp
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from checkpoint.store import load_checkpoint
from controller.adaptive_controller import AdaptiveController
from monitor.metrics_monitor import MetricsMonitor
from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model
from workers.models import build_model
from workers.worker import run_ps_worker, run_rar_worker


ROOT = Path(__file__).resolve().parents[1]
DEMO_LOG_ROOT = ROOT / "results" / "raw" / "demo"
DEMO_CHECKPOINT_ROOT = ROOT / "checkpoints" / "demo"
FILE_META_RE = re.compile(r"(?P<run>.+)_(?P<topology>ps|rar)_e(?P<epoch>\d+)_r(?P<rank>\d+)\.jsonl$")


class DemoStopped(Exception):
    pass


@dataclass
class EpochResult:
    params: dict
    summaries_by_rank: dict[int, dict]


class LiveLogRelay:
    def __init__(self, log_dir: Path, emit: Callable[[str, dict], None]) -> None:
        self._log_dir = log_dir
        self._emit = emit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._offsets: dict[Path, int] = {}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            for path in sorted(self._log_dir.glob("*.jsonl")):
                self._consume(path)
            time.sleep(0.15)

    def _consume(self, path: Path) -> None:
        offset = self._offsets.get(path, 0)
        with open(path, encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = self._parse_file_meta(path.name)
                if meta:
                    payload.setdefault("topology", meta["topology"])
                    payload.setdefault("display_epoch", meta["epoch"])
                    payload.setdefault("rank", meta["rank"])
                payload["log_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
                if payload.get("phase") == "train" and payload.get("rank", 0) == 0:
                    self._emit("train.iteration", payload)
                elif payload.get("phase") == "epoch_summary":
                    self._emit("worker.epoch_summary", payload)
            self._offsets[path] = handle.tell()

    @staticmethod
    def _parse_file_meta(filename: str) -> dict | None:
        match = FILE_META_RE.match(filename)
        if not match:
            return None
        return {
            "topology": match.group("topology"),
            "epoch": int(match.group("epoch")),
            "rank": int(match.group("rank")),
        }


def execute_run(run_id, request, emit: Callable[[str, dict], None], stop_event: threading.Event) -> dict[str, list[str]]:
    log_dir = DEMO_LOG_ROOT / run_id
    checkpoint_dir = DEMO_CHECKPOINT_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    relay = LiveLogRelay(log_dir, emit)
    relay.start()

    try:
        emit(
            "run.started",
            {
                "mode": request.mode,
                "model": request.model,
                "dataset": request.dataset,
                "epochs": request.epochs,
                "num_workers": request.num_workers,
                "initial_topology": request.initial_topology,
            },
        )

        params = build_model(request.model, seed=request.seed).get_params()
        monitor = MetricsMonitor()
        controller = None
        if request.mode == "hybrid":
            controller = AdaptiveController(
                {
                    "straggler_lag_ratio": 1.5,
                    "heartbeat_timeout_ms": 30_000,
                    "switching_cost_margin": 0.0,
                    "hysteresis_rounds": 1,
                    "min_bandwidth_mbs": 0.0,
                    "mode": request.initial_topology,
                },
                monitor,
                checkpoint_dir=str(checkpoint_dir),
            )

        current_topology = request.mode if request.mode != "hybrid" else controller.topology

        for epoch in range(1, request.epochs + 1):
            _raise_if_stopped(stop_event)

            if request.mode == "hybrid":
                previous_topology = controller.topology
                if epoch > 1:
                    decided_topology = controller.decide(params, round_num=epoch)
                    if decided_topology != previous_topology:
                        params, ckpt_meta = load_checkpoint(str(checkpoint_dir))
                        switch = controller.switch_log[-1]
                        emit(
                            "controller.switch",
                            {
                                "epoch": epoch,
                                "from": switch["from"],
                                "to": switch["to"],
                                "reason": switch["reason"],
                                "checkpoint": f"checkpoints/demo/{run_id}/ckpt_v{ckpt_meta['round']:06d}.npz",
                            },
                        )
                current_topology = controller.topology

            scenario = _scenario_for_epoch(request, epoch)
            emit(
                "run.epoch_started",
                {
                    "epoch": epoch,
                    "topology": current_topology,
                    "scenario": scenario,
                },
            )

            run_prefix = f"{run_id}_{current_topology}_e{epoch}"
            epoch_result = _run_distributed_epoch(
                topology=current_topology,
                request=request,
                run_prefix=run_prefix,
                params=params,
                log_dir=log_dir,
                stop_event=stop_event,
                straggler_delay=scenario["straggler_delay"],
                throttle_ms=scenario["throttle_ms"],
                port=6400 + epoch,
            )
            params = epoch_result.params
            summary = _aggregate_epoch(epoch, current_topology, epoch_result.summaries_by_rank)
            emit("train.epoch_summary", summary)
            telemetry = _record_epoch_telemetry(
                monitor,
                current_topology,
                epoch_result.summaries_by_rank,
            )
            emit("telemetry.snapshot", {"epoch": epoch, "topology": current_topology, **telemetry})

        emit("run.completed", {"final_topology": current_topology, "epochs": request.epochs})
    except DemoStopped:
        emit("run.stopped", {"message": "Run stopped by user."})
    except Exception as exc:
        emit("run.failed", {"error": str(exc)})
        raise
    finally:
        relay.stop()

    return {
        "logs": _collect_relative_paths(log_dir, "*.jsonl"),
        "checkpoints": _collect_relative_paths(checkpoint_dir, "*.npz"),
        "plots": [],
        "tables": [],
    }


def _run_distributed_epoch(
    topology: str,
    request,
    run_prefix: str,
    params: dict,
    log_dir: Path,
    stop_event: threading.Event,
    straggler_delay: float,
    throttle_ms: float,
    port: int,
) -> EpochResult:
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    params_q = ctx.Queue()
    processes = []

    config = {
        "dataset": request.dataset,
        "model": request.model,
        "lr": request.lr,
        "batch_size": request.batch_size,
        "epochs": 1,
        "seed": request.seed,
        "num_workers": request.num_workers,
        "data_dir": "data",
        "log_dir": str(log_dir),
        "ps_discipline": request.ps_discipline,
        "throttle_ms": throttle_ms,
    }

    if topology == "ps":
        from ps_engine.server import ps_server_process

        server = ctx.Process(
            target=ps_server_process,
            args=(request.num_workers, config, params, port),
            daemon=True,
        )
        server.start()
        time.sleep(0.8)
        processes.append(server)

        for rank in range(request.num_workers):
            worker_config = dict(config, run_id=f"{run_prefix}_r{rank}")
            params_sink = params_q if rank == 0 else None
            process = ctx.Process(
                target=run_ps_worker,
                args=(
                    rank,
                    request.num_workers,
                    worker_config,
                    params,
                    result_q,
                    port,
                    straggler_delay if rank == request.scenario.straggler_rank else 0.0,
                    params_sink,
                ),
                daemon=True,
            )
            process.start()
            processes.append(process)
    else:
        shared_bufs = make_shared_buffers(request.num_workers, param_shapes_from_model(build_model(request.model, seed=request.seed)))
        for rank in range(request.num_workers):
            worker_config = dict(config, run_id=f"{run_prefix}_r{rank}")
            params_sink = params_q if rank == 0 else None
            process = ctx.Process(
                target=run_rar_worker,
                args=(
                    rank,
                    request.num_workers,
                    worker_config,
                    params,
                    result_q,
                    shared_bufs,
                    straggler_delay if rank == request.scenario.straggler_rank else 0.0,
                    params_sink,
                ),
                daemon=True,
            )
            process.start()
            processes.append(process)

    _wait_for_epoch_completion(log_dir, run_prefix, request.num_workers, processes, stop_event, topology)
    summaries_by_rank = _collect_summaries(result_q, request.num_workers)
    if len(summaries_by_rank) < request.num_workers:
        summaries_by_rank = _collect_summaries_from_logs(log_dir, run_prefix, request.num_workers)
    new_params = params_q.get_nowait() if not params_q.empty() else params
    return EpochResult(params=new_params, summaries_by_rank=summaries_by_rank)


def _wait_for_epoch_completion(
    log_dir: Path,
    run_prefix: str,
    world_size: int,
    processes: list[mp.Process],
    stop_event: threading.Event,
    topology: str,
) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        _raise_if_stopped(stop_event)
        alive = [process for process in processes if process.is_alive()]
        if not alive or _all_epoch_summaries_present(log_dir, run_prefix, world_size):
            break
        time.sleep(0.2)
    else:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise TimeoutError(f"{topology.upper()} epoch timed out")

    for process in processes:
        process.join(timeout=0.5)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)


def _collect_summaries(result_q, world_size: int) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    end_at = time.time() + 10
    while len(summaries) < world_size and time.time() < end_at:
        try:
            item = result_q.get(timeout=0.5)
        except queue.Empty:
            continue
        rank = item["rank"]
        metrics = item.get("metrics", [])
        epoch_summaries = [metric for metric in metrics if metric.get("phase") == "epoch_summary"]
        if epoch_summaries:
            summaries[rank] = epoch_summaries[-1]
    return summaries


def _collect_summaries_from_logs(log_dir: Path, run_prefix: str, world_size: int) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    for rank in range(world_size):
        path = log_dir / f"{run_prefix}_r{rank}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        epoch_summaries = [row for row in rows if row.get("phase") == "epoch_summary"]
        if epoch_summaries:
            summaries[rank] = epoch_summaries[-1]
    return summaries


def _aggregate_epoch(epoch: int, topology: str, summaries_by_rank: dict[int, dict]) -> dict:
    summaries = list(summaries_by_rank.values())
    if not summaries:
        raise RuntimeError("No worker summaries were collected.")

    def avg(key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in summaries) / len(summaries)

    return {
        "epoch": epoch,
        "topology": topology,
        "worker_count": len(summaries),
        "train_loss": round(avg("train_loss"), 6),
        "train_acc": round(avg("train_acc"), 4),
        "val_loss": round(avg("val_loss"), 6),
        "val_acc": round(avg("val_acc"), 4),
        "avg_compute_ms": round(avg("avg_compute_ms"), 3),
        "avg_comm_ms": round(avg("avg_comm_ms"), 3),
        "elapsed_sec": round(max(float(item.get("elapsed_sec", 0.0)) for item in summaries), 2),
        "throughput_samples_sec": round(sum(float(item.get("throughput_samples_sec", 0.0)) for item in summaries), 1),
    }


def _record_epoch_telemetry(monitor: MetricsMonitor, topology: str, summaries_by_rank: dict[int, dict]) -> dict:
    monitor.clear()
    total_times = {
        rank: float(summary.get("avg_compute_ms", 0.0)) + float(summary.get("avg_comm_ms", 0.0))
        for rank, summary in summaries_by_rank.items()
    }
    if not total_times:
        return monitor.snapshot()

    fastest = min(total_times.values())
    slowest = max(total_times.values())
    spread = slowest / max(fastest, 1e-6)

    if spread < 1.15:
        clocks = {rank: 500 + index for index, rank in enumerate(sorted(total_times))}
    else:
        ordered_ranks = [rank for rank, _ in sorted(total_times.items(), key=lambda item: item[1])]
        clocks = {
            rank: max(10, int(500 * (0.45 ** position)))
            for position, rank in enumerate(ordered_ranks)
        }

    for rank, summary in summaries_by_rank.items():
        monitor.record(
            rank=rank,
            clock=clocks[rank],
            comm_ms=float(summary.get("avg_comm_ms", 0.0)),
            compute_ms=float(summary.get("avg_compute_ms", 0.0)),
            topology=topology,
        )
        monitor.heartbeat(rank)
    return monitor.snapshot()


def _scenario_for_epoch(request, epoch: int) -> dict:
    scenario = request.scenario
    straggler_delay = 0.0
    if epoch in set(scenario.straggler_epochs):
        straggler_delay = (scenario.base_compute_ms / 1000.0) * max(0.0, scenario.straggler_factor - 1.0)
    throttle_ms = scenario.throttle_ms if epoch in set(scenario.bandwidth_epochs) else 0.0
    return {
        "straggler_active": epoch in set(scenario.straggler_epochs),
        "straggler_rank": scenario.straggler_rank,
        "straggler_factor": scenario.straggler_factor,
        "straggler_delay": round(straggler_delay, 4),
        "bandwidth_active": epoch in set(scenario.bandwidth_epochs),
        "throttle_ms": throttle_ms,
    }


def _raise_if_stopped(stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise DemoStopped()


def _collect_relative_paths(directory: Path, pattern: str) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in directory.glob(pattern))


def _all_epoch_summaries_present(log_dir: Path, run_prefix: str, world_size: int) -> bool:
    for rank in range(world_size):
        path = log_dir / f"{run_prefix}_r{rank}.jsonl"
        if not path.exists():
            return False
        with open(path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            return False
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False
        if last.get("phase") != "epoch_summary":
            return False
    return True
