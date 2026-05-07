"""
E1 — Scalability experiment.

Runs PS and RAR at world_size in {2, 4, 8} across 3 seeds.
Computes:
  S(n) = system_throughput(n) / single_throughput(1)   # speedup
  E(n) = S(n) / n                                       # efficiency

Output
------
results/tables/e1_scalability.csv
results/plots/e1_speedup.png
"""

from __future__ import annotations
import csv
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZES = [2, 4, 8]
_SEEDS       = [42, 123, 456]
_EPOCHS      = 3


# ── Trial runners ──────────────────────────────────────────────────────────────

def _single_throughput(config: dict, seed: int) -> float:
    from workers.trainer import train_single
    cfg = dict(config, seed=seed, epochs=_EPOCHS,
               run_id=f"e1_single_seed{seed}")
    metrics = train_single(cfg, run_id=cfg["run_id"])
    summaries = [m for m in metrics if m.get("phase") == "epoch_summary"]
    return summaries[-1]["throughput_samples_sec"]


def _ps_throughput(config: dict, world_size: int, seed: int, port: int) -> float:
    from workers.models import build_model
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    cfg = dict(config, seed=seed, epochs=_EPOCHS, num_workers=world_size,
               ps_discipline="bsp")
    model       = build_model(cfg["model"], seed=seed)
    init_params = model.get_params()

    ctx    = mp.get_context("spawn")
    server = ctx.Process(
        target=ps_server_process,
        args=(world_size, cfg, init_params, port),
        daemon=True,
    )
    server.start()
    time.sleep(0.8)

    result_q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        wcfg = dict(cfg, run_id=f"e1_ps_w{world_size}_seed{seed}_r{rank}")
        p = ctx.Process(
            target=run_ps_worker,
            args=(rank, world_size, wcfg, init_params, result_q, port),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()

    total_tp = 0.0
    while not result_q.empty():
        r = result_q.get_nowait()
        sums = [m for m in r["metrics"] if m.get("phase") == "epoch_summary"]
        if sums:
            total_tp += sums[-1]["throughput_samples_sec"]
    return total_tp


def _rar_throughput(config: dict, world_size: int, seed: int) -> float:
    from workers.models import build_model
    from workers.worker import run_rar_worker
    from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model

    cfg   = dict(config, seed=seed, epochs=_EPOCHS, num_workers=world_size)
    model = build_model(cfg["model"], seed=seed)
    init_params = model.get_params()
    shapes      = param_shapes_from_model(model)

    ctx         = mp.get_context("spawn")
    shared_bufs = make_shared_buffers(world_size, shapes)

    result_q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        wcfg = dict(cfg, run_id=f"e1_rar_w{world_size}_seed{seed}_r{rank}")
        p = ctx.Process(
            target=run_rar_worker,
            args=(rank, world_size, wcfg, init_params, result_q, shared_bufs),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)

    total_tp = 0.0
    while not result_q.empty():
        r = result_q.get_nowait()
        sums = [m for m in r["metrics"] if m.get("phase") == "epoch_summary"]
        if sums:
            total_tp += sums[-1]["throughput_samples_sec"]
    return total_tp


# ── Main experiment ────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_speedup import plot_speedup

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    cfg = dict(config, dataset="cifar10", model="cnn",
               epochs=_EPOCHS, log_dir="results/raw")

    print("\n" + "=" * 60)
    print("  E1: Scalability  —  CIFAR-10 / CNN")
    print(f"  workers={_WORLD_SIZES}  seeds={_SEEDS}  epochs={_EPOCHS}")
    print("=" * 60)

    # ── Baseline (single worker) ───────────────────────────────────────────────
    print("\n[E1] Single-worker baseline...")
    baseline_tps = []
    for seed in _SEEDS:
        tp = _single_throughput(cfg, seed)
        baseline_tps.append(tp)
        print(f"  seed={seed}  throughput={tp:.1f} samples/sec")
    baseline = sum(baseline_tps) / len(baseline_tps)
    print(f"  baseline avg = {baseline:.1f} samples/sec")

    rows: list[dict] = []
    port  = 5650

    for world_size in _WORLD_SIZES:
        for mode in ["ps", "rar"]:
            tps: list[float] = []
            for seed in _SEEDS:
                print(f"\n[E1] {mode.upper()}  workers={world_size}  seed={seed}")
                try:
                    if mode == "ps":
                        tp = _ps_throughput(cfg, world_size, seed, port)
                        port += 1
                    else:
                        tp = _rar_throughput(cfg, world_size, seed)
                    tps.append(tp)
                    print(f"  throughput={tp:.1f} samples/sec")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    tps.append(0.0)
                time.sleep(1.5)   # let ZMQ sockets fully close

            n        = len(tps)
            avg_tp   = sum(tps) / n
            std_tp   = (sum((t - avg_tp) ** 2 for t in tps) / n) ** 0.5
            speedup  = avg_tp / baseline if baseline > 0 else 0.0
            eff      = speedup / world_size

            rows.append({
                "mode":             mode,
                "workers":          world_size,
                "throughput_mean":  round(avg_tp, 1),
                "throughput_std":   round(std_tp, 1),
                "speedup":          round(speedup, 3),
                "efficiency":       round(eff, 3),
            })
            print(f"  => S={speedup:.3f}  E={eff:.3f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = "results/tables/e1_scalability.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[E1] Table saved: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        plot_speedup(rows, baseline=baseline, out_dir="results/plots")
        print("[E1] Speedup plot saved.")
    except Exception as exc:
        print(f"[E1] Plot skipped: {exc}")

    print("\n[E1] DONE")
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'mode':<6} {'workers':>7} {'throughput':>12} {'speedup':>8} {'efficiency':>10}")
    print("-" * 48)
    for r in rows:
        print(
            f"{r['mode']:<6} {r['workers']:>7} "
            f"{r['throughput_mean']:>9.1f}+-{r['throughput_std']:<4.0f} "
            f"{r['speedup']:>8.3f} {r['efficiency']:>10.3f}"
        )
