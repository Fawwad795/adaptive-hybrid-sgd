"""
E2 — Straggler experiment.

One worker (rank = world_size - 1) is slowed by inserting a sleep of
  delay = compute_ms * (factor - 1)
before each gradient computation, simulating a 2×/3×/5× slower worker.

In BSP mode the whole system is gated on the slowest worker, so throughput
degrades proportionally. RAR suffers equally since it also uses BSP barriers.

Output
------
results/tables/e2_straggler.csv
results/plots/e2_straggler.png
"""

from __future__ import annotations
import csv
import json
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZE    = 4
_STRAGGLER_RANK = _WORLD_SIZE - 1   # always the last worker
_SEEDS         = [42, 123, 456]
_EPOCHS        = 2
_FACTORS       = [1, 2, 3, 5]       # slowdown multipliers
_LOG_DIR       = "results/raw"

# Approximate compute_ms per batch for LogReg on MNIST (measured from E1)
_BASE_COMPUTE_MS = 7.0


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_system_tp(prefix: str, world_size: int) -> float:
    total = 0.0
    for r in range(world_size):
        p = Path(f"{prefix}_r{r}.jsonl")
        if not p.exists():
            return 0.0
        rows = [json.loads(l) for l in open(p) if l.strip()]
        sums = [x for x in rows if x.get("phase") == "epoch_summary"]
        if not sums:
            return 0.0
        total += sums[-1]["throughput_samples_sec"]
    return total


def _base_config(seed: int) -> dict:
    return dict(dataset="mnist", model="logreg", lr=0.01, batch_size=64,
                epochs=_EPOCHS, seed=seed, num_workers=_WORLD_SIZE,
                data_dir="data", log_dir=_LOG_DIR)


def _run_ps_trial(seed: int, factor: int, port: int) -> float:
    from workers.models import build_model
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    delay_s = (_BASE_COMPUTE_MS / 1000.0) * (factor - 1)
    label   = f"e2_ps_f{factor}_seed{seed}"
    cfg     = dict(_base_config(seed), ps_discipline="bsp")
    model   = build_model("logreg", seed=seed)
    init_params = model.get_params()

    ctx    = mp.get_context("spawn")
    server = ctx.Process(target=ps_server_process,
                         args=(_WORLD_SIZE, cfg, init_params, port), daemon=True)
    server.start()
    time.sleep(0.8)

    result_q = ctx.Queue()
    procs = []
    for rank in range(_WORLD_SIZE):
        wcfg = dict(cfg, run_id=f"{label}_r{rank}")
        sd   = delay_s if rank == _STRAGGLER_RANK else 0.0
        p = ctx.Process(target=run_ps_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params,
                              result_q, port, sd),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()

    return _get_system_tp(f"{_LOG_DIR}/{label}", _WORLD_SIZE)


def _run_rar_trial(seed: int, factor: int) -> float:
    from workers.models import build_model
    from workers.worker import run_rar_worker
    from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model

    delay_s = (_BASE_COMPUTE_MS / 1000.0) * (factor - 1)
    label   = f"e2_rar_f{factor}_seed{seed}"
    cfg     = _base_config(seed)
    model   = build_model("logreg", seed=seed)
    init_params = model.get_params()
    shapes      = param_shapes_from_model(model)
    shared_bufs = make_shared_buffers(_WORLD_SIZE, shapes)

    ctx      = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for rank in range(_WORLD_SIZE):
        wcfg = dict(cfg, run_id=f"{label}_r{rank}")
        sd   = delay_s if rank == _STRAGGLER_RANK else 0.0
        p = ctx.Process(target=run_rar_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params,
                              result_q, shared_bufs, sd),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)

    return _get_system_tp(f"{_LOG_DIR}/{label}", _WORLD_SIZE)


# ── main ───────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_straggler import plot_straggler

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  E2: Straggler  —  MNIST / LogReg  (w={_WORLD_SIZE})")
    print(f"  factors={_FACTORS}  seeds={_SEEDS}  epochs={_EPOCHS}")
    print(f"  straggler=rank{_STRAGGLER_RANK}  base_compute={_BASE_COMPUTE_MS}ms")
    print("=" * 60)

    port = 5700
    rows: list[dict] = []

    for mode in ["ps", "rar"]:
        for factor in _FACTORS:
            tps: list[float] = []
            for seed in _SEEDS:
                print(f"\n[E2] {mode.upper()}  factor={factor}x  seed={seed}")
                try:
                    if mode == "ps":
                        tp = _run_ps_trial(seed, factor, port)
                        port += 1
                    else:
                        tp = _run_rar_trial(seed, factor)
                    tps.append(tp)
                    print(f"  system_tp={tp:.1f} samples/sec")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    tps.append(0.0)
                time.sleep(1.5)

            avg = sum(tps) / len(tps)
            std = (sum((t - avg) ** 2 for t in tps) / len(tps)) ** 0.5
            rows.append({
                "mode":             mode,
                "factor":           factor,
                "throughput_mean":  round(avg, 1),
                "throughput_std":   round(std, 1),
            })
            print(f"  => avg={avg:.1f}  std={std:.1f}")

    # Compute normalised throughput (relative to factor=1 for each mode)
    baselines = {r["mode"]: r["throughput_mean"]
                 for r in rows if r["factor"] == 1}
    for r in rows:
        base = baselines[r["mode"]]
        r["norm_throughput"] = round(r["throughput_mean"] / base, 4) if base > 0 else 0.0

    # Save CSV
    csv_path = "results/tables/e2_straggler.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[E2] Table saved: {csv_path}")

    # Plot
    try:
        plot_straggler(rows, out_dir="results/plots")
        print("[E2] Plot saved.")
    except Exception as exc:
        print(f"[E2] Plot skipped: {exc}")

    print("\n[E2] DONE")
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'mode':<6} {'factor':>7} {'tp_mean':>10} {'norm_tp':>8}")
    print("-" * 36)
    for r in rows:
        print(f"{r['mode']:<6} {r['factor']:>6}x "
              f"{r['throughput_mean']:>10.1f} {r['norm_throughput']:>8.3f}")
