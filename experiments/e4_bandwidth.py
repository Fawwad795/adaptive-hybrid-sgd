"""
E4 — Bandwidth experiment.

Simulates different network bandwidths by injecting a per-message sleep
proportional to gradient size / simulated_bandwidth.

For LogReg on MNIST: gradient size ≈ 31 KB (7850 float32 params).
  bandwidth_mbps → sleep_per_msg_ms = grad_bytes / (bandwidth_mbps * 1e6 / 8) * 1000

Configurations
--------------
  "10G"  : 10 000 Mbps  → ~0.025 ms/msg  (datacenter NVLink)
  "1G"   : 1  000 Mbps  → ~0.25  ms/msg  (datacenter Ethernet)
  "100M" :   100 Mbps   → ~2.5   ms/msg  (commodity LAN)
  "10M"  :    10 Mbps   → ~24.8  ms/msg  (WAN / throttled link)

This reveals the crossover: at low bandwidth PS (centralised) suffers because
the server handles N push + N pull messages per round; RAR's ring distributes
the load and transfers less per step.

Output
------
results/tables/e4_bandwidth.csv
results/plots/e4_bandwidth.png
"""

from __future__ import annotations
import csv
import json
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZE = 4
_SEEDS      = [42, 123, 456]
_EPOCHS     = 2
_LOG_DIR    = "results/raw"

# LogReg gradient size: (784×10 + 10) × 4 bytes ≈ 31 400 bytes
_GRAD_BYTES = (784 * 10 + 10) * 4

# Bandwidth configs: label → Mbps
_BANDWIDTHS = {
    "10G":  10_000,
    "1G":    1_000,
    "100M":    100,
    "10M":      10,
}


def _throttle_ms(bandwidth_mbps: float) -> float:
    """Per-message sleep that simulates one-way transfer time."""
    return _GRAD_BYTES / (bandwidth_mbps * 1e6 / 8) * 1000


def _rar_throttle_ms(bandwidth_mbps: float) -> float:
    # Each ring step transfers only 1/world_size of the gradient.
    return _GRAD_BYTES / _WORLD_SIZE / (bandwidth_mbps * 1e6 / 8) * 1000


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


def _base_config(seed: int, throttle_ms: float) -> dict:
    return dict(dataset="mnist", model="logreg", lr=0.01, batch_size=64,
                epochs=_EPOCHS, seed=seed, num_workers=_WORLD_SIZE,
                data_dir="data", log_dir=_LOG_DIR,
                throttle_ms=throttle_ms)


def _run_ps_trial(seed: int, throttle_ms: float, label: str, port: int) -> float:
    from workers.models import build_model
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    cfg = dict(_base_config(seed, throttle_ms), ps_discipline="bsp")
    model = build_model("logreg", seed=seed)
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
        p = ctx.Process(target=run_ps_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params, result_q, port),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()

    return _get_system_tp(f"{_LOG_DIR}/{label}", _WORLD_SIZE)


def _run_rar_trial(seed: int, throttle_ms: float, label: str) -> float:
    from workers.models import build_model
    from workers.worker import run_rar_worker
    from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model

    cfg = _base_config(seed, throttle_ms)
    model = build_model("logreg", seed=seed)
    init_params = model.get_params()
    shapes      = param_shapes_from_model(model)
    shared_bufs = make_shared_buffers(_WORLD_SIZE, shapes)

    ctx      = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for rank in range(_WORLD_SIZE):
        wcfg = dict(cfg, run_id=f"{label}_r{rank}")
        p = ctx.Process(target=run_rar_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params, result_q, shared_bufs),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)

    return _get_system_tp(f"{_LOG_DIR}/{label}", _WORLD_SIZE)


# ── main ───────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_bandwidth import plot_bandwidth

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  E4: Bandwidth  —  MNIST / LogReg  (w={_WORLD_SIZE})")
    print(f"  bandwidths={list(_BANDWIDTHS)}  seeds={_SEEDS}")
    print("=" * 60)

    port = 5800
    rows: list[dict] = []

    for bw_label, bw_mbps in _BANDWIDTHS.items():
        ps_t_ms  = _throttle_ms(bw_mbps)
        rar_t_ms = _rar_throttle_ms(bw_mbps)
        print(f"\n[E4] Bandwidth={bw_label} ({bw_mbps} Mbps)  ps_throttle={ps_t_ms:.3f}ms  rar_throttle={rar_t_ms:.3f}ms")

        for mode in ["ps", "rar"]:
            tps: list[float] = []
            for seed in _SEEDS:
                label = f"e4_{mode}_{bw_label}_seed{seed}"   # run_id stem only
                print(f"  {mode.upper()} seed={seed} ...", end="", flush=True)
                try:
                    if mode == "ps":
                        tp = _run_ps_trial(seed, ps_t_ms, label, port)
                        port += 1
                    else:
                        tp = _run_rar_trial(seed, rar_t_ms, label)
                    tps.append(tp)
                    print(f" tp={tp:.0f}")
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    tps.append(0.0)
                time.sleep(1.5)

            avg = sum(tps) / len(tps)
            std = (sum((t - avg) ** 2 for t in tps) / len(tps)) ** 0.5
            rows.append({
                "bandwidth":        bw_label,
                "bandwidth_mbps":   bw_mbps,
                "throttle_ms":      round(ps_t_ms if mode == "ps" else rar_t_ms, 4),
                "mode":             mode,
                "throughput_mean":  round(avg, 1),
                "throughput_std":   round(std, 1),
            })
            print(f"  => {mode.upper()} avg={avg:.1f} +-{std:.1f}")

    # Save CSV
    csv_path = "results/tables/e4_bandwidth.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[E4] Table saved: {csv_path}")

    # Plot
    try:
        plot_bandwidth(rows, out_dir="results/plots")
        print("[E4] Plot saved.")
    except Exception as exc:
        print(f"[E4] Plot skipped: {exc}")

    print("\n[E4] DONE")
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'bw':>5} {'mode':<5} {'tp_mean':>10} {'tp_std':>8}")
    print("-" * 32)
    for r in rows:
        print(f"{r['bandwidth']:>5} {r['mode']:<5} "
              f"{r['throughput_mean']:>10.1f} {r['throughput_std']:>8.1f}")
