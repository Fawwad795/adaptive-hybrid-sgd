"""
E6 — Adaptive Switching experiment.

Demonstrates the AdaptiveController switching between PS and RAR at epoch
boundaries in response to a mid-training straggler event.

Phase schedule (epochs 1-5):
  Epoch 1-2 : clean          (no straggler)
  Epoch 3-4 : straggler      (rank 3, 3x slowdown)
  Epoch 5   : clean          (straggler removed)

Three modes compared:
  hybrid    — AdaptiveController decides topology each epoch
  ps        — static Parameter Server for all epochs
  rar       — static Ring AllReduce for all epochs

The controller feeds on simulated per-epoch telemetry (lag_ratio):
  clean epoch    : all workers at similar clock  -> lag_ratio ~0   -> RAR
  straggler epoch: rank 3 at low clock          -> lag_ratio >2.0  -> PS

Output
------
results/tables/e6_adaptive.csv
results/tables/e6_switches.csv
results/plots/e6_adaptive.png
"""

from __future__ import annotations
import csv
import json
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZE      = 4
_SEEDS           = [42, 123, 456]
_N_EPOCHS        = 5
_LOG_DIR         = "results/raw"
_STRAGGLER_RANK  = 3
_STRAGGLER_FACTOR= 3.0
_BASE_COMPUTE_MS = 5.0
_STRAGGLER_DELAY = _BASE_COMPUTE_MS / 1000 * (_STRAGGLER_FACTOR - 1)  # seconds
_STRAGGLER_EPOCHS= frozenset([3, 4])   # straggler active in these epochs


# ── helpers ────────────────────────────────────────────────────────────────────

def _base_cfg(seed: int) -> dict:
    return dict(dataset="mnist", model="logreg", lr=0.01, batch_size=64,
                epochs=1, seed=seed, num_workers=_WORLD_SIZE,
                data_dir="data", log_dir=_LOG_DIR)


def _get_system_tp(run_prefix: str) -> float:
    total = 0.0
    for r in range(_WORLD_SIZE):
        p = Path(f"{_LOG_DIR}/{run_prefix}_r{r}.jsonl")
        if not p.exists():
            return 0.0
        rows = [json.loads(l) for l in open(p) if l.strip()]
        sums = [x for x in rows if x.get("phase") == "epoch_summary"]
        if sums:
            total += sums[-1].get("throughput_samples_sec", 0.0)
    return total


def _get_val_acc(run_prefix: str) -> float:
    p = Path(f"{_LOG_DIR}/{run_prefix}_r0.jsonl")
    if not p.exists():
        return 0.0
    rows = [json.loads(l) for l in open(p) if l.strip()]
    sums = [x for x in rows if x.get("phase") == "epoch_summary"]
    return sums[-1].get("val_acc", 0.0) if sums else 0.0


def _run_one_ps(init_params: dict, seed: int, port: int,
                run_prefix: str, straggler_delay: float) -> tuple:
    """One-epoch PS run. Returns (new_params, sys_tp)."""
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    cfg = dict(_base_cfg(seed), ps_discipline="bsp")
    ctx = mp.get_context("spawn")
    params_q = ctx.Queue()
    result_q = ctx.Queue()

    server = ctx.Process(target=ps_server_process,
                         args=(_WORLD_SIZE, cfg, init_params, port), daemon=True)
    server.start()
    time.sleep(0.8)

    procs = []
    for rank in range(_WORLD_SIZE):
        wcfg = dict(cfg, run_id=f"{run_prefix}_r{rank}")
        sd   = straggler_delay if rank == _STRAGGLER_RANK else 0.0
        pq   = params_q if rank == 0 else None
        p = ctx.Process(target=run_ps_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params,
                              result_q, port, sd, pq),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()

    new_params = params_q.get_nowait() if not params_q.empty() else init_params
    sys_tp = _get_system_tp(run_prefix)
    return new_params, sys_tp


def _run_one_rar(init_params: dict, seed: int,
                 run_prefix: str, straggler_delay: float) -> tuple:
    """One-epoch RAR run. Returns (new_params, sys_tp)."""
    from workers.worker import run_rar_worker
    from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model
    from workers.models import build_model

    cfg         = _base_cfg(seed)
    model       = build_model("logreg", seed=seed)
    shapes      = param_shapes_from_model(model)
    shared_bufs = make_shared_buffers(_WORLD_SIZE, shapes)

    ctx = mp.get_context("spawn")
    params_q = ctx.Queue()
    result_q = ctx.Queue()

    procs = []
    for rank in range(_WORLD_SIZE):
        wcfg = dict(cfg, run_id=f"{run_prefix}_r{rank}")
        sd   = straggler_delay if rank == _STRAGGLER_RANK else 0.0
        pq   = params_q if rank == 0 else None
        p = ctx.Process(target=run_rar_worker,
                        args=(rank, _WORLD_SIZE, wcfg, init_params,
                              result_q, shared_bufs, sd, pq),
                        daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)

    new_params = params_q.get_nowait() if not params_q.empty() else init_params
    sys_tp = _get_system_tp(run_prefix)
    return new_params, sys_tp


def _feed_telemetry(monitor, world_size: int, straggler_active: bool) -> None:
    """
    Inject simulated per-epoch telemetry into the MetricsMonitor so the
    AdaptiveController can evaluate it.

    Clean epoch     -> all workers at similar clock -> lag_ratio ~0   -> RAR
    Straggler epoch -> rank 3 lagging behind        -> lag_ratio >2.0 -> PS

    Lag formula: (max_clock - min_clock + 1) / (median_clock + 1)
      straggler: (500 - 10 + 1) / (100 + 1) = 491/101 = 4.86  [> 2.0 threshold]
      clean:     (503 - 500 + 1) / (501 + 1) = 4/502   = 0.008 [< 2.0 threshold]
    """
    monitor.clear()
    if straggler_active:
        monitor.record(0, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(1, clock=500, comm_ms=10.0, compute_ms=8.0)
        monitor.record(2, clock=100, comm_ms=12.0, compute_ms=10.0)
        monitor.record(3, clock=10,  comm_ms=30.0, compute_ms=24.0)
    else:
        for r in range(world_size):
            monitor.record(r, clock=500 + r, comm_ms=10.0, compute_ms=8.0)
    for r in range(world_size):
        monitor.heartbeat(r)


# ── per-seed runs ──────────────────────────────────────────────────────────────

def _run_hybrid(seed: int, base_port: int) -> tuple:
    """
    5-epoch adaptive run. The AdaptiveController chooses topology each epoch
    based on telemetry from the PREVIOUS epoch.

    Returns (epoch_records, switch_log, next_free_port).
    """
    from monitor.metrics_monitor import MetricsMonitor
    from controller.adaptive_controller import AdaptiveController
    from workers.models import build_model

    ctrl_cfg = {
        "straggler_lag_ratio":   1.5,    # 4-worker lag formula peaks ~1.63 under straggler
        "heartbeat_timeout_ms":  30_000,
        "switching_cost_margin": 0.0,
        "hysteresis_rounds":     1,      # allow switching every epoch
        "min_bandwidth_mbs":     0.0,
        "mode":                  "rar",  # initial topology
    }
    monitor    = MetricsMonitor()
    controller = AdaptiveController(ctrl_cfg, monitor)

    model  = build_model("logreg", seed=seed)
    params = model.get_params()
    port   = base_port

    epoch_records: list[dict] = []
    prev_straggler = False      # first epoch: assume clean conditions

    for epoch in range(1, _N_EPOCHS + 1):
        straggler_active = epoch in _STRAGGLER_EPOCHS
        straggler_delay  = _STRAGGLER_DELAY if straggler_active else 0.0

        # Feed telemetry reflecting PREVIOUS epoch's observed conditions
        _feed_telemetry(monitor, _WORLD_SIZE, prev_straggler)

        # Controller decides topology for THIS epoch
        topology = controller.decide(params, round_num=epoch)

        run_prefix = f"e6_hybrid_seed{seed}_e{epoch}"
        print(f"\n[E6] Hybrid seed={seed} epoch={epoch}/{_N_EPOCHS} "
              f"| topo={topology.upper()} | straggler={straggler_active}")

        try:
            if topology == "ps":
                new_params, sys_tp = _run_one_ps(
                    params, seed, port, run_prefix, straggler_delay)
                port += 1
            else:
                new_params, sys_tp = _run_one_rar(
                    params, seed, run_prefix, straggler_delay)

            val_acc = _get_val_acc(run_prefix)
            params  = new_params
            print(f"  tp={sys_tp:.0f}  val_acc={val_acc:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            sys_tp  = 0.0
            val_acc = 0.0

        epoch_records.append({
            "epoch":     epoch,
            "topology":  topology,
            "sys_tp":    sys_tp,
            "val_acc":   val_acc,
            "straggler": straggler_active,
        })
        prev_straggler = straggler_active
        time.sleep(1.0)

    switch_log = [
        {"round": s["round"], "from": s["from"], "to": s["to"]}
        for s in controller.switch_log
    ]
    return epoch_records, switch_log, port


def _run_static(mode: str, seed: int, base_port: int) -> tuple:
    """
    5-epoch static run (always PS or always RAR) with the same straggler
    schedule.  Returns (epoch_records, next_free_port).
    """
    from workers.models import build_model

    model  = build_model("logreg", seed=seed)
    params = model.get_params()
    port   = base_port

    epoch_records: list[dict] = []

    for epoch in range(1, _N_EPOCHS + 1):
        straggler_active = epoch in _STRAGGLER_EPOCHS
        straggler_delay  = _STRAGGLER_DELAY if straggler_active else 0.0

        run_prefix = f"e6_static_{mode}_seed{seed}_e{epoch}"
        print(f"\n[E6] Static-{mode.upper()} seed={seed} epoch={epoch}/{_N_EPOCHS} "
              f"| straggler={straggler_active}")

        try:
            if mode == "ps":
                new_params, sys_tp = _run_one_ps(
                    params, seed, port, run_prefix, straggler_delay)
                port += 1
            else:
                new_params, sys_tp = _run_one_rar(
                    params, seed, run_prefix, straggler_delay)

            val_acc = _get_val_acc(run_prefix)
            params  = new_params
            print(f"  tp={sys_tp:.0f}  val_acc={val_acc:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            sys_tp  = 0.0
            val_acc = 0.0

        epoch_records.append({
            "epoch":     epoch,
            "topology":  mode,
            "sys_tp":    sys_tp,
            "val_acc":   val_acc,
            "straggler": straggler_active,
        })
        time.sleep(1.0)

    return epoch_records, port


# ── main ───────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_switching import plot_switching

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  E6: Adaptive Switching  (w={_WORLD_SIZE}, epochs={_N_EPOCHS})")
    print(f"  Straggler: rank {_STRAGGLER_RANK}, {_STRAGGLER_FACTOR}x, "
          f"epochs {sorted(_STRAGGLER_EPOCHS)}")
    print(f"  seeds={_SEEDS}")
    print("=" * 60)

    port     = 5900
    all_rows: list[dict] = []
    all_sw:   list[dict] = []

    for seed in _SEEDS:
        print(f"\n{'=' * 40}")
        print(f"  Seed = {seed}")
        print("=" * 40)

        # Hybrid
        hybrid_recs, switch_log, port = _run_hybrid(seed, port)
        for r in hybrid_recs:
            all_rows.append({"seed": seed, "mode": "hybrid", **r})
        for s in switch_log:
            all_sw.append({"seed": seed, **s})

        # Static PS
        ps_recs, port = _run_static("ps", seed, port)
        for r in ps_recs:
            all_rows.append({"seed": seed, "mode": "ps", **r})

        # Static RAR
        rar_recs, port = _run_static("rar", seed, port)
        for r in rar_recs:
            all_rows.append({"seed": seed, "mode": "rar", **r})

        time.sleep(2.0)

    # ── Save tables ──────────────────────────────────────────────────────────
    csv_path = "results/tables/e6_adaptive.csv"
    fields   = ["seed", "mode", "epoch", "topology",
                "sys_tp", "val_acc", "straggler"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[E6] Table: {csv_path}")

    sw_path = "results/tables/e6_switches.csv"
    with open(sw_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "round", "from", "to"])
        writer.writeheader()
        writer.writerows(all_sw)
    print(f"[E6] Switches: {sw_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        plot_switching(all_rows, all_sw, out_dir="results/plots")
        print("[E6] Plot saved.")
    except Exception as exc:
        print(f"[E6] Plot skipped: {exc}")

    print("\n[E6] DONE")
    _print_summary(all_rows, all_sw)


def _print_summary(rows: list[dict], switches: list[dict]) -> None:
    from collections import defaultdict

    print(f"\n{'mode':<8} {'epoch':>5} {'topology':>8} {'avg_tp':>9} {'avg_acc':>8}")
    print("-" * 45)

    bucket: dict = defaultdict(list)
    for r in rows:
        bucket[(r["mode"], r["epoch"])].append((r["sys_tp"], r["val_acc"]))

    for mode in ["hybrid", "ps", "rar"]:
        for epoch in range(1, _N_EPOCHS + 1):
            key = (mode, epoch)
            if key not in bucket:
                continue
            tps, accs = zip(*bucket[key])
            avg_tp  = sum(tps)  / len(tps)
            avg_acc = sum(accs) / len(accs)
            topos   = [r["topology"] for r in rows
                       if r["mode"] == mode and r["epoch"] == epoch]
            topo_str = topos[0] if len(set(topos)) == 1 else "/".join(set(topos))
            print(f"{mode:<8} {epoch:>5} {topo_str:>8} {avg_tp:>9.0f} {avg_acc:>8.4f}")

    if switches:
        print("\nController switches (seed | round | from -> to):")
        for s in switches:
            print(f"  seed={s['seed']}  round={s['round']}  "
                  f"{s['from']} -> {s['to']}")
