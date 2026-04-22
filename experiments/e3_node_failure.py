"""
E3 — Node failure experiment.

One worker (rank = world_size - 1) is killed mid-training after epoch 1.
PS is run in *async* discipline so the remaining workers can continue
without waiting for the dead worker.

Metrics per trial
-----------------
  baseline_val_acc  — 4 workers, no failure, async, final epoch
  survivor_val_acc  — 3 survivors, final epoch (worker 3 killed after epoch 1)
  acc_drop          — baseline_val_acc - survivor_val_acc
  survivor_tp       — system throughput of 3 surviving workers

Output
------
results/tables/e3_node_failure.csv
results/plots/e3_recovery.png
"""

from __future__ import annotations
import csv
import json
import threading
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZE     = 4
_STRAGGLER_RANK = _WORLD_SIZE - 1
_SEEDS          = [42, 123, 456]
_EPOCHS         = 3          # enough to see recovery across epochs 2-3
_KILL_DELAY_S   = 22.0       # seconds after workers start → kills mid-training (epoch 1~2)
_LOG_DIR        = "results/raw"


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_final_val_acc(prefix: str) -> float:
    p = Path(f"{prefix}.jsonl")
    if not p.exists():
        return 0.0
    rows = [json.loads(l) for l in open(p) if l.strip()]
    sums = [x for x in rows if x.get("phase") == "epoch_summary"]
    return sums[-1]["val_acc"] if sums else 0.0


def _get_system_tp(prefix: str, world_size: int) -> float:
    total = 0.0
    for r in range(world_size):
        p = Path(f"{prefix}_r{r}.jsonl")
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        sums = [x for x in rows if x.get("phase") == "epoch_summary"]
        if sums:
            total += sums[-1]["throughput_samples_sec"]
    return total


def _base_config(seed: int) -> dict:
    return dict(dataset="mnist", model="logreg", lr=0.01, batch_size=64,
                epochs=_EPOCHS, seed=seed, num_workers=_WORLD_SIZE,
                ps_discipline="async", data_dir="data", log_dir=_LOG_DIR)


def _run_ps_trial(seed: int, port: int, kill_rank: int | None = None) -> None:
    """Spawn server + workers; optionally kill kill_rank after _KILL_DELAY_S."""
    from workers.models import build_model
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    label = f"e3_{'fail' if kill_rank is not None else 'base'}_seed{seed}"
    cfg   = dict(_base_config(seed))
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

    # Inject failure: kill the designated worker after a delay
    if kill_rank is not None:
        def _do_kill():
            target = procs[kill_rank]
            if target.is_alive():
                target.terminate()
                print(f"  [E3] killed worker rank={kill_rank} after {_KILL_DELAY_S}s")
        timer = threading.Timer(_KILL_DELAY_S, _do_kill)
        timer.daemon = True
        timer.start()

    # Wait for surviving workers
    for rank, p in enumerate(procs):
        if rank == kill_rank:
            p.join(timeout=10)   # already dead or will be soon
        else:
            p.join(timeout=600)

    # Server may block waiting for killed worker's STOP — terminate with timeout
    server.join(timeout=15)
    if server.is_alive():
        server.terminate()


# ── main ───────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_recovery import plot_recovery

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  E3: Node Failure  —  MNIST / LogReg  (w={_WORLD_SIZE}, async PS)")
    print(f"  seeds={_SEEDS}  epochs={_EPOCHS}  kill_delay={_KILL_DELAY_S}s")
    print(f"  killed_worker=rank{_STRAGGLER_RANK}")
    print("=" * 60)

    port = 5750
    rows: list[dict] = []

    for seed in _SEEDS:
        # ── Baseline: no failure ───────────────────────────────────────────────
        print(f"\n[E3] baseline  seed={seed}")
        _run_ps_trial(seed, port, kill_rank=None)
        port += 1
        base_acc = _get_final_val_acc(
            f"{_LOG_DIR}/e3_base_seed{seed}_r0")
        base_tp  = _get_system_tp(
            f"{_LOG_DIR}/e3_base_seed{seed}", _WORLD_SIZE)
        print(f"  base val_acc={base_acc:.4f}  base_tp={base_tp:.1f}")
        time.sleep(1.5)

        # ── Failure: kill worker _STRAGGLER_RANK after _KILL_DELAY_S s ────────
        print(f"\n[E3] failure   seed={seed}  (killing rank{_STRAGGLER_RANK})")
        _run_ps_trial(seed, port, kill_rank=_STRAGGLER_RANK)
        port += 1
        surv_acc = _get_final_val_acc(
            f"{_LOG_DIR}/e3_fail_seed{seed}_r0")
        surv_tp  = _get_system_tp(
            f"{_LOG_DIR}/e3_fail_seed{seed}", _WORLD_SIZE)
        print(f"  surv val_acc={surv_acc:.4f}  surv_tp={surv_tp:.1f}")
        time.sleep(1.5)

        rows.append({
            "seed":           seed,
            "base_val_acc":   round(base_acc,  4),
            "surv_val_acc":   round(surv_acc,  4),
            "acc_drop":       round(base_acc - surv_acc, 4),
            "base_tp":        round(base_tp,   1),
            "surv_tp":        round(surv_tp,   1),
            "tp_ratio":       round(surv_tp / base_tp, 4) if base_tp > 0 else 0.0,
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = "results/tables/e3_node_failure.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[E3] Table saved: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        plot_recovery(rows, out_dir="results/plots")
        print("[E3] Plot saved.")
    except Exception as exc:
        print(f"[E3] Plot skipped: {exc}")

    print("\n[E3] DONE")
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    avg = lambda key: sum(r[key] for r in rows) / len(rows)
    print(f"\n{'seed':>6} {'base_acc':>9} {'surv_acc':>9} {'acc_drop':>9} {'tp_ratio':>9}")
    print("-" * 48)
    for r in rows:
        print(f"{r['seed']:>6} {r['base_val_acc']:>9.4f} {r['surv_val_acc']:>9.4f} "
              f"{r['acc_drop']:>9.4f} {r['tp_ratio']:>9.4f}")
    print(f"{'avg':>6} {avg('base_val_acc'):>9.4f} {avg('surv_val_acc'):>9.4f} "
          f"{avg('acc_drop'):>9.4f} {avg('tp_ratio'):>9.4f}")
