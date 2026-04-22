"""
E5 — Model size experiment.

Compares LogReg (~7.8 K params, 31 KB grad) vs SmallCNN (~62 K params, 248 KB grad)
across PS and RAR at 2 workers.

Key metric: comm-to-compute ratio = avg_comm_ms / avg_compute_ms
  - High ratio → comm-bound → topology choice matters a lot
  - Low  ratio → compute-bound → topology choice matters less

Output
------
results/tables/e5_model_size.csv
results/plots/e5_model_size.png
"""

from __future__ import annotations
import csv
import json
import time
import multiprocessing as mp
from pathlib import Path

_WORLD_SIZE = 2
_SEEDS      = [42, 123, 456]
_EPOCHS     = 2
_LOG_DIR    = "results/raw"

# Model configs
_MODELS = {
    "logreg": {"dataset": "mnist",   "model": "logreg", "batch_size": 64,
               "param_count": 784 * 10 + 10,
               "grad_bytes":  (784 * 10 + 10) * 4},
    "cnn":    {"dataset": "cifar10", "model": "cnn",    "batch_size": 64,
               "param_count": 62_006,
               "grad_bytes":  62_006 * 4},
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_metrics(prefix: str) -> dict:
    """Return avg_comm_ms, avg_compute_ms, throughput from rank-0 log."""
    p = Path(f"{prefix}_r0.jsonl")
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in open(p) if l.strip()]
    sums = [r for r in rows if r.get("phase") == "epoch_summary"]
    if not sums:
        return {}
    last = sums[-1]
    return {
        "comm_ms":    last.get("avg_comm_ms", 0.0),
        "compute_ms": last.get("avg_compute_ms", 0.0),
        "val_acc":    last.get("val_acc", 0.0),
        "throughput": last.get("throughput_samples_sec", 0.0),
    }


def _get_system_tp(prefix: str) -> float:
    total = 0.0
    for r in range(_WORLD_SIZE):
        p = Path(f"{prefix}_r{r}.jsonl")
        if not p.exists():
            return 0.0
        rows = [json.loads(l) for l in open(p) if l.strip()]
        sums = [x for x in rows if x.get("phase") == "epoch_summary"]
        if sums:
            total += sums[-1].get("throughput_samples_sec", 0.0)
    return total


def _base_config(model_key: str, seed: int) -> dict:
    mc = _MODELS[model_key]
    return dict(lr=0.01, epochs=_EPOCHS, seed=seed,
                num_workers=_WORLD_SIZE, data_dir="data", log_dir=_LOG_DIR,
                dataset=mc["dataset"], model=mc["model"],
                batch_size=mc["batch_size"])


def _run_ps(model_key: str, seed: int, port: int) -> str:
    from workers.models import build_model
    from workers.worker import run_ps_worker
    from ps_engine.server import ps_server_process

    label = f"e5_ps_{model_key}_seed{seed}"
    cfg   = dict(_base_config(model_key, seed), ps_discipline="bsp")
    model = build_model(cfg["model"], seed=seed)
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
    return label


def _run_rar(model_key: str, seed: int) -> str:
    from workers.models import build_model
    from workers.worker import run_rar_worker
    from rar_engine.ring_allreduce import make_shared_buffers, param_shapes_from_model

    label = f"e5_rar_{model_key}_seed{seed}"
    cfg   = _base_config(model_key, seed)
    model = build_model(cfg["model"], seed=seed)
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
    return label


# ── main ───────────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    from analysis.plot_model_size import plot_model_size

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  E5: Model Size  —  LogReg vs CNN  (w={_WORLD_SIZE})")
    print(f"  seeds={_SEEDS}  epochs={_EPOCHS}")
    print("=" * 60)

    port = 5850
    rows: list[dict] = []

    for model_key, minfo in _MODELS.items():
        for mode in ["ps", "rar"]:
            comm_list, cmp_list, tp_list, acc_list = [], [], [], []

            for seed in _SEEDS:
                print(f"\n[E5] {mode.upper()}  model={model_key}  seed={seed}")
                try:
                    if mode == "ps":
                        label = _run_ps(model_key, seed, port)
                        port += 1
                    else:
                        label = _run_rar(model_key, seed)

                    m = _get_metrics(f"{_LOG_DIR}/{label}")
                    sys_tp = _get_system_tp(f"{_LOG_DIR}/{label}")
                    comm_list.append(m.get("comm_ms", 0.0))
                    cmp_list.append(m.get("compute_ms", 0.0))
                    tp_list.append(sys_tp)
                    acc_list.append(m.get("val_acc", 0.0))
                    print(f"  comm={m.get('comm_ms',0):.1f}ms  "
                          f"cmp={m.get('compute_ms',0):.1f}ms  "
                          f"tp={sys_tp:.0f}  val_acc={m.get('val_acc',0):.4f}")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    comm_list.append(0.0); cmp_list.append(0.0)
                    tp_list.append(0.0);   acc_list.append(0.0)
                time.sleep(1.5)

            def avg(lst): return sum(lst) / len(lst) if lst else 0.0

            avg_comm = avg(comm_list)
            avg_cmp  = avg(cmp_list)
            ratio    = avg_comm / avg_cmp if avg_cmp > 0 else 0.0

            rows.append({
                "model":        model_key,
                "mode":         mode,
                "param_count":  minfo["param_count"],
                "grad_kb":      round(minfo["grad_bytes"] / 1024, 1),
                "avg_comm_ms":  round(avg_comm, 3),
                "avg_compute_ms": round(avg_cmp, 3),
                "comm_compute_ratio": round(ratio, 4),
                "throughput":   round(avg(tp_list), 1),
                "val_acc":      round(avg(acc_list), 4),
            })
            print(f"  => comm/cmp ratio={ratio:.3f}  tp={avg(tp_list):.0f}")

    # Save CSV
    csv_path = "results/tables/e5_model_size.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[E5] Table saved: {csv_path}")

    # Plot
    try:
        plot_model_size(rows, out_dir="results/plots")
        print("[E5] Plot saved.")
    except Exception as exc:
        print(f"[E5] Plot skipped: {exc}")

    print("\n[E5] DONE")
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'model':<8} {'mode':<5} {'grad_kb':>8} {'comm_ms':>8} "
          f"{'cmp_ms':>7} {'ratio':>7} {'val_acc':>8}")
    print("-" * 58)
    for r in rows:
        print(f"{r['model']:<8} {r['mode']:<5} {r['grad_kb']:>8.1f} "
              f"{r['avg_comm_ms']:>8.2f} {r['avg_compute_ms']:>7.2f} "
              f"{r['comm_compute_ratio']:>7.3f} {r['val_acc']:>8.4f}")
