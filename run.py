"""
Unified CLI entry point for the Adaptive Hybrid SGD framework.

Usage examples
--------------
Phase 1 — single worker:
    python run.py --mode single --model logreg --dataset mnist --epochs 10 --seed 42
    python run.py --mode single --model cnn    --dataset cifar10 --epochs 5  --seed 42

Phase 2+ (stubs, to be filled in later phases):
    python run.py --mode ps     --num-workers 4
    python run.py --mode rar    --num-workers 4
    python run.py --mode hybrid --num-workers 4

Experiments:
    python run.py --exp e1
    python run.py --exp e2
    ... (all experiments run via experiments/eN_*.py)
"""

import argparse
import sys
import yaml
from pathlib import Path


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/default.yaml", overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ── Mode handlers ──────────────────────────────────────────────────────────────

def run_single(config: dict) -> None:
    from workers.trainer import train_single
    from analysis.plot_loss_curves import plot_convergence, plot_throughput

    run_id = f"single_{config['model']}_{config['dataset']}_seed{config['seed']}"
    train_single(config, run_id=run_id)

    log_path = str(Path(config.get("log_dir", "results/raw")) / f"{run_id}.jsonl")
    plot_convergence([log_path], out_dir="results/plots")
    plot_throughput([log_path],  out_dir="results/plots")
    print(f"\n[run] Plots saved to results/plots/")


def run_ps(config: dict) -> None:
    raise NotImplementedError("Phase 2 (Parameter Server) not yet implemented.")


def run_rar(config: dict) -> None:
    raise NotImplementedError("Phase 3 (Ring AllReduce) not yet implemented.")


def run_hybrid(config: dict) -> None:
    raise NotImplementedError("Phase 5 (Adaptive Hybrid) not yet implemented.")


# ── Experiment runners ─────────────────────────────────────────────────────────

_EXP_MODULES = {
    "e1": "experiments.e1_scalability",
    "e2": "experiments.e2_straggler",
    "e3": "experiments.e3_node_failure",
    "e4": "experiments.e4_bandwidth",
    "e5": "experiments.e5_model_size",
    "e6": "experiments.e6_adaptive",
}


def run_experiment(exp_id: str, config: dict) -> None:
    import importlib
    if exp_id not in _EXP_MODULES:
        print(f"Unknown experiment {exp_id!r}. Choose from: {list(_EXP_MODULES)}")
        sys.exit(1)
    mod = importlib.import_module(_EXP_MODULES[exp_id])
    mod.run(config)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Hybrid Distributed SGD Framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["single", "ps", "rar", "hybrid"],
                        default=None, help="Training mode")
    parser.add_argument("--exp", choices=list(_EXP_MODULES),
                        default=None, help="Run a named experiment (e1..e6)")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")

    # Per-run overrides (all optional — fall back to config file values)
    parser.add_argument("--model",       default=None, help="logreg | cnn")
    parser.add_argument("--dataset",     default=None, help="mnist | cifar10")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int, default=None, dest="batch_size")
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--seed",        type=int, default=None)

    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("mode", "exp", "config") and v is not None}
    config = load_config(args.config, overrides)

    if args.exp:
        run_experiment(args.exp, config)
    elif args.mode == "single":
        run_single(config)
    elif args.mode == "ps":
        run_ps(config)
    elif args.mode == "rar":
        run_rar(config)
    elif args.mode == "hybrid":
        run_hybrid(config)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
