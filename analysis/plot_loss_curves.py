"""
Fig B — Training loss and accuracy vs wall-clock time.

Usage:
    python analysis/plot_loss_curves.py --log results/raw/single_logreg_mnist_seed42.jsonl
    python analysis/plot_loss_curves.py --log results/raw/*.jsonl  (overlay multiple runs)
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless / CI runs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from workers.logger import read_log


# ── Colour cycle shared across figures ────────────────────────────────────────
_COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def _parse_log(path: str) -> tuple[list, list]:
    """
    Split a .jsonl log into:
        iter_rows  — per-iteration train rows
        epoch_rows — epoch-summary rows
    """
    rows = read_log(path)
    iter_rows  = [r for r in rows if r.get("phase") == "train"]
    epoch_rows = [r for r in rows if r.get("phase") == "epoch_summary"]
    return iter_rows, epoch_rows


def plot_convergence(
    log_paths: list[str],
    out_dir: str = "results/plots",
    show: bool = False,
) -> list[str]:
    """
    Generate two side-by-side subplots:
        Left  — training loss vs wall-clock seconds  (epoch summaries)
        Right — validation accuracy vs wall-clock seconds

    One line per log file, colour-coded.
    Returns list of saved file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Phase 1 — Single-worker convergence", fontsize=13, fontweight="bold")

    for idx, path in enumerate(log_paths):
        _, epoch_rows = _parse_log(path)
        if not epoch_rows:
            print(f"[plot] No epoch summaries found in {path}, skipping.")
            continue

        wall   = [r["wall_sec"] for r in epoch_rows]
        t_loss = [r["train_loss"] for r in epoch_rows]
        v_acc  = [r["val_acc"] for r in epoch_rows]

        label = Path(path).stem
        c     = _COLOURS[idx % len(_COLOURS)]

        ax_loss.plot(wall, t_loss, marker="o", markersize=4, color=c, label=label, linewidth=1.8)
        ax_acc.plot( wall, [a * 100 for a in v_acc],
                     marker="s", markersize=4, color=c, label=label, linewidth=1.8)

    for ax, ylabel, title in [
        (ax_loss, "Cross-entropy loss",    "Training Loss vs Wall-clock Time"),
        (ax_acc,  "Validation accuracy %", "Validation Accuracy vs Wall-clock Time"),
    ]:
        ax.set_xlabel("Wall-clock time (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()

    # Derive output name from first log path
    stem    = Path(log_paths[0]).stem if log_paths else "convergence"
    outpath = str(Path(out_dir) / f"loss_{stem}.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    saved.append(outpath)
    print(f"[plot] Saved -> {outpath}")

    if show:
        plt.show()
    plt.close(fig)
    return saved


def plot_throughput(
    log_paths: list[str],
    out_dir: str = "results/plots",
    show: bool = False,
) -> list[str]:
    """Bar chart: mean samples/sec per run (from epoch summaries)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    labels, values = [], []
    for path in log_paths:
        _, epoch_rows = _parse_log(path)
        if not epoch_rows:
            continue
        tpts = [r["throughput_samples_sec"] for r in epoch_rows if "throughput_samples_sec" in r]
        if tpts:
            labels.append(Path(path).stem)
            values.append(sum(tpts) / len(tpts))

    if not labels:
        return saved

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color=_COLOURS[:len(labels)], width=0.5, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_ylabel("Samples / second", fontsize=10)
    ax.set_title("Training Throughput (single worker)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()

    outpath = str(Path(out_dir) / "throughput_single.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    saved.append(outpath)
    print(f"[plot] Saved -> {outpath}")

    if show:
        plt.show()
    plt.close(fig)
    return saved


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from .jsonl logs")
    parser.add_argument("--log", nargs="+", required=True,
                        help="Path(s) to .jsonl log files (globs accepted)")
    parser.add_argument("--out-dir", default="results/plots")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    # Expand globs
    paths: list[str] = []
    for pattern in args.log:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    if not paths:
        print("No log files found.")
        return

    plot_convergence(paths, out_dir=args.out_dir, show=args.show)
    plot_throughput(paths, out_dir=args.out_dir, show=args.show)


if __name__ == "__main__":
    main()
