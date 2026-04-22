"""Fig D — Accuracy and throughput impact of node failure (E3)."""

from __future__ import annotations
from pathlib import Path


def plot_recovery(rows: list[dict], out_dir: str = "results/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seeds      = [r["seed"]          for r in rows]
    base_accs  = [r["base_val_acc"]  for r in rows]
    surv_accs  = [r["surv_val_acc"]  for r in rows]
    tp_ratios  = [r["tp_ratio"]      for r in rows]

    x = np.arange(len(seeds))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left: val_acc comparison ───────────────────────────────────────────────
    bars1 = ax1.bar(x - w/2, base_accs, w, label="Baseline (4 workers)",
                    color="#2196F3", alpha=0.85)
    bars2 = ax1.bar(x + w/2, surv_accs, w, label="After failure (3 workers)",
                    color="#FF5722", alpha=0.85)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Final val accuracy")
    ax1.set_title("Accuracy: Baseline vs After Failure")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_ylim(0.8, 0.95)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    avg_base = sum(base_accs) / len(base_accs)
    avg_surv = sum(surv_accs) / len(surv_accs)
    ax1.axhline(avg_base, color="#2196F3", ls="--", lw=1, alpha=0.6)
    ax1.axhline(avg_surv, color="#FF5722", ls="--", lw=1, alpha=0.6)

    # ── Right: throughput ratio ────────────────────────────────────────────────
    colors = ["#4CAF50" if r >= 0.75 else "#FF9800" for r in tp_ratios]
    ax2.bar(x, tp_ratios, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.4, label="Baseline (1.0)")
    ax2.axhline(0.75, color="gray", ls=":", lw=1, alpha=0.5, label="75% threshold")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Throughput ratio  (survivor / baseline)")
    ax2.set_title("Throughput After Node Failure")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("E3 Node Failure  —  MNIST / LogReg  (4 workers, async PS)",
                 fontweight="bold")
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e3_recovery.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_recovery] saved: {out_path}")


def main() -> None:
    import csv
    rows = []
    with open("results/tables/e3_node_failure.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k != "seed" else int(v))
                         for k, v in r.items()})
    plot_recovery(rows)


if __name__ == "__main__":
    main()
