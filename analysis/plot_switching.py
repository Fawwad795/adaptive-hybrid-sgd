"""Fig F — E6 Adaptive switching: topology timeline + throughput comparison."""

from __future__ import annotations
from pathlib import Path


def plot_switching(
    rows:     list[dict],
    switches: list[dict],
    out_dir:  str = "results/plots",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    epochs     = sorted({r["epoch"]  for r in rows})
    modes      = ["hybrid", "ps", "rar"]
    mode_labels= {"hybrid": "Adaptive Hybrid", "ps": "Static PS", "rar": "Static RAR"}
    colors     = {"ps": "#2196F3", "rar": "#4CAF50", "hybrid": "#9C27B0"}
    topo_color = {"ps": "#2196F3", "rar": "#4CAF50"}

    # Average per (mode, epoch) across seeds
    from collections import defaultdict
    bucket: dict = defaultdict(list)
    topo_bucket: dict = defaultdict(list)
    for r in rows:
        key = (r["mode"], r["epoch"])
        bucket[key].append(r["sys_tp"])
        topo_bucket[key].append(r["topology"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    fig, (ax_topo, ax_tp) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # ── Top panel: hybrid topology per epoch ────────────────────────────────
    ax_topo.set_title("Hybrid Topology Decision per Epoch  (avg across seeds)",
                       fontsize=11)
    ax_topo.set_xlim(0.5, len(epochs) + 0.5)
    ax_topo.set_ylim(-0.5, 1.5)
    ax_topo.set_yticks([0, 1])
    ax_topo.set_yticklabels(["PS", "RAR"], fontsize=10)
    ax_topo.set_xticks(epochs)
    ax_topo.set_xlabel("Epoch")
    ax_topo.grid(True, axis="x", alpha=0.3)

    # Straggler shading (epochs 3-4)
    ax_topo.axvspan(2.5, 4.5, color="orange", alpha=0.12, label="Straggler epochs 3-4")

    for epoch in epochs:
        topos = topo_bucket.get(("hybrid", epoch), ["rar"])
        # majority vote
        topo  = max(set(topos), key=topos.count)
        y_val = 0 if topo == "ps" else 1
        ax_topo.scatter(epoch, y_val,
                        color=topo_color.get(topo, "gray"),
                        s=200, zorder=5, edgecolors="black", linewidths=0.8)
        ax_topo.text(epoch, y_val + 0.18, topo.upper(),
                     ha="center", va="bottom", fontsize=8,
                     color=topo_color.get(topo, "gray"), fontweight="bold")

    # Draw switch arrows
    sw_epochs = sorted({s["round"] for s in switches})
    prev_topo = None
    for epoch in epochs:
        topos = topo_bucket.get(("hybrid", epoch), ["rar"])
        topo  = max(set(topos), key=topos.count)
        if prev_topo is not None and topo != prev_topo:
            ax_topo.annotate(
                "", xy=(epoch, 0 if topo == "ps" else 1),
                xytext=(epoch - 1, 1 if topo == "ps" else 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )
        prev_topo = topo

    ax_topo.legend(loc="upper right", fontsize=8)

    # ── Bottom panel: throughput comparison ─────────────────────────────────
    ax_tp.set_title("System Throughput per Epoch  (mean across seeds)", fontsize=11)
    ax_tp.axvspan(2.5, 4.5, color="orange", alpha=0.12, label="Straggler epochs 3-4")

    x = np.array(epochs)
    for mode in modes:
        ys = [avg(bucket.get((mode, ep), [0])) for ep in epochs]
        ax_tp.plot(x, ys,
                   marker="o" if mode == "hybrid" else ("s" if mode == "ps" else "^"),
                   color=colors[mode], label=mode_labels[mode], lw=2,
                   markersize=7 if mode == "hybrid" else 6,
                   zorder=3 if mode == "hybrid" else 2,
                   linestyle="-" if mode == "hybrid" else "--")

    ax_tp.set_xlabel("Epoch")
    ax_tp.set_ylabel("System throughput (samples/sec)")
    ax_tp.set_xticks(epochs)
    ax_tp.legend(fontsize=9)
    ax_tp.grid(True, alpha=0.3)

    # Annotate switch events from switch_log
    sw_rounds = sorted({s["round"] for s in switches})
    for rnd in sw_rounds:
        ax_tp.axvline(rnd - 0.5, color="red", ls=":", lw=1.2, alpha=0.7)
        ax_tp.text(rnd - 0.45,
                   ax_tp.get_ylim()[1] * 0.97 if ax_tp.get_ylim()[1] > 0 else 1,
                   "switch", color="red", fontsize=7, va="top")

    fig.suptitle(
        "E6 Adaptive Switching  —  MNIST / LogReg  (4 workers, BSP)",
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e6_adaptive.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_switching] saved: {out_path}")


def main() -> None:
    import csv
    rows: list[dict] = []
    with open("results/tables/e6_adaptive.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "seed":      int(r["seed"]),
                "mode":      r["mode"],
                "epoch":     int(r["epoch"]),
                "topology":  r["topology"],
                "sys_tp":    float(r["sys_tp"]),
                "val_acc":   float(r["val_acc"]),
                "straggler": r["straggler"] == "True",
            })
    switches: list[dict] = []
    sw_path = "results/tables/e6_switches.csv"
    if Path(sw_path).exists():
        with open(sw_path, newline="") as fh:
            for r in csv.DictReader(fh):
                switches.append({
                    "seed":  int(r["seed"]),
                    "round": int(r["round"]),
                    "from":  r["from"],
                    "to":    r["to"],
                })
    plot_switching(rows, switches)


if __name__ == "__main__":
    main()
