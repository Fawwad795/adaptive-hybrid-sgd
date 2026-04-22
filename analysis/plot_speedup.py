"""Fig A — Speedup vs number of workers (E1 scalability)."""

from __future__ import annotations
from pathlib import Path


def plot_speedup(
    rows: list[dict],
    baseline: float = 0.0,
    out_dir: str = "results/plots",
) -> None:
    """
    Parameters
    ----------
    rows    : list of dicts from e1_scalability.run(); each has
              {mode, workers, throughput_mean, throughput_std, speedup, efficiency}
    baseline: single-worker throughput (samples/sec) for annotation
    out_dir : directory to save the PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    modes  = sorted({r["mode"] for r in rows})
    colors = {"ps": "#2196F3", "rar": "#4CAF50", "hybrid": "#FF9800"}
    markers= {"ps": "o",       "rar": "s",        "hybrid": "^"}

    workers_all = sorted({r["workers"] for r in rows})
    x_ticks     = [1] + workers_all

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: Speedup ──────────────────────────────────────────────────────────
    ax1.plot([1, max(workers_all)], [1, max(workers_all)],
             "k--", lw=1, alpha=0.4, label="Linear ideal")

    for mode in modes:
        pts = sorted([(r["workers"], r["speedup"], r["throughput_std"] / max(baseline, 1))
                      for r in rows if r["mode"] == mode])
        xs  = [1] + [p[0] for p in pts]
        ys  = [1.0] + [p[1] for p in pts]
        ax1.plot(xs, ys, marker=markers.get(mode, "o"),
                 color=colors.get(mode, "gray"), label=mode.upper(), lw=2)

    ax1.set_xlabel("Number of workers")
    ax1.set_ylabel("Speedup S(n)")
    ax1.set_title("Speedup vs Workers")
    ax1.set_xticks(x_ticks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if baseline > 0:
        ax1.annotate(f"baseline {baseline:.0f} samp/s",
                     xy=(1, 1), xytext=(1.3, 0.85),
                     fontsize=8, color="gray")

    # ── Right: Efficiency ──────────────────────────────────────────────────────
    ax2.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4, label="Perfect (1.0)")

    for mode in modes:
        pts = sorted([(r["workers"], r["efficiency"]) for r in rows if r["mode"] == mode])
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        ax2.plot(xs, ys, marker=markers.get(mode, "o"),
                 color=colors.get(mode, "gray"), label=mode.upper(), lw=2)

    ax2.set_xlabel("Number of workers")
    ax2.set_ylabel("Efficiency E(n) = S(n)/n")
    ax2.set_title("Parallel Efficiency vs Workers")
    ax2.set_xticks(workers_all)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("E1 Scalability  —  MNIST / LogReg", fontweight="bold")
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e1_speedup.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_speedup] saved: {out_path}")


def main() -> None:
    import csv
    rows = []
    with open("results/tables/e1_scalability.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "mode":            r["mode"],
                "workers":         int(r["workers"]),
                "throughput_mean": float(r["throughput_mean"]),
                "throughput_std":  float(r["throughput_std"]),
                "speedup":         float(r["speedup"]),
                "efficiency":      float(r["efficiency"]),
            })
    plot_speedup(rows)


if __name__ == "__main__":
    main()
