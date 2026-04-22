"""Fig C — Throughput degradation under straggler slowdown (E2)."""

from __future__ import annotations
from pathlib import Path


def plot_straggler(rows: list[dict], out_dir: str = "results/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    modes   = sorted({r["mode"] for r in rows})
    colors  = {"ps": "#2196F3", "rar": "#4CAF50"}
    markers = {"ps": "o",       "rar": "s"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    factors_all = sorted({r["factor"] for r in rows})
    # Theoretical BSP lower bound: throughput = 1/factor
    ax.plot([1, max(factors_all)], [1.0, 1.0 / max(factors_all)],
            "k--", lw=1, alpha=0.4, label="BSP lower bound (1/f)")

    for mode in modes:
        mode_rows = sorted([r for r in rows if r["mode"] == mode],
                           key=lambda r: r["factor"])
        xs   = [r["factor"]        for r in mode_rows]
        ys   = [r["norm_throughput"] for r in mode_rows]
        stds = [r["throughput_std"] / max(r["throughput_mean"], 1)
                for r in mode_rows]
        ax.plot(xs, ys, marker=markers.get(mode, "o"),
                color=colors.get(mode, "gray"), label=mode.upper(), lw=2)
        ax.fill_between(xs,
                        [max(0, y - s) for y, s in zip(ys, stds)],
                        [y + s          for y, s in zip(ys, stds)],
                        color=colors.get(mode, "gray"), alpha=0.12)

    ax.set_xlabel("Straggler slowdown factor")
    ax.set_ylabel("Normalised throughput  (1 = no straggler)")
    ax.set_title("E2 Straggler  —  MNIST / LogReg  (4 workers, 1 straggler)")
    ax.set_xticks(factors_all)
    ax.set_xticklabels([f"{f}x" for f in factors_all])
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e2_straggler.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_straggler] saved: {out_path}")


def main() -> None:
    import csv
    rows = []
    with open("results/tables/e2_straggler.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "mode":            r["mode"],
                "factor":          int(r["factor"]),
                "throughput_mean": float(r["throughput_mean"]),
                "throughput_std":  float(r["throughput_std"]),
                "norm_throughput": float(r["norm_throughput"]),
            })
    plot_straggler(rows)


if __name__ == "__main__":
    main()
