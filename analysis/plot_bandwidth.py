"""Fig E — PS vs RAR throughput across bandwidth regimes (E4 'money plot')."""

from __future__ import annotations
from pathlib import Path


def plot_bandwidth(rows: list[dict], out_dir: str = "results/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Group by mode
    modes  = sorted({r["mode"] for r in rows})
    colors = {"ps": "#2196F3", "rar": "#4CAF50"}
    markers= {"ps": "o",       "rar": "s"}

    # Order bandwidths from low to high
    bw_order = ["10M", "100M", "1G", "10G"]
    bw_mbps  = {"10M": 10, "100M": 100, "1G": 1_000, "10G": 10_000}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for mode in modes:
        mode_rows = {r["bandwidth"]: r for r in rows if r["mode"] == mode}
        xs   = list(range(len(bw_order)))
        ys   = [mode_rows[bw]["throughput_mean"]  for bw in bw_order if bw in mode_rows]
        stds = [mode_rows[bw]["throughput_std"]   for bw in bw_order if bw in mode_rows]
        valid_x = [i for i, bw in enumerate(bw_order) if bw in mode_rows]

        ax.plot(valid_x, ys, marker=markers.get(mode, "o"),
                color=colors.get(mode, "gray"), label=mode.upper(), lw=2)
        ax.fill_between(valid_x,
                        [max(0, y - s) for y, s in zip(ys, stds)],
                        [y + s          for y, s in zip(ys, stds)],
                        color=colors.get(mode, "gray"), alpha=0.12)

    ax.set_xticks(range(len(bw_order)))
    ax.set_xticklabels(bw_order)
    ax.set_xlabel("Simulated bandwidth")
    ax.set_ylabel("System throughput (samples/sec)")
    ax.set_title("E4 Bandwidth  —  PS vs RAR Crossover  (MNIST / LogReg, 4 workers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate crossover if it exists
    ps_tps  = {r["bandwidth"]: r["throughput_mean"] for r in rows if r["mode"] == "ps"}
    rar_tps = {r["bandwidth"]: r["throughput_mean"] for r in rows if r["mode"] == "rar"}
    prev_winner = None
    for i, bw in enumerate(bw_order):
        if bw not in ps_tps or bw not in rar_tps:
            continue
        winner = "ps" if ps_tps[bw] > rar_tps[bw] else "rar"
        if prev_winner is not None and winner != prev_winner:
            ax.axvline(i - 0.5, color="red", ls="--", lw=1.5, alpha=0.6,
                       label=f"Crossover ~{bw}")
            ax.annotate("Crossover", xy=(i - 0.5, ax.get_ylim()[1] * 0.95),
                        xytext=(i - 0.3, ax.get_ylim()[1] * 0.95),
                        color="red", fontsize=9)
        prev_winner = winner

    ax.legend()
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e4_bandwidth.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_bandwidth] saved: {out_path}")


def main() -> None:
    import csv
    rows = []
    with open("results/tables/e4_bandwidth.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "bandwidth":       r["bandwidth"],
                "bandwidth_mbps":  int(r["bandwidth_mbps"]),
                "throttle_ms":     float(r["throttle_ms"]),
                "mode":            r["mode"],
                "throughput_mean": float(r["throughput_mean"]),
                "throughput_std":  float(r["throughput_std"]),
            })
    plot_bandwidth(rows)


if __name__ == "__main__":
    main()
