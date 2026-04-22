"""Fig E5 — Comm-to-compute ratio: LogReg vs SmallCNN across PS and RAR."""

from __future__ import annotations
from pathlib import Path


def plot_model_size(rows: list[dict], out_dir: str = "results/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    models = ["logreg", "cnn"]
    modes  = ["ps", "rar"]
    colors = {"ps": "#2196F3", "rar": "#4CAF50"}
    x      = np.arange(len(models))
    w      = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_comm, ax_cmp, ax_ratio = axes

    # Index rows by (model, mode)
    data = {(r["model"], r["mode"]): r for r in rows}

    for i, mode in enumerate(modes):
        comm_vals  = [data.get((m, mode), {}).get("avg_comm_ms",          0) for m in models]
        cmp_vals   = [data.get((m, mode), {}).get("avg_compute_ms",       0) for m in models]
        ratio_vals = [data.get((m, mode), {}).get("comm_compute_ratio",   0) for m in models]

        offset = (i - 0.5) * w
        ax_comm.bar(x + offset, comm_vals,  w, label=mode.upper(),
                    color=colors[mode], alpha=0.85)
        ax_cmp.bar( x + offset, cmp_vals,   w, label=mode.upper(),
                    color=colors[mode], alpha=0.85)
        ax_ratio.bar(x + offset, ratio_vals, w, label=mode.upper(),
                     color=colors[mode], alpha=0.85)

    labels = [f"{m}\n({data.get((m,'ps'),{}).get('grad_kb',0):.0f} KB)"
              for m in models]

    for ax, title, ylabel in [
        (ax_comm,  "Avg Comm Time per Batch",    "comm_ms"),
        (ax_cmp,   "Avg Compute Time per Batch", "compute_ms"),
        (ax_ratio, "Comm / Compute Ratio",       "ratio  (higher = comm-bound)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # Annotate ratio bars with values
    for i, mode in enumerate(modes):
        for j, m in enumerate(models):
            ratio = data.get((m, mode), {}).get("comm_compute_ratio", 0)
            offset = (i - 0.5) * w
            ax_ratio.text(j + offset, ratio + 0.01, f"{ratio:.2f}",
                          ha="center", va="bottom", fontsize=8)

    fig.suptitle("E5 Model Size  —  Comm-to-Compute Ratio  (2 workers, BSP)",
                 fontweight="bold")
    fig.tight_layout()

    out_path = str(Path(out_dir) / "e5_model_size.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model_size] saved: {out_path}")


def main() -> None:
    import csv
    rows = []
    with open("results/tables/e5_model_size.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "model":              r["model"],
                "mode":               r["mode"],
                "param_count":        int(r["param_count"]),
                "grad_kb":            float(r["grad_kb"]),
                "avg_comm_ms":        float(r["avg_comm_ms"]),
                "avg_compute_ms":     float(r["avg_compute_ms"]),
                "comm_compute_ratio": float(r["comm_compute_ratio"]),
                "throughput":         float(r["throughput"]),
                "val_acc":            float(r["val_acc"]),
            })
    plot_model_size(rows)


if __name__ == "__main__":
    main()
