"""
Single-worker SGD training loop (Phase 1).

train_single(config) → list[dict]
  - Trains for config["epochs"] epochs on the full dataset shard for rank 0.
  - Logs one dict per iteration: {epoch, iter, loss, acc, compute_ms, wall_sec}
  - Prints a one-line summary per epoch.
  - Returns the full metrics list (also written to a .jsonl file by RunLogger).

Used directly by `run.py --mode single` and as the local compute kernel
by distributed workers in Phases 2+.
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn

from workers.data_loader import get_shard, get_full_val_loader
from workers.models import LogReg, SmallCNN, build_model
from workers.logger import RunLogger


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── LogReg training step ───────────────────────────────────────────────────────

def _logreg_step(
    model: LogReg, X: np.ndarray, y: np.ndarray, lr: float
) -> tuple[float, float]:
    """One mini-batch forward + backward + update. Returns (loss, accuracy)."""
    grads = model.gradients(X, y)
    loss  = model.loss(X, y)
    model.update(grads, lr)
    preds = model.predict(X)
    acc   = (preds == y).mean()
    return float(loss), float(acc)


# ── CNN training step ──────────────────────────────────────────────────────────

def _cnn_step(
    model: SmallCNN,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    X: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    """One mini-batch forward + backward + update. Returns (loss, accuracy)."""
    optimizer.zero_grad()
    logits = model(X)
    loss   = criterion(logits, y)
    loss.backward()
    optimizer.step()
    preds = logits.argmax(dim=1)
    acc   = (preds == y).float().mean().item()
    return float(loss.item()), float(acc)


# ── Validation ────────────────────────────────────────────────────────────────

def evaluate_logreg(model: LogReg, loader) -> tuple[float, float]:
    all_X, all_y = [], []
    for Xb, yb in loader:
        all_X.append(_tensor_to_numpy(Xb).reshape(len(Xb), -1))
        all_y.append(_tensor_to_numpy(yb))
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    loss = model.loss(X, y)
    acc  = (model.predict(X) == y).mean()
    return float(loss), float(acc)


def evaluate_cnn(model: SmallCNN, loader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
    model.train()
    return total_loss / total, correct / total


# ── Main training function ─────────────────────────────────────────────────────

def train_single(config: dict, run_id: str | None = None) -> list[dict]:
    """
    Train on a single worker (rank=0, world_size=1).

    Parameters
    ----------
    config  : dict loaded from config/default.yaml (plus any overrides)
    run_id  : optional log file stem; auto-generated from config if None

    Returns
    -------
    metrics : list of per-iteration dicts (also persisted to .jsonl)
    """
    _set_seeds(config["seed"])

    model_name = config["model"].lower()
    dataset    = config["dataset"].lower()
    lr         = float(config["lr"])
    epochs     = int(config["epochs"])

    if run_id is None:
        run_id = f"single_{model_name}_{dataset}_seed{config['seed']}"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = get_shard(rank=0, world_size=1, config=config, train=True)
    val_loader   = get_full_val_loader(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name, seed=config["seed"])

    device = torch.device("cpu")
    optimizer = criterion = None
    if isinstance(model, SmallCNN):
        model = model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = nn.CrossEntropyLoss()

    # ── Logging ───────────────────────────────────────────────────────────────
    metrics: list[dict] = []
    logger  = RunLogger(run_id, config.get("log_dir", "results/raw"))

    print(f"\n{'-'*60}")
    print(f"  Training: {model_name.upper()} on {dataset.upper()}")
    print(f"  lr={lr}  batch={config['batch_size']}  epochs={epochs}  seed={config['seed']}")
    print(f"  Log: {logger.path()}")
    print(f"{'-'*60}")

    global_iter = 0
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        epoch_compute_ms = 0.0

        for Xb, yb in train_loader:
            t0 = time.perf_counter()

            if isinstance(model, LogReg):
                X_np = _tensor_to_numpy(Xb).reshape(len(Xb), -1).astype(np.float32)
                y_np = _tensor_to_numpy(yb).astype(np.int64)
                loss, acc = _logreg_step(model, X_np, y_np, lr)
            else:
                Xb, yb = Xb.to(device), yb.to(device)
                loss, acc = _cnn_step(model, optimizer, criterion, Xb, yb)

            compute_ms = (time.perf_counter() - t0) * 1000
            epoch_loss     += loss
            epoch_acc      += acc
            epoch_compute_ms += compute_ms
            n_batches      += 1
            global_iter    += 1

            row = {
                "phase": "train",
                "epoch": epoch,
                "iter":  global_iter,
                "loss":  round(loss, 6),
                "acc":   round(acc, 4),
                "compute_ms": round(compute_ms, 3),
            }
            logger.log(row)
            metrics.append(row)

        # ── Epoch summary ─────────────────────────────────────────────────────
        avg_loss = epoch_loss / n_batches
        avg_acc  = epoch_acc  / n_batches
        avg_cmp  = epoch_compute_ms / n_batches

        # Validation
        if isinstance(model, LogReg):
            val_loss, val_acc = evaluate_logreg(model, val_loader)
        else:
            val_loss, val_acc = evaluate_cnn(model, val_loader, device)

        elapsed   = time.perf_counter() - train_start
        n_samples = len(train_loader.dataset)
        throughput = n_samples * epoch / elapsed

        summary = {
            "phase": "epoch_summary",
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_acc":  round(avg_acc, 4),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 4),
            "avg_compute_ms": round(avg_cmp, 3),
            "elapsed_sec": round(elapsed, 2),
            "throughput_samples_sec": round(throughput, 1),
        }
        logger.log(summary)
        metrics.append(summary)

        print(
            f"  Epoch {epoch:>2}/{epochs} | "
            f"loss={avg_loss:.4f} acc={avg_acc*100:.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% | "
            f"{throughput:.0f} samples/s"
        )

    logger.close()
    print(f"{'-'*60}")
    print(f"  Done. Final val acc: {val_acc*100:.1f}%")
    print(f"  Log saved: {logger.path()}\n")
    return metrics
