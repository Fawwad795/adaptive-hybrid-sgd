"""
Phase 2+ — Distributed worker process.

Each worker:
1. Loads its assigned data shard.
2. Holds a local model replica initialised from shared params.
3. Computes gradients on each mini-batch.
4. Synchronises via the active topology (PS or RAR).
5. Emits per-iteration telemetry to the RunLogger.

Entry points
------------
run_ps_worker(rank, world_size, config, init_params, result_queue)
    — for Parameter Server mode (Phase 2)
run_rar_worker(rank, world_size, config, init_params, result_queue, shared_bufs)
    — for Ring AllReduce mode (Phase 3)
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn

from workers.data_loader import get_shard, get_full_val_loader
from workers.models import LogReg, SmallCNN, build_model
from workers.logger import RunLogger
from workers.trainer import evaluate_logreg, evaluate_cnn, _tensor_to_numpy


# ── Internal helpers ───────────────────────────────────────────────────────────

def _set_seeds(seed: int, rank: int) -> None:
    np.random.seed(seed + rank * 1000)
    torch.manual_seed(seed + rank * 1000)


def _compute_logreg_grad(model: LogReg, Xb, yb) -> tuple[dict, float, float]:
    X_np = _tensor_to_numpy(Xb).reshape(len(Xb), -1).astype(np.float32)
    y_np = _tensor_to_numpy(yb).astype(np.int64)
    loss  = model.loss(X_np, y_np)
    grads = model.gradients(X_np, y_np)
    preds = model.predict(X_np)
    acc   = float((preds == y_np).mean())
    return grads, float(loss), acc


def _compute_cnn_grad(model: SmallCNN, optimizer, criterion, Xb, yb, device):
    Xb, yb = Xb.to(device), yb.to(device)
    optimizer.zero_grad()
    logits = model(Xb)
    loss   = criterion(logits, yb)
    loss.backward()
    grads = model.gradients()
    preds = logits.argmax(dim=1)
    acc   = float((preds == yb).float().mean().item())
    return grads, float(loss.item()), acc


# ── PS Worker ──────────────────────────────────────────────────────────────────

def run_ps_worker(
    rank:        int,
    world_size:  int,
    config:      dict,
    init_params: dict[str, np.ndarray],
    result_queue,
    port:        int = 5555,
    straggler_delay: float = 0.0,
) -> None:
    """Training loop for a single PS worker."""
    from ps_engine.client import PSClient

    _set_seeds(config["seed"], rank)

    model_name = config["model"].lower()
    lr         = float(config["lr"])
    epochs     = int(config["epochs"])
    run_id     = config.get("run_id", f"ps_worker{rank}_{model_name}")

    train_loader = get_shard(rank=rank, world_size=world_size, config=config, train=True)
    val_loader   = get_full_val_loader(config)

    model = build_model(model_name, seed=config["seed"])
    model.set_params(init_params)

    device = torch.device("cpu")
    optimizer = criterion = None
    if isinstance(model, SmallCNN):
        model = model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = nn.CrossEntropyLoss()

    logger = RunLogger(run_id, config.get("log_dir", "results/raw"))
    client = PSClient(rank=rank, port=port,
                      throttle_ms=float(config.get("throttle_ms", 0.0)))
    client.connect()

    metrics: list[dict] = []
    global_iter  = 0
    clock        = 0
    train_start  = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        epoch_compute_ms, epoch_comm_ms  = 0.0, 0.0

        for Xb, yb in train_loader:
            if straggler_delay > 0:
                time.sleep(straggler_delay)

            t_compute = time.perf_counter()
            if isinstance(model, LogReg):
                grads, loss, acc = _compute_logreg_grad(model, Xb, yb)
            else:
                grads, loss, acc = _compute_cnn_grad(model, optimizer, criterion,
                                                      Xb, yb, device)
            compute_ms = (time.perf_counter() - t_compute) * 1000

            t_comm = time.perf_counter()
            client.push_gradient(grads, clock=clock)
            new_params = client.pull_params()
            comm_ms = (time.perf_counter() - t_comm) * 1000

            model.set_params(new_params)
            clock += 1

            epoch_loss       += loss
            epoch_acc        += acc
            epoch_compute_ms += compute_ms
            epoch_comm_ms    += comm_ms
            n_batches        += 1
            global_iter      += 1

            row = {
                "phase":      "train",
                "rank":       rank,
                "epoch":      epoch,
                "iter":       global_iter,
                "loss":       round(loss, 6),
                "acc":        round(acc, 4),
                "compute_ms": round(compute_ms, 3),
                "comm_ms":    round(comm_ms, 3),
            }
            logger.log(row)
            metrics.append(row)

        avg_loss = epoch_loss / n_batches
        avg_acc  = epoch_acc  / n_batches
        avg_cmp  = epoch_compute_ms / n_batches
        avg_comm = epoch_comm_ms / n_batches

        if isinstance(model, LogReg):
            val_loss, val_acc = evaluate_logreg(model, val_loader)
        else:
            val_loss, val_acc = evaluate_cnn(model, val_loader, device)

        elapsed    = time.perf_counter() - train_start
        n_samples  = len(train_loader.dataset)
        throughput = n_samples * epoch / elapsed

        summary = {
            "phase":       "epoch_summary",
            "rank":        rank,
            "epoch":       epoch,
            "train_loss":  round(avg_loss, 6),
            "train_acc":   round(avg_acc, 4),
            "val_loss":    round(val_loss, 6),
            "val_acc":     round(val_acc, 4),
            "avg_compute_ms": round(avg_cmp, 3),
            "avg_comm_ms":    round(avg_comm, 3),
            "elapsed_sec": round(elapsed, 2),
            "throughput_samples_sec": round(throughput, 1),
            "wall_sec":    round(elapsed, 2),
        }
        logger.log(summary)
        metrics.append(summary)

        if rank == 0:
            print(
                f"  [W{rank}] Epoch {epoch:>2}/{epochs} | "
                f"loss={avg_loss:.4f} acc={avg_acc*100:.1f}% | "
                f"val_acc={val_acc*100:.1f}% | "
                f"comm={avg_comm:.1f}ms cmp={avg_cmp:.1f}ms"
            )

    client.send_stop()
    client.close()
    logger.close()
    summaries = [m for m in metrics if m.get("phase") != "train"]
    result_queue.put({"rank": rank, "metrics": summaries})


# ── RAR Worker ─────────────────────────────────────────────────────────────────

def run_rar_worker(
    rank:        int,
    world_size:  int,
    config:      dict,
    init_params: dict[str, np.ndarray],
    result_queue,
    shared_bufs: dict,
    straggler_delay: float = 0.0,
) -> None:
    """Training loop for a single RAR worker."""
    from rar_engine.ring_allreduce import ring_allreduce

    _set_seeds(config["seed"], rank)

    model_name = config["model"].lower()
    lr         = float(config["lr"])
    epochs     = int(config["epochs"])
    run_id     = config.get("run_id", f"rar_worker{rank}_{model_name}")

    train_loader = get_shard(rank=rank, world_size=world_size, config=config, train=True)
    val_loader   = get_full_val_loader(config)

    model = build_model(model_name, seed=config["seed"])
    model.set_params(init_params)

    device = torch.device("cpu")
    optimizer = criterion = None
    if isinstance(model, SmallCNN):
        model = model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = nn.CrossEntropyLoss()

    logger = RunLogger(run_id, config.get("log_dir", "results/raw"))

    metrics: list[dict] = []
    global_iter  = 0
    train_start  = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        epoch_compute_ms, epoch_comm_ms  = 0.0, 0.0

        for Xb, yb in train_loader:
            if straggler_delay > 0:
                time.sleep(straggler_delay)

            t_compute = time.perf_counter()
            if isinstance(model, LogReg):
                grads, loss, acc = _compute_logreg_grad(model, Xb, yb)
            else:
                grads, loss, acc = _compute_cnn_grad(model, optimizer, criterion,
                                                      Xb, yb, device)
            compute_ms = (time.perf_counter() - t_compute) * 1000

            t_comm = time.perf_counter()
            avg_grads = ring_allreduce(grads, rank, world_size, shared_bufs,
                                       throttle_ms=float(config.get("throttle_ms", 0.0)))
            comm_ms   = (time.perf_counter() - t_comm) * 1000

            model.update(avg_grads, lr)

            epoch_loss       += loss
            epoch_acc        += acc
            epoch_compute_ms += compute_ms
            epoch_comm_ms    += comm_ms
            n_batches        += 1
            global_iter      += 1

            row = {
                "phase":      "train",
                "rank":       rank,
                "epoch":      epoch,
                "iter":       global_iter,
                "loss":       round(loss, 6),
                "acc":        round(acc, 4),
                "compute_ms": round(compute_ms, 3),
                "comm_ms":    round(comm_ms, 3),
            }
            logger.log(row)
            metrics.append(row)

        avg_loss = epoch_loss / n_batches
        avg_acc  = epoch_acc  / n_batches
        avg_cmp  = epoch_compute_ms / n_batches
        avg_comm = epoch_comm_ms / n_batches

        if isinstance(model, LogReg):
            val_loss, val_acc = evaluate_logreg(model, val_loader)
        else:
            val_loss, val_acc = evaluate_cnn(model, val_loader, device)

        elapsed    = time.perf_counter() - train_start
        n_samples  = len(train_loader.dataset)
        throughput = n_samples * epoch / elapsed

        summary = {
            "phase":       "epoch_summary",
            "rank":        rank,
            "epoch":       epoch,
            "train_loss":  round(avg_loss, 6),
            "train_acc":   round(avg_acc, 4),
            "val_loss":    round(val_loss, 6),
            "val_acc":     round(val_acc, 4),
            "avg_compute_ms": round(avg_cmp, 3),
            "avg_comm_ms":    round(avg_comm, 3),
            "elapsed_sec": round(elapsed, 2),
            "throughput_samples_sec": round(throughput, 1),
            "wall_sec":    round(elapsed, 2),
        }
        logger.log(summary)
        metrics.append(summary)

        if rank == 0:
            print(
                f"  [W{rank}] Epoch {epoch:>2}/{epochs} | "
                f"loss={avg_loss:.4f} acc={avg_acc*100:.1f}% | "
                f"val_acc={val_acc*100:.1f}% | "
                f"comm={avg_comm:.1f}ms cmp={avg_cmp:.1f}ms"
            )

    logger.close()
    summaries = [m for m in metrics if m.get("phase") != "train"]
    result_queue.put({"rank": rank, "metrics": summaries})
