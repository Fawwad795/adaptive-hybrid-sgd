"""
Data loading and deterministic sharding for MNIST and CIFAR-10.

get_shard(rank, world_size, config) → torch.utils.data.DataLoader
  - rank 0 in a world_size-1 setup returns the full dataset shard for that worker.
  - The same (rank, world_size, seed) triple always produces the same indices.

Performance note: data is pre-converted to float32 tensors at load time so that
each DataLoader __getitem__ is a fast tensor index (μs), not a PIL-Image pipeline
(~1ms × batch_size per fetch).
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision


# ── Pre-processed tensor dataset factory ──────────────────────────────────────

def _load_tensor_dataset(name: str, data_dir: str, train: bool):
    """
    Load and return (X_tensor, y_tensor) with data already normalised to float32.
    Bypasses per-sample PIL transform pipeline → ~50× faster DataLoader iteration.
    """
    name = name.lower()
    os.makedirs(data_dir, exist_ok=True)

    if name == "mnist":
        ds = torchvision.datasets.MNIST(root=data_dir, train=train, download=True)
        X = ds.data.float().unsqueeze(1) / 255.0          # (N,1,28,28)
        X = (X - 0.1307) / 0.3081
        y = ds.targets.long()
        return X, y

    if name == "cifar10":
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        X = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        X = (X - mean) / std
        y = torch.tensor(ds.targets, dtype=torch.long)
        return X, y

    raise ValueError(f"Unknown dataset: {name!r}. Choose 'mnist' or 'cifar10'.")


def _shard_indices(total: int, rank: int, world_size: int, seed: int) -> np.ndarray:
    """
    Deterministically partition `total` indices into `world_size` shards.
    Same (rank, world_size, seed) → identical output every time.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    shard_size = total // world_size
    start = rank * shard_size
    end   = start + shard_size if rank < world_size - 1 else total
    return indices[start:end]


def get_shard(rank: int, world_size: int, config: dict, train: bool = True) -> DataLoader:
    """
    Return a DataLoader for worker `rank`'s data shard.

    Parameters
    ----------
    rank        : worker index (0-based)
    world_size  : total number of workers (1 for single-worker mode)
    config      : dict with keys: dataset, seed, batch_size, data_dir
    train       : if True, use the training split; else validation split
    """
    X, y = _load_tensor_dataset(
        config["dataset"], config.get("data_dir", "data"), train
    )
    idx   = _shard_indices(len(X), rank, world_size, config["seed"])
    X_sh  = X[idx]
    y_sh  = y[idx]
    ds    = TensorDataset(X_sh, y_sh)
    return DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=train,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )


def get_full_val_loader(config: dict) -> DataLoader:
    """Full validation set (unsharded) for accuracy evaluation."""
    X, y = _load_tensor_dataset(
        config["dataset"], config.get("data_dir", "data"), train=False
    )
    return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False, num_workers=0)
