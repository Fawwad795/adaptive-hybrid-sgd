"""
Data loading and deterministic sharding for MNIST and CIFAR-10.

get_shard(rank, world_size, config) → torch.utils.data.DataLoader
  - rank 0 in a world_size-1 setup returns the full dataset shard for that worker.
  - The same (rank, world_size, seed) triple always produces the same indices.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# ── Per-dataset transform pipelines ────────────────────────────────────────────

_MNIST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])

_CIFAR10_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def _load_raw_dataset(name: str, data_dir: str, train: bool):
    """Download (if needed) and return a torchvision Dataset."""
    name = name.lower()
    os.makedirs(data_dir, exist_ok=True)
    if name == "mnist":
        return torchvision.datasets.MNIST(
            root=data_dir, train=train, download=True, transform=_MNIST_TRANSFORM
        )
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=_CIFAR10_TRANSFORM
        )
    raise ValueError(f"Unknown dataset: {name!r}. Choose 'mnist' or 'cifar10'.")


def _shard_indices(total: int, rank: int, world_size: int, seed: int) -> list[int]:
    """
    Deterministically partition `total` indices into `world_size` shards and
    return the shard for `rank`.

    Indices are shuffled once with `seed` so shards are balanced across classes,
    then split sequentially.  Same inputs → identical output every time.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total).tolist()
    shard_size = total // world_size
    start = rank * shard_size
    # Last worker absorbs any remainder.
    end = start + shard_size if rank < world_size - 1 else total
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
    dataset = _load_raw_dataset(config["dataset"], config.get("data_dir", "data"), train)
    indices = _shard_indices(len(dataset), rank, world_size, config["seed"])
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=config["batch_size"],
        shuffle=train,          # re-shuffle each epoch within the shard
        drop_last=False,
        num_workers=0,          # keep 0 for multiprocessing compatibility
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )
    return loader


def get_full_val_loader(config: dict) -> DataLoader:
    """Full validation set (unsharded) for accuracy evaluation."""
    dataset = _load_raw_dataset(config["dataset"], config.get("data_dir", "data"), train=False)
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
