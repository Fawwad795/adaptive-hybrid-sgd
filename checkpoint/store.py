"""
Atomic checkpoint store.

Write protocol: write to .tmp → fsync → os.rename (atomic on POSIX and NTFS).
Each checkpoint is a numpy .npz archive containing:
    - model parameters (arrays named by key)
    - metadata stored as a JSON blob in a special "__meta__" key

Usage
-----
save_checkpoint(params, meta, directory, version)
load_checkpoint(directory, version=None) -> (params, meta)
list_checkpoints(directory) -> list of (version, path)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np


_PREFIX = "ckpt_v"


def save_checkpoint(
    params:    dict[str, np.ndarray],
    meta:      dict,
    directory: str,
    version:   int,
) -> str:
    """Atomically save params + meta. Returns the final checkpoint path."""
    Path(directory).mkdir(parents=True, exist_ok=True)

    final_path = str(Path(directory) / f"{_PREFIX}{version:06d}.npz")
    tmp_path   = final_path + ".tmp"

    meta_bytes = json.dumps(meta).encode("utf-8")
    meta_array = np.frombuffer(meta_bytes, dtype=np.uint8)

    arrays = {k: v.astype(np.float32) for k, v in params.items()}
    arrays["__meta__"] = meta_array

    np.savez(tmp_path, **arrays)
    # np.savez appends .npz if not present
    actual_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(actual_tmp, final_path)

    return final_path


def load_checkpoint(
    directory: str,
    version:   int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Load a checkpoint.  If version is None, loads the latest.
    Returns (params, meta).
    """
    ckpts = list_checkpoints(directory)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {directory!r}")

    if version is None:
        _, path = ckpts[-1]
    else:
        matches = [(v, p) for v, p in ckpts if v == version]
        if not matches:
            raise FileNotFoundError(f"Checkpoint v{version} not found in {directory!r}")
        _, path = matches[0]

    data       = np.load(path, allow_pickle=False)
    meta_bytes = data["__meta__"].tobytes()
    meta       = json.loads(meta_bytes.decode("utf-8"))
    params     = {k: data[k] for k in data.files if k != "__meta__"}
    return params, meta


def list_checkpoints(directory: str) -> list[tuple[int, str]]:
    """Return sorted list of (version, path) tuples."""
    p = Path(directory)
    if not p.exists():
        return []
    results = []
    for f in p.glob(f"{_PREFIX}*.npz"):
        try:
            ver = int(f.stem[len(_PREFIX):])
            results.append((ver, str(f)))
        except ValueError:
            pass
    results.sort(key=lambda x: x[0])
    return results


def latest_version(directory: str) -> int | None:
    ckpts = list_checkpoints(directory)
    return ckpts[-1][0] if ckpts else None
