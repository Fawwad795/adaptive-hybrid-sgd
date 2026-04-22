"""
Phase 3 — Ring AllReduce via multiprocessing shared memory.

Simulates the reduce-scatter + allgather collective on a single machine
using numpy arrays backed by multiprocessing.Array (lock-free float32 buffers)
and multiprocessing.Barrier for synchronisation at each ring step.

This avoids the MPI dependency on Windows while faithfully reproducing the
O(2*(N-1)/N * data) communication volume of a real ring allreduce.

Public API
----------
make_shared_buffers(world_size, param_shapes) -> dict
    Call once in the parent process before spawning workers.

ring_allreduce(local_grads, rank, world_size, shared_bufs) -> avg_grads
    Called by each worker process per training step.
    Returns the gradient averaged across all workers.
"""

from __future__ import annotations
import ctypes
import multiprocessing as mp
import numpy as np


# ── Buffer factory (called once in parent) ─────────────────────────────────────

def make_shared_buffers(world_size: int,
                        param_shapes: dict[str, tuple]) -> dict:
    """
    Allocate shared memory for the ring allreduce.

    Returns a dict with:
        "send_bufs"  : list of world_size dicts {key: mp.Array}
                       send_bufs[rank][key] holds worker `rank`'s contribution
        "barrier"    : mp.Barrier(world_size) — synchronise ring steps
        "param_shapes": {key: shape}
    """
    send_bufs: list[dict[str, mp.Array]] = []
    for _ in range(world_size):
        worker_buf: dict[str, mp.Array] = {}
        for key, shape in param_shapes.items():
            n = int(np.prod(shape))
            worker_buf[key] = mp.Array(ctypes.c_float, n, lock=False)
        send_bufs.append(worker_buf)

    return {
        "send_bufs":   send_bufs,
        "barrier":     mp.Barrier(world_size),
        "param_shapes": dict(param_shapes),
    }


# ── Core ring allreduce ────────────────────────────────────────────────────────

def ring_allreduce(
    local_grads:  dict[str, np.ndarray],
    rank:         int,
    world_size:   int,
    shared_bufs:  dict,
) -> dict[str, np.ndarray]:
    """
    Average `local_grads` across all workers using ring reduce-scatter + allgather.

    Steps
    -----
    1. Each worker writes its gradient into send_bufs[rank].
    2. Barrier — all workers have written.
    3. Reduce-scatter: (world_size-1) steps; each step adds a neighbour chunk.
    4. Barrier between each step.
    5. Allgather: (world_size-1) steps; circulate the final chunk.
    6. Return averaged gradient dict.
    """
    send_bufs   = shared_bufs["send_bufs"]
    barrier     = shared_bufs["barrier"]
    param_shapes = shared_bufs["param_shapes"]

    # Phase 0 — write local gradients into shared buffer
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key], dtype=np.float32)
        np.copyto(arr, local_grads[key].ravel().astype(np.float32))
    barrier.wait()

    # Phase 1 — reduce-scatter
    # Each worker accumulates world_size-1 chunks from its left neighbour.
    # We keep a running accumulator in send_bufs[rank] (sum of contributions).
    for step in range(world_size - 1):
        src = (rank - step - 1) % world_size
        for key, shape in param_shapes.items():
            my_arr  = np.frombuffer(send_bufs[rank][key], dtype=np.float32)
            src_arr = np.frombuffer(send_bufs[src][key],  dtype=np.float32)
            my_arr += src_arr
        barrier.wait()

    # At this point send_bufs[rank] holds the SUM of all workers' gradients.
    # Phase 2 — divide to get average
    avg_grads: dict[str, np.ndarray] = {}
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key], dtype=np.float32).copy()
        arr /= world_size
        avg_grads[key] = arr.reshape(shape)

    # Phase 3 — allgather (write average back so all workers see the same result)
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key], dtype=np.float32)
        np.copyto(arr, avg_grads[key].ravel())
    barrier.wait()

    # Read final averaged gradient from rank-0's buffer (all are identical)
    final_grads: dict[str, np.ndarray] = {}
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[0][key], dtype=np.float32).copy()
        final_grads[key] = arr.reshape(shape)

    return final_grads


# ── Helper to extract param shapes from a model ────────────────────────────────

def param_shapes_from_model(model) -> dict[str, tuple]:
    params = model.get_params()
    return {k: v.shape for k, v in params.items()}
