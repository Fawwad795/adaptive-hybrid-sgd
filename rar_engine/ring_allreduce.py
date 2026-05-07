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
import time
import multiprocessing as mp
import numpy as np

# Use the spawn context so that Array/Barrier handles are transferable via
# pickle to child processes spawned with mp.get_context("spawn") on Linux.
# The default context on Linux is "fork", whose anonymous mmap objects are
# NOT pickle-safe and silently fail in spawned children.
_SPAWN = mp.get_context("spawn")


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
    send_bufs = []
    for _ in range(world_size):
        worker_buf: dict = {}
        for key, shape in param_shapes.items():
            n = int(np.prod(shape))
            # _SPAWN.Array uses named POSIX shared memory (/dev/shm/pymp-*)
            # which survives pickling into spawn children, unlike the default
            # fork-context anonymous mmap.
            worker_buf[key] = _SPAWN.Array(ctypes.c_float, n)
        send_bufs.append(worker_buf)

    return {
        "send_bufs":    send_bufs,
        "barrier":      _SPAWN.Barrier(world_size),
        "param_shapes": dict(param_shapes),
    }


# ── Core ring allreduce ────────────────────────────────────────────────────────

def ring_allreduce(
    local_grads:  dict[str, np.ndarray],
    rank:         int,
    world_size:   int,
    shared_bufs:  dict,
    throttle_ms:  float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Average `local_grads` across all workers using ring reduce-scatter + allgather.

    Steps
    -----
    1. Each worker writes its gradient into send_bufs[rank].
    2. Barrier — all workers have written.
    3. Reduce-scatter: (world_size-1) steps; each step snapshots a neighbour's
       buffer BEFORE the write phase, then accumulates into own buffer.
       Two barriers per step prevent the read/write race condition that existed
       in the original single-barrier design.
    4. Phase 2 — divide accumulated sum to get the average.
    5. Allgather — broadcast average back via rank-0's buffer.
    6. Return averaged gradient dict.
    """
    send_bufs    = shared_bufs["send_bufs"]
    barrier      = shared_bufs["barrier"]
    param_shapes = shared_bufs["param_shapes"]

    # Phase 0 — write local gradients into shared buffer
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key].get_obj(), dtype=np.float32)
        np.copyto(arr, local_grads[key].ravel().astype(np.float32))
    barrier.wait()

    # Phase 1 — reduce-scatter
    # Each step uses a snapshot taken BEFORE any worker writes, eliminating
    # the race where rank r reads send_bufs[src] while src is also writing it.
    for step in range(world_size - 1):
        if throttle_ms > 0:
            time.sleep(throttle_ms / 1000.0)
        src = (rank - step - 1) % world_size

        # Snapshot src's buffer while no writes are in flight (all workers are
        # here before any has started writing in this step).
        snapshots = {
            key: np.frombuffer(send_bufs[src][key].get_obj(), dtype=np.float32).copy()
            for key in param_shapes
        }
        barrier.wait()   # barrier 1: everyone has their snapshot

        for key in param_shapes:
            my_arr = np.frombuffer(send_bufs[rank][key].get_obj(), dtype=np.float32)
            my_arr += snapshots[key]
        barrier.wait()   # barrier 2: all writes done before next step reads

    # At this point send_bufs[rank] holds the SUM of all workers' gradients.
    # Phase 2 — divide to get average
    avg_grads: dict[str, np.ndarray] = {}
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key].get_obj(), dtype=np.float32).copy()
        arr /= world_size
        avg_grads[key] = arr.reshape(shape)

    # Phase 3 — allgather (write average back so all workers see the same result)
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[rank][key].get_obj(), dtype=np.float32)
        np.copyto(arr, avg_grads[key].ravel())
    barrier.wait()

    # Read final averaged gradient from rank-0's buffer (all are identical)
    final_grads: dict[str, np.ndarray] = {}
    for key, shape in param_shapes.items():
        arr = np.frombuffer(send_bufs[0][key].get_obj(), dtype=np.float32).copy()
        final_grads[key] = arr.reshape(shape)

    return final_grads


# ── Helper to extract param shapes from a model ────────────────────────────────

def param_shapes_from_model(model) -> dict[str, tuple]:
    params = model.get_params()
    return {k: v.shape for k, v in params.items()}
