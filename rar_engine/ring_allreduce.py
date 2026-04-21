"""
Phase 3 — Ring AllReduce (mpi4py-based).

NOT YET IMPLEMENTED — stub for skeleton completeness.

Architecture (to be implemented):
    - Reduce-scatter phase: each worker sends one chunk to right neighbour,
      receives one chunk from left neighbour, accumulates; repeat (n-1) times.
    - Allgather phase: circulate the fully-reduced chunks around the ring.
    - BSP discipline only (natural fit for collective operations).
    - Validates gradient equality against PS baseline under controlled seeds.
"""

raise NotImplementedError("Phase 3: Ring AllReduce not yet implemented.")
