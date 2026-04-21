"""
Phase 2+ — Distributed worker process.

NOT YET IMPLEMENTED — stub for skeleton completeness.

Each worker:
    1. Loads its assigned data shard.
    2. Holds a local model replica.
    3. Computes gradients on each mini-batch.
    4. Synchronizes via the currently active topology (PS or RAR).
    5. Emits heartbeat + telemetry to the metrics monitor.
"""

raise NotImplementedError("Phase 2: Distributed worker not yet implemented.")
