"""
Atomic checkpoint store.

Atomic write protocol:  write to .tmp → fsync → os.rename (atomic on POSIX/NTFS).
Each checkpoint stores model parameters + versioned membership metadata.

Used in Phase 5 (topology switching safety) and Phase 3/6 (failure recovery).

NOT YET IMPLEMENTED beyond the stub — will be fleshed out in Phase 4.
"""

raise NotImplementedError("Checkpoint store not yet implemented.")
