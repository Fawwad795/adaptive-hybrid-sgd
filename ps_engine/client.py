"""
Phase 2 — Worker-side Parameter Server client (ZeroMQ DEALER socket).

Usage
-----
client = PSClient(rank=0, port=5555)
client.connect()
client.push_gradient(grad_dict, clock=3)
params = client.pull_params()   # blocks until server replies
client.send_stop()
client.close()
"""

from __future__ import annotations
import json
import time
import numpy as np
import zmq

from ps_engine.serializer import pack, unpack


class PSClient:
    """DEALER socket that communicates with PSServer (ROUTER)."""

    def __init__(self, rank: int, port: int = 5555,
                 timeout_ms: int = 300_000,
                 throttle_ms: float = 0.0) -> None:
        self.rank        = rank
        self.port        = port
        self.timeout_ms  = timeout_ms
        self.throttle_ms = throttle_ms   # artificial comm delay (bandwidth simulation)
        self._ctx: zmq.Context | None   = None
        self._sock: zmq.Socket | None   = None

    def connect(self) -> None:
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.connect(f"tcp://127.0.0.1:{self.port}")

    def close(self) -> None:
        if self._sock:
            self._sock.close()
        if self._ctx:
            self._ctx.term()

    # ── Send gradient ──────────────────────────────────────────────────────────

    def push_gradient(self, grads: dict[str, np.ndarray], clock: int = 0) -> None:
        if self.throttle_ms > 0:
            time.sleep(self.throttle_ms / 1000.0)
        meta    = json.dumps({"rank": self.rank, "clock": clock}).encode()
        payload = pack(grads)
        self._sock.send_multipart([b"", b"GRAD", meta, payload])

    # ── Receive updated parameters ─────────────────────────────────────────────

    def pull_params(self) -> dict[str, np.ndarray]:
        """Block until the server sends back updated params."""
        parts    = self._sock.recv_multipart()
        msg_type = parts[1].decode()
        if msg_type == "PARAMS":
            return unpack(parts[3])
        raise RuntimeError(f"Unexpected message type from PS: {msg_type!r}")

    # ── Signal done ────────────────────────────────────────────────────────────

    def send_stop(self) -> None:
        meta = json.dumps({"rank": self.rank}).encode()
        self._sock.send_multipart([b"", b"STOP", meta, b""])
        # Drain the STOP_ACK
        try:
            self._sock.recv_multipart()
        except zmq.Again:
            pass
