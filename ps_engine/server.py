"""
Phase 2 — Parameter Server (ZeroMQ ROUTER socket).

Protocol (ROUTER/DEALER over TCP):
    Worker → Server: [identity, b"", type_bytes, meta_json, payload_bytes]
        type GRAD : gradient update for current clock
        type STOP : worker finished training

    Server → Worker: [identity, b"", type_bytes, meta_json, payload_bytes]
        type PARAMS  : updated model parameters
        type STOP_ACK: acknowledge stop

Consistency disciplines:
    bsp   — wait for ALL world_size gradients, then aggregate and broadcast
    ssp   — wait for at least (world_size - staleness) gradients; lag-bounded
    async — aggregate each gradient immediately, reply right away

The server runs in its own process (launched via multiprocessing.Process).
"""

from __future__ import annotations
import json
import numpy as np
import zmq

from ps_engine.serializer import pack, unpack


# ── Message helpers ────────────────────────────────────────────────────────────

def _send(sock: zmq.Socket, identity: bytes, msg_type: str,
          meta: dict, payload: bytes = b"") -> None:
    sock.send_multipart([identity, b"",
                         msg_type.encode(),
                         json.dumps(meta).encode(),
                         payload])


def _recv(sock: zmq.Socket) -> tuple[bytes, str, dict, bytes]:
    parts    = sock.recv_multipart()
    identity = parts[0]
    msg_type = parts[2].decode()
    meta     = json.loads(parts[3])
    payload  = parts[4] if len(parts) > 4 else b""
    return identity, msg_type, meta, payload


# ── Aggregation ────────────────────────────────────────────────────────────────

def _average_grads(grad_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    avg: dict[str, np.ndarray] = {}
    for key in grad_list[0]:
        avg[key] = np.stack([g[key] for g in grad_list], axis=0).mean(axis=0)
    return avg


# ── Server class ───────────────────────────────────────────────────────────────

class PSServer:
    """
    Parameter-Server process.

    Parameters
    ----------
    world_size  : number of workers
    config      : training config (lr, ps_discipline, ssp_staleness, …)
    init_params : initial model parameters {name: np.ndarray}
    port        : TCP port to bind (default 5555)
    """

    def __init__(self, world_size: int, config: dict,
                 init_params: dict[str, np.ndarray], port: int = 5555) -> None:
        self.world_size   = world_size
        self.lr           = float(config.get("lr", 0.01))
        self.discipline   = config.get("ps_discipline", "bsp").lower()
        self.staleness    = int(config.get("ssp_staleness", 2))
        self.params       = {k: v.astype(np.float32).copy() for k, v in init_params.items()}
        self.port         = port
        self.global_clock = 0

    def run(self) -> None:
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(f"tcp://127.0.0.1:{self.port}")
        try:
            if self.discipline == "bsp":
                self._run_bsp(sock)
            elif self.discipline == "ssp":
                self._run_ssp(sock)
            else:
                self._run_async(sock)
        finally:
            sock.close()
            ctx.term()

    # ── BSP ────────────────────────────────────────────────────────────────────

    def _run_bsp(self, sock: zmq.Socket) -> None:
        active = self.world_size
        while active > 0:
            pending_grads: list[dict] = []
            pending_ids:   list[bytes] = []

            collected = 0
            while collected < active:
                identity, msg_type, meta, payload = _recv(sock)
                if msg_type == "STOP":
                    _send(sock, identity, "STOP_ACK", {})
                    active -= 1
                    continue
                pending_grads.append(unpack(payload))
                pending_ids.append(identity)
                collected += 1

            if not pending_grads:
                continue

            avg = _average_grads(pending_grads)
            for key in self.params:
                self.params[key] -= self.lr * avg[key]
            self.global_clock += 1

            out_payload = pack(self.params)
            out_meta    = {"clock": self.global_clock}
            for iid in pending_ids:
                _send(sock, iid, "PARAMS", out_meta, out_payload)

    # ── SSP ────────────────────────────────────────────────────────────────────

    def _run_ssp(self, sock: zmq.Socket) -> None:
        active         = self.world_size
        buffered_grads: dict[bytes, dict] = {}
        min_workers    = max(1, self.world_size - self.staleness)

        while active > 0:
            identity, msg_type, meta, payload = _recv(sock)

            if msg_type == "STOP":
                _send(sock, identity, "STOP_ACK", {})
                active -= 1
                buffered_grads.pop(identity, None)
                continue

            buffered_grads[identity] = unpack(payload)

            if len(buffered_grads) >= min_workers:
                grad_list  = list(buffered_grads.values())
                responders = list(buffered_grads.keys())
                buffered_grads.clear()

                avg = _average_grads(grad_list)
                for key in self.params:
                    self.params[key] -= self.lr * avg[key]
                self.global_clock += 1

                out_payload = pack(self.params)
                out_meta    = {"clock": self.global_clock}
                for iid in responders:
                    _send(sock, iid, "PARAMS", out_meta, out_payload)

    # ── Async ──────────────────────────────────────────────────────────────────

    def _run_async(self, sock: zmq.Socket) -> None:
        active = self.world_size
        while active > 0:
            identity, msg_type, meta, payload = _recv(sock)

            if msg_type == "STOP":
                _send(sock, identity, "STOP_ACK", {})
                active -= 1
                continue

            grad = unpack(payload)
            for key in self.params:
                self.params[key] -= self.lr * grad[key]
            self.global_clock += 1

            _send(sock, identity, "PARAMS",
                  {"clock": self.global_clock}, pack(self.params))


# ── Process entry point ────────────────────────────────────────────────────────

def ps_server_process(world_size: int, config: dict,
                      init_params: dict[str, np.ndarray],
                      port: int = 5555) -> None:
    """Top-level callable for multiprocessing.Process(target=...)."""
    server = PSServer(world_size, config, init_params, port=port)
    server.run()
