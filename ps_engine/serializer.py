"""
Serializer for gradient/parameter dicts transmitted over ZeroMQ.

Wire format:
    4 bytes (big-endian uint32) — header length
    N bytes                     — JSON header: {key: {dtype, shape}, ...}
    M bytes                     — concatenated float32 array payloads
"""

from __future__ import annotations
import json
import struct
import numpy as np


def pack(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize {name: ndarray} → bytes."""
    header: dict[str, dict] = {}
    chunks: list[bytes] = []
    for key, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float32)
        header[key] = {"dtype": "float32", "shape": list(arr.shape)}
        chunks.append(arr.tobytes())
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(hdr_bytes)) + hdr_bytes + b"".join(chunks)


def unpack(data: bytes) -> dict[str, np.ndarray]:
    """Deserialize bytes → {name: ndarray}."""
    hlen = struct.unpack(">I", data[:4])[0]
    header: dict = json.loads(data[4 : 4 + hlen])
    offset = 4 + hlen
    result: dict[str, np.ndarray] = {}
    for key, meta in header.items():
        shape = tuple(meta["shape"])
        nbytes = int(np.prod(shape)) * 4  # float32 = 4 bytes
        arr = np.frombuffer(data[offset : offset + nbytes], dtype=np.float32)
        result[key] = arr.reshape(shape).copy()
        offset += nbytes
    return result
