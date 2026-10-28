"""Lossless float32 request snapshots; legacy JSON is read-only compatible."""

import json
import struct
import zlib
import numpy as np

MAGIC = b"SF2\0"
MAX_BYTES = 8 * 1024 * 1024


def pack(snapshot):
    meta = {k: v for k, v in snapshot.items() if k != "features"}
    ids = list(snapshot["features"])
    meta["feature_ids"] = ids
    x = np.asarray([snapshot["features"][i] for i in ids], dtype="<f4").reshape(
        len(ids), 1158
    )
    header = json.dumps(meta, separators=(",", ":")).encode()
    return MAGIC + zlib.compress(
        struct.pack("<I", len(header)) + header + x.tobytes(), 1
    )


def unpack(payload):
    if isinstance(payload, str):
        return json.loads(payload)
    if not payload.startswith(MAGIC):
        raise ValueError("Unknown snapshot encoding")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload[len(MAGIC) :], MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or not decoder.eof or len(raw) < 4:
        raise ValueError("Invalid or oversized snapshot")
    size = struct.unpack("<I", raw[:4])[0]
    meta = json.loads(raw[4 : 4 + size])
    ids = meta.pop("feature_ids")
    x = np.frombuffer(raw[4 + size :], dtype="<f4").reshape(len(ids), 1158)
    meta["features"] = dict(zip(ids, x.tolist()))
    return meta
