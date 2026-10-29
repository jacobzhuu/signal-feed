"""Isolated content-v1 catalog journaling and exact offline replay. Never enabled by API."""

from dataclasses import dataclass
import hashlib
import json
import os
import errno
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import uuid
import numpy as np
from .settings import ENCODER_MODEL, ENCODER_REVISION

VERSION = "content-catalog-replay-v1"


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def algorithm():
    base = Path(__file__).parent
    return {
        name: file_digest(base / name) for name in ("serving.py", "catalog_replay.py")
    }


def write_bytes(file, data):
    with file.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def synced_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def catalog_rows(items):
    rows = []
    for a in items:
        row = dict(
            id=str(a["id"]),
            title=a["title"],
            published=float(a["published"]),
            score=int(a["score"]),
            deleted=int(a.get("deleted", 0)),
            updated=float(a.get("updated", 0)),
            url=a.get("url", ""),
            domain=a.get("domain", ""),
        )
        rows.append(row)
    if (
        any(a["deleted"] or a["score"] < 0 for a in rows)
        or not np.isfinite([(a["published"], a["updated"]) for a in rows]).all()
    ):
        raise ValueError("Unsupported catalog metadata")
    if len({a["id"] for a in rows}) != len(rows):
        raise ValueError("Duplicate catalog identity")
    return rows


@dataclass
class Catalog:
    version: str
    rows: list
    vectors: np.ndarray
    # Hold actual references to prevent object ID reuse in the hot capture cache.
    source_items: object = None
    source_vectors: object = None


class ReplayStore:
    def __init__(self, root):
        if os.environ.get("SIGNALFEED_TEST_EVENTS") != "1":
            raise ValueError("Replay prototype requires an isolated test environment")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        for name in ("catalogs", "requests"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
        self._catalog = None
        self.last_request = None
        self.source = algorithm()
        self.lock = threading.RLock()

    def register(self, items, vectors):
        with self.lock:
            return self._register(items, vectors)

    def _register(self, items, vectors):
        if (
            vectors.dtype != np.float32
            or vectors.shape != (len(items), 384)
            or vectors.flags.writeable
        ):
            raise ValueError(
                "Catalog requires immutable float32 384-dimensional vectors"
            )
        rows = catalog_rows(items)
        old = self._catalog
        if (
            old is not None
            and old.source_items is items
            and old.source_vectors is vectors
            and old.rows == rows
        ):
            return old.version
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite vectors")
        header = dict(
            version=VERSION,
            encoder=dict(model=ENCODER_MODEL, revision=ENCODER_REVISION),
            availability="service-visible nondeleted catalog at refresh; history is penalized, not excluded",
            items=rows,
        )
        body = canonical(header)
        version = digest(body + vectors.astype("<f4", copy=False).tobytes(order="C"))
        dest = self.root / "catalogs" / version
        if not dest.exists():
            with tempfile.TemporaryDirectory(
                prefix=".catalog-", dir=self.root / "catalogs"
            ) as tmp:
                folder = Path(tmp)
                write_bytes(folder / "catalog.json", body)
                with (folder / "vectors.npy").open("xb") as stream:
                    np.save(
                        stream, vectors.astype("<f4", copy=False), allow_pickle=False
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                write_bytes(
                    folder / "manifest.json",
                    canonical(
                        dict(
                            version=version,
                            shape=list(vectors.shape),
                            files={
                                name: file_digest(folder / name)
                                for name in ("catalog.json", "vectors.npy")
                            },
                        )
                    ),
                )
                synced_directory(folder)
                try:
                    os.rename(folder, dest)
                except OSError as exc:
                    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                        raise
                synced_directory(dest.parent)
        # Verify existing objects before reuse, including a prior crashed writer.
        loaded = load_catalog(self.root, version)
        self._catalog = Catalog(version, rows, loaded.vectors, items, vectors)
        return version

    def capture(self, *, items, vectors, context, scores, returned):
        with self.lock:
            return self._capture(
                items=items,
                vectors=vectors,
                context=context,
                scores=scores,
                returned=returned,
            )

    def _capture(self, *, items, vectors, context, scores, returned):
        if algorithm() != self.source:
            raise ValueError("Source changed during capture")
        version = self.register(items, vectors)
        limit = context["limit"]
        if context["model_version"] != "content-v1" or not 1 <= limit <= 100:
            raise ValueError("Unsupported recorded policy/budget")
        ids = [a["id"] for a in items]
        top100 = [ids[i] for i in np.argsort(-scores, kind="stable")[:100]]
        if returned != top100[:limit]:
            raise ValueError("Actual response differs from the claimed content policy")
        request_id = uuid.uuid4().hex
        explicit = np.asarray(context.pop("explicit"), dtype="<f4")
        context["explicit_vector"] = explicit.tolist()
        record = dict(
            version=VERSION,
            request_id=request_id,
            isolated_capture=True,
            algorithm=self.source,
            numerical_environment=dict(numpy=np.__version__),
            catalog_version=version,
            candidate_budget=100,
            score_sha256=digest(np.asarray(scores, dtype="<f8").tobytes()),
            top100=top100,
            returned=returned,
            **context,
        )
        validate_request(record, self._catalog)
        data = canonical(record)
        envelope = canonical(dict(sha256=digest(data), record=record))
        self._publish(request_id, envelope)
        self.last_request = request_id
        return request_id

    def _publish(self, request_id, envelope):
        destination = self.root / "requests" / f"{request_id}.json"
        # Exclusive hard-link publishes only complete fsynced files, never replaces.
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".request-", delete=False
        ) as temp:
            pending = Path(temp.name)
            try:
                temp.write(envelope)
                temp.flush()
                os.fsync(temp.fileno())
                os.link(pending, destination)
                synced_directory(destination.parent)
            finally:
                pending.unlink(missing_ok=True)


class SQLiteReplayStore(ReplayStore):
    """Isolated synchronous durable request journal; never a production DB."""

    def __init__(self, root):
        super().__init__(root)
        self.connection = sqlite3.connect(
            self.root / "requests.sqlite", timeout=15, check_same_thread=False
        )
        try:
            mode = self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            self.connection.execute("PRAGMA synchronous=FULL")
            if (
                mode != "wal"
                or self.connection.execute("PRAGMA synchronous").fetchone()[0] != 2
            ):
                raise ValueError("Durable WAL contract unavailable")
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS records("
                    "id TEXT PRIMARY KEY,envelope BLOB NOT NULL) WITHOUT ROWID"
                )
            synced_directory(self.root)
        except BaseException:
            self.connection.close()
            raise

    def _publish(self, request_id, envelope):
        with self.connection:
            self.connection.execute(
                "INSERT INTO records(id,envelope) VALUES(?,?)", (request_id, envelope)
            )

    def close(self):
        with self.lock:
            self.connection.close()


def load_catalog(root, version):
    if not re.fullmatch("[a-f0-9]{64}", version):
        raise ValueError("Invalid catalog reference")
    folder = Path(root) / "catalogs" / version
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if manifest["version"] != version or set(manifest["files"]) != {
        "catalog.json",
        "vectors.npy",
    }:
        raise ValueError("Invalid catalog manifest")
    for name, checksum in manifest["files"].items():
        if file_digest(folder / name) != checksum:
            raise ValueError("Catalog file checksum mismatch")
    header = json.loads((folder / "catalog.json").read_bytes())
    x = np.load(folder / "vectors.npy", allow_pickle=False)
    rows = catalog_rows(header["items"])
    if (
        header["version"] != VERSION
        or x.dtype != np.float32
        or x.shape != (len(rows), 384)
        or list(x.shape) != manifest["shape"]
        or not np.isfinite(x).all()
        or header["encoder"] != dict(model=ENCODER_MODEL, revision=ENCODER_REVISION)
    ):
        raise ValueError("Catalog contract mismatch")
    if (
        digest(canonical(header) + x.astype("<f4", copy=False).tobytes(order="C"))
        != version
    ):
        raise ValueError("Catalog identity mismatch")
    x.setflags(write=False)
    return Catalog(version, rows, x)


def validate_request(r, catalog):
    ids = {a["id"] for a in catalog.rows}
    h = r["history_newest_first"]
    full = r["observed_unique_history_newest_first"]
    if (
        r["version"] != VERSION
        or r["catalog_version"] != catalog.version
        or r["model_version"] != "content-v1"
        or r["candidate_budget"] != 100
        or not 1 <= r["limit"] <= 100
        or r["algorithm"] != algorithm()
        or r["numerical_environment"]["numpy"] != np.__version__
    ):
        raise ValueError("Request policy/source contract mismatch")
    if (
        len(h) > 20
        or len(set(h)) != len(h)
        or not set(h) <= ids
        or len({v["item_id"] for v in full}) != len(full)
        or full != sorted(full, key=lambda v: (-v["time"], v["item_id"]))
        or any(not np.isfinite(v["time"]) or v["time"] > r["created"] for v in full)
        or h != [v["item_id"] for v in full[:20] if v["item_id"] in ids]
        or r["positive_actions"] < len(full)
        or any(
            not isinstance(r[k], int) or r[k] < 0
            for k in (
                "positive_actions",
                "real_positive_actions",
                "test_positive_actions",
            )
        )
        or r["real_positive_actions"] + r["test_positive_actions"]
        != r["positive_actions"]
    ):
        raise ValueError("Invalid point-in-time history")
    explicit = np.asarray(r["explicit_vector"], dtype="float32")
    cf = r["collaborative"]
    if (
        explicit.shape != (384,)
        or not np.isfinite(explicit).all()
        or not np.isfinite([r["created"], r["catalog_refreshed_at"]]).all()
        or not 0 <= r["catalog_refreshed_at"] <= r["created"]
        or any(not isinstance(v, int) or v <= 0 for v in cf.values())
    ):
        raise ValueError("Invalid point-in-time inputs")
    for key in ("top100", "returned"):
        if (
            not isinstance(r[key], list)
            or len(set(r[key])) != len(r[key])
            or not set(r[key]) <= ids
        ):
            raise ValueError("Invalid stored ordering")
    if (
        len(r["top100"]) != min(100, len(ids))
        or r["returned"] != r["top100"][: r["limit"]]
    ):
        raise ValueError("Invalid stored budget")


def content_scores(catalog, r):
    """Independent reproduction of serving's unchanged full content formula."""
    x = catalog.vectors
    index = {a["id"]: i for i, a in enumerate(catalog.rows)}
    h = [index[i] for i in r["history_newest_first"]]
    profile = x[h].mean(0) if h else np.zeros(384, dtype="float32")
    profile = profile + np.asarray(r["explicit_vector"], dtype="float32")
    profile = profile / max(np.linalg.norm(profile), 1e-8)
    affinity = x @ profile
    fresh = np.array(
        [
            np.exp(-max(0, r["created"] - a["published"]) / 86400 / 3)
            for a in catalog.rows
        ]
    )
    pop = np.log1p([a["score"] for a in catalog.rows])
    pop = pop / max(1, pop.max())
    score = 0.6 * affinity + 0.25 * pop + 0.15 * fresh
    if r["collaborative"]:
        scale = max(r["collaborative"].values())
        for item, count in r["collaborative"].items():
            if item in index:
                score[index[item]] += 0.15 * count / scale
    score[h] -= 0.4
    return score


class ReplayReader:
    def __init__(self, root):
        self.root = Path(root)
        self.catalogs = {}

    def replay(self, request_id):
        if not re.fullmatch("[a-f0-9]{32}", request_id):
            raise ValueError("Invalid request identity")
        envelope = self.envelope(request_id)
        r = envelope["record"]
        if r["request_id"] != request_id or digest(canonical(r)) != envelope["sha256"]:
            raise ValueError("Request checksum mismatch")
        version = r["catalog_version"]
        if version not in self.catalogs:
            self.catalogs[version] = load_catalog(self.root, version)
        catalog = self.catalogs[version]
        validate_request(r, catalog)
        scores = content_scores(catalog, r)
        top100 = [
            catalog.rows[i]["id"] for i in np.argsort(-scores, kind="stable")[:100]
        ]
        if (
            digest(np.asarray(scores, dtype="<f8").tobytes()) != r["score_sha256"]
            or top100 != r["top100"]
        ):
            raise ValueError("Recomputed policy differs from capture")
        return dict(
            request_id=request_id,
            top100=top100,
            returned=top100[: r["limit"]],
            scores=scores,
            catalog_version=version,
            source_verified=True,
        )

    def envelope(self, request_id):
        return json.loads((self.root / "requests" / f"{request_id}.json").read_bytes())


class SQLiteReplayReader(ReplayReader):
    def envelope(self, request_id):
        uri = (self.root / "requests.sqlite").resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=15)
        try:
            connection.execute("PRAGMA query_only=ON")
            row = connection.execute(
                "SELECT envelope FROM records WHERE id=?", (request_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise FileNotFoundError(request_id)
        return json.loads(row[0])
