"""All supported online publication operations share one cross-process lock."""

from contextlib import contextmanager
import fcntl
import json
import os
import tempfile
from .settings import path


@contextmanager
def publication_lock(root=None):
    root = path("online-models") if root is None else root
    with (root / "publication.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another training/publication operation is running"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_json(file, value):
    fd, name = tempfile.mkstemp(prefix=file.name + ".", dir=file.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, file)
        directory = os.open(file.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)
