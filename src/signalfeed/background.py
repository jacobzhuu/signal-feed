"""Bounded daemon work: no unbounded executor queue when a dependency stalls."""

import threading
from time import monotonic


class Work:
    def __init__(self):
        self.done = threading.Event()
        self.result = None
        self.error = None

    def wait(self, timeout):
        if not self.done.wait(max(0, timeout)):
            return False, None
        if self.error is not None:
            raise self.error
        return True, self.result


class BoundedRunner:
    def __init__(self, capacity):
        self.slots = threading.BoundedSemaphore(capacity)
        self.condition = threading.Condition()
        self.closed = False
        self.active = 0

    def open(self):
        with self.condition:
            if self.active:
                raise RuntimeError("Cannot reopen runner with outstanding work")
            self.closed = False

    def close(self, timeout=1):
        deadline = monotonic() + timeout
        with self.condition:
            self.closed = True
            while self.active and monotonic() < deadline:
                self.condition.wait(max(0, deadline - monotonic()))
            return self.active == 0

    def submit(self, call):
        with self.condition:
            if self.closed or not self.slots.acquire(blocking=False):
                return None
            self.active += 1
        job = Work()

        def run():
            try:
                job.result = call()
            except Exception as exc:
                job.error = exc
            finally:
                self.slots.release()
                job.done.set()
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

        try:
            threading.Thread(target=run, daemon=True).start()
        except Exception:
            self.slots.release()
            with self.condition:
                self.active -= 1
                self.condition.notify_all()
            raise
        return job
