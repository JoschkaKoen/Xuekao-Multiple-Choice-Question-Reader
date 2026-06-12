"""Live logging — a FileHandler that flushes per record (so `tail -f run.log`
works), attached to the ROOT logger so reused modules (logging to "course_reader")
land in the file too. Plus a thread-safe progress/cost counter.
"""

from __future__ import annotations

import logging
import sys
import threading

log = logging.getLogger("xuekao")


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s", "%H:%M:%S")
    fh = _FlushFileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)
    for noisy in ("httpx", "httpcore", "openai", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Progress:
    """Thread-safe done/total counter that logs a running RMB cost each tick."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self._lock = threading.Lock()

    def tick(self, label: str = "") -> None:
        import cost
        import usage
        with self._lock:
            self.done += 1
            try:
                spent = cost.compute_cost(usage.snapshot())[0]
            except Exception:  # noqa: BLE001
                spent = 0.0
            log.info("progress %d/%d  ¥%.3f  %s", self.done, self.total, spent, label)
