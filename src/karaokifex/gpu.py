"""GPU housekeeping shared by the model-running steps."""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

WAIT_SAY = 5 * 60  # seconds between "waiting for the GPU" lines while another run holds it


def free_gpu_memory() -> None:
    """Release cached CUDA memory once a model is no longer needed, so the next one fits."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SharedGpu:
    """A lock file that several karaokifex runs on one machine share, so their models take turns on the GPU.

    Two songs made at once each keep their own models in video memory; running two at the same time can
    fill it, and then both crawl (the driver pages memory out) instead of either finishing. Held with
    `with`, it waits for the other runs' turns, however long they take.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> SharedGpu:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a+b")
        began, said = time.monotonic(), None
        while not self._try():
            # said as it starts waiting and every WAIT_SAY after, so whoever watches the output for a run gone quiet
            # (Karaokino's batch stops one that is quiet and idle) sees a run waiting its turn, not one stuck
            waited = time.monotonic() - began
            if said is None or waited - said >= WAIT_SAY:
                log.info("waiting for the GPU: another run is using it (%.0f min so far)", waited / 60)
                said = waited
        return self

    def _try(self) -> bool:
        """The lock taken, or False after a while without it (Windows: 10 s; elsewhere a second)"""
        if sys.platform == "win32":
            import msvcrt
            self._file.seek(0)
            try:
                msvcrt.locking(self._file.fileno(), msvcrt.LK_LOCK, 1)  # gives up after 10 s
                return True
            except OSError:
                return False
        import fcntl
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            time.sleep(1)
            return False

    def __exit__(self, *_exc: object) -> None:
        if sys.platform == "win32":
            import msvcrt
            self._file.seek(0)
            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()
        self._file = None
