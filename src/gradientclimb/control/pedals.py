from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PedalAction:
    gas: bool
    brake: bool
    duration: float

    def __post_init__(self):
        if type(self.gas) is not bool or type(self.brake) is not bool:
            raise ValueError("Pedals must be independent booleans")
        if not math.isfinite(self.duration) or not 0 < self.duration <= 2:
            raise ValueError("A control lease must last between 0 and 2 seconds")

    @property
    def code(self) -> int:
        return int(self.gas) | (int(self.brake) << 1)

    @classmethod
    def from_code(cls, code: int, duration: float = 0.1):
        if type(code) is not int or code not in range(4):
            raise ValueError("Expected a joint pedal state in 0..3")
        return cls(bool(code & 1), bool(code & 2), duration)


class InputBackend(Protocol):
    def set_pedals(self, gas: bool, brake: bool) -> None: ...


class PedalController:
    """Leased independent controls. Unknown UI/focus/expired lease releases both.

    Consecutive submit calls preserve an existing hold while adding the other pedal.
    A watchdog releases abandoned holds. The live backend additionally checks foreground
    window identity. This cannot protect against OS/process termination; use short leases.
    """

    def __init__(self, backend: InputBackend, is_playing: Callable[[], bool]):
        self.backend = backend
        self.is_playing = is_playing
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._deadline = 0.0
        self.fault: str | None = None
        self.release_failure: str | None = None
        self.trace: list[dict] = []
        self.backend.set_pedals(False, False)
        self._thread = threading.Thread(target=self._watchdog, daemon=True)
        self._thread.start()

    def submit(self, action: PedalAction) -> None:
        with self._lock:
            try:
                if self._stop.is_set() or not self.is_playing():
                    raise RuntimeError("Input refused: gameplay/focus is not verified")
                self.backend.set_pedals(action.gas, action.brake)
            except Exception as error:
                self._trip(error)
                raise
            now = time.perf_counter()
            self._deadline = now + action.duration
            self.trace.append(
                {
                    "timestamp_ns": time.perf_counter_ns(),
                    "gas": action.gas,
                    "brake": action.brake,
                    "duration_seconds": action.duration,
                }
            )

    def release(self):
        with self._lock:
            self.backend.set_pedals(False, False)
            self._deadline = 0.0

    def _watchdog(self):
        while not self._stop.wait(0.01):
            try:
                with self._lock:
                    if self._deadline and (
                        time.perf_counter() >= self._deadline or not self.is_playing()
                    ):
                        self.release()
            except Exception as error:  # noqa: BLE001 - watchdog must latch backend faults
                self._trip(error)

    def _trip(self, error: Exception):
        self._stop.set()
        self.fault = f"{type(error).__name__}: {error}"
        try:
            self.release()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve original fault and disable input
            self.release_failure = f"{type(cleanup_error).__name__}: {cleanup_error}"

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
