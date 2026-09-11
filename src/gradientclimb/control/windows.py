"""Ordinary SendInput pedal transitions, guarded by the selected game window.

No focus changes or menu clicks are implemented. Key bindings must be explicitly
supplied from an observed game configuration. Constructing a backend sends no input.
The full INPUT union is necessary: sizeof(INPUT) is 40 on Windows x64, not 32.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time

from gradientclimb.capture.windows import WindowGuard, WindowTarget


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_uint32),
        ("wParamL", ctypes.c_uint16),
        ("wParamH", ctypes.c_uint16),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("payload",)
    _fields_ = [("type", ctypes.c_uint32), ("payload", _INPUTUNION)]


KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
EXTENDED_KEYS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E})


def keyboard_event(virtual_key: int, pressed: bool) -> INPUT:
    if type(virtual_key) is not int or not 1 <= virtual_key <= 254:
        raise ValueError("Virtual key must be an integer in 1..254")
    flags = 0 if pressed else KEYEVENTF_KEYUP
    if virtual_key in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    event = INPUT()
    event.type = 1
    event.ki = KEYBDINPUT(virtual_key, 0, flags, 0, 0)
    return event


class _NativeInput:
    def __init__(self):
        if os.name != "nt":
            raise OSError("SendInput is only available on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SendInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(INPUT), ctypes.c_int]
        self.user32.SendInput.restype = ctypes.c_uint32
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_int16

    def is_down(self, virtual_key: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    def send(self, events: list[INPUT]) -> int:
        if not events:
            return 0
        array = (INPUT * len(events))(*events)
        return int(self.user32.SendInput(len(events), array, ctypes.sizeof(INPUT)))


class WindowsPedalBackend:
    """Use through PedalController, whose short leases provide a release watchdog.

    Identity/focus are checked immediately before every held/pressed command. On
    failure, owned keys are released even if focus changed; key-up cleanup does not
    press keys in another window. OS focus can still change after the check, so this
    is not an atomic input-targeting guarantee. No physical key owned by the user is
    deliberately released. Process termination cannot guarantee key-up delivery.
    """

    def __init__(
        self, target: WindowTarget, *, gas_vk: int, brake_vk: int, guard=None, sender=None
    ):
        keyboard_event(gas_vk, False)
        keyboard_event(brake_vk, False)
        if gas_vk == brake_vk:
            raise ValueError("Gas and brake require distinct keys")
        self.gas_vk, self.brake_vk = gas_vk, brake_vk
        self.guard = guard if guard is not None else WindowGuard(target)
        self.sender = sender if sender is not None else _NativeInput()
        self._held: set[int] = set()
        self._lock = threading.RLock()
        self._faulted = False
        self.trace: list[dict] = []

    def _release(self):
        keys = sorted(self._held)
        if keys:
            started = time.perf_counter_ns()
            count = self.sender.send([keyboard_event(key, False) for key in keys])
            self.trace.append(
                {
                    "started_ns": started,
                    "completed_ns": time.perf_counter_ns(),
                    "events": [(key, False) for key in keys],
                    "delivered": count,
                }
            )
            if count != len(keys):
                self._faulted = True
                raise RuntimeError("SendInput could not release all owned keys; input is disabled")
            self._held.clear()

    def set_pedals(self, gas: bool, brake: bool) -> None:
        if type(gas) is not bool or type(brake) is not bool:
            raise ValueError("Pedals must be independent booleans")
        with self._lock:
            if not (gas or brake):
                self._release()
                return
            if self._faulted:
                raise RuntimeError("Input backend faulted; recreate only after checking key state")
            try:
                self.guard.validate(require_foreground=True)
                desired = {
                    key for key, down in ((self.gas_vk, gas), (self.brake_vk, brake)) if down
                }
                added = desired - self._held
                if any(self.sender.is_down(key) for key in added):
                    raise RuntimeError("A requested pedal key is already held outside this backend")
                # Releases precede additions; held keys are never released/repressed unnecessarily.
                transitions = [(key, False) for key in sorted(self._held - desired)]
                transitions += [(key, True) for key in (self.gas_vk, self.brake_vk) if key in added]
                if not transitions:
                    return
                self.guard.validate(require_foreground=True)
                self._held |= added  # Conservatively track every possibly delivered key-down.
                started = time.perf_counter_ns()
                count = self.sender.send([keyboard_event(*event) for event in transitions])
                self.trace.append(
                    {
                        "started_ns": started,
                        "completed_ns": time.perf_counter_ns(),
                        "events": transitions,
                        "delivered": count,
                    }
                )
                if count != len(transitions):
                    self._faulted = True
                    raise RuntimeError("SendInput delivered only part of the pedal transition")
                self._held = desired
            except Exception:
                self._release()
                raise

    def close(self):
        with self._lock:
            self._release()
            self._faulted = True
