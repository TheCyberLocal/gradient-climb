"""Public Windows desktop touch input for two explicitly verified pedal regions.

Use only through PedalController with a fresh gameplay guard. There is no menu
navigation or automatic target selection. OS acceptance is not game acknowledgment.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
import time
from dataclasses import asdict, dataclass

from gradientclimb.artifacts import canonical_json
from gradientclimb.capture.windows import ClientRect, WindowGuard, WindowTarget

INRANGE, INCONTACT = 0x2, 0x4
DOWN, UPDATE, UP, CANCELED = 0x10000, 0x20000, 0x40000, 0x8000
TOUCH_FEEDBACK_DEFAULT, TOUCH_FEEDBACK_INDIRECT, TOUCH_FEEDBACK_NONE = 1, 2, 3


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class RECT(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("left", "top", "right", "bottom")]


class POINTER_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerType", ctypes.c_uint32),
        ("pointerId", ctypes.c_uint32),
        ("frameId", ctypes.c_uint32),
        ("pointerFlags", ctypes.c_uint32),
        ("sourceDevice", ctypes.c_void_p),
        ("hwndTarget", ctypes.c_void_p),
        ("ptPixelLocation", POINT),
        ("ptHimetricLocation", POINT),
        ("ptPixelLocationRaw", POINT),
        ("ptHimetricLocationRaw", POINT),
        ("dwTime", ctypes.c_uint32),
        ("historyCount", ctypes.c_uint32),
        ("InputData", ctypes.c_int32),
        ("dwKeyStates", ctypes.c_uint32),
        ("PerformanceCount", ctypes.c_uint64),
        ("ButtonChangeType", ctypes.c_uint32),
    ]


class POINTER_TOUCH_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerInfo", POINTER_INFO),
        ("touchFlags", ctypes.c_uint32),
        ("touchMask", ctypes.c_uint32),
        ("rcContact", RECT),
        ("rcContactRaw", RECT),
        ("orientation", ctypes.c_uint32),
        ("pressure", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class TouchPedalProfile:
    client_rect: ClientRect
    normalized_size: tuple[int, int]
    gas_xy: tuple[int, int]
    brake_xy: tuple[int, int]
    reference_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.client_rect, ClientRect)
            or len(self.normalized_size) != 2
            or any(type(v) is not int or v <= 0 for v in self.normalized_size)
            or len(self.reference_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.reference_sha256)
        ):
            raise ValueError(
                "An explicit physical rectangle, image size and reference hash are required"
            )
        for xy in (self.gas_xy, self.brake_xy):
            if len(xy) != 2 or any(type(v) is not int for v in xy):
                raise ValueError("Pedal centers must be integer image coordinates")
            if not all(0 <= v < size for v, size in zip(xy, self.normalized_size, strict=True)):
                raise ValueError("Pedal centers must be inside the verified reference")
        if self.gas_xy == self.brake_xy:
            raise ValueError("Pedal centers must be distinct")

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()

    def physical_point(self, gas: bool):
        x, y = self.gas_xy if gas else self.brake_xy
        width, height = self.normalized_size
        rect = self.client_rect
        return rect.left + round(x * rect.width / width), rect.top + round(y * rect.height / height)


@dataclass(frozen=True)
class TouchPositionEvidence:
    """Caller-produced evidence from fresh pixels, not an inferred permission."""

    profile_sha256: str
    source_rect: ClientRect
    frame_started_ns: int
    playing: bool
    positions_verified: bool


def touch_contact(contact_id: int, xy: tuple[int, int], flags: int) -> POINTER_TOUCH_INFO:
    item = POINTER_TOUCH_INFO()
    item.pointerInfo.pointerType = 2  # PT_TOUCH
    item.pointerInfo.pointerId = contact_id
    item.pointerInfo.pointerFlags = flags
    item.pointerInfo.ptPixelLocation = POINT(*xy)
    # Optional contact area/orientation/pressure omitted: Windows supplies defaults.
    return item


class _NativeTouchInput:
    def __init__(self, feedback_mode=TOUCH_FEEDBACK_NONE):
        if os.name != "nt":
            raise OSError("Desktop touch input requires Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.InitializeTouchInjection.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.user32.InitializeTouchInjection.restype = ctypes.c_int32
        self.user32.InjectTouchInput.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(POINTER_TOUCH_INFO),
        ]
        self.user32.InjectTouchInput.restype = ctypes.c_int32
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_int16
        self.initialized = False
        self.feedback_mode = feedback_mode

    def is_down(self, key):
        return bool(self.user32.GetAsyncKeyState(key) & 0x8000)

    def inject(self, contacts):
        if not self.initialized:
            if not self.user32.InitializeTouchInjection(2, self.feedback_mode):
                raise ctypes.WinError(ctypes.get_last_error())
            self.initialized = True
        array = (POINTER_TOUCH_INFO * len(contacts))(*contacts)
        # Microsoft requires retrying the same frame on ERROR_NOT_READY. Bound
        # retries; never retry an arbitrary failure or silently lose a release.
        for attempt in range(3):
            if self.user32.InjectTouchInput(len(contacts), array):
                return True
            error = ctypes.get_last_error()
            if error != 21 or attempt == 2:
                raise ctypes.WinError(error)
            time.sleep(0.001)
        return False


class WindowsTouchPedalBackend:
    """Two contact IDs (gas=1, brake=2), pinned points, no focus or menu actions.

    Renew commands frequently through PedalController: held contacts receive UPDATE
    frames. Its watchdog must use the same fresh evidence/focus condition. This
    backend does not create an unattended control loop or prove emulator support.
    """

    def __init__(
        self,
        target: WindowTarget,
        profile: TouchPedalProfile,
        evidence,
        *,
        guard=None,
        sender=None,
        max_observation_age_seconds=0.45,
        clock=time.perf_counter,
        feedback_mode=TOUCH_FEEDBACK_NONE,
    ):
        if profile.client_rect != target.client_rect:
            raise ValueError("Touch profile must pin the selected physical client rectangle")
        if (
            not math.isfinite(max_observation_age_seconds)
            or not 0 < max_observation_age_seconds <= 0.5
        ):
            raise ValueError("Touch evidence age must be in (0, 0.5] seconds")
        if not callable(evidence):
            raise TypeError("A fresh pixel evidence callback is required")
        if type(feedback_mode) is not int or feedback_mode not in (1, 2, 3):
            raise ValueError("Touch feedback must be DEFAULT=1, INDIRECT=2, or NONE=3")
        self.profile, self.evidence = profile, evidence
        self.guard = guard if guard is not None else WindowGuard(target)
        self.sender = sender if sender is not None else _NativeTouchInput(feedback_mode)
        self.feedback_mode = feedback_mode
        self.max_age, self.clock = max_observation_age_seconds, clock
        self._points = {1: profile.physical_point(True), 2: profile.physical_point(False)}
        left, top, right, bottom = profile.client_rect.bbox
        if any(not (left <= x < right and top <= y < bottom) for x, y in self._points.values()):
            raise ValueError("Mapped pedal points must remain inside the physical client rectangle")
        self._held = set()
        self._faulted = False
        self._lock = threading.RLock()
        self.trace = []

    @property
    def input_encoding(self):
        return {
            "mode": "touch",
            "api": "InjectTouchInput",
            "touch_feedback_mode": self.feedback_mode,
            "touch_feedback_name": {1: "default", 2: "indirect", 3: "none"}[self.feedback_mode],
            "feedback_scope": "this process's injected contacts; no system settings changed",
            "profile": asdict(self.profile),
            "profile_sha256": self.profile.sha256,
            "physical_points": self._points,
            "delivery_semantics": "OS API accepted frame; no game acknowledgment",
        }

    def _verify(self):
        target = self.guard.validate(require_foreground=True)
        evidence = self.evidence()
        if target.client_rect != self.profile.client_rect:
            raise RuntimeError("Touch target moved or resized; fresh profile required")
        if (
            not isinstance(evidence, TouchPositionEvidence)
            or evidence.playing is not True
            or evidence.positions_verified is not True
            or evidence.profile_sha256 != self.profile.sha256
            or evidence.source_rect != self.profile.client_rect
            or type(evidence.frame_started_ns) is not int
            or not 0 <= self.clock() - evidence.frame_started_ns / 1e9 <= self.max_age
        ):
            raise RuntimeError("Fresh playing/pedal-position evidence is required")
        if self.guard.validate(require_foreground=True).client_rect != self.profile.client_rect:
            raise RuntimeError("Touch target changed during evidence verification")

    def _send(self, changes):
        contacts = [touch_contact(key, self._points[key], flags) for key, flags in changes]
        row = {
            "started_ns": time.perf_counter_ns(),
            "input_mode": "touch",
            "accepted": None,
            "contacts": [{"id": k, "xy": self._points[k], "flags": f} for k, f in changes],
        }
        try:
            accepted = self.sender.inject(contacts)
            row["accepted"] = accepted
            if accepted is not True:
                raise RuntimeError("InjectTouchInput did not accept the contact frame")
        except Exception as exc:
            self._faulted = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["completed_ns"] = time.perf_counter_ns()
            self.trace.append(row)

    def _release(self, cancel=False):
        if self._held:
            # Do not send a new contact or move an existing contact on cleanup.
            flags = UP | (CANCELED if cancel else 0)
            self._send([(key, flags) for key in sorted(self._held)])
            self._held.clear()

    def set_pedals(self, gas: bool, brake: bool):
        with self._lock:
            try:
                if type(gas) is not bool or type(brake) is not bool:
                    raise ValueError("Pedals must be independent booleans")
                if not (gas or brake):
                    self._release()
                    return
                if self._faulted:
                    raise RuntimeError("Touch backend faulted; new presses disabled")
                self._verify()
                desired = {key for key, down in ((1, gas), (2, brake)) if down}
                changes = [(key, UP) for key in sorted(self._held - desired)]
                changes += [
                    (key, INRANGE | INCONTACT | (UPDATE if key in self._held else DOWN))
                    for key in sorted(desired)
                ]
                self._held |= desired  # Conservatively release every possibly accepted contact.
                self._send(changes)
                self._held = desired
            except Exception:
                self._faulted = True
                self._release(cancel=True)
                raise

    def close(self):
        with self._lock:
            self._faulted = True
            self._release(cancel=True)
