"""Read-only Windows window identity and physical client geometry.

Only public window APIs and process metadata are used. No process memory is read.
Discovery never changes focus, starts a process, or sends an input event.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import PureWindowsPath

import psutil


@dataclass(frozen=True)
class ClientRect:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("A capture rectangle must have positive dimensions")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


@dataclass(frozen=True)
class WindowTarget:
    hwnd: int
    pid: int
    process_created_at: float
    title: str
    executable: str
    client_rect: ClientRect

    def as_dict(self) -> dict:
        return asdict(self)


class WindowUnavailable(RuntimeError):
    """The selected game cannot be verified as a visible foreground target."""


class _Rect(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("left", "top", "right", "bottom")]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class WindowsAPI:
    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows window discovery is only available on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_ssize_t)
        signatures = {
            "EnumWindows": ([self._callback_type, ctypes.c_ssize_t], ctypes.c_int),
            "IsWindow": ([ctypes.c_void_p], ctypes.c_int),
            "IsWindowVisible": ([ctypes.c_void_p], ctypes.c_int),
            "IsIconic": ([ctypes.c_void_p], ctypes.c_int),
            "GetWindowTextLengthW": ([ctypes.c_void_p], ctypes.c_int),
            "GetWindowTextW": ([ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int], ctypes.c_int),
            "GetWindowThreadProcessId": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
                ctypes.c_uint32,
            ),
            "GetClientRect": ([ctypes.c_void_p, ctypes.POINTER(_Rect)], ctypes.c_int),
            "ClientToScreen": ([ctypes.c_void_p, ctypes.POINTER(_Point)], ctypes.c_int),
            "GetForegroundWindow": ([], ctypes.c_void_p),
            "SetThreadDpiAwarenessContext": ([ctypes.c_void_p], ctypes.c_void_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.user32, name)
            function.argtypes, function.restype = args, result

    @contextmanager
    def physical_pixels(self):
        # Per-thread context avoids changing another application's DPI settings.
        previous = self.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        if not previous:
            raise WindowUnavailable("Unable to establish physical pixel coordinates")
        try:
            yield
        finally:
            self.user32.SetThreadDpiAwarenessContext(previous)

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    def title(self, hwnd: int) -> str:
        length = self.user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buffer, len(buffer))
        return buffer.value

    def describe(self, hwnd: int) -> WindowTarget:
        if (
            not self.user32.IsWindow(hwnd)
            or not self.user32.IsWindowVisible(hwnd)
            or self.user32.IsIconic(hwnd)
        ):
            raise WindowUnavailable("Target window disappeared, is hidden, or is minimized")
        pid = ctypes.c_uint32()
        if not self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
            raise WindowUnavailable("Cannot resolve target window process")
        try:
            process = psutil.Process(pid.value)
            with process.oneshot():
                executable = process.exe()
                created = process.create_time()
        except (psutil.Error, OSError) as exc:
            raise WindowUnavailable("Cannot verify target process metadata") from exc
        with self.physical_pixels():
            rect, origin = _Rect(), _Point()
            if not self.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                raise WindowUnavailable("Cannot obtain client rectangle")
            if not self.user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                raise WindowUnavailable("Cannot convert client rectangle to screen coordinates")
        try:
            client = ClientRect(origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top)
        except ValueError as exc:
            raise WindowUnavailable("Target client area is empty") from exc
        return WindowTarget(hwnd, pid.value, created, self.title(hwnd), executable, client)

    def discover(self, title_prefix: str, process_name: str) -> list[WindowTarget]:
        if not title_prefix or not process_name:
            raise ValueError("Discovery requires an explicit title prefix and process name")
        targets = []

        @self._callback_type
        def visit(hwnd, _):
            if self.title(hwnd).casefold().startswith(title_prefix.casefold()):
                try:
                    target = self.describe(int(hwnd))
                    if (
                        PureWindowsPath(target.executable).name.casefold()
                        == process_name.casefold()
                    ):
                        targets.append(target)
                except WindowUnavailable:
                    pass
            return 1

        if not self.user32.EnumWindows(visit, 0):
            raise WindowUnavailable("Window enumeration failed")
        return targets


def discover_windows(
    title_prefix: str = "Hill Climb Racing", process_name: str = "crosvm.exe"
) -> list[WindowTarget]:
    return WindowsAPI().discover(title_prefix, process_name)


class WindowGuard:
    """Pin a selected window; require reselection after process restart/handle reuse."""

    def __init__(self, target: WindowTarget, *, api=None):
        self.target = target
        self.api = api if api is not None else WindowsAPI()

    def validate(self, *, require_foreground: bool = True) -> WindowTarget:
        current = self.api.describe(self.target.hwnd)
        expected = self.target
        if (
            current.hwnd != expected.hwnd
            or current.pid != expected.pid
            or current.process_created_at != expected.process_created_at
            or current.executable.casefold() != expected.executable.casefold()
            or current.title != expected.title
        ):
            raise WindowUnavailable("Target identity changed; explicit rediscovery is required")
        if require_foreground and self.api.foreground() != expected.hwnd:
            raise WindowUnavailable("Target is not the foreground window")
        return current
