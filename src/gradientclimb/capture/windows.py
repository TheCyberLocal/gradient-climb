"""Windows window identity, physical client geometry and the game's restart path.

Only public window APIs and process metadata are used. No process memory is read.
Discovery never changes focus, starts a process, or sends an input event. The only
state-changing calls in this module are ``request_close`` (a WM_CLOSE window
message, the emulator's own exit path) and ``bring_to_foreground``; both exist for
the adapter's application-restart recovery and neither sends a click or a key to
the game's content.
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
            "PostMessageW": (
                [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t],
                ctypes.c_int,
            ),
            "ShowWindow": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
            "SetForegroundWindow": ([ctypes.c_void_p], ctypes.c_int),
            "BringWindowToTop": ([ctypes.c_void_p], ctypes.c_int),
            "AttachThreadInput": ([ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.user32, name)
            function.argtypes, function.restype = args, result
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.GetCurrentThreadId.argtypes = []
        self.kernel32.GetCurrentThreadId.restype = ctypes.c_uint32

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

    def foreground_summary(self) -> dict:
        """Best-effort title and process image of the current foreground window.

        Recorded as evidence whenever the game loses the foreground, so a click that
        opened another application (a store page in a browser) is attributable.
        """
        hwnd = self.foreground()
        summary: dict = {"hwnd": hwnd, "title": None, "executable": None}
        if not hwnd:
            return summary
        try:
            summary["title"] = self.title(hwnd)
        except (OSError, ValueError):
            pass
        pid = ctypes.c_uint32()
        if self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) and pid.value:
            try:
                summary["executable"] = PureWindowsPath(psutil.Process(pid.value).exe()).name
            except (psutil.Error, OSError):
                pass
        return summary

    def visible(self, hwnd: int) -> bool:
        return bool(
            self.user32.IsWindow(hwnd)
            and self.user32.IsWindowVisible(hwnd)
            and not self.user32.IsIconic(hwnd)
        )

    def request_close(self, hwnd: int) -> bool:
        """Post WM_CLOSE to a top-level window; no input event reaches its content."""
        return bool(self.user32.PostMessageW(hwnd, 0x0010, 0, 0))

    def bring_to_foreground(self, hwnd: int) -> bool:
        """Restore and activate a window (the manual step an operator performs by hand)."""
        foreground = self.foreground()
        if foreground == hwnd:
            return True
        current = self.kernel32.GetCurrentThreadId()
        other = self.user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
        attached = (
            bool(other)
            and other != current
            and bool(self.user32.AttachThreadInput(current, other, 1))
        )
        try:
            self.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            self.user32.SetForegroundWindow(hwnd)
            self.user32.BringWindowToTop(hwnd)
        finally:
            if attached:
                self.user32.AttachThreadInput(current, other, 0)
        return self.foreground() == hwnd

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
            raise WindowUnavailable(
                f"Target is not the foreground window; {foreground_note(self.api)}"
            )
        return current


def foreground_note(api) -> str:
    """Describe the foreground window for guard evidence; never raises."""
    try:
        summary = api.foreground_summary()
    except Exception:  # noqa: BLE001 - evidence is best effort, the fault itself is not
        return "foreground unknown"
    return (
        f"foreground is {summary.get('title')!r} ({summary.get('executable') or 'unknown process'})"
    )
