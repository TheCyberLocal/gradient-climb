"""Frame sinks shared by selected-policy replay and the live training observer.

A sink receives already rendered PIL frames; it never touches simulation or
learning state. This module deliberately imports neither Torch nor Tk at module
level so CI can exercise the headless sinks without a display.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from PIL import Image


class SinkClosed(RuntimeError):
    """The sink can no longer accept frames (window closed, encoder exited)."""


class FrameSink(Protocol):
    kind: str
    paced: bool  # True when the sink wants real-time pacing between frames (a window).

    def open(self, width: int, height: int, fps: float) -> None: ...

    def write(self, image: Image.Image) -> None: ...

    def close(self) -> dict[str, Any]: ...


class NullSink:
    """Count frames without displaying or storing them (CI and overhead protocol)."""

    kind = "null"
    paced = False

    def __init__(self) -> None:
        self.frames = 0
        self.size: tuple[int, int] | None = None
        self.fps: float | None = None

    def open(self, width: int, height: int, fps: float) -> None:
        self.size, self.fps = (int(width), int(height)), float(fps)

    def write(self, image: Image.Image) -> None:
        self.frames += 1

    def close(self) -> dict[str, Any]:
        return {"kind": self.kind, "frames": self.frames, "size": self.size, "fps": self.fps}


class FfmpegSink:
    """Encode raw RGB frames into an H.264 MP4 through a local FFmpeg process.

    ``close()`` is bounded by ``close_timeout`` seconds (stdin close, stderr
    drain and process exit together); an encoder that stalls at close time is
    killed and reported. A stall while frames are still being written blocks the
    writing thread inside ``write()`` and is outside this bound (documented).
    """

    kind = "ffmpeg"
    paced = False
    close_timeout = 30.0

    def __init__(self, path: str | Path, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("ffmpeg")
        if not self.executable:
            raise RuntimeError("FFmpeg is required for --video")
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("Refusing to replace an existing research video")
        self.process: subprocess.Popen | None = None
        self.frames = 0
        self.size: tuple[int, int] | None = None
        self.fps: float | None = None
        self._report: dict[str, Any] | None = None
        self._started = False

    def open(self, width: int, height: int, fps: float) -> None:
        if self.process is not None:
            raise RuntimeError("FFmpeg sink is already open")
        self.size, self.fps = (int(width), int(height)), float(fps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                self.executable,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.size[0]}x{self.size[1]}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._started = True

    def write(self, image: Image.Image) -> None:
        if self.process is None or self.process.stdin is None:
            raise SinkClosed("FFmpeg sink is not open")
        if self.process.poll() is not None:
            raise SinkClosed(f"FFmpeg exited early with code {self.process.returncode}")
        if tuple(image.size) != self.size:
            raise ValueError(f"Frame size {image.size} does not match the declared {self.size}")
        try:
            self.process.stdin.write(image.convert("RGB").tobytes())
        except (BrokenPipeError, OSError) as error:
            raise SinkClosed(f"FFmpeg pipe failed: {error}") from error
        self.frames += 1

    def close(self) -> dict[str, Any]:
        if self._report is not None:
            return self._report
        report: dict[str, Any] = {
            "kind": self.kind,
            "frames": self.frames,
            "video": str(self.path),
            "size": self.size,
            "fps": self.fps,
        }
        process, self.process = self.process, None
        if process is None:
            if not self._started:
                # Never opened (an earlier sink failed first): there is no file to report.
                report.update(video=None, error="FFmpeg encoder was never started")
            self._report = report
            return report
        try:
            # communicate() closes stdin, drains stderr and waits under one
            # deadline; reading stderr to EOF first would block for as long as
            # the encoder hangs.
            try:
                _, stderr = process.communicate(timeout=self.close_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError(
                    f"FFmpeg did not finish encoding within {self.close_timeout:g} s"
                ) from None
            if process.returncode != 0:
                error = stderr.decode(errors="replace") if stderr else ""
                raise RuntimeError(f"FFmpeg failed: {error}")
        finally:
            if process.stderr is not None:
                process.stderr.close()
        self._report = report
        return report


class TkSink:
    """Display frames in a Tk window owned by the thread that calls ``open``."""

    kind = "tk"
    paced = True

    def __init__(self, title: str = "GradientClimb | live training observer") -> None:
        self.title = title
        self.window = None
        self.label = None
        self.frames = 0
        self.closed_by_user = False
        self._tk = None
        self._image_tk = None

    @staticmethod
    def probe() -> None:
        """Open and destroy a hidden root window on the calling thread.

        Raises ``RuntimeError`` when Tk or ``PIL.ImageTk`` is missing or no
        display can be opened, so a headed run fails before any run directory
        exists instead of training for its whole budget without a window.
        """
        try:
            import tkinter as tk

            from PIL import ImageTk  # noqa: F401  (import is the check)
        except ImportError as error:
            raise RuntimeError(f"Tk display support is unavailable: {error}") from error
        try:
            window = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(f"Tk cannot open a window: {error}") from error
        try:
            window.withdraw()
            window.update_idletasks()
        finally:
            window.destroy()

    def open(self, width: int, height: int, fps: float) -> None:
        import tkinter as tk

        from PIL import ImageTk

        self._tk, self._image_tk = tk, ImageTk
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.protocol("WM_DELETE_WINDOW", self._request_close)
        self.label = tk.Label(self.window)
        self.label.pack()

    def _request_close(self) -> None:
        # Destroy at once: a window that is no longer pumped would otherwise stay
        # on screen unresponsive until the run ends.
        self.closed_by_user = True
        window, self.window = self.window, None
        if window is not None:
            with contextlib.suppress(Exception):
                window.destroy()

    def write(self, image: Image.Image) -> None:
        if self.closed_by_user:
            raise SinkClosed("Tk window closed by the user")
        if self.window is None:
            raise SinkClosed("Tk window is not open")
        try:
            photo = self._image_tk.PhotoImage(image)
            self.label.configure(image=photo)
            self.label.image = photo
            self.window.update()
        except self._tk.TclError as error:
            raise SinkClosed(f"Tk window unavailable: {error}") from error
        self.frames += 1

    def close(self) -> dict[str, Any]:
        window, self.window = self.window, None
        if window is not None:
            # A dead Tcl interpreter must not mask the frame report.
            with contextlib.suppress(Exception):
                window.destroy()
        return {"kind": self.kind, "frames": self.frames, "closed_by_user": self.closed_by_user}


class FanoutSink:
    """Forward every frame to several sinks; a sink that reports SinkClosed is closed at once."""

    kind = "fanout"

    def __init__(self, sinks: list[FrameSink]) -> None:
        self.sinks = list(sinks)
        if not self.sinks:
            raise ValueError("At least one frame sink is required")
        self.active = list(self.sinks)
        self.errors: list[str] = []
        self.frames = 0
        self.paced = any(getattr(sink, "paced", False) for sink in self.sinks)
        # The longest bounded close among the sinks (an encoder flush); callers
        # that join the owning thread add it to their own stop timeout.
        self.close_timeout = max(
            (float(getattr(sink, "close_timeout", 0.0) or 0.0) for sink in self.sinks), default=0.0
        )
        self._reports: dict[int, dict[str, Any]] = {}

    def open(self, width: int, height: int, fps: float) -> None:
        for sink in self.sinks:
            sink.open(width, height, fps)

    def _close_one(self, sink: FrameSink) -> dict[str, Any]:
        if id(sink) in self._reports:
            return self._reports[id(sink)]
        try:
            report = sink.close()
        except Exception as error:  # Close every sink and report every failure.
            self.errors.append(f"{sink.kind}: {error}")
            logging.getLogger(__name__).warning("Frame sink close failed", exc_info=True)
            report = {"kind": sink.kind, "frames": getattr(sink, "frames", 0), "error": str(error)}
        self._reports[id(sink)] = report
        return report

    def write(self, image: Image.Image) -> None:
        for sink in list(self.active):
            try:
                sink.write(image)
            except SinkClosed as error:
                self.active.remove(sink)
                self.errors.append(f"{sink.kind}: {error}")
                # Release its resources now (a Tk window, an encoder pipe); the
                # remaining sinks keep receiving frames.
                self._close_one(sink)
        if not self.active:
            raise SinkClosed("Every frame sink has closed")
        self.frames += 1

    def close(self) -> dict[str, Any]:
        reports = [self._close_one(sink) for sink in self.sinks]
        return {
            "kind": self.kind,
            "frames": self.frames,
            "sinks": reports,
            "errors": list(self.errors),
        }
