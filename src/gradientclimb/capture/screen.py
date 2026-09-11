"""Bounded timestamped screen capture for a previously verified game window.

The API timing is a capture-call round trip, not measured display-to-policy latency.
This module does not launch, focus, resize, or otherwise control the target window.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageGrab

from .windows import ClientRect, WindowGuard, WindowUnavailable


@dataclass(frozen=True)
class CapturedFrame:
    rgb: np.ndarray
    started_ns: int
    completed_ns: int
    utc_started_at: str
    source_rect: ClientRect
    backend: str
    total_latency_ms: float

    @property
    def timestamp_ns(self) -> int:
        """Capture interval midpoint; actual presentation time is unknown."""
        return (self.started_ns + self.completed_ns) // 2

    @property
    def capture_latency_ms(self) -> float:
        return (self.completed_ns - self.started_ns) / 1e6

    def metadata(self) -> dict:
        return {
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "timestamp_ns": self.timestamp_ns,
            "utc_started_at": self.utc_started_at,
            "source_rect": asdict(self.source_rect),
            "backend": self.backend,
            "width": int(self.rgb.shape[1]),
            "height": int(self.rgb.shape[0]),
            "capture_latency_ms": self.capture_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


def crop_rectangle(rect: ClientRect, crop: tuple[float, float, float, float]) -> ClientRect:
    if len(crop) != 4 or not all(math.isfinite(v) for v in crop):
        raise ValueError("Crop must be four finite normalized coordinates")
    left, top, right, bottom = crop
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError("Crop must be within [0,1] with positive area")
    x0, y0 = round(rect.width * left), round(rect.height * top)
    x1, y1 = round(rect.width * right), round(rect.height * bottom)
    return ClientRect(rect.left + x0, rect.top + y0, x1 - x0, y1 - y0)


class WindowCapture:
    def __init__(
        self,
        guard: WindowGuard,
        *,
        backend: str = "mss",
        crop: tuple[float, float, float, float] = (0, 0, 1, 1),
        output_size: tuple[int, int] | None = None,
    ):
        if backend not in {"mss", "pillow"}:
            raise ValueError("Capture backend must be mss or pillow")
        crop_rectangle(guard.target.client_rect, crop)
        if output_size is not None and (
            len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size)
        ):
            raise ValueError("Output size must contain positive integer width and height")
        self.guard, self.backend, self.crop = guard, backend, crop
        self.output_size = output_size
        self._mss = None

    def _read(self, rect: ClientRect) -> np.ndarray:
        if self.backend == "pillow":
            return np.asarray(ImageGrab.grab(bbox=rect.bbox, all_screens=True).convert("RGB"))
        if self._mss is None:
            try:
                import mss
            except ImportError as exc:
                raise RuntimeError("Install gradientclimb[capture] or choose pillow") from exc
            self._mss = mss.mss()
        shot = self._mss.grab(asdict(rect))
        # MSS exposes BGRA bytes; create an independent contiguous RGB observation.
        return np.asarray(shot, dtype=np.uint8)[..., [2, 1, 0]].copy()

    def grab(self) -> CapturedFrame:
        total_start = time.perf_counter_ns()
        before = self.guard.validate(require_foreground=True)
        rect = crop_rectangle(before.client_rect, self.crop)
        utc = datetime.now(UTC).isoformat()
        started = time.perf_counter_ns()
        # Match physical-pixel geometry even on mixed-DPI desktop setups.
        with self.guard.api.physical_pixels():
            rgb = self._read(rect)
        completed = time.perf_counter_ns()
        after = self.guard.validate(require_foreground=True)
        if after.client_rect != before.client_rect:
            raise WindowUnavailable("Window moved/resized during capture; frame discarded")
        if rgb.shape != (rect.height, rect.width, 3):
            raise WindowUnavailable(
                "Captured image dimensions differ from verified client geometry"
            )
        if self.output_size is not None:
            rgb = np.asarray(
                Image.fromarray(rgb).resize(self.output_size, Image.Resampling.BILINEAR)
            ).copy()
        return CapturedFrame(
            rgb,
            started,
            completed,
            utc,
            rect,
            self.backend,
            (time.perf_counter_ns() - total_start) / 1e6,
        )

    def benchmark(
        self,
        *,
        frame_count: int = 120,
        target_fps: float | None = None,
        output_dir: str | Path | None = None,
        record_every: int = 1,
        max_record_bytes: int = 256 * 1024 * 1024,
    ) -> dict:
        """Measure capture; optionally save bounded PNGs and per-frame timestamp JSONL.

        A supplied directory must not exist. Disk encoding/writing counts in the
        end-to-end throughput. Missed schedule deadlines are not claimed dropped
        source frames: neither backend exposes compositor frame/drop counters.
        """
        if type(frame_count) is not int or not 1 <= frame_count <= 10000:
            raise ValueError("frame_count must be an integer in 1..10000")
        if target_fps is not None and (not math.isfinite(target_fps) or target_fps <= 0):
            raise ValueError("target_fps must be finite and positive")
        if type(record_every) is not int or record_every < 1 or max_record_bytes < 1:
            raise ValueError("Recording limits must be positive")
        destination = Path(output_dir) if output_dir is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
        process = psutil.Process()
        cpu_before = process.cpu_times()
        start = time.perf_counter()
        latencies, total_latencies = [], []
        saved_bytes, saved_count, missed, completed_count = 0, 0, 0, 0
        recording_limit_hit = False
        status, failure = "completed", None
        last = None
        manifest = (
            (destination / "frames.jsonl").open("x", encoding="utf-8") if destination else None
        )
        try:
            for index in range(frame_count):
                if target_fps is not None:
                    deadline = start + index / target_fps
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif index and -remaining >= 1 / target_fps:
                        missed += 1
                frame = self.grab()
                last = frame
                completed_count += 1
                latencies.append(frame.capture_latency_ms)
                total_latencies.append(frame.total_latency_ms)
                entry = {"frame_index": index, **frame.metadata(), "path": None}
                if destination and index % record_every == 0 and not recording_limit_hit:
                    buffer = io.BytesIO()
                    Image.fromarray(frame.rgb).save(buffer, format="PNG")
                    data = buffer.getvalue()
                    if saved_bytes + len(data) <= max_record_bytes:
                        name = f"frame-{index:06d}.png"
                        (destination / name).write_bytes(data)
                        saved_bytes += len(data)
                        saved_count += 1
                        entry.update(
                            path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
                        )
                    else:
                        recording_limit_hit = True
                if manifest is not None:
                    manifest.write(json.dumps(entry, allow_nan=False) + "\n")
                    manifest.flush()
        except Exception as exc:
            status, failure = "failed", str(exc)
            raise
        finally:
            if manifest is not None:
                manifest.close()
            elapsed = time.perf_counter() - start
            cpu_after = process.cpu_times()
            cpu_seconds = cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
            summary = {
                "schema_version": "capture-benchmark-1",
                "status": status,
                "failure": failure,
                "backend": self.backend,
                "target": self.guard.target.as_dict(),
                "requested_frames": frame_count,
                "captured_frames": completed_count,
                "elapsed_seconds": elapsed,
                "end_to_end_fps": completed_count / elapsed,
                "capture_latency_ms_mean": float(np.mean(latencies)) if latencies else None,
                "capture_latency_ms_p95": float(np.percentile(latencies, 95))
                if latencies
                else None,
                "total_latency_ms_mean": float(np.mean(total_latencies))
                if total_latencies
                else None,
                "process_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
                "process_rss_bytes": process.memory_info().rss,
                "requested_fps": target_fps,
                "missed_schedule_deadlines": missed,
                "source_dropped_frames": None,
                "gpu_impact": None,
                "recorded_frames": saved_count,
                "recorded_bytes": saved_bytes,
                "recording_limit_hit": recording_limit_hit,
                "output_resolution": list(last.rgb.shape[1::-1]) if last else None,
                "crop": list(self.crop),
                "limitations": [
                    "API round-trip timing is not display-to-observation latency.",
                    "Source frame identifiers, dropped frames, and GPU impact are unavailable.",
                    "Foreground checks cannot detect every popup or momentary occlusion.",
                ],
            }
            if destination is not None:
                (destination / "benchmark.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
        return summary

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
