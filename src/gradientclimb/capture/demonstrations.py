"""Bounded, read-only human demonstration capture and causal offline pairing.

Only the two declared arrow controls are sampled; no hooks or input injection are
used. Polls report observed key state, never exact OS delivery or human intent.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
import threading
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .windows import WindowGuard, WindowUnavailable

DEMONSTRATION_VERSION = "human-demonstration-3.0"


@dataclass(frozen=True)
class DemonstrationLimits:
    seconds: float = 300
    frames_per_second: float = 10
    controls_per_second: float = 100
    maximum_frames: int = 10000
    maximum_payload_bytes: int = 512 * 1024**2
    maximum_frame_age_seconds: float = 0.45
    maximum_poll_gap_seconds: float = 0.05

    def __post_init__(self):
        for name in (
            "seconds",
            "frames_per_second",
            "controls_per_second",
            "maximum_frame_age_seconds",
            "maximum_poll_gap_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.seconds > 1800 or self.frames_per_second > 60 or self.controls_per_second > 500:
            raise ValueError("Demonstration duration and sample rates exceed bounded limits")
        for name in ("maximum_frames", "maximum_payload_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


class DemonstrationWindowGuard(WindowGuard):
    """Pin geometry and sanitize unrelated foreground information from demo logs."""

    def validate(self, *, require_foreground=True):
        try:
            current = super().validate(require_foreground=require_foreground)
        except WindowUnavailable:
            raise WindowUnavailable("Demonstration target lost focus or identity") from None
        if current.client_rect != self.target.client_rect:
            raise WindowUnavailable("Demonstration target geometry changed")
        return current


class NativePedalReader:
    """Read only Right/Left key-down bits, guarded before and after each poll."""

    def __init__(self, guard, *, key_state=None, clock_ns=time.perf_counter_ns):
        self.guard, self.clock_ns = guard, clock_ns
        if key_state is None:
            if os.name != "nt":
                raise OSError("Native demonstration control sampling requires Windows")
            api = ctypes.WinDLL("user32", use_last_error=True)
            api.GetAsyncKeyState.argtypes = [ctypes.c_int]
            api.GetAsyncKeyState.restype = ctypes.c_short
            key_state = api.GetAsyncKeyState
        self.key_state = key_state

    def __call__(self):
        self.guard.validate(require_foreground=True)
        started = self.clock_ns()
        gas = bool(self.key_state(0x27) & 0x8000)  # VK_RIGHT; ignore process-global low bit.
        gas_observed = self.clock_ns()
        brake = bool(self.key_state(0x25) & 0x8000)  # VK_LEFT; no other key is inspected.
        completed = self.clock_ns()
        self.guard.validate(require_foreground=True)
        return {
            "started_ns": started,
            "completed_ns": completed,
            "gas_observed_ns": gas_observed,
            "brake_observed_ns": completed,
            "gas": gas,
            "brake": brake,
        }


class PayloadBudget:
    """Reserve original bytes and canonical copies before writing either stream."""

    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.reserved_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if self.reserved_bytes + 2 * size > self.maximum_bytes:
                raise RuntimeError("Demonstration payload disk budget exhausted")
            self.reserved_bytes += 2 * size


class BoundedJournal:
    def __init__(self, path, budget):
        self.path, self.budget = Path(path), budget
        self.stream = self.path.open("xb")
        self.lock = threading.Lock()
        self.closed = False

    def write(self, row):
        data = (json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode()
        with self.lock:
            if self.closed:
                raise RuntimeError("Demonstration journal is closed")
            self.budget.reserve(len(data))
            if self.stream.write(data) != len(data):
                raise OSError("Partial demonstration journal write")
            self.stream.flush()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                try:
                    self.stream.flush()
                    os.fsync(self.stream.fileno())
                finally:
                    self.stream.close()


def _failure(exc):
    # WindowGuard can include unrelated foreground titles; never persist those
    # in this scoped demonstration dataset, even with an injected guard/capture.
    return (
        "WindowUnavailable: demonstration target unavailable"
        if isinstance(exc, WindowUnavailable)
        else f"{type(exc).__name__}: {exc}"
    )


class ControlSampler:
    def __init__(
        self, reader, journal, *, frequency, max_gap_seconds, clock_ns=time.perf_counter_ns
    ):
        self.reader, self.journal = reader, journal
        self.frequency, self.max_gap_ns = frequency, int(max_gap_seconds * 1e9)
        self.clock_ns = clock_ns
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="human-pedal-observer", daemon=True)
        self.samples = self.transitions = self.gaps = 0
        self.error = None
        self.previous = None
        self.first_started_ns = None

    def _run(self):
        try:
            while not self.stop.is_set():
                tick = self.clock_ns()
                row = self.reader()
                if self.stop.is_set():
                    break
                if (
                    type(row.get("started_ns")) is not int
                    or type(row.get("completed_ns")) is not int
                    or not 0 <= row["started_ns"] <= row["completed_ns"] <= self.clock_ns()
                    or type(row.get("gas")) is not bool
                    or type(row.get("brake")) is not bool
                ):
                    raise ValueError("Invalid observed control sample")
                previous = self.previous
                if previous and row["started_ns"] <= previous["completed_ns"]:
                    raise ValueError("Control samples are repeated or nonmonotonic")
                gap = row["started_ns"] - previous["completed_ns"] if previous else None
                changes = [
                    name for name in ("gas", "brake") if previous and row[name] != previous[name]
                ]
                row = {
                    **row,
                    "sample_index": self.samples,
                    "changed_controls": changes,
                    "transition_earliest_ns": previous["started_ns"] if changes else None,
                    "transition_latest_ns": row["completed_ns"] if changes else None,
                    "simultaneous_change_order": "unknown" if len(changes) > 1 else None,
                    "gap_since_previous_ns": gap,
                    "gap_censored": gap is not None and gap > self.max_gap_ns,
                    "source": "focused_key_state_poll; exact delivery and between-poll edges unknown",
                }
                self.journal.write(row)
                if self.first_started_ns is None:
                    self.first_started_ns = row["started_ns"]
                self.samples += 1
                self.transitions += bool(changes)
                self.gaps += bool(row["gap_censored"])
                self.previous = row
                elapsed = (self.clock_ns() - tick) / 1e9
                self.stop.wait(max(0, 1 / self.frequency - elapsed))
        except BaseException as exc:  # noqa: BLE001 - Transport worker termination to the owner.
            self.error = _failure(exc)
            self.stop.set()

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        # Closed journals prohibit late writes even if an OS read stalls.
        self.journal.close()
        return not self.thread.is_alive()


def record_demonstration(
    run,
    *,
    capture,
    controls,
    host_check,
    limits=None,
    classify=None,
    clock_ns=time.perf_counter_ns,
    wait=time.sleep,
):
    """Record one human-controlled session; do not infer episodes or inject input.

    `capture` and `controls` must independently enforce current focus and pinned
    geometry. Classifier output is diagnostic; unknown UI causes no navigation.
    """
    limits = limits or DemonstrationLimits()
    host_check()
    budget = PayloadBudget(limits.maximum_payload_bytes)
    directory = run.directory / "demonstration"
    directory.mkdir(exist_ok=False)
    frames = BoundedJournal(directory / "frames.jsonl", budget)
    try:
        samples = BoundedJournal(directory / "controls.jsonl", budget)
    except BaseException:
        frames.close()
        raise
    sampler = ControlSampler(
        controls,
        samples,
        frequency=limits.controls_per_second,
        max_gap_seconds=limits.maximum_poll_gap_seconds,
        clock_ns=clock_ns,
    )
    start = clock_ns()
    previous_started = previous_completed = previous_hash = None
    count = duplicates = missed_slots = 0
    playing_frames = unknown_frames = 0
    state_counts = {}
    frame_payload_bytes = 0
    failure = None
    reason, status = "session_time_limit", "completed"
    cleanup_errors = []
    sampler.start()
    try:
        while count < limits.maximum_frames:
            if sampler.error:
                raise RuntimeError(sampler.error)
            now = clock_ns()
            if now - start >= limits.seconds * 1e9:
                break
            host_check()
            frame = capture()
            ready = clock_ns()
            if not 0 <= frame.started_ns <= frame.completed_ns <= ready:
                raise ValueError("Capture timestamps are invalid")
            if previous_completed is not None and frame.started_ns <= previous_completed:
                raise ValueError("Capture timestamps are repeated or nonmonotonic")
            if ready - frame.started_ns > limits.maximum_frame_age_seconds * 1e9:
                raise RuntimeError("Demonstration capture is stale")
            if sampler.error:
                raise RuntimeError(sampler.error)
            raw_hash = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
            duplicate = raw_hash == previous_hash
            interval = frame.started_ns - previous_started if previous_started is not None else None
            period_ns = 1e9 / limits.frames_per_second
            missed = max(0, math.floor(interval / period_ns) - 1) if interval is not None else 0
            classify_started = clock_ns()
            state = classify(frame) if classify else {"state": "unlabeled", "variant": None}
            classify_completed = clock_ns()
            data = io.BytesIO()
            Image.fromarray(frame.rgb).save(data, format="PNG")
            blob = data.getvalue()
            budget.reserve(len(blob))
            name = f"frame-{count:06d}.png"
            with (directory / name).open("xb") as stream:
                if stream.write(blob) != len(blob):
                    raise OSError("Partial demonstration frame write")
                stream.flush()
            row = {
                **frame.metadata(),
                "frame_index": count,
                "observation_ready_ns": ready,
                "recorded_ns": clock_ns(),
                "path": name,
                "file_sha256": hashlib.sha256(blob).hexdigest(),
                "pixels_sha256": raw_hash,
                "identical_to_previous_pixels": duplicate,
                "capture_interval_ns": interval,
                "missed_capture_schedule_slots": missed,
                "source_compositor_drops": None,
                "state_evidence": state,
                "state_classification_latency_ms": (classify_completed - classify_started) / 1e6,
                "actor_model_latency_ms": None,
                "timing_semantics": "Capture-call interval; actual display and human reaction times unknown",
            }
            frames.write(row)
            count += 1
            state_name = state.get("state", "unknown")
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
            playing_frames += state_name == "playing"
            unknown_frames += state_name in {"unknown", "unlabeled"}
            duplicates += duplicate
            missed_slots += missed
            frame_payload_bytes += len(blob)
            previous_started, previous_completed, previous_hash = (
                frame.started_ns,
                frame.completed_ns,
                raw_hash,
            )
            elapsed = (clock_ns() - now) / 1e9
            remaining = max(0, limits.seconds - (clock_ns() - start) / 1e9)
            wait(min(remaining, max(0, 1 / limits.frames_per_second - elapsed)))
        else:
            reason = "frame_limit"
    except KeyboardInterrupt as exc:
        status, reason, failure = "cancelled", "operator_interruption", _failure(exc)
    except Exception as exc:  # noqa: BLE001 - Preserve completed evidence after backend failures.
        status, reason, failure = "failed", "capture_control_or_storage_failure", _failure(exc)
    finally:
        try:
            if not sampler.close():
                cleanup_errors.append("Control sampler did not finish within the bounded join")
        except Exception as exc:  # noqa: BLE001 - Record cleanup failure and still close frames.
            cleanup_errors.append(_failure(exc))
        try:
            frames.close()
        except Exception as exc:  # noqa: BLE001 - Preserve summary when capture cleanup fails.
            cleanup_errors.append(_failure(exc))
    if sampler.error and failure is None:
        status, reason, failure = "failed", "control_sampling_failure", sampler.error
    if cleanup_errors:
        status = "failed"
    ended = clock_ns()
    summary = {
        "version": DEMONSTRATION_VERSION,
        "status": status,
        "stop_reason": reason,
        "failure": failure,
        "cleanup_errors": cleanup_errors,
        "frames": count,
        "control_samples": sampler.samples,
        "observed_control_changes": sampler.transitions,
        "control_gap_intervals": sampler.gaps,
        "identical_pixel_frames": duplicates,
        "missed_capture_schedule_slots": missed_slots,
        "source_compositor_drops": None,
        "observed_seconds": (ended - start) / 1e9,
        "capture_session_started_ns": start,
        "capture_session_ended_ns": ended,
        "capture_wall_clock_seconds": (ended - start) / 1e9,
        "first_control_sample_started_ns": sampler.first_started_ns,
        "last_control_sample_completed_ns": sampler.previous["completed_ns"]
        if sampler.previous
        else None,
        "control_sampling_span_seconds": (
            (sampler.previous["completed_ns"] - sampler.first_started_ns) / 1e9
            if sampler.previous
            else None
        ),
        "frames_labeled_playing": playing_frames,
        "frames_labeled_unknown": unknown_frames,
        "state_counts": state_counts,
        "real_interaction_seconds": None,
        "real_interaction_missing_reason": "Requires reviewed episode segmentation; UI template states and frame times are retained",
        "payload_reserved_bytes_including_canonical_copies": budget.reserved_bytes,
        "frame_payload_bytes": frame_payload_bytes,
        "injected_input_events": 0,
        "episode_count": None,
        "completed_segmented_episodes": 0,
        "episode_count_missing_reason": "Unsegmented recording; experience count is unknown",
        "episode_segmentation": "Unlabeled recording session; natural outcomes require review",
        "action_semantics": "Observed focused arrow key states, not intended/delivered game actions",
        "last_control_interval_censored": True,
        "qualification_evidence": False,
        "imitation_eligibility": "Requires playing segmentation, recorded causal action pairs and declared training partition",
    }
    # Register complete and partial payloads alike, after all writers are closed.
    # A partially written journal is retained verbatim, never repaired in place.
    run.annotate(demonstration=summary)
    for path in sorted(directory.iterdir()):
        if path.is_file():
            run.register_artifact(path, "human_demonstration_payload", {"session_status": status})
    return summary


def causal_action_pairs(
    frame_rows, control_rows, *, maximum_lag_seconds=0.1, maximum_poll_gap_seconds=0.05
):
    """Pair each available frame only with a subsequent recorded control sample.

    Sparse/unbracketed intervals stay ineligible. This proves ordering of recorded
    times only, not that the human reacted to the particular captured frame.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (maximum_lag_seconds, maximum_poll_gap_seconds)):
        raise ValueError("Pairing latency and gap bounds must be positive")
    starts, ends = [], []
    for row in control_rows:
        if (
            type(row["started_ns"]) is not int
            or type(row["completed_ns"]) is not int
            or not 0 <= row["started_ns"] <= row["completed_ns"]
            or type(row["gas"]) is not bool
            or type(row["brake"]) is not bool
        ):
            raise ValueError("Invalid control interval")
        if ends and row["started_ns"] <= ends[-1]:
            raise ValueError("Control intervals must be strictly ordered and nonoverlapping")
        starts.append(row["started_ns"])
        ends.append(row["completed_ns"])
    pairs = []
    for frame in frame_rows:
        ready = frame["observation_ready_ns"]
        if (
            any(
                type(frame[key]) is not int
                for key in ("started_ns", "completed_ns", "observation_ready_ns")
            )
            or not frame["started_ns"] <= frame["completed_ns"] <= ready
        ):
            raise ValueError("Observation-ready time precedes acquisition completion")
        index = bisect_left(starts, ready)
        preceding_index = bisect_right(ends, frame["started_ns"]) - 1
        candidate = control_rows[index] if index < len(control_rows) else None
        covered = (
            candidate is not None
            and index > 0
            and (
                starts[index] - ends[index - 1] <= maximum_poll_gap_seconds * 1e9
                and ends[index - 1] <= ready
                and candidate["completed_ns"] - candidate["started_ns"]
                <= maximum_poll_gap_seconds * 1e9
            )
        )
        eligible = bool(
            frame.get("state_evidence", {}).get("state") == "playing"
            and covered
            and (candidate["completed_ns"] - ready) <= maximum_lag_seconds * 1e9
        )
        pairs.append(
            {
                "frame_index": frame["frame_index"],
                "eligible": eligible,
                "target_control_sample_index": index if eligible else None,
                "preceding_control_sample_index": preceding_index if preceding_index >= 0 else None,
                "gas": candidate["gas"] if eligible else None,
                "brake": candidate["brake"] if eligible else None,
                "censored_or_gap": not eligible,
                "observation_ready_ns": ready,
                "target_sample_started_ns": candidate["started_ns"] if eligible else None,
            }
        )
    return pairs


def read_complete_journal(path):
    """Recover complete rows without modifying a possibly interrupted journal."""
    data = Path(path).read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines()]
    return {"rows": rows, "trailing_partial_bytes": len(data) - len(complete)}
