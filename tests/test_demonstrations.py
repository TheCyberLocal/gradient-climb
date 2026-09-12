"""Read-only recording faults and temporal ordering use synthetic pixels/keys."""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gradientclimb.capture.demonstrations import (
    BoundedJournal,
    ControlSampler,
    DemonstrationLimits,
    DemonstrationWindowGuard,
    NativePedalReader,
    PayloadBudget,
    causal_action_pairs,
    read_complete_journal,
    record_demonstration,
)
from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect, WindowTarget, WindowUnavailable
from gradientclimb.experiments import RunRecorder, verify_run


class FakeRun:
    def __init__(self, directory):
        self.directory = directory
        self.artifacts = []
        self.summary = {}

    def register_artifact(self, path, kind, metadata):
        self.artifacts.append((path, kind, metadata))

    def annotate(self, **kwargs):
        self.summary.update(kwargs)


def frame(*, old=False):
    started = time.perf_counter_ns() - (1_000_000_000 if old else 100_000)
    return CapturedFrame(
        np.zeros((12, 16, 3), np.uint8),
        started,
        started + 50_000,
        "2026-09-12T00:00:00Z",
        ClientRect(0, 0, 16, 12),
        "synthetic",
        0.05,
    )


def controls():
    started = time.perf_counter_ns()
    return {
        "started_ns": started,
        "completed_ns": time.perf_counter_ns(),
        "gas": True,
        "brake": False,
    }


def record(run, **changes):
    kwargs = {
        "capture": frame,
        "controls": controls,
        "host_check": lambda: None,
        "limits": DemonstrationLimits(seconds=1, frames_per_second=60, maximum_frames=3),
        "classify": lambda _: {"state": "playing"},
    }
    kwargs.update(changes)
    return record_demonstration(run, **kwargs)


def test_native_reader_only_inspects_two_declared_keys_and_ignores_low_bit():
    calls, checks = [], []
    guard = SimpleNamespace(validate=lambda **kw: checks.append(kw))

    def key(vk):
        calls.append(vk)
        return 0x8000 if vk == 0x27 else 1

    observed = NativePedalReader(guard, key_state=key)()
    assert calls == [0x27, 0x25] and checks == [{"require_foreground": True}] * 2
    assert observed["gas"] and not observed["brake"]
    assert observed["started_ns"] <= observed["gas_observed_ns"] <= observed["brake_observed_ns"]


def test_out_of_focus_never_samples_keys_and_focus_change_discards_sample():
    calls = []

    def lost(**kwargs):
        raise WindowUnavailable("A private foreground title must not be retained")

    with pytest.raises(WindowUnavailable):
        NativePedalReader(SimpleNamespace(validate=lost), key_state=lambda key: calls.append(key))()
    assert not calls
    checks = [0]

    def during(**kwargs):
        checks[0] += 1
        if checks[0] == 2:
            lost()

    with pytest.raises(WindowUnavailable):
        NativePedalReader(SimpleNamespace(validate=during), key_state=lambda key: 0)()


def test_demo_guard_pins_geometry_and_sanitizes_foreground_names():
    target = WindowTarget(1, 2, 3.0, "Hill Climb Racing", "game.exe", ClientRect(1, 2, 100, 60))
    api = SimpleNamespace(
        describe=lambda _: target,
        foreground=lambda: 10,
        foreground_summary=lambda: {"title": "private-title"},
    )
    guard = DemonstrationWindowGuard(target, api=api)
    with pytest.raises(WindowUnavailable, match="lost focus") as error:
        guard.validate()
    assert "private-title" not in str(error.value)
    api.foreground = lambda: 1
    moved = WindowTarget(1, 2, 3.0, "Hill Climb Racing", "game.exe", ClientRect(2, 2, 100, 60))
    api.describe = lambda _: moved
    with pytest.raises(WindowUnavailable, match="geometry"):
        guard.validate()


def test_recording_has_monotonic_frames_independent_controls_and_no_injected_input(tmp_path):
    run = FakeRun(tmp_path)
    result = record(run)
    assert result["status"] == "completed" and result["frames"] == 3
    assert result["injected_input_events"] == 0
    assert result["episode_count"] is None
    assert result["completed_segmented_episodes"] == 0
    assert "unknown" in result["episode_count_missing_reason"]
    assert result["control_samples"] > 0 and result["identical_pixel_frames"] == 2
    assert result["source_compositor_drops"] is None and result["last_control_interval_censored"]
    assert not result["cleanup_errors"]
    rows = read_complete_journal(tmp_path / "demonstration/frames.jsonl")["rows"]
    assert all(r["started_ns"] <= r["completed_ns"] <= r["observation_ready_ns"] for r in rows)
    assert all(r["state_evidence"]["state"] == "playing" for r in rows)
    assert not any(t.name == "human-pedal-observer" for t in threading.enumerate())


@pytest.mark.parametrize(
    "fault,expected",
    [
        (KeyboardInterrupt(), "cancelled"),
        (OSError("capture failure"), "failed"),
        (WindowUnavailable("private foreground title"), "failed"),
    ],
)
def test_interrupt_capture_and_window_closure_preserve_completed_work(tmp_path, fault, expected):
    count = [0]

    def capture():
        count[0] += 1
        if count[0] > 1:
            raise fault
        return frame()

    run = FakeRun(tmp_path)
    result = record(run, capture=capture)
    assert result["status"] == expected and result["frames"] == 1
    assert run.summary["demonstration"]["frames"] == 1
    assert "private foreground title" not in json.dumps(result)
    assert run.artifacts and not result["cleanup_errors"]


def test_stale_capture_rejected_before_frame_is_written(tmp_path):
    result = record(FakeRun(tmp_path), capture=lambda: frame(old=True))
    assert result["status"] == "failed" and result["frames"] == 0
    assert "stale" in result["failure"]
    assert not list((tmp_path / "demonstration").glob("*.png"))


def test_repeated_frame_timestamp_is_not_mislabeled_new_capture(tmp_path):
    fixed = frame()
    result = record(FakeRun(tmp_path), capture=lambda: fixed)
    assert result["status"] == "failed" and result["frames"] == 1
    assert "repeated or nonmonotonic" in result["failure"]


def test_control_sampler_failure_stops_capture_and_is_not_success(tmp_path):
    count = [0]

    def fail():
        count[0] += 1
        if count[0] > 1:
            raise RuntimeError("control sample fault")
        return controls()

    result = record(FakeRun(tmp_path), controls=fail)
    assert result["status"] == "failed" and "control sample fault" in result["failure"]
    assert result["control_samples"] == 1 and not result["cleanup_errors"]


def test_disk_budget_and_runtime_host_pressure_stop_without_erasing_frames(tmp_path):
    run = FakeRun(tmp_path)
    result = record(run, limits=DemonstrationLimits(maximum_payload_bytes=1, maximum_frames=3))
    assert result["status"] == "failed" and result["frames"] == 0
    assert result["payload_reserved_bytes_including_canonical_copies"] <= 1


def test_runtime_host_failure_preserves_previous_capture(tmp_path):
    checks = [0]

    def host():
        checks[0] += 1
        if checks[0] > 2:
            raise RuntimeError("Host disk margin")

    result = record(FakeRun(tmp_path), host_check=host)
    assert result["frames"] == 1 and result["status"] == "failed"
    assert "Host disk margin" in result["failure"]


def test_partial_journal_write_detected_and_partial_tail_recovered_without_mutation(tmp_path):
    path = tmp_path / "partial.jsonl"
    data = b'{"index":0}\n{"index":'
    path.write_bytes(data)
    recovered = read_complete_journal(path)
    assert recovered == {"rows": [{"index": 0}], "trailing_partial_bytes": 9}
    assert path.read_bytes() == data
    path.write_bytes(b"not-json\n")
    with pytest.raises(ValueError):
        read_complete_journal(path)
    journal = BoundedJournal(tmp_path / "write.jsonl", PayloadBudget(10000))
    original = journal.stream
    journal.stream = SimpleNamespace(write=lambda data: original.write(data[:2]))
    with pytest.raises(OSError, match="Partial"):
        journal.write({"index": 1})
    journal.stream = original
    journal.close()
    with pytest.raises(RuntimeError, match="closed"):
        journal.write({"index": 2})


def test_multiple_control_changes_keep_unknown_order_and_poll_gaps(tmp_path):
    journal = BoundedJournal(tmp_path / "controls.jsonl", PayloadBudget(100000))
    index = [0]

    def read():
        i = index[0]
        index[0] += 1
        now = time.perf_counter_ns()
        if i == 2:
            raise RuntimeError("done")
        return {"started_ns": now, "completed_ns": now, "gas": bool(i), "brake": bool(i)}

    sampler = ControlSampler(read, journal, frequency=100, max_gap_seconds=0.001)
    sampler.start()
    assert sampler.stop.wait(1)
    assert sampler.close()
    rows = read_complete_journal(journal.path)["rows"]
    assert rows[1]["changed_controls"] == ["gas", "brake"]
    assert rows[1]["simultaneous_change_order"] == "unknown" and rows[1]["gap_censored"]


def pair_frames():
    return [
        {
            "frame_index": 0,
            "started_ns": 10,
            "completed_ns": 20,
            "observation_ready_ns": 25,
            "state_evidence": {"state": "playing"},
        }
    ]


def pair_controls():
    return [
        {"started_ns": 5, "completed_ns": 6, "gas": False, "brake": False},
        {"started_ns": 23, "completed_ns": 24, "gas": False, "brake": True},
        {"started_ns": 26, "completed_ns": 27, "gas": True, "brake": False},
    ]


def test_causal_pairing_uses_only_actions_sampled_after_observation_ready():
    paired = causal_action_pairs(pair_frames(), pair_controls())[0]
    assert paired["eligible"] and paired["target_control_sample_index"] == 2
    assert paired["preceding_control_sample_index"] == 0
    assert paired["gas"] and not paired["brake"]
    assert paired["target_sample_started_ns"] >= paired["observation_ready_ns"]


@pytest.mark.parametrize("change", ["unknown", "menu", "future", "gap", "overlap"])
def test_missing_or_nonplaying_action_targets_remain_censored(change):
    frames, controls = pair_frames(), pair_controls()
    options = {}
    if change == "unknown":
        frames[0]["state_evidence"] = {"state": "unknown"}
    if change == "menu":
        frames[0]["state_evidence"] = {"state": "tune"}
    if change == "future":
        controls = controls[:2]
    if change == "gap":
        options["maximum_poll_gap_seconds"] = 1e-9
    if change == "overlap":
        controls[1]["completed_ns"] = 25
        frames[0]["observation_ready_ns"] = 24
    result = causal_action_pairs(frames, controls, **options)[0]
    assert not result["eligible"] and result["gas"] is None and result["censored_or_gap"]


def test_real_recorder_seals_demo_as_zero_episode_nonqualification(tmp_path):
    root = tmp_path / "artifacts"
    with RunRecorder(root, "synthetic-demo-test", {"synthetic": True}) as run:
        result = record(run, limits=DemonstrationLimits(maximum_frames=1, frames_per_second=60))
        run.finalize(status=result["status"], episode_count=0, demonstration=result)
    assert verify_run(root, run.run_id)["valid"]
    stored = json.loads((run.directory / "run.json").read_text())
    assert stored["summary"]["episode_count"] == 0
    assert stored["summary"]["demonstration"]["episode_count"] is None
    assert stored["summary"]["demonstration"]["completed_segmented_episodes"] == 0
    assert stored["summary"]["demonstration"]["qualification_evidence"] is False


def test_foreground_arming_does_not_capture_or_inspect_keys():
    spec = importlib.util.spec_from_file_location(
        "demo_script", Path(__file__).resolve().parents[1] / "scripts/record_human_demonstration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now, checks = [0.0], []
    guard = SimpleNamespace(
        target=SimpleNamespace(hwnd=2),
        api=SimpleNamespace(foreground=lambda: 2 if now[0] >= 0.2 else 1),
        validate=lambda **kwargs: checks.append(kwargs),
    )
    module.wait_for_foreground(
        guard, 1, clock=lambda: now[0], wait=lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    assert checks[-1] == {"require_foreground": True}
    guard.api.foreground = lambda: 1
    with pytest.raises(TimeoutError):
        module.wait_for_foreground(
            guard,
            0.2,
            clock=lambda: now[0],
            wait=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
