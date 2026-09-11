"""Supervised probe regressions; all capture, input, focus and clocks are mocked."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.experiments import RunRecorder, list_runs, verify_run
from gradientclimb.experiments.schemas import TrajectoryRecord


def probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts/collect_control_probe.py"
    spec = importlib.util.spec_from_file_location("control_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classifier_requires_geometry_and_both_anchors():
    probe = probe_module()
    reference = np.full((581, 1034, 3), [180, 210, 235], dtype=np.uint8)
    classify = probe.make_playing_classifier(reference)
    assert classify(reference)[0]
    assert not classify(np.zeros_like(reference))[0]
    changed = reference.copy()
    changed[44:106, 970:1023] = 0
    assert not classify(changed)[0]
    assert not classify(reference[:-1])[0]
    assert not classify(reference.astype(np.float32))[0]


def test_pedal_labels_require_both_matches_inside_bounded_search():
    probe = probe_module()
    reference = np.full((581, 1034, 3), 190, dtype=np.uint8)
    rng = np.random.default_rng(4)
    for left, top, right, bottom in probe.PEDAL_BOXES:
        reference[top:bottom, left:right] = rng.integers(
            0, 256, (bottom - top, right - left, 3), dtype=np.uint8
        )
    classify = probe.make_pedal_classifier(reference)
    assert classify(reference)[0]
    shifted = np.full_like(reference, 190)
    for left, top, right, bottom in probe.PEDAL_BOXES:
        shifted[top + 4 : bottom + 4, left - 3 : right - 3] = reference[top:bottom, left:right]
    assert classify(shifted)[0]
    assert all(match["offset_xy"] == [-3, 4] for match in classify(shifted)[1])
    shifted[500:550, 890:965] = 190
    assert not classify(shifted)[0]
    assert not classify(reference[:-1])[0]


def test_persistence_uses_canonical_trajectory_schema(tmp_path):
    probe = probe_module()
    recorded = []

    class Recorder:
        directory = tmp_path

        def trajectory(self, row):
            recorded.append(
                TrajectoryRecord.model_validate(
                    {"run_id": "mock", "timestamp": datetime.now(UTC), **row}
                )
            )

        def register_artifact(self, *args):
            pass

    command = [{"gas": True, "brake": True, "timestamp_ns": 2_000_000_000, "duration_seconds": 0.4}]
    os_trace = [{"events": [[39, True], [37, True]], "delivered": 2}]
    assert probe.persist_probe(Recorder(), [], [], command, os_trace, 1.0) == []
    assert recorded[0].gas and recorded[0].brake
    assert json.loads((tmp_path / "os-pedal-transitions.json").read_text()) == os_trace


@pytest.mark.parametrize("failure", ["capture", "submit", "escape"])
@pytest.mark.parametrize("input_mode", ["vk", "touch"])
def test_failure_preserves_control_traces_frames_and_sealed_run(
    tmp_path, monkeypatch, failure, input_mode
):
    probe = probe_module()
    reference = np.full((581, 1034, 3), [180, 210, 235], dtype=np.uint8)
    reference_path = tmp_path / "reference.png"
    Image.fromarray(reference).save(reference_path)
    now = [100.0]
    monkeypatch.setattr(probe.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    target = SimpleNamespace(
        as_dict=lambda: {"target": "mock"}, client_rect=ClientRect(0, 0, 1034, 581)
    )
    monkeypatch.setattr(probe, "discover_windows", lambda: [target])
    monkeypatch.setattr(
        probe, "WindowGuard", lambda target: SimpleNamespace(validate=lambda: target)
    )
    monkeypatch.setattr(
        probe,
        "RunRecorder",
        lambda root, *args, **kwargs: RunRecorder(tmp_path / "store", *args, **kwargs),
    )
    backends = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.trace, self.closed = [], False
            self.input_encoding = {"mode": kwargs.get("input_mode", "touch")}
            self.evidence = args[2] if len(args) == 3 else None
            self.sender = SimpleNamespace(is_down=lambda key: failure == "escape")
            backends.append(self)

        def close(self):
            self.closed = True
            self.trace.append({"events": [[39, False], [37, False]], "delivered": 2})

    class Capture:
        def __init__(self, *args, **kwargs):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            self.count += 1
            if failure == "capture" and self.count == 2:
                raise OSError("mock capture failure")
            now[0] += 0.04
            end = int(now[0] * 1e9)
            return CapturedFrame(
                reference,
                end - 1000,
                end,
                datetime.now(UTC).isoformat(),
                ClientRect(0, 0, 1034, 581),
                "mock",
                0.001,
            )

    class Controller:
        def __init__(self, backend, allowed):
            self.backend, self.allowed, self.trace, self.fault = backend, allowed, [], None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.release()

        def release(self):
            self.backend.close()

        def submit(self, action):
            assert self.allowed()
            if self.backend.evidence:
                evidence = self.backend.evidence()
                assert evidence.playing and evidence.positions_verified
                assert evidence.source_rect == target.client_rect
            self.backend.trace.append(
                {"events": [[39, action.gas], [37, action.brake]], "delivered": 2}
            )
            if failure == "submit":
                raise OSError("mock submit failure")
            self.trace.append(
                {
                    "gas": action.gas,
                    "brake": action.brake,
                    "timestamp_ns": int(now[0] * 1e9),
                    "duration_seconds": action.duration,
                }
            )

    monkeypatch.setattr(probe, "WindowsPedalBackend", Backend)
    monkeypatch.setattr(probe, "WindowsTouchPedalBackend", Backend)
    monkeypatch.setattr(probe, "WindowCapture", Capture)
    monkeypatch.setattr(probe, "PedalController", Controller)
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_control_probe.py",
            "--reference",
            str(reference_path),
            "--wait-seconds",
            "1",
            "--input-mode",
            input_mode,
        ],
    )
    probe.main()
    run = list_runs(tmp_path / "store")[0]
    assert run["status"] == "failed"
    assert backends[0].closed
    assert verify_run(tmp_path / "store", run["run_id"])["valid"]
    assert run["summary"]["persistence_errors"] == []
    if failure != "escape":
        assert run["summary"]["frames"] == 1
        assert f"mock {failure} failure" in run["summary"]["failure_traceback"]
    directory = tmp_path / "store/runs" / run["run_id"]
    assert json.loads((directory / "os-pedal-transitions.json").read_text()) == backends[0].trace
