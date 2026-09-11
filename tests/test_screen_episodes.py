"""Native episode orchestration tests with synthetic frames and no OS interaction."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect


def episode_module():
    spec = importlib.util.spec_from_file_location(
        "screen_episodes_test",
        Path(__file__).resolve().parents[1] / "scripts/run_screen_episodes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(module, *, fail_capture=False):
    clock = [1.0]
    module.time = SimpleNamespace(perf_counter=lambda: clock[0])
    sent, released = [], []
    controller = SimpleNamespace(
        fault=None,
        submit=lambda action: sent.append(action.code),
        release=lambda: released.append(True),
    )

    def frame():
        stamp = int(clock[0] * 1e9)
        return SimpleNamespace(
            state="playing",
            frame=CapturedFrame(
                np.zeros((2, 2, 3), dtype=np.uint8),
                stamp,
                stamp,
                "test",
                ClientRect(0, 0, 2, 2),
                "fake",
                0,
            ),
        )

    def observe():
        if fail_capture:
            raise RuntimeError("capture lost")
        clock[0] += 0.03
        return frame()

    adapter = SimpleNamespace(observe=observe, is_playing=lambda: True)

    def measure(*args, **kwargs):
        clock[0] += 0.03
        return SimpleNamespace(hud={"valid": False}, as_dict=lambda: {"missing": True})

    bridge = SimpleNamespace(reset=lambda: None, observe=measure)
    return adapter, controller, SimpleNamespace(trace=[]), bridge, frame(), sent, released


def test_bounded_episode_releases_and_preserves_unknown_score():
    module = episode_module()
    adapter, controller, backend, bridge, first, sent, released = fixture(module)
    summary, rows, images = module.collect_episode(
        adapter,
        controller,
        backend,
        bridge,
        lambda _: 3,
        first,
        seconds=0.12,
        deadline=10,
    )
    assert sent and all(code == 3 for code in sent)
    assert released and summary["error"] is None
    assert summary["reason"] == "episode_time_limit"
    assert summary["distance"] is None and summary["observed_hud_max"] is None
    assert rows and images
    assert all(row["elapsed_seconds"] < 0.12 for row in rows)


def test_capture_failure_retains_partial_frames_and_releases():
    module = episode_module()
    adapter, controller, backend, bridge, first, sent, released = fixture(module, fail_capture=True)
    summary, rows, images = module.collect_episode(
        adapter,
        controller,
        backend,
        bridge,
        lambda _: 1,
        first,
        seconds=1,
        deadline=10,
    )
    assert sent == [1] and released
    assert len(rows) == len(images) == 1
    assert summary["reason"] == "episode_failure" and "capture lost" in summary["error"]


def test_invalid_policy_never_reaches_backend():
    module = episode_module()
    adapter, controller, backend, bridge, first, sent, released = fixture(module)
    summary, rows, _ = module.collect_episode(
        adapter,
        controller,
        backend,
        bridge,
        lambda _: 9,
        first,
        seconds=1,
        deadline=10,
    )
    assert not sent and released and not rows
    assert "invalid independent pedal" in summary["error"]


def test_terminal_agreement_requires_fresh_spaced_consecutive_valid_readings():
    module = episode_module()
    current = {"valid": True, "distance_meters": 229}
    collector = module.ResultScoreCollector(SimpleNamespace(read=lambda *a, **k: current.copy()))

    def seen(stamp):
        return SimpleNamespace(
            state="result",
            frame=SimpleNamespace(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                metadata=lambda: {"timestamp_ns": stamp},
            ),
        )

    assert collector(seen(1_000_000_000)) is False
    assert collector(seen(1_000_000_000)) is False
    assert collector(seen(1_100_000_000)) is False
    current["valid"] = False
    assert collector(seen(1_200_000_000)) is False
    current["valid"] = True
    assert collector(seen(1_300_000_000)) is False
    current["distance_meters"] = 228
    assert collector(seen(1_500_000_000)) is False
    assert collector(seen(1_650_000_000)) is True
    assert collector.accepted[0]["distance_meters"] == 228
    assert collector.accepted[0]["agreement_interval_seconds"] == 0.15
    assert len(collector.readings) == 7


def test_parked_score_uses_confirmed_boundary_and_preserves_unknown():
    module = episode_module()
    frame = SimpleNamespace(rgb=None, metadata=lambda: {"timestamp_ns": 100})
    reader = SimpleNamespace(read=lambda *a, **k: {"valid": True, "distance_meters": 49})
    summary = {"distance": None, "observed_hud_max": 100, "error": None}
    module.annotate_parked_score(summary, SimpleNamespace(state="playing", frame=frame), reader)
    assert summary["distance"] is None
    module.annotate_parked_score(summary, SimpleNamespace(state="paused", frame=frame), reader)
    assert summary["distance"] == 49
    assert "release-to-pause" in summary["score_semantics"]


def test_unreadable_terminal_is_dismissible_but_never_scored_after_bounded_attempts():
    module = episode_module()
    collector = module.ResultScoreCollector(
        SimpleNamespace(read=lambda *a, **k: {"valid": False, "distance_meters": None})
    )
    for step in range(12):
        stamp = (100 + step) * 100_000_000
        observation = SimpleNamespace(
            state="result",
            frame=SimpleNamespace(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                metadata=lambda stamp=stamp: {"timestamp_ns": stamp},
            ),
        )
        assert collector(observation) is (step == 11)
    assert not collector.accepted and len(collector.exhausted) == 1
    collector.reset()
    assert collector.attempts == 0 and collector.pending is None


def test_interrupt_retains_partial_episode_and_releases():
    module = episode_module()
    adapter, controller, backend, bridge, first, sent, released = fixture(module)

    def interrupt():
        raise KeyboardInterrupt("operator interrupted")

    adapter.observe = interrupt
    summary, rows, images = module.collect_episode(
        adapter, controller, backend, bridge, lambda _: 1, first, seconds=1, deadline=10
    )
    assert sent == [1] and released and len(rows) == len(images) == 1
    assert "KeyboardInterrupt" in summary["error"]


def test_attempt_classification_uses_registered_vocabulary():
    from types import SimpleNamespace

    from run_screen_episodes import classify_attempt, terminal_cause

    paused = SimpleNamespace(state="paused")
    tune = SimpleNamespace(state="tune")
    truncated = {
        "reason": "episode_time_limit",
        "distance": 120,
        "paused_reading": {"valid": True},
        "terminal_readings": [],
    }
    assert classify_attempt(truncated, paused) == "success_truncated_scored"
    assert terminal_cause(truncated) == "truncated_horizon"
    unscored = {**truncated, "distance": None, "paused_reading": {"valid": False}}
    assert classify_attempt(unscored, paused) == "success_unscored"
    natural = {
        "reason": "observed_result",
        "distance": 289,
        "accepted_terminal_readings": [{"distance_meters": 289}],
        "terminal_readings": [{"ui_variant": "driver_down_native"}],
    }
    assert classify_attempt(natural, tune) == "success_natural_scored"
    assert terminal_cause(natural) == "driver_down"
    fuel = {**natural, "terminal_readings": [{"ui_variant": "out_of_fuel_discovery"}]}
    assert terminal_cause(fuel) == "out_of_fuel"
    unlabeled = {**natural, "terminal_readings": []}
    assert terminal_cause(unlabeled) == "natural_unlabeled"
    halted = {**natural, "park_error": "RuntimeError: Unrecognized advertisement; no click"}
    assert classify_attempt(halted, None) == "unknown_failure"
    focus = {**natural, "error": "RuntimeError: Target is not the foreground window"}
    assert classify_attempt(focus, None) == "recoverable_failure"
    assert terminal_cause(focus) == "driver_down"
    assert classify_attempt({"reason": "observed_result", "distance": None}, None) == (
        "unknown_failure"
    )
