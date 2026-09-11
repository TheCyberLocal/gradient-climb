"""Mocked screen sessions: no native input, capture, desktop or timing waits."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.control import PedalAction, PedalController
from gradientclimb.control.session import ScreenEpisodeSession, SessionLimits, VerifiedControl
from gradientclimb.perception.states import StateEvidence, UIState


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Backend:
    def __init__(self):
        self.events = []

    def set_pedals(self, gas, brake):
        self.events.append((gas, brake))


class Recorder:
    def __init__(self):
        self.trajectories, self.evaluations = [], []

    def trajectory(self, row):
        self.trajectories.append(row)

    def evaluation(self, row):
        self.evaluations.append(row)


def build(states, actions=None, **overrides):
    clock, backend, recorder = Clock(), Backend(), Recorder()
    states = iter(states)
    current = [None]
    clicks, histories = [], []
    actions = iter(actions or [PedalAction(True, True, 0.1)] * 20)

    def capture():
        clock.value += 0.001
        current[0] = next(states)
        end = int(clock() * 1e9)
        return CapturedFrame(
            np.zeros((10, 20, 3), dtype=np.uint8),
            end - 100,
            end,
            datetime.now(UTC).isoformat(),
            ClientRect(0, 0, 20, 10),
            "mock",
            0.001,
        )

    def encode(frames, history):
        histories.append((len(frames), [item.action.code for item in history]))
        assert frames[-1].rgb.flags.writeable is False
        return frames[-1].rgb

    def locate(name, frame, evidence):
        return VerifiedControl(name, evidence.state, frame.timestamp_ns, (1, 1, 5, 5), 1.0)

    kwargs = {
        "controller": PedalController(backend, lambda: True),
        "capture": capture,
        "classify": lambda pixels: current[0],
        "encode_observation": encode,
        "policy": lambda observation: next(actions),
        "verify_focus": lambda: True,
        "locate_control": locate,
        "click_control": clicks.append,
        "clock": clock,
        "sleep": clock.sleep,
        "recorder": recorder,
        "limits": SessionLimits(max_frames=20, max_episodes=1),
    }
    kwargs.update(overrides)
    return ScreenEpisodeSession(**kwargs), backend, recorder, clicks, histories


def evidence(state, **kwargs):
    return StateEvidence(state, 1.0, **kwargs)


def test_simultaneous_controls_history_durations_and_terminal_metrics():
    completed = []
    session, backend, recorder, clicks, histories = build(
        [evidence(UIState.PLAYING)] * 4 + [evidence(UIState.RESULT)],
        [PedalAction.from_code(code, 0.1) for code in [0, 1, 3, 2]],
        episode_metrics=lambda *args: {"distance": 12.5},
        on_session_end=completed.append,
    )
    result = session.run()
    assert (result.status, result.reason, result.completed_episodes) == (
        "completed",
        "episode_limit",
        1,
    )
    assert [row["gas"] for row in recorder.trajectories] == [False, True, True, False]
    assert [row["brake"] for row in recorder.trajectories] == [False, False, True, True]
    assert [row["action_duration_seconds"] for row in recorder.trajectories] == pytest.approx(
        [0.1] * 4
    )
    assert histories == [(1, []), (2, [0]), (3, [0, 1]), (4, [0, 1, 3])]
    assert recorder.evaluations[0]["results"]["metrics"] == {"distance": 12.5}
    assert completed[0]["actions"] == 4
    assert clicks == [] and backend.events[-1] == (False, False)


def test_verified_restart_ad_wait_close_and_new_episode_history():
    states = [
        evidence(UIState.PLAYING),
        evidence(UIState.RESULT, restart_visible=True),
        evidence(UIState.ADVERTISEMENT),
        evidence(UIState.ADVERTISEMENT),
        evidence(UIState.ADVERTISEMENT, legitimate_close_visible=True),
        evidence(UIState.STARTING),
        evidence(UIState.PLAYING),
        evidence(UIState.RESULT),
    ]
    session, backend, _, clicks, histories = build(states, limits=SessionLimits(max_episodes=2))
    result = session.run()
    assert result.completed_episodes == 2
    assert [control.name for control in clicks] == ["restart", "legitimate_ad_close"]
    assert histories == [(1, []), (1, [])]
    assert backend.events[-1] == (False, False)


@pytest.mark.parametrize(
    "state",
    [
        StateEvidence(UIState.PLAYING, 0.5),
        evidence(UIState.UNEXPECTED),
        evidence(UIState.MAIN_MENU),
        evidence(UIState.SELECTION),
    ],
)
def test_unknown_low_confidence_and_store_menus_stop_without_clicks(state):
    session, backend, _, clicks, _ = build([state])
    result = session.run()
    assert result.reason == "unrecognized_or_low_confidence_state"
    assert result.actions == 0 and clicks == []
    assert all(event == (False, False) for event in backend.events)


@pytest.mark.parametrize("bad_name", ["purchase", "store", "restart"])
def test_mismatched_control_never_reaches_click_callback(bad_name):
    session, backend, _, clicks, _ = build(
        [evidence(UIState.ADVERTISEMENT, legitimate_close_visible=True)],
        locate_control=lambda name, frame, ev: VerifiedControl(
            bad_name, ev.state, frame.timestamp_ns, (1, 1, 5, 5), 1
        ),
    )
    assert session.run().reason == "verified_control_mismatch"
    assert clicks == [] and backend.events[-1] == (False, False)


@pytest.mark.parametrize(
    "change",
    [
        {"frame_timestamp_ns": 1},
        {"bounds_xyxy": (-1, 0, 5, 5)},
        {"bounds_xyxy": (0, 0, 30, 10)},
        {"confidence": float("nan")},
        {"confidence": 0.5},
        {"state": UIState.PLAYING},
    ],
)
def test_stale_ambiguous_or_outside_control_is_refused(change):
    def locate(name, frame, ev):
        return replace(
            VerifiedControl(name, ev.state, frame.timestamp_ns, (1, 1, 5, 5), 1), **change
        )

    session, _, _, clicks, _ = build(
        [evidence(UIState.RESULT, restart_visible=True)], locate_control=locate
    )
    assert session.run().reason == "verified_control_mismatch"
    assert clicks == []


@pytest.mark.parametrize(
    "callback", ["capture", "classify", "verify_focus", "encode_observation", "policy"]
)
def test_callback_failure_closes_controller(callback):
    def broken(*args):
        raise OSError("mock callback failure")

    session, backend, _, clicks, _ = build([evidence(UIState.PLAYING)], **{callback: broken})
    result = session.run()
    assert result.status == "failed" and "mock callback failure" in result.error
    assert clicks == [] and backend.events[-1] == (False, False)


def test_slow_policy_cannot_act_on_stale_pixels():
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)])

    def slow(observation):
        session.clock.value += 1
        return PedalAction(True, True, 0.1)

    session.policy = slow
    assert session.run().reason == "stale_observation"
    assert all(event == (False, False) for event in backend.events)


@pytest.mark.parametrize("kind", ["old", "future", "repeat", "bad_pixels"])
def test_capture_timestamp_and_pixels_are_validated(kind):
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)] * 3)
    original = session.capture
    previous = [None]

    def capture():
        frame = original()
        if kind == "old":
            return replace(frame, started_ns=1, completed_ns=2)
        if kind == "future":
            return replace(frame, completed_ns=frame.completed_ns + 1_000_000)
        if kind == "bad_pixels":
            return replace(frame, rgb=np.zeros((0, 20, 3), dtype=np.uint8))
        if previous[0] is None:
            previous[0] = frame
        return previous[0]

    session.capture = capture
    result = session.run()
    assert result.reason in {
        "stale_observation",
        "invalid_capture",
        "nonmonotonic_or_repeated_frame",
    }
    assert backend.events[-1] == (False, False)


def test_invalid_action_and_latched_watchdog_stop_input():
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)], policy=lambda observation: 3)
    assert session.run().reason == "invalid_policy_action"
    assert backend.events[-1] == (False, False)
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)])
    session.controller.fault = "simulated watchdog failure"
    assert session.run().reason == "controller_watchdog_fault"
    assert all(event == (False, False) for event in backend.events)


def test_uncloseable_ad_and_repeated_restart_are_bounded():
    session, backend, _, clicks, _ = build(
        [evidence(UIState.ADVERTISEMENT)] * 30, limits=SessionLimits(max_wait_seconds=0.05)
    )
    assert session.run().reason == "ui_transition_wait_limit"
    assert clicks == [] and backend.events[-1] == (False, False)
    session, _, _, clicks, _ = build(
        [evidence(UIState.RESULT, restart_visible=True)] * 30,
        limits=SessionLimits(max_wait_seconds=0.05),
    )
    assert session.run().reason == "ui_transition_wait_limit" and len(clicks) == 1


def test_focus_loss_stop_and_deadline_after_focus_callback_release():
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)])
    session.sleep = lambda seconds: session.stop()
    result = session.run()
    assert result.status == "cancelled" and backend.events[-1] == (False, False)
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)], verify_focus=lambda: False)
    assert session.run().reason == "foreground_not_verified"
    assert backend.events[-1] == (False, False)
    session, backend, _, _, _ = build(
        [evidence(UIState.PLAYING)], limits=SessionLimits(max_seconds=0.05)
    )

    def delayed_focus():
        session.clock.value += 0.1
        return True

    session.verify_focus = delayed_focus
    assert session.run().reason == "session_time_limit"
    assert all(event == (False, False) for event in backend.events)


@pytest.mark.parametrize("callback", ["episode_metrics", "on_episode_end", "on_session_end"])
def test_metrics_callback_failure_occurs_with_controls_released(callback):
    session, backend, _, _, _ = build([evidence(UIState.PLAYING), evidence(UIState.RESULT)])

    def broken(*args):
        assert backend.events[-1] == (False, False)
        raise ValueError("metrics unavailable")

    setattr(session, callback, broken)
    result = session.run()
    assert result.status == "failed" and "metrics unavailable" in result.error
    assert backend.events[-1] == (False, False)


def test_release_failure_is_retained_and_single_use_is_enforced():
    session, backend, _, _, _ = build(
        [evidence(UIState.PLAYING)], limits=SessionLimits(max_frames=1)
    )
    close = session.controller.close

    def bad_close():
        close()
        raise OSError("release delivery failed")

    session.controller.close = bad_close
    result = session.run()
    assert (
        result.reason == "controller_release_failed"
        and "release delivery failed" in result.release_failure
    )
    assert backend.events[-1] == (False, False)
    with pytest.raises(RuntimeError, match="single-use"):
        session.run()


def test_terminal_without_result_restart_is_never_clicked():
    session, _, _, clicks, _ = build([evidence(UIState.GAME_OVER, restart_visible=True)])
    assert session.run().reason == "restart_requires_recognized_result" and clicks == []


def test_focus_loss_while_pedals_are_held_releases_before_end_callback():
    session, backend, _, _, _ = build([evidence(UIState.PLAYING)])
    session.verify_focus = lambda: backend.events[-1] != (True, True)
    result = session.run()
    assert result.reason == "foreground_not_verified"
    assert result.actions == 1 and backend.events[-1] == (False, False)


@pytest.mark.parametrize("role", ["restart", "legitimate_ad_close"])
def test_control_count_limits_stop_further_clicks(role):
    state = (
        evidence(UIState.RESULT, restart_visible=True)
        if role == "restart"
        else evidence(UIState.ADVERTISEMENT, legitimate_close_visible=True)
    )
    session, _, _, clicks, _ = build(
        [state, evidence(UIState.STARTING), state],
        limits=SessionLimits(max_restarts=1, max_advertisement_closes=1),
    )
    assert session.run().reason in {"restart_limit", "advertisement_close_limit"}
    assert len(clicks) == 1


def test_stop_at_frame_limit_and_time_limit():
    session, backend, _, _, _ = build(
        [evidence(UIState.PLAYING)], limits=SessionLimits(max_frames=1)
    )
    result = session.run()
    assert result.reason == "frame_limit" and result.frames == 1
    assert backend.events[-1] == (False, False)
    session, backend, _, _, _ = build(
        [evidence(UIState.PLAYING)], limits=SessionLimits(max_seconds=0.05)
    )
    result = session.run()
    assert result.reason == "session_time_limit" and result.elapsed_seconds < 0.08
    assert backend.events[-1] == (False, False)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_seconds": float("nan")},
        {"max_frames": 0},
        {"poll_seconds": -1},
        {"confidence_threshold": 2},
    ],
)
def test_invalid_limits_are_refused(limits):
    with pytest.raises(ValueError):
        SessionLimits(**limits)
