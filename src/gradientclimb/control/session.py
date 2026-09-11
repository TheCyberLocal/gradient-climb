"""Bounded screen-episode orchestration with supplied, validated perception.

This module has no native UI implementation and is not a live-game entry point.
Every screen encoder, classifier and named-control detector is supplied by the
caller. Their real-game accuracy must be validated before live deployment.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from gradientclimb.artifacts import canonical_json
from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.perception.states import StateEvidence, UIState, permitted_action

from .pedals import PedalAction, PedalController


@dataclass(frozen=True)
class VerifiedControl:
    """A named control detected in exactly one fresh classified frame."""

    name: Literal[
        "restart",
        "legitimate_ad_close",
        "pause_episode",
        "restart_paused",
        "resume_paused",
        "decline_revive_offer",
        "decline_bonus_offer",
        "continue_result",
        "start_episode",
    ]
    state: UIState
    frame_timestamp_ns: int
    bounds_xyxy: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class ActionInterval:
    action: PedalAction
    frame_timestamp_ns: int
    dispatched_ns: int
    ended_ns: int
    end_reason: str


@dataclass(frozen=True)
class SessionResult:
    status: str
    reason: str
    frames: int
    actions: int
    completed_episodes: int
    restarts: int
    advertisement_closes: int
    elapsed_seconds: float
    episode_results: tuple[dict, ...] = ()
    error: str | None = None
    release_failure: str | None = None
    deployment_status: str = "live_perception_and_orchestration_unvalidated"


@dataclass(frozen=True)
class SessionLimits:
    max_seconds: float = 60
    max_frames: int = 10000
    max_episodes: int = 10
    max_restarts: int = 10
    max_advertisement_closes: int = 10
    max_wait_seconds: float = 90
    max_observation_age_seconds: float = 0.5
    poll_seconds: float = 0.02
    history_length: int = 4
    confidence_threshold: float = 0.97

    def __post_init__(self):
        for name in (
            "max_seconds",
            "max_wait_seconds",
            "max_observation_age_seconds",
            "poll_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_frames",
            "max_episodes",
            "max_restarts",
            "max_advertisement_closes",
            "history_length",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite and in [0,1]")


class _Halt(Exception):
    def __init__(self, reason: str, status: str = "halted"):
        self.reason, self.status = reason, status


class ScreenEpisodeSession:
    """Run bounded episodes, release controls on every exit, and return evidence.

    The supplied controller is closed when the session ends. ``capture`` returns
    CapturedFrame using the same monotonic clock as ``clock``. ``classify`` takes
    RGB pixels. ``encode_observation`` receives tuples of current-episode frames
    and completed ActionIntervals; ``policy`` must return a PedalAction.

    Only verified ``restart`` and ``legitimate_ad_close`` control names can reach
    ``click_control``. That callback must independently guard native window
    identity/foreground and map frame coordinates to the correct client area.
    Callbacks are synchronous and must return promptly; the controller watchdog
    still expires held pedals if a callback stalls.
    """

    def __init__(
        self,
        controller: PedalController,
        capture: Callable[[], CapturedFrame],
        classify: Callable[[np.ndarray], StateEvidence],
        encode_observation: Callable[[tuple[CapturedFrame, ...], tuple[ActionInterval, ...]], Any],
        policy: Callable[[Any], PedalAction],
        verify_focus: Callable[[], bool],
        locate_control: Callable[[str, CapturedFrame, StateEvidence], VerifiedControl | None],
        click_control: Callable[[VerifiedControl], None],
        *,
        limits: SessionLimits | None = None,
        recorder: Any = None,
        episode_metrics: Callable[[CapturedFrame, StateEvidence, tuple[ActionInterval, ...]], dict]
        | None = None,
        on_episode_end: Callable[[dict], None] | None = None,
        on_session_end: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.controller = controller
        self.capture, self.classify = capture, classify
        self.encode_observation, self.policy = encode_observation, policy
        self.verify_focus = verify_focus
        self.locate_control, self.click_control = locate_control, click_control
        self.limits = limits or SessionLimits()
        self.recorder = recorder
        self.episode_metrics = episode_metrics
        self.on_episode_end, self.on_session_end = on_episode_end, on_session_end
        self.clock, self.sleep = clock, sleep
        self.frames: deque[CapturedFrame] = deque(maxlen=self.limits.history_length)
        self.action_history: deque[ActionInterval] = deque(maxlen=self.limits.history_length)
        self._stop = threading.Event()
        self._used = False

    def stop(self):
        """Request a stop; the next bounded polling point releases both pedals."""
        self._stop.set()

    def _guard(self):
        if self._stop.is_set():
            raise _Halt("stop_requested", "cancelled")
        if self.controller.fault or self.controller.release_failure:
            raise _Halt("controller_watchdog_fault", "failed")
        if self.verify_focus() is not True:
            raise _Halt("foreground_not_verified")
        # A focus verifier may take time or discover a watchdog fault itself.
        if self.controller.fault or self.controller.release_failure:
            raise _Halt("controller_watchdog_fault", "failed")
        if self.clock() - self._started >= self.limits.max_seconds:
            raise _Halt("session_time_limit", "completed")

    def _fresh(self, frame: CapturedFrame):
        now_ns = int(self.clock() * 1e9)
        if not isinstance(frame, CapturedFrame):
            raise _Halt("invalid_capture", "failed")
        if (
            not isinstance(frame.rgb, np.ndarray)
            or frame.rgb.dtype != np.uint8
            or frame.rgb.ndim != 3
            or frame.rgb.shape[2] != 3
            or min(frame.rgb.shape[:2]) < 1
            or type(frame.started_ns) is not int
            or type(frame.completed_ns) is not int
            or not 0 <= frame.started_ns <= frame.completed_ns <= now_ns
        ):
            raise _Halt("invalid_capture", "failed")
        if (now_ns - frame.timestamp_ns) / 1e9 > self.limits.max_observation_age_seconds:
            raise _Halt("stale_observation")

    def _wait(self, duration: float):
        end = self.clock() + duration
        while self.clock() < end:
            self._guard()
            self.sleep(min(self.limits.poll_seconds, end - self.clock()))
        self._guard()

    def _click(self, name: str, frame: CapturedFrame, evidence: StateEvidence):
        self.controller.release()
        control = self.locate_control(name, frame, evidence)
        if not isinstance(control, VerifiedControl):
            raise _Halt("verified_control_missing")
        height, width = frame.rgb.shape[:2]
        bounds = control.bounds_xyxy
        if (
            name not in {"restart", "legitimate_ad_close"}
            or control.name != name
            or control.state != evidence.state
            or control.frame_timestamp_ns != frame.timestamp_ns
            or not math.isfinite(control.confidence)
            or not self.limits.confidence_threshold <= control.confidence <= 1
            or len(bounds) != 4
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] < bounds[2] <= width
            or not 0 <= bounds[1] < bounds[3] <= height
        ):
            raise _Halt("verified_control_mismatch")
        self._guard()
        self._fresh(frame)
        self.click_control(control)

    def run(self) -> SessionResult:
        if self._used:
            raise RuntimeError("A screen session is single-use")
        self._used = True
        self._started = self.clock()
        counts = {
            "frames": 0,
            "actions": 0,
            "completed_episodes": 0,
            "restarts": 0,
            "advertisement_closes": 0,
        }
        episode_results: list[dict] = []
        status, reason, error, release_failure = "completed", "frame_limit", None, None
        previous_frame_ns = -1
        episode_active = False
        episode_start = self.clock()
        wait_started: float | None = None
        pending_click: str | None = None
        try:
            self.controller.release()
            while counts["frames"] < self.limits.max_frames:
                self._guard()
                frame = self.capture()
                self._guard()
                self._fresh(frame)
                if frame.completed_ns <= previous_frame_ns:
                    raise _Halt("nonmonotonic_or_repeated_frame")
                previous_frame_ns = frame.completed_ns
                counts["frames"] += 1
                evidence = self.classify(frame.rgb)
                if not isinstance(evidence, StateEvidence):
                    raise _Halt("invalid_state_evidence", "failed")
                evidence = StateEvidence(
                    evidence.state,
                    evidence.confidence,
                    evidence.legitimate_close_visible,
                    evidence.restart_visible,
                )
                permission = permitted_action(evidence, self.limits.confidence_threshold)
                self._guard()
                self._fresh(frame)
                if (
                    evidence.confidence >= self.limits.confidence_threshold
                    and evidence.state in {UIState.GAME_OVER, UIState.RESULT, UIState.RETURN}
                    and episode_active
                ):
                    self.controller.release()
                    metrics = (
                        self.episode_metrics(frame, evidence, tuple(self.action_history))
                        if self.episode_metrics
                        else {}
                    )
                    if not isinstance(metrics, dict):
                        raise TypeError("episode_metrics must return a dictionary")
                    canonical_json(metrics)
                    result = {
                        "episode_id": str(counts["completed_episodes"]),
                        "termination_state": evidence.state.value,
                        "elapsed_seconds": self.clock() - episode_start,
                        "metrics": metrics,
                    }
                    episode_results.append(result)
                    counts["completed_episodes"] += 1
                    episode_active = False
                    if self.recorder is not None:
                        self.recorder.evaluation(
                            {"results": result, "protocol": "screen-session-unvalidated"}
                        )
                    if self.on_episode_end is not None:
                        self.on_episode_end(result)
                    if counts["completed_episodes"] >= self.limits.max_episodes:
                        raise _Halt("episode_limit", "completed")
                if permission == "release_and_halt":
                    raise _Halt("unrecognized_or_low_confidence_state")
                if permission == "policy":
                    pending_click, wait_started = None, None
                    if not episode_active:
                        self.frames.clear()
                        self.action_history.clear()
                        episode_active, episode_start = True, self.clock()
                    pixels = frame.rgb.copy()
                    pixels.flags.writeable = False
                    frame = replace(frame, rgb=pixels)
                    self.frames.append(frame)
                    observation = self.encode_observation(
                        tuple(self.frames), tuple(self.action_history)
                    )
                    action = self.policy(observation)
                    if not isinstance(action, PedalAction):
                        raise _Halt("invalid_policy_action", "failed")
                    # Revalidate even an object constructed outside the normal initializer.
                    action = PedalAction(action.gas, action.brake, action.duration)
                    self._guard()
                    self._fresh(frame)
                    dispatched = self.clock()
                    self.controller.submit(action)
                    counts["actions"] += 1
                    action_end = "lease_elapsed"
                    try:
                        self._wait(action.duration)
                    except BaseException:
                        action_end = "session_interrupted"
                        raise
                    finally:
                        ended = self.clock()
                        interval = ActionInterval(
                            action,
                            frame.timestamp_ns,
                            int(dispatched * 1e9),
                            int(ended * 1e9),
                            action_end,
                        )
                        self.action_history.append(interval)
                        if self.recorder is not None:
                            self.recorder.trajectory(
                                {
                                    "episode_id": str(counts["completed_episodes"]),
                                    "step": counts["actions"] - 1,
                                    "timestamp": frame.utc_started_at,
                                    "elapsed_seconds": max(0, dispatched - self._started),
                                    "gas": action.gas,
                                    "brake": action.brake,
                                    "action_duration_seconds": min(
                                        action.duration, max(0, ended - dispatched)
                                    ),
                                    "observation": {
                                        "ui_state": evidence.state.value,
                                        "state_confidence": evidence.confidence,
                                        "capture_timestamp_ns": frame.timestamp_ns,
                                        "action_dispatched_ns": interval.dispatched_ns,
                                        "action_interval_ended_ns": interval.ended_ns,
                                        "requested_lease_seconds": action.duration,
                                        "duration_semantics": "bounded command lease; actual OS key duration is not measured",
                                        "pixels_sha256": hashlib.sha256(
                                            frame.rgb.tobytes()
                                        ).hexdigest(),
                                        "frame_shape": list(frame.rgb.shape),
                                        "end_reason": action_end,
                                    },
                                }
                            )
                    continue
                self.controller.release()
                if wait_started is None:
                    wait_started = self.clock()
                if self.clock() - wait_started >= self.limits.max_wait_seconds:
                    raise _Halt("ui_transition_wait_limit")
                if permission == "verified_restart":
                    # A death/return label alone is insufficient for navigation.
                    if evidence.state != UIState.RESULT:
                        raise _Halt("restart_requires_recognized_result")
                    if pending_click != "restart":
                        if counts["restarts"] >= self.limits.max_restarts:
                            raise _Halt("restart_limit")
                        self._click("restart", frame, evidence)
                        counts["restarts"] += 1
                        pending_click = "restart"
                elif permission == "close_verified_ad":
                    if pending_click != "legitimate_ad_close":
                        if counts["advertisement_closes"] >= self.limits.max_advertisement_closes:
                            raise _Halt("advertisement_close_limit")
                        self._click("legitimate_ad_close", frame, evidence)
                        counts["advertisement_closes"] += 1
                        pending_click = "legitimate_ad_close"
                elif evidence.state == UIState.STARTING:
                    pending_click = None
                self._wait(self.limits.poll_seconds)
        except _Halt as halt:
            status, reason = halt.status, halt.reason
        except KeyboardInterrupt:
            status, reason = "cancelled", "keyboard_interrupt"
        except Exception as exc:  # noqa: BLE001 - preserve failure evidence after releasing controls
            status, reason, error = (
                "failed",
                "callback_or_control_failure",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                self.controller.close()
            except Exception as exc:  # noqa: BLE001 - report failed release; never resume input
                status, reason = "failed", "controller_release_failed"
                release_failure = f"{type(exc).__name__}: {exc}"
        result = SessionResult(
            status,
            reason,
            **counts,
            elapsed_seconds=max(0, self.clock() - self._started),
            episode_results=tuple(episode_results),
            error=error,
            release_failure=release_failure,
        )
        if self.on_session_end is not None:
            try:
                self.on_session_end(asdict(result))
            except Exception as exc:  # noqa: BLE001 - controller is already closed; retain reporting failure
                result = replace(
                    result,
                    status="failed",
                    reason="session_metrics_callback_failure",
                    error=f"{type(exc).__name__}: {exc}",
                )
        return result
