"""Bounded end-to-end actual-game baseline episodes, with local pixel records.

This is a supervised integration/evaluation entry point, not a training claim.
All menu actions come from the separately verified native adapter's allowlist.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.windows import discover_windows
from gradientclimb.control.game_adapter import (
    FOREGROUND_LOSS_MARKER,
    STUCK_RESET_MARKERS,
    NativeGameAdapter,
)
from gradientclimb.control.host import HostBudgetGuard, HostLimits
from gradientclimb.control.native_protocol import (
    longest_scored_streak,
    validate_limits,
    validate_protocol,
)
from gradientclimb.control.pedals import PedalAction, PedalController
from gradientclimb.control.windows import WindowsPedalBackend
from gradientclimb.evaluation.native_summary import VERSION as NATIVE_SUMMARY_VERSION
from gradientclimb.evaluation.native_summary import describe_native_attempts
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile
from gradientclimb.perception.scoring import PausedDistanceReader, ResultDistanceReader
from gradientclimb.perception.screen_features import FEATURE_NAMES, ScreenFeatureBridge


def os_state_at(trace, timestamp_ns):
    code, last_time = 0, None
    for event in trace:
        if event["completed_ns"] > timestamp_ns:
            break
        if event.get("delivered") != len(event["events"]):
            return None, None
        for key, down in event["events"]:
            bit = 1 if key == 0x27 else 2
            code = code | bit if down else code & ~bit
        last_time = event["completed_ns"]
    return code, (timestamp_ns - last_time) / 1e9 if last_time is not None else None


class ResultScoreCollector:
    """Seek spaced agreement; retain bounded failures as unknown before dismissal.

    Only the adapter's independently verified RESULT authorizes dismissal. A
    failed numeric reading never authorizes an optimizer update or a zero score.
    """

    def __init__(self, reader):
        self.reader = reader
        self.readings, self.frames, self.accepted, self.exhausted = [], [], [], []
        self.reset()

    def reset(self):
        self.pending = None
        self.first_attempt_ns, self.attempts = None, 0

    def __call__(self, observation):
        self.frames.append((observation.frame.rgb.copy(), observation.frame.metadata()))
        if self.reader is None:
            return True
        reading = {
            **observation.frame.metadata(),
            **self.reader.read(
                observation.frame.rgb, result_state_confirmed=observation.state == "result"
            ),
            # The recognized result variant is the only terminal-cause evidence today;
            # it is retained as evidence, not asserted as a validated cause label.
            "ui_state": observation.state,
            "ui_variant": getattr(observation, "variant", None),
            "ui_confidence": getattr(observation, "confidence", None),
        }
        self.readings.append(reading)
        if self.first_attempt_ns is None:
            self.first_attempt_ns = reading["timestamp_ns"]
        self.attempts += 1
        exhausted = self.attempts >= 12 or (
            reading["timestamp_ns"] - self.first_attempt_ns >= 3_000_000_000
        )

        def unresolved():
            if exhausted:
                self.exhausted.append(
                    {
                        "timestamp_ns": reading["timestamp_ns"],
                        "attempts": self.attempts,
                        "reason": "Bounded numeric read attempts exhausted; distance remains unknown",
                    }
                )
                self.pending = None
                return True
            return False

        if not reading["valid"]:
            self.pending = None
            return unresolved()
        if self.pending and self.pending["distance_meters"] == reading["distance_meters"]:
            separation = (reading["timestamp_ns"] - self.pending["timestamp_ns"]) / 1e9
            if separation >= 0.15:
                self.accepted.append(
                    {
                        **reading,
                        "agreement_interval_seconds": separation,
                        "first_agreeing_timestamp_ns": self.pending["timestamp_ns"],
                    }
                )
                self.pending = None
                return True
            return unresolved()
        self.pending = reading
        return unresolved()


def adapter_profile_id(path):
    return json.loads(Path(path).read_text(encoding="utf-8")).get("profile_id", "unknown")


STABILIZING_PITCH_THRESHOLD_RADIANS = 0.35


def stabilizing_action(screen, *, pitch_threshold=STABILIZING_PITCH_THRESHOLD_RADIANS):
    """Registered real-baselines-2.0 stabilizing control: gas when level and grounded.

    Coast (neutral) when the body pitch magnitude exceeds the threshold or when
    both wheels are measured clear of the terrain; never brake. The body axis is
    modulo pi, so this is a level-versus-tilted rule without a nose-direction
    sign. Unknown wheel state counts as grounded; unknown pitch falls back to gas.
    Parameters were fixed a priori and are never tuned on evaluation episodes.
    """
    from gradientclimb.experiments.objectives import body_pitch_radians, wheels_clear

    names, values, valid = FEATURE_NAMES, screen.values, screen.valid
    if wheels_clear(names, values, valid):
        return 0
    pitch = body_pitch_radians(names, values, valid)
    if pitch is not None and abs(pitch) > pitch_threshold:
        return 0
    return 1


def annotate_parked_score(summary, parked, paused_reader):
    """Attach the actual paused-boundary score, never an earlier HUD maximum."""
    if parked is None or parked.state != "paused" or summary.get("error"):
        return
    reading = {
        **parked.frame.metadata(),
        **paused_reader.read(parked.frame.rgb, paused_state_confirmed=True),
    }
    summary["paused_reading"] = reading
    if reading["valid"]:
        summary["distance"] = reading["distance_meters"]
        summary["score_semantics"] = (
            "displayed progress at verified paused boundary; includes release-to-pause delay"
        )


def collect_episode(
    adapter,
    controller,
    backend,
    bridge,
    choose_action,
    first,
    *,
    seconds,
    deadline,
    on_tick=None,
    stall_seconds=None,
    max_frames=2000,
):
    """Hold leases only after fresh classified capture and explicit policy output."""
    bridge.reset()
    start = time.perf_counter()
    episode_end = start + seconds
    end = min(episode_end, deadline)
    clock_reason = "session_deadline" if deadline < episode_end else "episode_time_limit"
    rows, images = [], []
    observed_values = []
    current = first
    reason = clock_reason
    failure = None
    unknown_since = None
    last_progress_time = start
    last_progress = None
    progress_continuous = True
    try:
        for step in range(max_frames):
            if on_tick:
                on_tick()
            if current.state != "playing":
                controller.release()
                reason = f"observed_{current.state}"
                images.append(
                    (
                        current.frame.rgb.copy(),
                        {**current.frame.metadata(), "state": current.state, "step": step},
                    )
                )
                if current.state == "unknown":
                    unknown_since = unknown_since or time.perf_counter()
                    if time.perf_counter() - unknown_since < 2.0:
                        current = adapter.observe()
                        continue
                break
            if time.perf_counter() >= end:
                break
            unknown_since = None
            reason = clock_reason
            actual_code, age = os_state_at(backend.trace, current.frame.started_ns)
            screen = bridge.observe(
                current.frame.rgb,
                current.frame.timestamp_ns,
                previous_action_code=actual_code,
                action_age_seconds=age,
                episode_elapsed_seconds=max(0.0, current.frame.timestamp_ns / 1e9 - start),
            )
            observation_ready_ns = int(time.perf_counter() * 1e9)
            code = choose_action(screen)
            decision_ready_ns = int(time.perf_counter() * 1e9)
            if type(code) is not int or code not in range(4):
                raise ValueError("Policy returned an invalid independent pedal state")
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            dispatched = adapter.is_playing()
            lease = min(0.4, remaining)
            input_started_ns = int(time.perf_counter() * 1e9)
            if dispatched:
                controller.submit(PedalAction.from_code(code, lease))
            else:
                controller.release()
            row = {
                "step": step,
                **current.frame.metadata(),
                "state": current.state,
                "elapsed_seconds": time.perf_counter() - start,
                "requested_code": code,
                "dispatched": dispatched,
                "maximum_lease_seconds": lease if dispatched else 0.0,
                "preceding_os_code": actual_code,
                "screen": screen.as_dict(),
                "observation_ready_ns": observation_ready_ns,
                "decision_ready_ns": decision_ready_ns,
                "input_started_ns": input_started_ns,
                "input_completed_ns": int(time.perf_counter() * 1e9),
            }
            rows.append(row)
            if step % 5 == 0:
                images.append(
                    (
                        current.frame.rgb.copy(),
                        {**current.frame.metadata(), "step": step, "state": current.state},
                    )
                )
            if screen.hud.get("valid"):
                progress = screen.hud["hud_displayed_progress_meters"]
                observed_values.append(progress)
                if last_progress is None or progress > last_progress or not progress_continuous:
                    last_progress_time = time.perf_counter()
                last_progress = max(progress, last_progress or 0)
                progress_continuous = True
                if (
                    stall_seconds is not None
                    and time.perf_counter() - last_progress_time >= stall_seconds
                ):
                    reason = "verified_progress_stall"
                    break
            else:
                progress_continuous = False
            if controller.fault:
                raise RuntimeError(controller.fault)
            if time.perf_counter() >= end:
                break
            current = adapter.observe()
        else:
            reason = "frame_limit"
    except BaseException:  # noqa: BLE001 - retain partial rows on interrupts, then stop safely
        reason, failure = "episode_failure", traceback.format_exc()
    finally:
        try:
            controller.release()
        except BaseException:  # noqa: BLE001 - retain release failure alongside partial frames
            reason, failure = "release_failure", (failure or "") + traceback.format_exc()
    return (
        {
            "reason": reason,
            "requested_horizon_seconds": seconds,
            "endpoint_contract": "native-endpoint-3.0",
            "observed_seconds": time.perf_counter() - start,
            "frames": len(rows),
            "actions_dispatched": sum(row["dispatched"] for row in rows),
            "accepted_hud_frames": len(observed_values),
            "observed_hud_max": max(observed_values) if observed_values else None,
            "distance": None,
            "error": failure,
            "score_semantics": "HUD diagnostic only; qualified terminal-distance reader required",
        },
        rows,
        images,
    )


def save_episode(run, index, summary, rows, images):
    directory = run.directory / f"episode-{index:03d}"
    directory.mkdir()
    path = directory / "observations.jsonl"
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    run.register_artifact(path, "screen_episode_observations")
    for n, (rgb, metadata) in enumerate(images):
        path = directory / f"sample-{n:04d}.png"
        Image.fromarray(rgb).save(path)
        run.register_artifact(path, "real_game_frame", metadata)
    for row in rows:
        code = row["requested_code"]
        run.trajectory(
            {
                "episode_id": f"episode-{index:03d}",
                "step": row["step"],
                "gas": bool(code & 1),
                "brake": bool(code & 2),
                "action_duration_seconds": row["maximum_lease_seconds"],
                "observation": {
                    "capture_timestamp_ns": row["timestamp_ns"],
                    "dispatched": row["dispatched"],
                    "preceding_os_code": row["preceding_os_code"],
                    "duration_semantics": "maximum requested lease; use raw OS trace for intervals",
                },
                "elapsed_seconds": row.get("canonical_elapsed_seconds", row["elapsed_seconds"]),
            }
        )
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    run.register_artifact(path, "screen_episode_summary")


UNKNOWN_HALT_MARKERS = (
    "Unrecognized advertisement",
    "Advertisement without legitimate control",
    "Unknown screen persisted without recognized context",
    "Unrecognized/unauthorized reset state",
    "did not recover",
    "Unexpected or timed-out reset transition",
)


POST_CLICK_WINDOW_SECONDS = 5.0


def should_restart(error, adapter=None):
    """Whether a failed reset left the game stuck on a screen the adapter may not touch.

    Only ``STUCK_RESET_MARKERS`` qualify: each is raised after the full bounded
    advertisement wait with no click in between. A foreground loss, a latched
    guard fault, a deadline or click limit, or a post-click transition timeout
    never qualifies: those halt the session with their evidence intact.
    """
    if not error or FOREGROUND_LOSS_MARKER in error or "latched" in error:
        return False
    if adapter is not None and getattr(adapter, "_latched", False):
        return False
    return any(marker in error for marker in STUCK_RESET_MARKERS)


def unintended_action_events(adapter, *, first_row=0, window_seconds=POST_CLICK_WINDOW_SECONDS):
    """Effect-based unintended actions derived from the adapter traces alone.

    The allowlist bounds what is clicked; this bounds what a click did. For every
    accepted click (from trace index ``first_row`` on) a latched foreground-loss
    guard event recorded at or after the click's start and within
    ``window_seconds`` of its delivery is an unintended action, whichever thread
    or method observed the loss and whatever exception the runner finally saw.
    """
    events = []
    losses = [
        event
        for event in getattr(adapter, "guard_trace", [])
        if event.get("latched") and FOREGROUND_LOSS_MARKER in event["reason"]
    ]
    for row in adapter.trace[first_row:]:
        if not row.get("accepted"):
            continue
        delivered = row.get("accepted_ns", row.get("completed_ns"))
        started = row.get("started_ns", delivered)
        for loss in losses:
            if loss["timestamp_ns"] < started:
                continue
            elapsed = max(0.0, (loss["timestamp_ns"] - delivered) / 1e9)
            if elapsed <= window_seconds:
                events.append(
                    {
                        "seconds_after_click": elapsed,
                        "click": {
                            key: row.get(key)
                            for key in (
                                "name",
                                "state",
                                "variant",
                                "bounds_xyxy",
                                "source_pixels_sha256",
                            )
                        },
                        "guard_reason": loss["reason"],
                    }
                )
                break
    return events


def unintended_action_evidence(adapter, *, first_row=0, window_seconds=POST_CLICK_WINDOW_SECONDS):
    """The most recent effect-based unintended action for an attempt, or None."""
    events = unintended_action_events(adapter, first_row=first_row, window_seconds=window_seconds)
    return events[-1] if events else None


def attempt_restart(adapter, launch, *, phase, error_text, rows, budget, on_observation=None):
    """Restart the game after a stuck reset; True when it settled on a known state.

    ``budget`` carries ``deadline``, ``restart_seconds``, ``max_restarts`` and the
    mutable ``used`` count. Nothing here decides whether the error qualifies beyond
    ``should_restart``; the adapter refuses on its own after any latched fault.
    """
    if launch is None or not should_restart(error_text, adapter):
        return False
    now = time.perf_counter()
    if budget["used"] >= budget["max_restarts"] or budget["deadline"] - now < 30:
        return False
    budget["used"] += 1
    reason = error_text.strip().splitlines()[-1][:160]
    row = {"phase": phase, "index": budget["used"], "reason": reason, "ok": False, "seconds": None}
    rows.append(row)
    started = time.perf_counter()
    try:
        settled = adapter.restart_app(
            launch,
            max_seconds=min(budget["restart_seconds"], budget["deadline"] - started),
            reason=f"{phase}: {reason}",
            on_observation=on_observation,
        )
        row["ok"], row["settled_state"] = True, settled.state
    except Exception:  # noqa: BLE001 - a failed restart halts the session with evidence
        row["error"] = traceback.format_exc()
    finally:
        row["seconds"] = time.perf_counter() - started
    return row["ok"]


def classify_attempt(summary, parked, *, halted_error=None, unintended=None):
    """Terminal classification registered by native-reliability-2.0; never post hoc.

    ``unintended_action`` (registered by 2.2) takes precedence: an allowlisted click
    whose effect was a foreground loss to another application.
    """
    if unintended:
        return "unintended_action"
    error = summary.get("error") or summary.get("park_error") or halted_error
    if error:
        if any(marker in error for marker in UNKNOWN_HALT_MARKERS):
            return "unknown_failure"
        return "recoverable_failure"
    if parked is None:
        return "unknown_failure"
    reason = summary.get("reason")
    if reason in {"session_deadline", "frame_limit", "verified_progress_stall"}:
        return "administrative_interruption"
    if reason == "episode_time_limit" and parked.state == "paused":
        reading = summary.get("paused_reading") or {}
        if reading.get("valid") and summary.get("distance") is not None:
            return "success_truncated_scored"
        return "success_unscored"
    if reason in {"observed_result", "observed_revive_offer"} and parked.state == "tune":
        if summary.get("accepted_terminal_readings") and summary.get("distance") is not None:
            return "success_natural_scored"
        return "success_unscored"
    if parked.state in {"paused", "tune"}:
        return "success_unscored"
    return "unknown_failure"


def terminal_cause(summary):
    """Evidence-backed terminal cause; unknown unless the result variant was recognized."""
    variants = {r.get("ui_variant") for r in summary.get("terminal_readings", []) if r}
    if any(v and "driver_down" in v for v in variants):
        return "driver_down"
    if any(v and "out_of_fuel" in v for v in variants):
        return "out_of_fuel"
    if summary.get("reason") in {"observed_result", "observed_revive_offer"}:
        return "natural_unlabeled"
    if summary.get("reason") == "episode_time_limit":
        return "truncated_horizon"
    if summary.get("reason") in {"session_deadline", "frame_limit", "verified_progress_stall"}:
        return summary["reason"]
    if summary.get("error"):
        return "aborted"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Canonical artifact root, independent of the pinned source checkout",
    )
    parser.add_argument(
        "--ui-profile", type=Path, default=Path("configs/perception/hcr-reset-ui.json")
    )
    parser.add_argument(
        "--measurement-profile",
        type=Path,
        default=Path("configs/perception/hcr-discovery-wrapper.json"),
    )
    parser.add_argument("--hud", type=Path, required=True)
    parser.add_argument("--result-reader", type=Path)
    parser.add_argument("--policy", type=Path, help="Frozen learned checkpoint; no updates")
    parser.add_argument("--policy-kind", choices=["screen-linear", "screen-body-student"])
    parser.add_argument("--parent-run", help="Sealed canonical run that registered the checkpoint")
    parser.add_argument(
        "--baseline",
        choices=[
            "always_gas",
            "random",
            "neutral",
            "stabilizing",
            "alternating_gas_random",
            "interleaved_baselines",
        ],
        default="always_gas",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=8)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument(
        "--long-run",
        action="store_true",
        help="Explicit long-run mode: up to 900 gameplay seconds, 60 s verified-progress stall handling",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-backend", choices=["dxcam", "mss", "pillow"], default="dxcam")
    parser.add_argument(
        "--inspect", action="store_true", help="Classify one current frame; never navigate or drive"
    )
    parser.add_argument(
        "--experiment-id",
        help="Canonical experiment identifier; defaults to the pilot/evaluation identifiers",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Registered protocol definition whose hash and versions are frozen into the run",
    )
    parser.add_argument("--note", default="", help="Operator note recorded in the configuration")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Label the run exploratory (for example advertisement discovery); never study data",
    )
    parser.add_argument(
        "--reset-seconds",
        type=float,
        default=90.0,
        help="Per-reset overall deadline (<= 120 s) covering menus and advertisements",
    )
    parser.add_argument(
        "--restart-shortcut",
        type=Path,
        help=(
            "Shortcut that relaunches the game; enables application-restart recovery after "
            "a stuck reset (unknown screen, or an advertisement without a verified control "
            "past its bounded wait). Never used after a foreground loss."
        ),
    )
    parser.add_argument("--restart-seconds", type=float, default=150.0)
    parser.add_argument("--max-restarts", type=int, default=6)
    parser.add_argument(
        "--probe-restart",
        action="store_true",
        help=(
            "Exploratory only: perform exactly one application restart from the current "
            "screen, seal its trace and exit; no episode, no click"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.reset_seconds <= 120:
        raise ValueError("Reset deadline must be within 1..120 seconds")
    if not 30 <= args.restart_seconds <= 300 or not 0 <= args.max_restarts <= 12:
        raise ValueError("Restart recovery limits are out of bounds")
    launch = None
    if args.restart_shortcut is not None:
        if not args.restart_shortcut.is_file():
            raise FileNotFoundError(f"Restart shortcut not found: {args.restart_shortcut}")
        launch = lambda: os.startfile(str(args.restart_shortcut))
    if args.probe_restart and (not args.exploratory or launch is None or args.inspect):
        raise ValueError("--probe-restart requires --exploratory and --restart-shortcut")
    if any((args.policy, args.policy_kind, args.parent_run)) and not all(
        (args.policy, args.policy_kind, args.parent_run)
    ):
        raise ValueError("Frozen evaluation requires policy path, kind and parent run together")
    validate_limits(args.episodes, args.episode_seconds, args.max_seconds, long_run=args.long_run)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else None
    if protocol:
        validate_protocol(protocol, args)
    host = HostBudgetGuard(args.root)
    host.check(force=True)
    targets = discover_windows()
    if len(targets) != 1:
        raise RuntimeError("Expected exactly one visible game window")
    target = targets[0]
    host.target = target
    host.check(force=True)
    profile = MeasurementProfile(**json.loads(args.measurement_profile.read_text()))
    bridge = ScreenFeatureBridge(HCRPixelMeasurer(profile), HUDDigitReader.from_manifest(args.hud))
    result_reader = (
        ResultDistanceReader.from_manifest(args.result_reader) if args.result_reader else None
    )
    paused_reader = PausedDistanceReader.from_gameplay_manifest(args.hud)
    rng = np.random.default_rng(args.seed)
    scripted = {
        "random": lambda _: int(rng.integers(4)),
        "always_gas": lambda _: 1,
        "neutral": lambda _: 0,
        "stabilizing": stabilizing_action,
    }
    schedules = {
        "alternating_gas_random": ("always_gas", "random"),
        "interleaved_baselines": ("random", "always_gas", "stabilizing"),
    }

    def attempt_policy_name(index):
        cycle = schedules.get(args.baseline)
        return cycle[index % len(cycle)] if cycle else args.baseline

    choose = scripted[attempt_policy_name(0)]
    checkpoint_hash = None
    algorithm = args.baseline
    if args.policy:
        checkpoint_hash = sha256_file(args.policy)
        parent = load_run(args.root, args.parent_run)
        if not verify_run(args.root, args.parent_run)["valid"] or not any(
            item["sha256"] == checkpoint_hash and item["kind"] == "checkpoint"
            for item in parent["artifact_manifest"]
        ):
            raise ValueError("Policy must match a registered checkpoint in the sealed parent run")
        if args.policy_kind == "screen-linear":
            from gradientclimb.algorithms.screen_search import ScreenLinearPolicy

            data = json.loads(args.policy.read_text())
            policy = ScreenLinearPolicy.from_dict(data["policy"])
        else:
            import torch

            from gradientclimb.algorithms.screen_distillation import ScreenBodyStudent

            torch.set_num_threads(1)
            policy = ScreenBodyStudent.load(args.policy)
        if policy.schema_id != bridge.schema_id:
            raise ValueError("Frozen policy and live screen feature schemas differ")
        choose = lambda screen: (
            policy.action(screen, FEATURE_NAMES)
            if screen.supports(["body_sin_2angle", "body_cos_2angle"])
            else 0
        )
        algorithm = args.policy_kind
    if protocol:
        recovery = (protocol.get("recovery") or {}) if isinstance(protocol, dict) else {}
        if recovery.get("application_restart") and not args.exploratory:
            # A registered recovery is part of the frozen protocol: it must be
            # enabled and its bounds must not exceed the registered ones.
            if launch is None:
                raise ValueError(
                    "Protocol declares application restart; --restart-shortcut required"
                )
            if args.restart_seconds > float(recovery.get("max_seconds", args.restart_seconds)):
                raise ValueError("Restart bound exceeds the registered recovery limit")
            if args.max_restarts > int(recovery.get("max_restarts_per_session", args.max_restarts)):
                raise ValueError("Restart count exceeds the registered recovery limit")
    config = {
        "artifact_root": args.root.resolve().as_posix(),
        "baseline": args.baseline if not args.policy else None,
        "policy_kind": args.policy_kind,
        "checkpoint_hash": checkpoint_hash,
        "parent_run": args.parent_run,
        "requested_episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
        "long_run": args.long_run,
        "stall_seconds": 60 if args.long_run else None,
        "endpoint_contract": "native-endpoint-3.0",
        "statistics_contract": NATIVE_SUMMARY_VERSION,
        "host_limits": vars(HostLimits()),
        "max_session_seconds": args.max_seconds,
        "ui_profile_sha256": sha256_file(args.ui_profile),
        "measurement_profile_sha256": sha256_file(args.measurement_profile),
        "hud_manifest_sha256": sha256_file(args.hud),
        "result_reader_sha256": sha256_file(args.result_reader) if args.result_reader else None,
        "screen_schema": bridge.schema,
        "screen_schema_id": bridge.schema_id,
        "target": target.as_dict(),
        "input_mode": "scancode",
        "capture_backend": args.capture_backend,
        "max_lease_seconds": 0.4,
        "scope": "frozen learned-policy evaluation"
        if args.policy
        else "scripted baseline evaluation",
        "inspect_only": args.inspect,
        "probe_restart": args.probe_restart,
        "exploratory": args.exploratory,
        "operator_note": args.note,
        "reset_deadline_seconds": args.reset_seconds,
        "initial_playing_policy": "pause and restart before the first attempt",
        "protocol_path": str(args.protocol) if args.protocol else None,
        "protocol_sha256": sha256_file(args.protocol) if args.protocol else None,
        "protocol_version": protocol.get("protocol_version") if protocol else None,
        "attempt_policy_schedule": [attempt_policy_name(i) for i in range(args.episodes)]
        if not args.policy
        else None,
        "restart_recovery": {
            "enabled": launch is not None,
            "shortcut": str(args.restart_shortcut) if args.restart_shortcut else None,
            "shortcut_sha256": sha256_file(args.restart_shortcut)
            if args.restart_shortcut
            else None,
            "max_seconds": args.restart_seconds,
            "max_restarts": args.max_restarts,
            "policy": (
                "WM_CLOSE to the pinned game window plus the shortcut, only after a stuck "
                "reset (one of STUCK_RESET_MARKERS, each raised after the full bounded "
                "advertisement wait); never after a foreground loss or any latched guard "
                "fault; never a click inside the game"
            ),
            "stuck_markers": list(STUCK_RESET_MARKERS),
        },
        "unintended_action_definition": (
            f"foreground loss to another window within {POST_CLICK_WINDOW_SECONDS:g} s of an "
            "accepted click (effect-based), in addition to the click allowlist"
        ),
    }
    if protocol and protocol.get("cycle2_versions") and not args.exploratory:
        from gradientclimb.experiments.objectives import ExperimentVersions, stamp_versions

        declared = dict(protocol["cycle2_versions"])
        declared["observation_schema"] = f"{bridge.schema['version']}@{bridge.schema_id[:16]}"
        declared["ui_profile"] = (
            f"{adapter_profile_id(args.ui_profile)}@{config['ui_profile_sha256'][:16]}"
        )
        config = stamp_versions(config, ExperimentVersions.model_validate(declared))
    adapter = None
    backend = WindowsPedalBackend(target, gas_vk=0x27, brake_vk=0x25, input_mode="scancode")
    controller = PedalController(backend, lambda: adapter is not None and adapter.is_playing())
    adapter = NativeGameAdapter(
        target,
        args.ui_profile,
        reference_root=args.root,
        release_pedals=controller.release,
        capture_backend=args.capture_backend,
        runtime_check=host.check,
    )
    summaries, reset_frames, restart_frames = [], [], []
    terminal = ResultScoreCollector(result_reader)
    error = None
    experiment_id = args.experiment_id or (
        "real-screen-policy-evaluation" if args.policy else "real-screen-episode-pilot"
    )
    attempt_outcomes = []
    with RunRecorder(
        args.root,
        experiment_id,
        config,
        seed=args.seed,
        algorithm=algorithm,
        environment="actual_hill_climb_racing",
        parent_run=args.parent_run,
        parent_checkpoint=checkpoint_hash,
    ) as run:
        if args.policy:
            run.register_artifact(args.policy, "evaluated_checkpoint")
        for path, kind in (
            (args.ui_profile, "ui_profile"),
            (args.measurement_profile, "measurement_profile"),
            (args.hud, "hud_manifest"),
        ):
            run.register_artifact(path, kind)
        for variant in adapter.recognizer.profile.variants:
            run.register_artifact(args.root / variant.file, "ui_reference")
        for glyph in bridge.hud_reader.glyphs:
            run.register_artifact(args.hud.parent / glyph["file"], "hud_glyph")
        if args.result_reader:
            for path in args.result_reader.parent.rglob("*"):
                if (
                    path.is_file()
                    and (path.suffix in {".png", ".json"})
                    and path.name
                    not in {"run.json", "run-start.json", "seal.json", "source-state.json"}
                    and "files" not in path.relative_to(args.result_reader.parent).parts
                ):
                    run.register_artifact(path, "result_reader_dependency")
        start = time.perf_counter()
        deadline = start + args.max_seconds
        attempt_states = {}
        restart_states = {}
        restart_budget = {
            "deadline": deadline,
            "restart_seconds": args.restart_seconds,
            "max_restarts": args.max_restarts,
            "used": 0,
        }

        def restart_observation(observation):
            # Boot frames of a relaunch are evidence of the restart, never
            # unknown-state incidence of the reset flow.
            restart_states[observation.state] = restart_states.get(observation.state, 0) + 1
            if len(restart_frames) < 40 and (
                not restart_frames or restart_frames[-1][1]["state"] != observation.state
            ):
                restart_frames.append(
                    (
                        observation.frame.rgb.copy(),
                        {**observation.frame.metadata(), "state": observation.state},
                    )
                )

        def try_restart(phase, error_text, rows):
            return attempt_restart(
                adapter,
                launch,
                phase=phase,
                error_text=error_text,
                rows=rows,
                budget=restart_budget,
                on_observation=restart_observation,
            )

        def reset_observation(observation):
            attempt_states[observation.state] = attempt_states.get(observation.state, 0) + 1
            if len(reset_frames) < 120 and (
                not reset_frames
                or reset_frames[-1][1]["state"] != observation.state
                or (
                    observation.state == "unknown"
                    and observation.frame.timestamp_ns - reset_frames[-1][1]["timestamp_ns"]
                    >= 1_000_000_000
                )
            ):
                reset_frames.append(
                    (
                        observation.frame.rgb.copy(),
                        {**observation.frame.metadata(), "state": observation.state},
                    )
                )

        try:
            with adapter, controller:
                if args.probe_restart:
                    probe_rows = []
                    adapter.observe()
                    settled = adapter.restart_app(
                        launch,
                        max_seconds=min(args.restart_seconds, deadline - time.perf_counter()),
                        reason="exploratory restart probe from the current screen",
                        on_observation=restart_observation,
                    )
                    probe_rows.append({"settled_state": settled.state, "variant": settled.variant})
                    print(json.dumps({"probe_restart": probe_rows[-1]}), flush=True)
                elif args.inspect:
                    seen = adapter.observe()
                    reset_observation(seen)
                    print(
                        json.dumps(
                            {
                                "state": seen.state,
                                "variant": seen.variant,
                                "controls": [c.name for c in seen.controls],
                            }
                        ),
                        flush=True,
                    )
                else:
                    initial = adapter.observe()
                    if initial.state == "playing":
                        # A game left mid-episode by the operator is never data: pause
                        # and restart so the first attempt starts at a fresh boundary.
                        adapter.reset(
                            truncate=True,
                            start_next=False,
                            max_seconds=min(args.reset_seconds, deadline - time.perf_counter()),
                            on_terminal=terminal,
                            on_observation=reset_observation,
                        )
                        terminal.reset()
                    for index in range(args.episodes):
                        remaining = deadline - time.perf_counter()
                        if remaining < 1:
                            break
                        attempt_states.clear()
                        choose = scripted.get(attempt_policy_name(index), choose)
                        attempt_started = time.perf_counter()
                        menu_rows_before = len(adapter.trace)
                        terminal.reset()
                        halted_error = None
                        summary, rows, images, parked, first = None, [], [], None, None
                        restart_rows, start_error_before_restart = [], None
                        restart_states.clear()
                        try:
                            first = adapter.reset(
                                allow_initial_start=True,
                                max_seconds=min(args.reset_seconds, remaining),
                                on_terminal=terminal,
                                on_observation=reset_observation,
                            )
                        except Exception:  # noqa: BLE001 - classify, persist, then stop safely
                            halted_error = traceback.format_exc()
                            if try_restart("start", halted_error, restart_rows):
                                start_error_before_restart = halted_error
                                remaining = deadline - time.perf_counter()
                                if remaining >= 1:
                                    try:
                                        first = adapter.reset(
                                            allow_initial_start=True,
                                            max_seconds=min(args.reset_seconds, remaining),
                                            on_terminal=terminal,
                                            on_observation=reset_observation,
                                        )
                                        halted_error = None
                                    except Exception:  # noqa: BLE001 - a second failure halts
                                        halted_error = traceback.format_exc()
                        start_reset_seconds = time.perf_counter() - attempt_started
                        terminal.reset()
                        terminal_start = len(terminal.readings)
                        accepted_start = len(terminal.accepted)
                        episode_run_elapsed = run.elapsed_seconds
                        if first is not None:
                            summary, rows, images = collect_episode(
                                adapter,
                                controller,
                                backend,
                                bridge,
                                choose,
                                first,
                                seconds=args.episode_seconds,
                                deadline=deadline,
                                stall_seconds=60 if args.long_run else None,
                                max_frames=30000 if args.long_run else 2000,
                            )
                        else:
                            summary = {
                                "reason": "start_failure",
                                "observed_seconds": 0.0,
                                "frames": 0,
                                "actions_dispatched": 0,
                                "accepted_hud_frames": 0,
                                "observed_hud_max": None,
                                "distance": None,
                                "error": halted_error,
                                "score_semantics": "no episode started",
                            }
                        for row in rows:
                            row["canonical_elapsed_seconds"] = (
                                episode_run_elapsed + row["elapsed_seconds"]
                            )
                        # Park before evidence persistence or policy updates, so the
                        # next episode cannot run while previous data is written.
                        remaining = deadline - time.perf_counter()
                        park_started = time.perf_counter()
                        try:
                            if summary["error"]:
                                raise RuntimeError(summary["error"])
                            if remaining >= 1:
                                parked = adapter.reset(
                                    truncate=summary["reason"]
                                    in {"episode_time_limit", "verified_progress_stall"},
                                    start_next=False,
                                    max_seconds=min(args.reset_seconds, remaining),
                                    on_terminal=terminal,
                                    on_observation=reset_observation,
                                )
                        except Exception:  # noqa: BLE001 - retain the halt with the attempt
                            summary["park_error"] = traceback.format_exc()
                            if halted_error is None and try_restart(
                                "park", summary["park_error"], restart_rows
                            ):
                                # The relaunched game is the parked state for the next
                                # attempt; the attempt keeps its terminal classification
                                # and the restart is reported alongside it.
                                parked = adapter.latest
                                summary["parked_via_restart"] = True
                                summary["park_error_before_restart"] = summary.pop("park_error")
                            else:
                                halted_error = halted_error or summary["park_error"]
                        finally:
                            annotate_parked_score(summary, parked, paused_reader)
                            summary["terminal_readings"] = terminal.readings[terminal_start:]
                            summary["accepted_terminal_readings"] = terminal.accepted[
                                accepted_start:
                            ]
                            accepted_results = [
                                r["distance_meters"] for r in summary["accepted_terminal_readings"]
                            ]
                            if accepted_results and len(set(accepted_results)) == 1:
                                summary["distance"] = accepted_results[0]
                                summary["score_semantics"] = (
                                    "two agreeing fresh right-side result-field readings at least 0.15s apart"
                                )
                            menu_rows = adapter.trace[menu_rows_before:]
                            unintended = unintended_action_evidence(
                                adapter, first_row=menu_rows_before
                            )
                            if unintended and not halted_error:
                                halted_error = unintended["guard_reason"]
                            summary["attempt"] = {
                                "index": index,
                                "policy": attempt_policy_name(index)
                                if not args.policy
                                else args.policy_kind,
                                "classification": classify_attempt(
                                    summary,
                                    parked,
                                    halted_error=halted_error,
                                    unintended=unintended,
                                ),
                                "terminal_cause": terminal_cause(summary),
                                "parked_state": parked.state if parked is not None else None,
                                "start_reset_seconds": start_reset_seconds,
                                "park_reset_seconds": time.perf_counter() - park_started,
                                "attempt_seconds": time.perf_counter() - attempt_started,
                                "menu_clicks": [row["name"] for row in menu_rows],
                                "advertisement_closes": sum(
                                    row["name"] == "legitimate_ad_close" for row in menu_rows
                                ),
                                "advertisement_frames": attempt_states.get("advertisement", 0),
                                "unknown_frames": attempt_states.get("unknown", 0),
                                "reset_state_counts": dict(attempt_states),
                                "stale_captures": len(adapter.capture_trace),
                                "app_restarts": len(restart_rows),
                                "restarts": restart_rows,
                                "restart_state_counts": dict(restart_states),
                                "start_error_before_restart": start_error_before_restart,
                                "parked_via_restart": bool(summary.get("parked_via_restart")),
                                "unintended_action": unintended,
                            }
                            attempt_outcomes.append(summary["attempt"])
                            save_episode(run, index, summary, rows, images)
                            summaries.append(summary)
                            if summary["distance"] is not None:
                                run.metric(
                                    "distance",
                                    summary["distance"],
                                    index,
                                    score_semantics=summary["score_semantics"],
                                )
                            run.metric(
                                "episode_observed_seconds", summary["observed_seconds"], index
                            )
                            run.metric(
                                "attempt_seconds", summary["attempt"]["attempt_seconds"], index
                            )
                        print(json.dumps({"episode": index, **summary}), flush=True)
                        if halted_error:
                            raise RuntimeError(halted_error)
        except BaseException:  # noqa: BLE001 - retain interrupts and release every input path
            error = traceback.format_exc()
        finally:
            for cleanup in (controller.close, backend.close):
                try:
                    cleanup()
                except BaseException:  # noqa: BLE001 - attempt every release and retain each failure
                    error = (error or "") + traceback.format_exc()
        for name, data in (
            ("os-pedal-transitions", backend.trace),
            ("requested-pedal-leases", controller.trace),
            ("menu-transitions", adapter.trace),
            ("os-menu-transitions", adapter.sender.trace),
            ("capture-diagnostics", adapter.capture_trace),
            ("guard-diagnostics", adapter.guard_trace),
            ("app-restarts", adapter.restart_trace),
            ("numeric-score-exhaustions", terminal.exhausted),
            ("host-budget", host.trace),
        ):
            path = run.directory / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            run.register_artifact(path, name)
        for prefix, frames in (
            ("reset", reset_frames),
            ("terminal", terminal.frames),
            ("restart", restart_frames),
        ):
            for index, (rgb, metadata) in enumerate(frames):
                path = run.directory / f"{prefix}-{index:03d}.png"
                Image.fromarray(rgb).save(path)
                run.register_artifact(path, f"{prefix}_frame", metadata)
        session_unintended = unintended_action_events(adapter)
        if session_unintended and not error:
            error = "Unintended action detected: " + session_unintended[-1]["guard_reason"]
        scored = [s for s in summaries if s["distance"] is not None and not s["error"]]
        # Preserve the existing scored-episode selection for legacy summary keys.
        # The additive inventory below also retains observed scores when a later
        # fault prevents lifecycle success. Neither summary implies qualification.
        statistics = describe_native_attempts(scored, len(scored))["observed_result_distance"]
        native_descriptives = describe_native_attempts(
            summaries,
            0 if args.inspect or args.probe_restart else args.episodes,
            attempt_outcomes=attempt_outcomes,
        )
        classifications = {}
        for outcome in attempt_outcomes:
            key = outcome["classification"]
            classifications[key] = classifications.get(key, 0) + 1
        not_attempted = (
            max(0, args.episodes - len(attempt_outcomes))
            if not (args.inspect or args.probe_restart)
            else 0
        )
        longest_success_run = longest_scored_streak(attempt_outcomes)
        reliability = {
            "attempts_requested": args.episodes if not (args.inspect or args.probe_restart) else 0,
            "attempts_completed": len(attempt_outcomes),
            "not_attempted": not_attempted,
            "classifications": classifications,
            "longest_consecutive_scored_successes": longest_success_run,
            "advertisement_closes": sum(o["advertisement_closes"] for o in attempt_outcomes),
            "advertisement_encounters": sum(
                1 for o in attempt_outcomes if o["advertisement_frames"]
            ),
            "manual_interventions": 0,
            # Session-wide and trace-derived: every accepted click in the session
            # (including the pre-attempt truncating reset) followed within the
            # window by a latched foreground loss, whichever thread observed it.
            "unintended_actions": len(session_unintended),
            "unintended_action_evidence": session_unintended
            or (
                f"no latched foreground loss within {POST_CLICK_WINDOW_SECONDS:g} s of any "
                "accepted click; every click is an allowlisted named control (menu-transitions)"
            ),
            "unintended_action_window_seconds": POST_CLICK_WINDOW_SECONDS,
            "app_restarts": sum(o.get("app_restarts", 0) for o in attempt_outcomes),
            "restart_failures": sum(
                1 for o in attempt_outcomes for r in o.get("restarts", []) if not r["ok"]
            ),
            "restart_recovery_enabled": launch is not None,
        }
        if summaries:
            run.evaluation(
                {
                    "episodes": len(summaries),
                    "checkpoint_hash": checkpoint_hash,
                    "protocol": "actual-screen-frozen-episode-1",
                    "results": {
                        "episodes": summaries,
                        "summary": {"distance": statistics},
                        "scored_episodes": len(scored),
                        "excluded_episodes": len(summaries) - len(scored),
                        "mean_distance": statistics["mean"] if statistics else None,
                        "median_distance": statistics["median"] if statistics else None,
                        "game_seed_control": False,
                        "scope": config["scope"],
                        "attempt_outcomes": attempt_outcomes,
                        "reliability": reliability,
                        "native_descriptive_summary": native_descriptives,
                    },
                }
            )
        run.finalize(
            status="failed" if error else "completed",
            episode_count=len(summaries),
            episodes=len(summaries),
            episode_summaries=summaries,
            scored_episodes=len(scored),
            distance_statistics=statistics,
            native_descriptive_summary=native_descriptives,
            reliability=reliability,
            error=error,
            observed_session_seconds=time.perf_counter() - start,
            qualification_evidence=False,
        )
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "status": "failed" if error else "completed",
                    "episodes": len(summaries),
                    "error": error,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
