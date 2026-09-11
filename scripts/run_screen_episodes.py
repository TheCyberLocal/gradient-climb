"""Bounded end-to-end actual-game baseline episodes, with local pixel records.

This is a supervised integration/evaluation entry point, not a training claim.
All menu actions come from the separately verified native adapter's allowlist.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.windows import discover_windows
from gradientclimb.control.game_adapter import NativeGameAdapter
from gradientclimb.control.pedals import PedalAction, PedalController
from gradientclimb.control.windows import WindowsPedalBackend
from gradientclimb.evaluation.benchmark import summarize
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
):
    """Hold leases only after fresh classified capture and explicit policy output."""
    bridge.reset()
    start = time.perf_counter()
    end = min(start + seconds, deadline)
    rows, images = [], []
    observed_values = []
    current = first
    reason = "episode_time_limit"
    failure = None
    unknown_since = None
    try:
        for step in range(2000):
            if on_tick:
                on_tick()
            if time.perf_counter() >= end:
                break
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
            unknown_since = None
            reason = "episode_time_limit"
            actual_code, age = os_state_at(backend.trace, current.frame.started_ns)
            screen = bridge.observe(
                current.frame.rgb,
                current.frame.timestamp_ns,
                previous_action_code=actual_code,
                action_age_seconds=age,
                episode_elapsed_seconds=max(0.0, current.frame.timestamp_ns / 1e9 - start),
            )
            code = choose_action(screen)
            if type(code) is not int or code not in range(4):
                raise ValueError("Policy returned an invalid independent pedal state")
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            dispatched = adapter.is_playing()
            lease = min(0.4, remaining)
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
                observed_values.append(screen.hud["hud_displayed_progress_meters"])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--baseline", choices=["always_gas", "random", "neutral"], default="always_gas"
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=8)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-backend", choices=["dxcam", "mss", "pillow"], default="dxcam")
    parser.add_argument(
        "--inspect", action="store_true", help="Classify one current frame; never navigate or drive"
    )
    args = parser.parse_args()
    if any((args.policy, args.policy_kind, args.parent_run)) and not all(
        (args.policy, args.policy_kind, args.parent_run)
    ):
        raise ValueError("Frozen evaluation requires policy path, kind and parent run together")
    if not (
        1 <= args.episodes <= 20
        and 1 <= args.episode_seconds <= 120
        and 1 <= args.max_seconds <= 3600
    ):
        raise ValueError("Episode/count/session limits are out of bounds")
    targets = discover_windows()
    if len(targets) != 1:
        raise RuntimeError("Expected exactly one visible game window")
    target = targets[0]
    profile = MeasurementProfile(**json.loads(args.measurement_profile.read_text()))
    bridge = ScreenFeatureBridge(HCRPixelMeasurer(profile), HUDDigitReader.from_manifest(args.hud))
    result_reader = (
        ResultDistanceReader.from_manifest(args.result_reader) if args.result_reader else None
    )
    paused_reader = PausedDistanceReader.from_gameplay_manifest(args.hud)
    rng = np.random.default_rng(args.seed)
    choose = lambda _: (
        int(rng.integers(4))
        if args.baseline == "random"
        else (1 if args.baseline == "always_gas" else 0)
    )
    checkpoint_hash = None
    algorithm = args.baseline
    if args.policy:
        checkpoint_hash = sha256_file(args.policy)
        parent = load_run(Path("artifacts"), args.parent_run)
        if not verify_run(Path("artifacts"), args.parent_run)["valid"] or not any(
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
    config = {
        "baseline": args.baseline if not args.policy else None,
        "policy_kind": args.policy_kind,
        "checkpoint_hash": checkpoint_hash,
        "parent_run": args.parent_run,
        "requested_episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
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
    }
    adapter = None
    backend = WindowsPedalBackend(target, gas_vk=0x27, brake_vk=0x25, input_mode="scancode")
    controller = PedalController(backend, lambda: adapter is not None and adapter.is_playing())
    adapter = NativeGameAdapter(
        target,
        args.ui_profile,
        release_pedals=controller.release,
        capture_backend=args.capture_backend,
    )
    summaries, reset_frames = [], []
    terminal = ResultScoreCollector(result_reader)
    error = None
    with RunRecorder(
        "artifacts",
        "real-screen-policy-evaluation" if args.policy else "real-screen-episode-pilot",
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
            run.register_artifact(Path("artifacts") / variant.file, "ui_reference")
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

        def reset_observation(observation):
            if len(reset_frames) < 60 and (
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
                if args.inspect:
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
                    for index in range(args.episodes):
                        remaining = deadline - time.perf_counter()
                        if remaining < 1:
                            break
                        terminal.reset()
                        first = adapter.reset(
                            allow_initial_start=True,
                            max_seconds=min(60.0, remaining),
                            on_terminal=terminal,
                            on_observation=reset_observation,
                        )
                        terminal.reset()
                        terminal_start = len(terminal.readings)
                        accepted_start = len(terminal.accepted)
                        episode_run_elapsed = run.elapsed_seconds
                        summary, rows, images = collect_episode(
                            adapter,
                            controller,
                            backend,
                            bridge,
                            choose,
                            first,
                            seconds=args.episode_seconds,
                            deadline=deadline,
                        )
                        for row in rows:
                            row["canonical_elapsed_seconds"] = (
                                episode_run_elapsed + row["elapsed_seconds"]
                            )
                        # Park before evidence persistence or policy updates, so the
                        # next episode cannot run while previous data is written.
                        remaining = deadline - time.perf_counter()
                        parked = None
                        try:
                            if summary["error"]:
                                raise RuntimeError(summary["error"])
                            if remaining >= 1:
                                parked = adapter.reset(
                                    truncate=summary["reason"] == "episode_time_limit",
                                    start_next=False,
                                    max_seconds=min(60.0, remaining),
                                    on_terminal=terminal,
                                    on_observation=reset_observation,
                                )
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
                        print(json.dumps({"episode": index, **summary}), flush=True)
        except Exception:  # noqa: BLE001 - release and persist all native traces on every failure
            error = traceback.format_exc()
        finally:
            for cleanup in (controller.close, backend.close):
                try:
                    cleanup()
                except Exception:  # noqa: BLE001 - attempt every release and retain each failure
                    error = (error or "") + traceback.format_exc()
        for name, data in (
            ("os-pedal-transitions", backend.trace),
            ("requested-pedal-leases", controller.trace),
            ("menu-transitions", adapter.trace),
            ("os-menu-transitions", adapter.sender.trace),
            ("capture-diagnostics", adapter.capture_trace),
            ("guard-diagnostics", adapter.guard_trace),
            ("numeric-score-exhaustions", terminal.exhausted),
        ):
            path = run.directory / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            run.register_artifact(path, name)
        for prefix, frames in (("reset", reset_frames), ("terminal", terminal.frames)):
            for index, (rgb, metadata) in enumerate(frames):
                path = run.directory / f"{prefix}-{index:03d}.png"
                Image.fromarray(rgb).save(path)
                run.register_artifact(path, f"{prefix}_frame", metadata)
        scored = [s for s in summaries if s["distance"] is not None and not s["error"]]
        statistics = summarize([s["distance"] for s in scored]) if scored else None
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
