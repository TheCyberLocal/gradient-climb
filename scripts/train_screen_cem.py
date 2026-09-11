"""Governed actual-screen CEM. Importing or validating this script sends no input.

Use --execute only for an authorized supervised integration or governed run.
All times are measured from entry into this file, before heavy package imports.
"""

from __future__ import annotations

import time

ENTRY_CLOCK = time.perf_counter()

# The governed clock deliberately precedes third-party initialization.
import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from run_screen_episodes import (
    ResultScoreCollector,
    annotate_parked_score,
    collect_episode,
    save_episode,
)

from gradientclimb.algorithms.screen_search import EpisodeCEM, ScreenLinearPolicy
from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.windows import discover_windows
from gradientclimb.control.game_adapter import NativeGameAdapter
from gradientclimb.control.pedals import PedalController
from gradientclimb.control.windows import WindowsPedalBackend
from gradientclimb.experiments import RunRecorder
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile
from gradientclimb.perception.scoring import PausedDistanceReader, ResultDistanceReader
from gradientclimb.perception.screen_features import FEATURE_NAMES, ScreenFeatureBridge

CHECKPOINT_SECONDS = (300, 600, 1200, 1800, 2700, 3600)
PARK_RESERVE_SECONDS = 10.0


class ScreenCEMProtocol(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True, allow_inf_nan=False)
    training_seconds: int = 3600
    maximum_episode_seconds: int = Field(default=60, ge=1, le=120)
    maximum_episodes: int = Field(default=120, ge=1, le=1000)
    seed: int = Field(default=42, ge=0)
    population: int = Field(default=8, ge=4, le=64)
    elite_count: int = Field(default=3, ge=2)
    initial_std: float = Field(default=0.5, gt=0, le=5)
    minimum_std: float = Field(default=0.08, gt=0, le=5)
    smoothing: float = Field(default=0.5, gt=0, le=1)
    features: list[str] = Field(min_length=1, max_length=64)
    required_features: list[str] = Field(min_length=1)
    feature_scales: list[float] = Field(min_length=1)
    history: int = Field(default=4, ge=1, le=16)
    checkpoint_times: list[int]

    @model_validator(mode="after")
    def coherent(self):
        if self.training_seconds != 3600 or tuple(self.checkpoint_times) != CHECKPOINT_SECONDS:
            raise ValueError("The governed protocol requires 3600 seconds and its six checkpoints")
        if self.elite_count >= self.population or self.minimum_std > self.initial_std:
            raise ValueError("Inconsistent CEM population or deviation")
        if len(set(self.features)) != len(self.features) or set(self.features) - set(FEATURE_NAMES):
            raise ValueError("Policy features must be unique names in the screen schema")
        if set(self.required_features) - set(self.features):
            raise ValueError("Required features must belong to the selected policy schema")
        if len(self.feature_scales) != len(self.features) or any(
            v <= 0 for v in self.feature_scales
        ):
            raise ValueError("Every selected feature requires a finite positive scale")
        return self


class BudgetExpired(RuntimeError):
    pass


class GovernedClock:
    def __init__(self, seconds, *, started=None, clock=time.perf_counter, sleep=time.sleep):
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 10 <= seconds <= 3600
        ):
            raise ValueError("Governed duration must be a finite number in 10..3600 seconds")
        self.clock, self.sleep, self.seconds = clock, sleep, float(seconds)
        self.started = clock() if started is None else started
        if not math.isfinite(self.started) or self.started > clock():
            raise ValueError("Clock origin must be a finite past monotonic timestamp")
        self.deadline = self.started + self.seconds

    @property
    def elapsed(self):
        return max(0.0, self.clock() - self.started)

    @property
    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def require_active(self):
        if self.clock() >= self.deadline:
            raise BudgetExpired("Governed wall-clock deadline reached; no new input permitted")


class DeadlineInput:
    """Check at the ordinary OS packet boundary; release-only cleanup stays allowed."""

    def __init__(self, wrapped, budget):
        self.wrapped, self.budget = wrapped, budget

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def send(self, events):
        release_only = all(
            (event.type == 1 and bool(event.ki.dwFlags & 2))
            or (event.type == 0 and event.mi.dwFlags == 4)
            for event in events
        )
        if not release_only:
            self.budget.require_active()
        return self.wrapped.send(events)


def require_clean_source(root=PROJECT, *, runner=subprocess.run):
    def git(*arguments):
        result = runner(
            ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode:
            raise RuntimeError("Cannot verify clean committed source")
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    if not revision or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Screen training requires a clean committed checkout")
    return revision


def manifest_dependencies(path):
    """Read a finite local hash-pinned manifest tree, including every glyph image."""
    found = {}

    def visit(filename, expected=None, depth=0):
        filename = Path(filename).resolve(strict=True)
        if depth > 8 or len(found) > 2000 or not filename.is_file():
            raise ValueError("Manifest dependency tree exceeds its file/depth bound")
        digest = sha256_file(filename)
        if expected is not None and digest != expected:
            raise ValueError("Manifest dependency hash mismatch")
        if filename in found:
            return
        found[filename] = digest
        if filename.suffix.lower() != ".json":
            return

        def references(value):
            if isinstance(value, dict):
                if "file" in value and "sha256" in value:
                    child = (filename.parent / value["file"]).resolve()
                    if not child.is_relative_to(filename.parent):
                        raise ValueError("Manifest dependency escapes its directory")
                    visit(child, value["sha256"], depth + 1)
                else:
                    for nested in value.values():
                        references(nested)
            elif isinstance(value, list):
                for nested in value:
                    references(nested)

        references(json.loads(filename.read_text(encoding="utf-8")))

    visit(path)
    return found


def write_json_artifact(run, name, value, kind, metadata=None):
    path = run.directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return run.register_artifact(path, kind, metadata)


class CheckpointSchedule:
    """Snapshot when a boundary is observed; preserve target, snapshot, and I/O times."""

    def __init__(self, budget):
        self.budget = budget
        self.targets = [t for t in CHECKPOINT_SECONDS if t <= budget.seconds]
        self.snapshots, self.written = [], 0

    def pulse(self, state, phase):
        while len(self.snapshots) < len(self.targets):
            target = self.targets[len(self.snapshots)]
            if self.budget.elapsed < target:
                break
            # JSON roundtrip detaches the optimizer arrays and pending population.
            snapshot = json.loads(json.dumps(state(), allow_nan=False))
            self.snapshots.append(
                {
                    **snapshot,
                    "format": "screen-cem-json-v1",
                    "target_seconds": target,
                    "snapshot_elapsed_seconds": self.budget.elapsed,
                    "phase": phase,
                    "timing_semantics": "first observed boundary; not asserted exact target time",
                }
            )

    def flush(self, run):
        while self.written < len(self.snapshots):
            row = self.snapshots[self.written]
            begin = self.budget.elapsed
            row = {**row, "persistence_started_elapsed_seconds": begin}
            path = run.directory / f"checkpoint-{row['target_seconds']:04d}s.json"
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(row, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            run.register_artifact(
                path,
                "checkpoint",
                {
                    "format": "screen-cem-json-v1",
                    "target_seconds": row["target_seconds"],
                    "training_elapsed_seconds": row["snapshot_elapsed_seconds"],
                    "persistence_started_elapsed_seconds": begin,
                    "persistence_completed_elapsed_seconds": self.budget.elapsed,
                    "final": row["target_seconds"] == self.budget.seconds,
                    "evaluation_required": True,
                    "parent_run": run.run_id,
                },
            )
            self.written += 1


def eligible_score(summary, parked, accepted, *, horizon, first_timestamp_ns):
    invalid = {"eligible": False, "score": None, "source": None}
    if summary.get("error") or summary.get("park_error") or parked is None:
        return {**invalid, "reason": "Episode or parking failed"}
    reason = summary.get("reason")
    if reason == "episode_time_limit":
        observed_seconds = summary.get("observed_seconds")
        if (
            type(observed_seconds) not in (int, float)
            or not math.isfinite(observed_seconds)
            or observed_seconds < horizon
            or parked.state != "paused"
        ):
            return {**invalid, "reason": "The declared truncation horizon was not completed"}
        reading = summary.get("paused_reading", {})
        if reading.get("field") != "paused_gameplay_displayed_progress":
            return {**invalid, "reason": "Paused boundary requires its scoped reader"}
        source = "verified_paused_frame"
    elif reason in {"observed_result", "observed_revive_offer"}:
        if parked.state != "tune" or len(accepted) != 1:
            return {**invalid, "reason": "Natural episode lacks one stable terminal boundary"}
        reading = accepted[0]
        interval = reading.get("agreement_interval_seconds")
        if (
            type(interval) not in (int, float)
            or not math.isfinite(interval)
            or interval < 0.15
            or reading.get("first_agreeing_timestamp_ns", -1) < first_timestamp_ns
            or reading.get("field") != "right_side_result_distance"
        ):
            return {**invalid, "reason": "Terminal evidence is stale or unconfirmed"}
        source = "stable_terminal_distance"
    else:
        return {**invalid, "reason": "Aborted/unknown episodes cannot update CEM"}
    score = reading.get("distance_meters")
    if (
        reading.get("valid") is not True
        or type(score) is not int
        or score < 0
        or reading.get("timestamp_ns", -1) < first_timestamp_ns
        or reading.get("timestamp_ns", math.inf) > parked.frame.completed_ns
    ):
        return {**invalid, "reason": "Missing valid nonnegative integer boundary score"}
    return {"eligible": True, "score": score, "source": source, "evidence": reading}


def governed_train(run, protocol, runtime, budget, *, collect=collect_episode, save=save_episode):
    """Testable orchestration over explicit runtime dependencies; no implicit native I/O."""
    budget.require_active()
    initial = np.zeros((2, 2 * len(protocol.features) + 1))
    initial[:, -1] = [1.0, -1.0]
    optimizer = EpisodeCEM(
        initial,
        seed=protocol.seed,
        population=protocol.population,
        elite_count=protocol.elite_count,
        initial_std=protocol.initial_std,
        minimum_std=protocol.minimum_std,
        smoothing=protocol.smoothing,
    )
    schedule = CheckpointSchedule(budget)
    summaries, generations, phases = [], [], []
    candidate = None
    error = None
    stop_reason = "governed_deadline"
    invalid_streak = 0
    runtime.reset_frames = []
    last_reset_save = [-math.inf, None]

    def state():
        return {
            "optimizer_state": optimizer.state_dict(),
            "policy": ScreenLinearPolicy(
                runtime.bridge.schema_id,
                tuple(protocol.features),
                optimizer.incumbent,
                protocol.feature_scales,
            ).as_dict(),
            "candidate": candidate,
            "episode_attempts": len(summaries),
            "feature_schema": runtime.bridge.schema,
            "parent_run": run.run_id,
            "source_lineage": getattr(runtime, "lineage", {}),
        }

    def pulse(phase):
        schedule.pulse(state, phase)

    def phase(name):
        phases.append({"phase": name, "elapsed_seconds": budget.elapsed})
        pulse(name)

    def reset_observation(observation):
        pulse("menu_observation")
        if observation.state != last_reset_save[1] or budget.elapsed - last_reset_save[0] >= 1:
            if len(runtime.reset_frames) >= 4000:
                raise RuntimeError("Reset evidence frame bound reached")
            runtime.reset_frames.append(
                (
                    observation.frame.rgb.copy(),
                    {
                        **observation.frame.metadata(),
                        "state": observation.state,
                        "governed_elapsed_seconds": budget.elapsed,
                    },
                )
            )
            last_reset_save[:] = [budget.elapsed, observation.state]

    def reset(**kwargs):
        budget.require_active()
        return runtime.adapter.reset(
            max_seconds=min(60.0, budget.remaining),
            on_terminal=runtime.terminal,
            on_observation=reset_observation,
            **kwargs,
        )

    def idle_to_deadline():
        runtime.controller.release()
        phase("stationary_budget_remainder")
        while budget.remaining:
            runtime.adapter._guard()
            pulse("stationary_budget_remainder")
            schedule.flush(run)
            budget.sleep(min(0.1, budget.remaining))

    try:
        phase("model_initialized")
        for index in range(protocol.maximum_episodes):
            runtime.controller.release()
            if budget.remaining < protocol.maximum_episode_seconds + PARK_RESERVE_SECONDS:
                idle_to_deadline()
                break
            candidate_id, weights = optimizer.ask()
            candidate = {"id": candidate_id, "weights": weights.tolist(), "attempt": index}
            phase("candidate_selected")
            schedule.flush(run)
            runtime.terminal.reset()
            phase("starting_episode")
            first = reset(allow_initial_start=True)
            # Menu/ad costs consume the same budget. Never rank a shortened
            # budget-cutoff episode against a full-horizon candidate.
            if budget.remaining < protocol.maximum_episode_seconds + PARK_RESERVE_SECONDS:
                phase("insufficient_full_horizon")
                reset(truncate=True, start_next=False)
                idle_to_deadline()
                break
            accepted_start = len(runtime.terminal.accepted)
            reading_start = len(getattr(runtime.terminal, "readings", []))
            exhaustion_start = len(getattr(runtime.terminal, "exhausted", []))
            runtime.terminal.reset()
            policy = ScreenLinearPolicy(
                runtime.bridge.schema_id,
                tuple(protocol.features),
                weights,
                protocol.feature_scales,
            )
            missing_required = [0]

            def choose(observation, policy=policy, missing_required=missing_required):
                budget.require_active()
                if not observation.supports(protocol.required_features):
                    missing_required[0] += 1
                    return 0
                return policy.action(observation, FEATURE_NAMES)

            phase("episode_collection")
            canonical_elapsed_origin = getattr(run, "elapsed_seconds", budget.elapsed)
            summary, rows, images = collect(
                runtime.adapter,
                runtime.controller,
                runtime.backend,
                runtime.bridge,
                choose,
                first,
                seconds=protocol.maximum_episode_seconds,
                deadline=budget.deadline - PARK_RESERVE_SECONDS,
                on_tick=lambda: pulse("episode_tick"),
            )
            for row in rows:
                row["canonical_elapsed_seconds"] = canonical_elapsed_origin + row["elapsed_seconds"]
            summary.update(
                candidate_id=candidate_id,
                attempt=index,
                missing_required_feature_frames=missing_required[0],
            )
            parked = None
            try:
                runtime.controller.release()
                if summary.get("error"):
                    raise RuntimeError(summary["error"])
                phase("parking_episode")
                parked = reset(truncate=summary["reason"] == "episode_time_limit", start_next=False)
                if parked.state not in {"paused", "tune"}:
                    raise RuntimeError("Episode did not reach a stationary evidence boundary")
                phase("scoring_episode")
                annotate_parked_score(summary, parked, runtime.paused_reader)
                if parked.state == "paused":
                    images.append(
                        (
                            parked.frame.rgb.copy(),
                            {
                                **parked.frame.metadata(),
                                "state": "paused",
                                "role": "score_boundary",
                            },
                        )
                    )
            except BaseException:
                summary["park_error"] = traceback.format_exc()
                raise
            finally:
                runtime.controller.release()
                summary["accepted_terminal_readings"] = runtime.terminal.accepted[accepted_start:]
                summary["terminal_readings"] = getattr(runtime.terminal, "readings", [])[
                    reading_start:
                ]
                summary["terminal_reader_exhaustion"] = getattr(runtime.terminal, "exhausted", [])[
                    exhaustion_start:
                ]
                summary["score_decision"] = eligible_score(
                    summary,
                    parked,
                    runtime.terminal.accepted[accepted_start:],
                    horizon=protocol.maximum_episode_seconds,
                    first_timestamp_ns=first.frame.started_ns,
                )
                if summary["score_decision"]["eligible"]:
                    summary["distance"] = summary["score_decision"]["score"]
                    summary["score_semantics"] = summary["score_decision"]["source"]
                summary["governed_elapsed_seconds"] = budget.elapsed
                phase("persisting_episode")
                summaries.append(summary)
                save(run, index, summary, rows, images)
                pulse("episode_persisted")
                schedule.flush(run)
            decision = summary["score_decision"]
            if decision["eligible"]:
                # Neither pending generation scores nor parameters may change
                # after the governed deadline, even if final persistence runs late.
                budget.require_active()
                phase("optimizer_update")
                budget.require_active()
                update_started = budget.elapsed
                generation = optimizer.tell(candidate_id, decision["score"])
                run.metric("optimizer_seconds", budget.elapsed - update_started, step=index)
                run.metric(
                    "training_episode_distance",
                    decision["score"],
                    step=index,
                    score_source=decision["source"],
                    candidate_id=candidate_id,
                )
                if generation is not None:
                    generations.append({**generation, "elapsed_seconds": budget.elapsed})
                    run.metric(
                        "generation_mean_training_distance",
                        generation["mean_score"],
                        step=optimizer.generation,
                    )
                invalid_streak = 0
            else:
                invalid_streak += 1
                if invalid_streak >= 3:
                    raise RuntimeError("Three ineligible scores; CEM candidate remains pending")
            pulse("after_optimizer_or_ineligible_score")
            schedule.flush(run)
            run.telemetry(phase="stationary_episode_boundary", governed_elapsed=budget.elapsed)
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "attempt": index,
                        "candidate_id": candidate_id,
                        "score": decision["score"],
                        "eligible": decision["eligible"],
                        "elapsed_seconds": budget.elapsed,
                    }
                ),
                flush=True,
            )
        else:
            stop_reason = "maximum_episode_attempts"
    except BudgetExpired:
        stop_reason = "governed_deadline"
    except BaseException:  # noqa: BLE001 - preserve partial evidence on interrupt too
        stop_reason, error = "failure", traceback.format_exc()
    finally:
        try:
            runtime.controller.release()
        except BaseException:  # noqa: BLE001 - retain release failures during final cleanup
            error = (error or "") + traceback.format_exc()
            stop_reason = "release_failure"
        stopped = budget.elapsed
        # No native action or optimizer update follows this point. Final storage
        # is allowed to finish and is explicitly recorded as post-budget cleanup.
        pulse("governed_stop")
        schedule.flush(run)
        write_json_artifact(
            run,
            "final-screen-cem-state.json",
            state(),
            "screen_cem_state",
            {"elapsed_seconds": budget.elapsed, "stop_reason": stop_reason},
        )
        write_json_artifact(run, "governed-phases.json", phases, "phase_timing")
        write_json_artifact(run, "cem-generations.json", generations, "optimizer_generations")
    return {
        "episodes": len(summaries),
        "episode_summaries": summaries,
        "optimizer_updates": optimizer.generation,
        "eligible_episodes": optimizer.completed_episodes,
        "environment_steps": sum(s.get("frames", 0) for s in summaries),
        "error": error,
        "stop_reason": stop_reason,
        "governed_elapsed_at_stop": stopped,
        "requested_training_seconds": budget.seconds,
        "checkpoint_targets_observed": [row["target_seconds"] for row in schedule.snapshots],
        "completed_full_governed_hour": budget.seconds == 3600 and stopped >= 3600 and not error,
        "qualification_evidence": False,
        "score_semantics": "training selection scores; independent checkpoint evaluation required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definition", type=Path, default=Path("experiments/definitions/real-screen-cem.json")
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
    parser.add_argument("--result-reader", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument("--capture-backend", choices=["dxcam", "mss", "pillow"], default="dxcam")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    budget = GovernedClock(args.seconds, started=ENTRY_CLOCK)
    protocol_data = json.loads(args.definition.read_text(encoding="utf-8"))
    if args.episode_seconds is not None:
        if args.seconds == 3600:
            raise ValueError("Episode-horizon override is available only for shortened integration")
        protocol_data["maximum_episode_seconds"] = args.episode_seconds
    protocol = ScreenCEMProtocol.model_validate(protocol_data)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "capture_backend": args.capture_backend,
                    "requested_seconds": args.seconds,
                    "protocol": protocol.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return
    revision = require_clean_source()
    budget.require_active()
    dependencies = {**manifest_dependencies(args.hud), **manifest_dependencies(args.result_reader)}
    for path in (
        args.definition,
        args.ui_profile,
        args.measurement_profile,
        PROJECT / "requirements-lock.txt",
        PROJECT / "pyproject.toml",
        Path(__file__),
        PROJECT / "scripts/run_screen_episodes.py",
    ):
        dependencies[path.resolve()] = sha256_file(path)
    profile = MeasurementProfile(**json.loads(args.measurement_profile.read_text()))
    bridge = ScreenFeatureBridge(
        HCRPixelMeasurer(profile), HUDDigitReader.from_manifest(args.hud), history=protocol.history
    )
    terminal = ResultScoreCollector(ResultDistanceReader.from_manifest(args.result_reader))
    paused_reader = PausedDistanceReader.from_gameplay_manifest(args.hud)
    targets = discover_windows()
    if len(targets) != 1:
        raise RuntimeError("Expected one visible inspected game window")
    target = targets[0]
    config = {
        **protocol.model_dump(mode="json"),
        "requested_seconds": budget.seconds,
        "integration_only": budget.seconds != 3600,
        "clean_start_git_sha": revision,
        "governed_origin": "file entry before heavy imports; interpreter launch excluded",
        "entry_monotonic_seconds": budget.started,
        "elapsed_before_recorder_seconds": budget.elapsed,
        "dependency_sha256": {str(k): v for k, v in dependencies.items()},
        "target": target.as_dict(),
        "screen_schema": bridge.schema,
        "screen_schema_id": bridge.schema_id,
        "input_mode": "extended_scancode",
        "capture_backend": args.capture_backend,
        "park_reserve_seconds": PARK_RESERVE_SECONDS,
        "score_gate": "stable terminal pair or full-horizon verified paused readout",
        "unknown_score_policy": "retain pending candidate; stop after three invalid attempts",
        "initial_biases": {"gas": 1.0, "brake": -1.0},
    }
    adapter = controller = backend = runtime = None
    summary = {"episodes": 0, "error": None, "qualification_evidence": False}
    with RunRecorder(
        args.artifacts,
        "real-screen-cem",
        config,
        seed=protocol.seed,
        algorithm="sequential_episode_cem",
        environment="actual_hill_climb_racing",
        source_root=PROJECT,
        vehicle_profile=protocol_data.get("vehicle"),
        map_profile=protocol_data.get("map"),
    ) as run:
        try:
            budget.require_active()
            backend = WindowsPedalBackend(target, gas_vk=0x27, brake_vk=0x25, input_mode="scancode")
            backend.sender = DeadlineInput(backend.sender, budget)
            controller = PedalController(
                backend,
                lambda: budget.remaining > 0 and adapter is not None and adapter.is_playing(),
            )
            adapter = NativeGameAdapter(
                target,
                args.ui_profile,
                args.reference_root,
                release_pedals=controller.release,
                capture_backend=args.capture_backend,
            )
            adapter.sender.api = DeadlineInput(adapter.sender.api, budget)
            runtime = SimpleNamespace(
                adapter=adapter,
                backend=backend,
                controller=controller,
                bridge=bridge,
                terminal=terminal,
                paused_reader=paused_reader,
                reset_frames=[],
                lineage={
                    "clean_start_git_sha": revision,
                    "ui_profile_sha256": dependencies[args.ui_profile.resolve()],
                    "result_reader_sha256": dependencies[args.result_reader.resolve()],
                    "hud_manifest_sha256": dependencies[args.hud.resolve()],
                },
            )
            for variant in adapter.recognizer.profile.variants:
                path = (args.reference_root / variant.file).resolve()
                dependencies[path] = variant.sha256
            for module in tuple(sys.modules.values()):
                filename = getattr(module, "__file__", None)
                if filename and getattr(module, "__name__", "").startswith("gradientclimb."):
                    path = Path(filename).resolve()
                    if path.is_relative_to(PROJECT / "src") and path.suffix == ".py":
                        dependencies[path] = sha256_file(path)
            for path, digest in dependencies.items():
                if sha256_file(path) != digest:
                    raise RuntimeError("A frozen source/profile dependency changed during setup")
                run.register_artifact(
                    path,
                    "screen_training_dependency",
                    {"original_path": str(path), "sha256": digest},
                )
            write_json_artifact(
                run,
                "dependency-index.json",
                {str(k): v for k, v in dependencies.items()},
                "dependency_index",
            )
            if require_clean_source() != revision:
                raise RuntimeError("Source revision changed during setup")
            budget.require_active()
            with adapter, controller:
                initial = adapter.observe()
                if initial.state not in {"paused", "tune"}:
                    raise RuntimeError("Training must start from verified PAUSED or Tune")
                summary = governed_train(run, protocol, runtime, budget)
        except BaseException:  # noqa: BLE001 - retain native traces on every interrupt/failure
            summary["error"] = (summary.get("error") or "") + traceback.format_exc()
        finally:
            for cleanup in (
                controller.close if controller else None,
                backend.close if backend else None,
            ):
                if cleanup:
                    try:
                        cleanup()
                    except BaseException:  # noqa: BLE001 - attempt all release cleanup paths
                        summary["error"] = (summary.get("error") or "") + traceback.format_exc()
        traces = {
            "os-pedal-transitions": backend.trace if backend else [],
            "requested-pedal-leases": controller.trace if controller else [],
            "menu-transitions": adapter.trace if adapter else [],
            "raw-mouse-input": adapter.sender.trace if adapter else [],
            "adapter-guard-trace": adapter.guard_trace if adapter else [],
            "discarded-captures": adapter.capture_trace if adapter else [],
            "terminal-readings": terminal.readings,
            "accepted-terminal-readings": terminal.accepted,
            "terminal-reader-exhaustion": getattr(terminal, "exhausted", []),
        }
        for name, rows in traces.items():
            write_json_artifact(run, f"{name}.json", rows, name)
        for prefix, frames in (
            ("reset", runtime.reset_frames if runtime else []),
            ("terminal", terminal.frames),
        ):
            for index, (rgb, metadata) in enumerate(frames):
                path = run.directory / f"{prefix}-{index:04d}.png"
                Image.fromarray(rgb).save(path)
                run.register_artifact(path, f"{prefix}_frame", metadata)
        summary["actual_elapsed_before_finalize"] = budget.elapsed
        summary["post_budget_persistence_seconds"] = max(0.0, budget.elapsed - budget.seconds)
        run.finalize(status="failed" if summary.get("error") else "completed", **summary)
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "error": summary.get("error"),
                    "eligible_episodes": summary.get("eligible_episodes", 0),
                    "elapsed_seconds": budget.elapsed,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
