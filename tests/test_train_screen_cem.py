"""Governed screen trainer tests: synthetic clocks/frames, no native execution."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.control.game_adapter import mouse_click_events
from gradientclimb.control.windows import keyboard_event


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "screen_cem_trainer_test",
        Path(__file__).resolve().parents[1] / "scripts/train_screen_cem.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Run:
    run_id = "test-run"

    def __init__(self, path):
        self.directory = path
        self.artifacts, self.metrics, self.saved = [], [], []

    def register_artifact(self, path, kind, metadata=None):
        self.artifacts.append({"path": path, "kind": kind, "metadata": metadata})

    def metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def telemetry(self, **kwargs):
        pass


def protocol(trainer, **overrides):
    data = json.loads(
        (
            Path(__file__).resolve().parents[1] / "experiments/definitions/real-screen-cem.json"
        ).read_text()
    )
    data.update(maximum_episode_seconds=8, maximum_episodes=4, population=4, elite_count=2)
    data.update(overrides)
    return trainer.ScreenCEMProtocol.model_validate(data)


def reading(score=0, *, terminal=False, timestamp_ns=2_000_000_000):
    return {
        "valid": True,
        "distance_meters": score,
        "timestamp_ns": timestamp_ns,
        "field": "right_side_result_distance" if terminal else "paused_gameplay_displayed_progress",
        "agreement_interval_seconds": 0.2,
        "first_agreeing_timestamp_ns": timestamp_ns - 200_000_000,
    }


def test_deadline_blocks_new_key_and_mouse_packets_but_allows_owned_release(trainer):
    clock = Clock()
    budget = trainer.GovernedClock(10, clock=clock, sleep=clock.sleep)
    packets = []
    sender = trainer.DeadlineInput(
        SimpleNamespace(send=lambda events: packets.append(events)), budget
    )
    sender.send([keyboard_event(0x27, True)])
    clock.now = 10
    for events in (
        [keyboard_event(0x27, True)],
        mouse_click_events((30, 30), ClientRect(0, 0, 100, 100)),
    ):
        with pytest.raises(trainer.BudgetExpired):
            sender.send(events)
    sender.send([keyboard_event(0x27, False)])
    sender.send([mouse_click_events((30, 30), ClientRect(0, 0, 100, 100))[-1]])
    assert len(packets) == 3


@pytest.mark.parametrize("duration", [True, 0, -10, float("nan"), float("inf"), 3601])
def test_bad_clock_bounds_are_rejected(trainer, duration):
    with pytest.raises(ValueError):
        trainer.GovernedClock(duration)


def test_checkpoints_retain_actual_crossing_and_detached_pending_state(trainer, tmp_path):
    clock = Clock()
    budget = trainer.GovernedClock(3600, clock=clock, sleep=clock.sleep)
    schedule = trainer.CheckpointSchedule(budget)
    original = {"policy": {"weights": [1]}, "candidate": {"id": "2:3"}}
    clock.now = 302.25
    schedule.pulse(lambda: original, "episode_tick")
    original["policy"]["weights"][0] = 99
    clock.now = 607.5
    schedule.pulse(lambda: original, "parked")
    run = Run(tmp_path)
    schedule.flush(run)
    schedule.flush(run)
    assert len(run.artifacts) == 2
    first = json.loads((tmp_path / "checkpoint-0300s.json").read_text())
    assert first["policy"]["weights"] == [1]
    assert first["target_seconds"] == 300 and first["snapshot_elapsed_seconds"] == 302.25
    assert first["persistence_started_elapsed_seconds"] == 607.5
    assert first["candidate"]["id"] == "2:3"


@pytest.mark.parametrize(
    "case", ["raw_hud", "abort", "shortened", "stale", "boolean", "nan", "wrong_field"]
)
def test_unqualified_scores_cannot_update_search(trainer, case):
    summary = {
        "reason": "episode_time_limit",
        "observed_seconds": 8.1,
        "paused_reading": reading(80),
        "distance": 999,
        "observed_hud_max": 999,
    }
    parked = SimpleNamespace(state="paused", frame=SimpleNamespace(completed_ns=3_000_000_000))
    if case == "raw_hud":
        summary.pop("paused_reading")
    elif case == "abort":
        summary["error"] = "capture failed"
    elif case == "shortened":
        summary["observed_seconds"] = 7.99
    elif case == "stale":
        summary["paused_reading"]["timestamp_ns"] = 0
    elif case == "boolean":
        summary["paused_reading"]["distance_meters"] = True
    elif case == "nan":
        summary["observed_seconds"] = float("nan")
    else:
        summary["paused_reading"]["field"] = "gameplay_hud"
    assert not trainer.eligible_score(summary, parked, [], horizon=8, first_timestamp_ns=1)[
        "eligible"
    ]


def test_measured_zero_and_stable_terminal_are_valid_but_not_interchangeable(trainer):
    parked = SimpleNamespace(state="paused", frame=SimpleNamespace(completed_ns=3_000_000_000))
    summary = {"reason": "episode_time_limit", "observed_seconds": 8, "paused_reading": reading(0)}
    assert (
        trainer.eligible_score(summary, parked, [], horizon=8, first_timestamp_ns=1)["score"] == 0
    )
    summary = {"reason": "observed_revive_offer"}
    parked.state = "tune"
    evidence = [reading(203, terminal=True)]
    assert (
        trainer.eligible_score(summary, parked, evidence, horizon=60, first_timestamp_ns=1)["score"]
        == 203
    )
    evidence[0]["agreement_interval_seconds"] = 0.1
    assert not trainer.eligible_score(summary, parked, evidence, horizon=60, first_timestamp_ns=1)[
        "eligible"
    ]


def runtime_fixture(trainer, clock, *, valid=True, failure=False, features=True):
    current = {"attempt": 0, "parked": True}
    released, codes = [], []

    def observation(state):
        stamp = int(clock() * 1e9)
        return SimpleNamespace(
            state=state,
            frame=CapturedFrame(
                np.zeros((2, 2, 3), dtype=np.uint8),
                stamp,
                stamp,
                "mock",
                ClientRect(0, 0, 2, 2),
                "mock",
                0.0,
            ),
        )

    def reset(**kwargs):
        clock.sleep(0.2)
        current["parked"] = kwargs.get("start_next") is False
        obs = observation("paused" if current["parked"] else "playing")
        kwargs["on_observation"](obs)
        return obs

    def read(*args, **kwargs):
        return {**reading(current["attempt"] * 10, timestamp_ns=int(clock() * 1e9)), "valid": valid}

    runtime = SimpleNamespace(
        adapter=SimpleNamespace(reset=reset, _guard=lambda: None),
        controller=SimpleNamespace(release=lambda: released.append(clock())),
        backend=SimpleNamespace(trace=[]),
        bridge=SimpleNamespace(schema_id="synthetic", schema={"source": "mock_pixels"}),
        terminal=SimpleNamespace(reset=lambda: None, accepted=[]),
        paused_reader=SimpleNamespace(read=read),
    )

    def collect(adapter, controller, backend, bridge, choose, first, *, seconds, deadline, on_tick):
        current["attempt"] += 1
        for _ in range(3):
            on_tick()
            screen = SimpleNamespace(
                schema_id="synthetic",
                supports=lambda names: features,
                values=np.zeros(len(trainer.FEATURE_NAMES)),
                valid=np.ones(len(trainer.FEATURE_NAMES), dtype=bool),
            )
            codes.append(choose(screen))
            clock.sleep(seconds / 3)
        return (
            {
                "reason": "episode_failure" if failure else "episode_time_limit",
                "error": "mock capture loss" if failure else None,
                "observed_seconds": seconds,
                "frames": 3,
                "distance": None,
            },
            [{"step": 0, "elapsed_seconds": 0.1}],
            [],
        )

    def save(run, index, summary, rows, images):
        assert current["parked"] or summary.get("error")
        run.saved.append({"summary": json.loads(json.dumps(summary)), "rows": rows})
        clock.sleep(0.1)

    return runtime, collect, save, codes, released


def test_complete_generation_uses_scoped_scores_and_saves_only_while_parked(trainer, tmp_path):
    clock = Clock()
    runtime, collect, save, codes, released = runtime_fixture(trainer, clock)
    run = Run(tmp_path)
    summary = trainer.governed_train(
        run,
        protocol(trainer),
        runtime,
        trainer.GovernedClock(100, clock=clock, sleep=clock.sleep),
        collect=collect,
        save=save,
    )
    assert summary["eligible_episodes"] == 4 and summary["optimizer_updates"] == 1
    assert summary["error"] is None and len(run.saved) == 4 and released
    assert codes[:3] == [1, 1, 1]
    assert [s["summary"]["distance"] for s in run.saved] == [10, 20, 30, 40]
    state = json.loads((tmp_path / "final-screen-cem-state.json").read_text())
    restored = trainer.EpisodeCEM.from_state_dict(state["optimizer_state"])
    assert restored.generation == 1 and restored.completed_episodes == 4
    assert state["policy"]["schema_id"] == "synthetic"


def test_missing_body_features_release_policy_and_unknown_scores_keep_candidate_pending(
    trainer, tmp_path
):
    clock = Clock()
    runtime, collect, save, codes, _ = runtime_fixture(trainer, clock, valid=False, features=False)
    run = Run(tmp_path)
    summary = trainer.governed_train(
        run,
        protocol(trainer),
        runtime,
        trainer.GovernedClock(100, clock=clock, sleep=clock.sleep),
        collect=collect,
        save=save,
    )
    assert set(codes) == {0}
    assert summary["eligible_episodes"] == summary["optimizer_updates"] == 0
    assert "Three ineligible" in summary["error"]
    assert [s["summary"]["candidate_id"] for s in run.saved] == ["0:0"] * 3


def test_partial_episode_failure_is_saved_and_never_optimized(trainer, tmp_path):
    clock = Clock()
    runtime, collect, save, _, released = runtime_fixture(trainer, clock, failure=True)
    run = Run(tmp_path)
    summary = trainer.governed_train(
        run,
        protocol(trainer),
        runtime,
        trainer.GovernedClock(100, clock=clock, sleep=clock.sleep),
        collect=collect,
        save=save,
    )
    assert len(run.saved) == 1 and run.saved[0]["rows"] and released
    assert not run.saved[0]["summary"]["score_decision"]["eligible"]
    assert summary["eligible_episodes"] == 0 and "mock capture loss" in summary["error"]


def test_insufficient_horizon_waits_without_starting_or_learning(trainer, tmp_path):
    clock = Clock()
    runtime, collect, save, codes, released = runtime_fixture(trainer, clock)
    run = Run(tmp_path)
    summary = trainer.governed_train(
        run,
        protocol(trainer),
        runtime,
        trainer.GovernedClock(10, clock=clock, sleep=clock.sleep),
        collect=collect,
        save=save,
    )
    assert not codes and not run.saved and released
    assert summary["governed_elapsed_at_stop"] == 10 and summary["eligible_episodes"] == 0


def test_checkpoint_phase_crossing_deadline_cannot_update_optimizer(trainer, tmp_path, monkeypatch):
    clock = Clock()
    runtime, collect, save, _, _ = runtime_fixture(trainer, clock)
    run = Run(tmp_path)
    pulse = trainer.CheckpointSchedule.pulse

    def cross_at_update(self, state, phase):
        if phase == "optimizer_update":
            clock.now = 100
        return pulse(self, state, phase)

    monkeypatch.setattr(trainer.CheckpointSchedule, "pulse", cross_at_update)
    summary = trainer.governed_train(
        run,
        protocol(trainer),
        runtime,
        trainer.GovernedClock(100, clock=clock, sleep=clock.sleep),
        collect=collect,
        save=save,
    )
    assert summary["eligible_episodes"] == summary["optimizer_updates"] == 0
    assert len(run.saved) == 1 and run.saved[0]["summary"]["score_decision"]["eligible"]
    assert summary["stop_reason"] == "governed_deadline"


def test_clean_source_and_manifest_hashes_are_mandatory(trainer, tmp_path):
    def dirty(command, **kwargs):
        return SimpleNamespace(
            returncode=0, stdout="abc" if "rev-parse" in command else " M source.py"
        )

    with pytest.raises(RuntimeError, match="clean committed"):
        trainer.require_clean_source(tmp_path, runner=dirty)
    glyph = tmp_path / "digit.png"
    glyph.write_bytes(b"synthetic")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"glyphs": [{"file": glyph.name, "sha256": trainer.sha256_file(glyph)}]})
    )
    assert len(trainer.manifest_dependencies(manifest)) == 2
    glyph.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer.manifest_dependencies(manifest)


@pytest.mark.parametrize("backend", ["dxcam", "mss", "pillow"])
def test_capture_backend_is_explicit_in_nonexecuting_plan(trainer, monkeypatch, capsys, backend):
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "train_screen_cem.py",
            "--hud",
            "unused.json",
            "--result-reader",
            "unused-result.json",
            "--capture-backend",
            backend,
        ],
    )
    trainer.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["capture_backend"] == backend and plan["execute"] is False
