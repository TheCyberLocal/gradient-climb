"""Failure boundaries preserve completed work, uncertain operations, and prior checkpoints."""

import json

import pytest
import torch

from gradientclimb.algorithms import ActorCritic, checkpoints, load_policy, ppo
from gradientclimb.benchmarks.runner import run_training
from gradientclimb.experiments import RunRecorder, list_runs, load_run, recorder, verify_run
from gradientclimb.visualization import observer
from gradientclimb.visualization.sinks import NullSink, SinkClosed

TINY = {
    "num_envs": 2,
    "rollout_steps": 2,
    "hidden_size": 8,
    "minibatch_size": 2,
    "epochs": 1,
    "max_steps": 1,
    "checkpoint_times": [],
}


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interruption_preserves_completed_steps_updates_checkpoint_and_observer(
    tmp_path, monkeypatch, failure
):
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    factory = ppo.make_environment

    def failing_factory(**kwargs):
        env = factory(**kwargs)
        original = env.step
        calls = 0

        def step(actions):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise failure("injected environment fault")
            return original(actions)

        env.step = step
        return env

    monkeypatch.setattr(ppo, "make_environment", failing_factory)
    with pytest.raises(failure, match="injected environment fault"):
        run_training(tmp_path, "ppo", 10, 0, TINY, observer={"display": "none"})
    record = list_runs(tmp_path)[0]
    assert record["status"] == ("cancelled" if failure is KeyboardInterrupt else "failed")
    assert record["environment_steps"] == record["training_steps"] == 6
    assert record["episodes"] == 6 and record["optimizer_updates"] > 0
    assert record["summary"]["accounting"]["in_flight_operation"] == "environment_step"
    assert record["summary"]["accounting"]["counter_basis"] == "observed_completed_operations"
    assert record["summary"]["observer"]["thread_joined"]
    assert "injected environment fault" in record["summary"]["traceback"]
    directory = tmp_path / "runs" / record["run_id"]
    model = load_policy(directory / record["summary"]["last_valid_checkpoint"])
    assert model.training_state["environment_steps"] == 6
    assert model.checkpoint_manifest["continuation"] == "warm_start"
    assert model.checkpoint_manifest["exact_resume"] is False
    assert verify_run(tmp_path, record["run_id"])["valid"]


def test_completed_environment_step_count_survives_bootstrap_inference_failure(
    tmp_path, monkeypatch
):
    original = ActorCritic.distribution_value
    calls = 0

    def fail_after_environment(self, observations):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("bootstrap inference failed")
        return original(self, observations)

    monkeypatch.setattr(ActorCritic, "distribution_value", fail_after_environment)
    with pytest.raises(RuntimeError, match="bootstrap inference failed"):
        run_training(tmp_path, "ppo", 10, 0, TINY)
    record = list_runs(tmp_path)[0]
    assert record["environment_steps"] == record["episodes"] == 2
    assert record["optimizer_updates"] == 0


def test_partially_mutated_optimizer_retains_previous_checkpoint(tmp_path, monkeypatch):
    original = torch.optim.Adam.step

    def partial_step(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt("interrupted inside optimizer")

    monkeypatch.setattr(torch.optim.Adam, "step", partial_step)
    with pytest.raises(KeyboardInterrupt, match="inside optimizer"):
        run_training(tmp_path, "ppo", 10, 0, TINY)
    record = list_runs(tmp_path)[0]
    assert record["environment_steps"] == 4 and record["optimizer_updates"] == 0
    assert record["summary"]["accounting"]["in_flight_operation"] == "optimizer_step"
    assert "may be partial" in record["summary"]["checkpoint_recovery"]
    saved = [a for a in record["artifact_manifest"] if a["kind"] == "checkpoint"]
    assert len(saved) == 1 and saved[0]["metadata"]["stage"] == "initial"
    model = load_policy(tmp_path / "runs" / record["run_id"] / saved[0]["path"])
    assert model.training_state["optimizer_updates"] == 0
    assert verify_run(tmp_path, record["run_id"])["valid"]


@pytest.mark.parametrize("stage", ["serialize", "publish"])
def test_atomic_checkpoint_failure_keeps_previous_bytes(tmp_path, monkeypatch, stage):
    model = ActorCritic(4, 8)
    path = tmp_path / "policy.pt"
    model.save(path)
    before = path.read_bytes()

    def partial_save(value, stream):
        stream.write(b"partial checkpoint")
        raise OSError("disk pressure")

    def fail_publish(*args):
        raise KeyboardInterrupt("between checkpoint stages")

    if stage == "serialize":
        monkeypatch.setattr(torch, "save", partial_save)
    else:
        monkeypatch.setattr(checkpoints.os, "replace", fail_publish)
    with pytest.raises((OSError, KeyboardInterrupt)):
        model.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))
    load_policy(path)


def test_v1_checkpoint_stays_readable_with_explicit_legacy_manifest(tmp_path):
    model = ActorCritic(4, 8)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "ppo",
            "config": {},
            "observation_dim": 4,
            "hidden_size": 8,
            "model_state": model.state_dict(),
            "training_state": {},
        },
        path,
    )
    loaded = load_policy(path)
    assert loaded.checkpoint_manifest == {
        "continuation": "legacy_warm_start",
        "exact_resume": False,
    }


def test_checkpoint_registration_fault_keeps_source_exception_and_valid_checkpoint(
    tmp_path, monkeypatch
):
    original = RunRecorder.register_artifact

    def fail_final_registration(self, path, kind, metadata=None):
        if kind == "checkpoint" and metadata.get("stage") == "final":
            raise OSError("interrupted between publish and registration")
        return original(self, path, kind, metadata)

    monkeypatch.setattr(RunRecorder, "register_artifact", fail_final_registration)
    with pytest.raises(OSError, match="between publish and registration"):
        run_training(tmp_path, "ppo", 0.2, 0, TINY)
    record = list_runs(tmp_path)[0]
    assert record["environment_steps"] > 0
    assert record["summary"]["failure_phase"] == "checkpoint_registration"
    checkpoints_saved = [a for a in record["artifact_manifest"] if a["kind"] == "checkpoint"]
    assert [a["metadata"]["stage"] for a in checkpoints_saved] == ["initial", "interrupted"]
    assert verify_run(tmp_path, record["run_id"])["valid"]


def test_cem_interruption_uses_same_completed_operation_accounting(tmp_path, monkeypatch):
    from gradientclimb.algorithms import cem

    original = cem.make_environment

    def factory(**kwargs):
        env = original(**kwargs)
        step = env.step
        calls = 0

        def failing_step(actions):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("CEM environment failure")
            return step(actions)

        env.step = failing_step
        return env

    monkeypatch.setattr(cem, "make_environment", factory)
    with pytest.raises(RuntimeError, match="CEM environment failure"):
        run_training(
            tmp_path,
            "cem",
            10,
            0,
            {
                "population": 4,
                "episodes_per_candidate": 1,
                "max_steps": 1,
                "checkpoint_times": [],
            },
        )
    record = list_runs(tmp_path)[0]
    assert record["environment_steps"] == 4 and record["episodes"] == 4
    assert record["optimizer_updates"] == 1
    assert record["summary"]["accounting"]["in_flight_operation"] == "environment_step"
    assert verify_run(tmp_path, record["run_id"])["valid"]


def snapshot(steps=8):
    return {
        "environment_steps": steps,
        "training_steps": steps,
        "episodes": 2,
        "optimizer_updates": 3,
        "training_clock_seconds": 1.0,
        "phase": "inference",
        "counter_basis": "observed_completed_operations",
        "in_flight_operation": None,
        "checkpoint_safe": True,
    }


def test_partial_progress_tail_is_lower_bound_and_remains_unsealed(tmp_path):
    run = RunRecorder(tmp_path, "fault-test", {}, telemetry_interval_seconds=0)
    run.record_progress(snapshot())
    run._streams["progress"].write('{"environment_steps":999')
    run._streams["progress"].flush()
    loaded = load_run(tmp_path, run.run_id)
    assert loaded["environment_steps"] == 8
    assert loaded["summary"]["accounting"]["counter_basis"].endswith("lower_bound")
    assert not verify_run(tmp_path, run.run_id)["valid"]
    run._abandon()


def test_disk_failure_during_finalization_preserves_primary_exception(tmp_path, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError("no disk space for parquet")

    monkeypatch.setattr(recorder.pq, "ParquetWriter", disk_full)
    with (
        pytest.raises(RuntimeError, match="original learner failure") as caught,
        RunRecorder(tmp_path, "fault-test", {}, telemetry_interval_seconds=0) as run,
    ):
        run.record_progress(snapshot())
        raise RuntimeError("original learner failure")
    assert "finalization also failed" in " ".join(caught.value.__notes__)
    assert recorder.read_progress(run.directory)["environment_steps"] == 8
    assert all(stream.closed for stream in run._streams.values())
    assert not verify_run(tmp_path, run.run_id)["valid"]
    assert json.loads((run.directory / "failure.json").read_text())["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "fault", [SinkClosed("window closed"), RuntimeError("video writer failed")]
)
def test_optional_observer_failure_does_not_erase_learner(tmp_path, monkeypatch, fault):
    class FaultSink(NullSink):
        def write(self, frame):
            raise fault

    monkeypatch.setattr(observer, "build_sinks", lambda config: ([FaultSink()], None))
    record = run_training(tmp_path, "ppo", 0.3, 0, TINY, observer={"display": "none"})
    assert record["status"] == "completed" and record["environment_steps"] > 0
    assert record["summary"]["learner_status"] == "completed"
    assert record["summary"]["observer"]["status"] == "degraded"
    assert verify_run(tmp_path, record["run_id"])["valid"]
