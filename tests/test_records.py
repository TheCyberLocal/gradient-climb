"""Scientific integrity, provenance and independent dashboard reader tests."""

import json

import pytest
from pydantic import ValidationError

from gradientclimb.artifacts import hash_config, sha256_file
from gradientclimb.experiments import (
    RunRecorder,
    connect_database,
    list_runs,
    load_run,
    query_metrics,
    run_synthetic,
    verify_run,
)
from gradientclimb.experiments.schemas import SCHEMA_MODELS, MetricRecord


def test_stable_configuration_hash():
    assert hash_config({"b": [2, 3], "a": 1}) == hash_config({"a": 1, "b": [2, 3]})
    assert hash_config({"a": 1}) != hash_config({"a": 2})
    with pytest.raises(ValueError):
        hash_config({"invalid": float("nan")})


def test_full_run_roundtrip_and_artifact_snapshot(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"model parameters\x00\x01")
    config = {"learning_rate": 0.001, "nested": {"a": 1}}
    with RunRecorder(tmp_path, "EXP-001", config, seed=17, telemetry_interval_seconds=0) as run:
        config["nested"]["a"] = 999
        run.metric("reward", 12.5, step=42, split="validation")
        artifact = run.register_artifact(checkpoint, "checkpoint")
        checkpoint.write_bytes(b"different model")
        run.evaluation({"results": {"distance": 17.2}, "seed": 123, "episodes": 3})
        run.trajectory(
            {
                "episode_id": "one",
                "step": 0,
                "gas": True,
                "brake": True,
                "action_duration_seconds": 0.1,
            }
        )
        result = run.finalize(environment_steps=42, episodes=1, checkpoint_hash=artifact["sha256"])
    assert result["configuration"]["nested"]["a"] == 1
    assert result["seed"] == 17
    assert result["machine_fingerprint"]
    assert result["git_sha"]
    assert result["python_version"]
    assert result["duration"] > 0
    assert result["environment_steps_per_second"] > 0
    assert result["evaluation_results"][0]["results"]["distance"] == 17.2
    assert sha256_file(run.directory / artifact["path"]) == artifact["sha256"]
    assert verify_run(tmp_path, run.run_id)["valid"]
    assert load_run(tmp_path, run.run_id) == result
    assert list_runs(tmp_path)[0]["run_id"] == run.run_id
    metrics = query_metrics(tmp_path, run.run_id)
    assert len(metrics) == 1
    assert metrics[0]["value"] == 12.5
    assert metrics[0]["dimensions"] == {"split": "validation"}
    with connect_database(tmp_path) as connection:
        assert connection.execute("SELECT gas AND brake FROM trajectories").fetchone()[0]
        assert connection.execute("SELECT count(*) FROM telemetry").fetchone()[0] >= 2


def test_schema_rejects_invalid_values_and_unknown_fields():
    with pytest.raises(ValidationError):
        MetricRecord(
            run_id="x",
            name="bad",
            value=float("nan"),
            step=-1,
            elapsed_seconds=0,
            timestamp="2026-01-01T00:00:00Z",
            surprise=True,
        )
    assert set(SCHEMA_MODELS) == {
        "experiments",
        "runs",
        "metrics",
        "system_telemetry",
        "evaluations",
        "trajectories",
        "artifacts",
        "model_lineage",
        "benchmarks",
        "simulator_calibration",
    }
    for model in SCHEMA_MODELS.values():
        assert model.model_json_schema()["additionalProperties"] is False


def test_finalized_recorder_rejects_all_mutations(tmp_path):
    with RunRecorder(tmp_path, "immutable", {}, telemetry_interval_seconds=0) as run:
        pass
    with pytest.raises(RuntimeError, match="immutable"):
        run.metric("late", 1)
    with pytest.raises(RuntimeError, match="immutable"):
        run.evaluation({"quality": 1})
    with pytest.raises(RuntimeError, match="immutable"):
        run.finalize()
    before = (run.directory / "run.json").read_bytes()
    assert load_run(tmp_path, run.run_id)["status"] == "completed"
    assert before == (run.directory / "run.json").read_bytes()


def test_hash_verification_detects_tampering(tmp_path):
    result = run_synthetic(tmp_path, seed=0, gain=0.1, steps=4)
    path = tmp_path / "runs" / result["run_id"] / "config.json"
    path.write_text('{"tampered": true}', encoding="utf-8")
    verified = verify_run(tmp_path, result["run_id"])
    assert not verified["valid"]
    assert any("config.json" in error for error in verified["errors"])


def test_nested_metadata_and_evaluations_are_input_snapshots(tmp_path):
    metadata = {"layers": [8, 4]}
    result = {"results": {"quality": {"mean": 2.0}}}
    artifact_metadata = {"history": [1, 2]}
    checkpoint = tmp_path / "snapshot.bin"
    checkpoint.write_bytes(b"snapshot")
    with RunRecorder(
        tmp_path, "snapshot", {}, policy_architecture=metadata, telemetry_interval_seconds=0
    ) as run:
        run.evaluation(result)
        run.register_artifact(checkpoint, "checkpoint", artifact_metadata)
        metadata["layers"].append(99)
        result["results"]["quality"]["mean"] = 999
        artifact_metadata["history"].append(99)
    record = load_run(tmp_path, run.run_id)
    assert record["policy_architecture"] == {"layers": [8, 4]}
    assert record["evaluation_results"][0]["results"]["quality"]["mean"] == 2.0
    assert record["artifact_manifest"][0]["metadata"] == {"history": [1, 2]}


def test_failed_experiment_is_retained(tmp_path):
    with (
        pytest.raises(RuntimeError, match="test failure"),
        RunRecorder(tmp_path, "negative-result", {}, telemetry_interval_seconds=0) as run,
    ):
        run.metric("loss", 99)
        raise RuntimeError("test failure")
    record = load_run(tmp_path, run.run_id)
    assert record["status"] == "failed"
    assert record["summary"]["error"] == "test failure"
    assert verify_run(tmp_path, run.run_id)["valid"]


def test_independent_readers_see_live_and_sealed_metrics(tmp_path):
    with RunRecorder(tmp_path, "live", {}, telemetry_interval_seconds=0) as run:
        run.metric("live", 1, step=0)
        assert query_metrics(tmp_path)[0]["value"] == 1
        with connect_database(tmp_path) as reader:
            run.metric("live", 2, step=1)
            # The reader owns a consistent snapshot; a new reader gets new data.
            assert reader.execute("SELECT count(*) FROM metrics").fetchone()[0] == 1
            assert len(query_metrics(tmp_path)) == 2
    assert len(query_metrics(tmp_path)) == 2


def test_two_synthetic_gains_are_distinguishable_and_reproducible(tmp_path):
    slow = run_synthetic(tmp_path, seed=7, gain=0.05, steps=10)
    fast = run_synthetic(tmp_path, seed=7, gain=0.4, steps=10)
    repeat = run_synthetic(tmp_path, seed=7, gain=0.4, steps=10)
    assert fast["summary"]["final_quality"] > slow["summary"]["final_quality"]
    assert repeat["summary"]["final_quality"] == fast["summary"]["final_quality"]
    assert len(list_runs(tmp_path)) == 3
    assert len(query_metrics(tmp_path)) == 60
    assert all(verify_run(tmp_path, record["run_id"])["valid"] for record in [slow, fast, repeat])


def test_empty_database_and_invalid_run_path(tmp_path):
    assert list_runs(tmp_path) == []
    assert query_metrics(tmp_path) == []
    with pytest.raises(ValueError):
        load_run(tmp_path, "../../etc")
    with pytest.raises(ValueError):
        RunRecorder(tmp_path, "invalid", {}, seed=-1)
    assert list_runs(tmp_path) == []


def test_exported_schemas_match_models():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "schemas"
    for name, model in SCHEMA_MODELS.items():
        assert json.loads((root / f"{name}.schema.json").read_text()) == model.model_json_schema()
