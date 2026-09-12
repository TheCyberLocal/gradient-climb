"""Paired headed-overhead protocol: summaries, validation and a sealed record."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from gradientclimb.experiments import load_run, verify_run

SPEC = importlib.util.spec_from_file_location(
    "measure_headed_overhead",
    Path(__file__).resolve().parents[1] / "scripts/measure_headed_overhead.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


HEALTHY_OBSERVER = {
    "frames_rendered": 40,
    "snapshots_received": 3,
    "snapshots_loaded": 3,
    "observer_episodes": 5,
    "error": None,
    "record_error": None,
    "stopped_cleanly": True,
    "sink_errors": [],
    "video_skipped_reason": None,
}


def record(run_id, seed, steps, clock, observer=False, observer_summary=None, **config):
    configuration = {"algorithm": "ppo", "seconds": 60.0, "seed": seed, "num_envs": 8, **config}
    if observer:
        configuration["observer"] = {"enabled": True, "display": "none"}
    summary = {"training_clock_seconds": clock}
    if observer:
        summary["observer"] = (
            dict(HEALTHY_OBSERVER) if observer_summary is None else observer_summary
        )
    return {
        "run_id": run_id,
        "seed": seed,
        "status": "completed",
        "experiment_id": "headed-overhead-pilot",
        "configuration": configuration,
        "environment_steps": steps,
        "summary": summary,
    }


def synthetic_pairs():
    return [
        {
            "seed": 0,
            "off": record("off-0", 0, 1000, 10.0),
            "on": record("on-0", 0, 900, 10.0, True),
        },
        {
            "seed": 1,
            "off": record("off-1", 1, 1200, 10.0),
            "on": record("on-1", 1, 1140, 10.0, True),
        },
        {"seed": 2, "off": record("off-2", 2, 800, 10.0), "on": record("on-2", 2, 760, 10.0, True)},
    ]


def test_paired_differences_summarizes_on_minus_off():
    result = MODULE.paired_differences(synthetic_pairs())
    steps = result["environment_steps_difference"]
    assert steps["n"] == 3 and steps["mean"] == pytest.approx(-66.6667, rel=1e-4)
    assert steps["ci95_low"] is not None and steps["ci95_high"] is not None
    rate = result["environment_steps_per_second_difference"]
    assert rate["n"] == 3 and rate["mean"] == pytest.approx(-6.66667, rel=1e-4)
    relative = result["relative_environment_steps_change"]
    assert relative["mean"] == pytest.approx((-0.1 - 0.05 - 0.05) / 3)
    assert result["sign_convention"] == MODULE.SIGN_CONVENTION
    assert [row["off_run"] for row in result["pairs"]] == ["off-0", "off-1", "off-2"]
    assert result["pairs"][0]["environment_steps_difference"] == -100
    assert result["pairs"][0]["observer"]["frames_rendered"] == 40
    assert result["pairs"][0]["observer"]["stopped_cleanly"] is True
    # The CI label must describe a seed-level paired bootstrap, never the
    # episode-level label summarize() uses for one policy.
    assert result["ci_method"] == MODULE.CI_METHOD
    for key in (
        "environment_steps_difference",
        "environment_steps_per_second_difference",
        "relative_environment_steps_change",
    ):
        assert result[key]["ci_method"] == MODULE.CI_METHOD
        assert "one trained policy" not in result[key]["ci_method"]
        assert result[key]["unit_of_analysis"] == "paired seed (on minus off)"
    assert json.dumps(result, allow_nan=False)


def test_pair_validation_rejects_config_mismatch_and_wrong_seed():
    off, on = record("off", 0, 10, 1.0), record("on", 0, 9, 1.0, True)
    MODULE.validate_pair(off, on)
    with pytest.raises(ValueError):
        MODULE.validate_pair(off, record("on", 1, 9, 1.0, True))
    with pytest.raises(ValueError):
        MODULE.validate_pair(off, record("on", 0, 9, 1.0, True, num_envs=16))
    with pytest.raises(ValueError):
        MODULE.validate_pair(record("off", 0, 10, 1.0, True), on)
    with pytest.raises(ValueError):
        MODULE.validate_pair(off, record("on", 0, 9, 1.0, False))
    incomplete = record("on", 0, 9, 1.0, True)
    incomplete["status"] = "running"
    with pytest.raises(ValueError):
        MODULE.validate_pair(off, incomplete)
    with pytest.raises(ValueError):
        MODULE.paired_differences([])


@pytest.mark.parametrize(
    "broken",
    [
        {"error": "RuntimeError: boom"},
        {"record_error": "ValueError: nan"},
        {"stopped_cleanly": False},
        {"sink_errors": ["tk: Tk window closed by the user"]},
        {"frames_rendered": 0},
        {"snapshots_received": 0},
        None,
    ],
)
def test_pair_validation_requires_a_healthy_observer(broken):
    off = record("off", 0, 10, 1.0)
    healthy = record("on", 0, 9, 1.0, True)
    MODULE.validate_pair(off, healthy)
    summary = None if broken is None else {**HEALTHY_OBSERVER, **broken}
    on = record("on", 0, 9, 1.0, True, observer_summary=summary)
    if broken is None:
        del on["summary"]["observer"]
    with pytest.raises(ValueError, match="observer"):
        MODULE.validate_pair(off, on)
    with pytest.raises(ValueError, match="observer"):
        MODULE.paired_differences([{"seed": 0, "off": off, "on": on}])


def test_dry_run_prints_plan_without_training(tmp_path, capsys):
    result = MODULE.main(["--root", str(tmp_path), "--seeds", "1", "2", "--seconds", "5"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry-run" and printed["runs"] == 0
    assert printed["plan"] == result["plan"]
    assert result["plan"]["seeds"] == [1, 2] and result["plan"]["seconds"] == 5.0
    assert result["plan"]["order"] == {"1": ["off", "on"], "2": ["on", "off"]}
    assert result["plan"]["observer"]["display"] == "none"
    assert result["plan"]["config"] == {"num_envs": 64}
    assert result["plan"]["ci_method"] == MODULE.CI_METHOD
    assert not (tmp_path / "runs").exists()
    with pytest.raises(ValueError):
        MODULE.main(["--root", str(tmp_path), "--seeds", "41000", "--seconds", "5"])


def test_envs_flag_never_silently_overridden_by_config(tmp_path):
    config_path = tmp_path / "eight.json"
    config_path.write_text(json.dumps({"num_envs": 8}))
    base = ["--root", str(tmp_path), "--seeds", "1", "--seconds", "5", "--config"]
    with pytest.raises(ValueError, match="conflicts"):
        MODULE.main([*base, str(config_path), "--envs", "64"])
    result = MODULE.main([*base, str(config_path)])
    assert result["plan"]["num_envs"] == 8 and result["plan"]["config"] == {"num_envs": 8}
    result = MODULE.main([*base, str(config_path), "--envs", "8"])
    assert result["plan"]["num_envs"] == 8
    config_path.write_text(json.dumps({"observer": {"display": "none"}}))
    with pytest.raises(ValueError):
        MODULE.main([*base, str(config_path)])


def test_execution_directories_are_unique_per_execution(tmp_path):
    digest = "a" * 64
    first = MODULE.execution_directory(tmp_path, digest)
    second = MODULE.execution_directory(tmp_path, digest)
    assert first != second and first.is_dir() and second.is_dir()
    assert first.parent == second.parent == tmp_path / "headed-overhead"
    assert digest[:12] in first.name and digest[:12] in second.name


def test_in_process_pair_seals_verifiable_protocol_run(tmp_path):
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "rollout_steps": 16,
                "hidden_size": 16,
                "minibatch_size": 64,
                "epochs": 2,
                "max_steps": 30,
                "checkpoint_times": [],
            }
        )
    )
    result = MODULE.main(
        [
            "--root",
            str(tmp_path),
            "--execute",
            "--in-process",
            "--seeds",
            "3",
            "--seconds",
            "2",
            "--envs",
            "8",
            "--config",
            str(config_path),
            "--observer-interval",
            "0.5",
            "--observer-fps",
            "100",
        ]
    )
    assert result["mode"] == "executed" and result["n_pairs"] == 1
    protocol = load_run(tmp_path, result["protocol_run"])
    assert verify_run(tmp_path, protocol["run_id"])["valid"]
    assert protocol["status"] == "completed"
    assert protocol["configuration"]["protocol"] == MODULE.PROTOCOL
    # A measurement record, never a PPO/CEM training run.
    assert protocol["algorithm"] == MODULE.PROTOCOL_ALGORITHM
    assert protocol["algorithm"] not in {"ppo", "cem"}
    assert protocol["metadata"]["arm_algorithm"] == "ppo"
    assert protocol["metadata"]["pair_seeds"] == [3]
    summary = protocol["summary"]
    assert summary["protocol"] == MODULE.PROTOCOL
    assert summary["ci_method"] == MODULE.CI_METHOD
    assert summary["environment_steps_difference"]["ci_method"] == MODULE.CI_METHOD
    assert summary["environment_steps_difference"]["n"] == 1
    assert summary["environment_steps_per_second_difference"]["n"] == 1
    pair = summary["pairs"][0]
    assert pair["seed"] == 3
    off, on = load_run(tmp_path, pair["off_run"]), load_run(tmp_path, pair["on_run"])
    assert "observer" not in off["configuration"] and "observer" in on["configuration"]
    assert on["summary"]["observer"]["frames_rendered"] > 0
    assert (
        pair["environment_steps_difference"] == on["environment_steps"] - off["environment_steps"]
    )
    for run_id in (pair["off_run"], pair["on_run"]):
        assert verify_run(tmp_path, run_id)["valid"]
    assert (
        on["summary"]["observer"]["stopped_cleanly"] and on["summary"]["observer"]["error"] is None
    )
    directory = Path(result["state_directory"])
    assert directory.parent == tmp_path / "headed-overhead"
    assert (directory / "plan.json").exists() and (directory / "outcome.json").exists()
    outcome = json.loads((directory / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["protocol_run"] == protocol["run_id"]
    assert outcome["plan_sha256"] == result["plan_sha256"]
    assert protocol["metadata"]["state_directory"] == str(directory)
    assert any(a["kind"] == "paired_runs" for a in protocol["artifact_manifest"])
