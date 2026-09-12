import builtins
import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradientclimb.algorithms.progress import TrainingProgress
from gradientclimb.experiments.command_clock import clock_receipt, validate_clock


def test_receipt_declares_early_entry_boundary_and_rejects_future_origin():
    receipt = clock_receipt(time.monotonic(), time.time())
    assert receipt["boundary"] == "python_entry_before_project_imports"
    assert "project_and_framework_imports" in receipt["included"]
    with pytest.raises(ValueError, match="prior finite"):
        validate_clock({**receipt, "started_monotonic": time.monotonic() + 100})
    with pytest.raises(ValueError, match="another process"):
        validate_clock({**receipt, "process_id": receipt["process_id"] + 10000})


def test_parallel_episodes_count_experience_and_inflight_decisions_separately():
    progress = TrainingProgress()
    progress.configure_simulator(SimpleNamespace(action_duration=0.06, substeps=3))
    progress.decisions_completed(1000)
    progress.simulator_step_completed(1000, 1000)
    progress.decisions_completed(1000)  # Next env operation failed before it returned.
    measured = progress.snapshot()["experience"]
    assert measured["simulator_episodes"] == measured["simulator_transitions"] == 1000
    assert measured["physics_steps"] == 3000
    assert measured["policy_decisions"] == 2000
    assert measured["simulator_seconds"] == 60
    assert measured["actor_rendered_frames"] == measured["real_game_interaction_seconds"] == 0


def test_spent_command_budget_prevents_training_but_retains_publication_receipts(tmp_path):
    from gradientclimb.benchmarks.runner import run_training
    from gradientclimb.experiments import verify_run

    receipt = clock_receipt(time.monotonic() - 20, time.time() - 20)
    result = run_training(
        tmp_path,
        "cem",
        1,
        31,
        {"population": 4, "episodes_per_candidate": 1, "max_steps": 1},
        command_clock=receipt,
    )
    assert result["environment_steps"] == result["optimizer_updates"] == 0
    assert result["summary"]["model_config"]["clock_contract"] == "command-clock-3.0"
    folder = tmp_path / "runs" / result["run_id"]
    costs = [json.loads(x.read_text()) for x in folder.glob("policy-*-cost.json")]
    assert costs and all(not x["within_declared_budget"] for x in costs)
    assert all(x["command_elapsed_seconds"] >= 20 for x in costs)
    assert verify_run(tmp_path, result["run_id"])["valid"]


def test_outer_receipt_survives_project_import_failure_without_overwrite(tmp_path, monkeypatch):
    target = tmp_path / "entry.json"
    script = Path(__file__).resolve().parents[1] / "scripts/train_with_command_clock.py"
    original = builtins.__import__

    def failed_import(name, *args, **kwargs):
        if name.startswith("gradientclimb"):
            assert target.is_file()
            raise ImportError("injected project import failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", [str(script), "--entry-receipt", str(target), "train"])
    monkeypatch.setattr(builtins, "__import__", failed_import)
    with pytest.raises(ImportError, match="injected"):
        runpy.run_path(str(script), run_name="__main__")
    raw = target.read_bytes()
    assert json.loads(raw)["stage"] == "entry_recorded_before_project_imports"
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script), run_name="__main__")
    assert target.read_bytes() == raw


@pytest.mark.parametrize("seconds", [float("inf"), float("nan"), -1, 0])
def test_nonfinite_or_nonpositive_budget_never_creates_run(tmp_path, seconds):
    from gradientclimb.benchmarks.runner import run_training

    with pytest.raises(ValueError, match="finite"):
        run_training(tmp_path, "cem", seconds, 0)
    assert not (tmp_path / "runs").exists()
