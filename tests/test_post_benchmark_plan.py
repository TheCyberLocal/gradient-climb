"""The post-hour dispatcher must reject ambiguous clocks before launching jobs."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "post_benchmark", Path(__file__).resolve().parents[1] / "scripts/run_post_benchmark.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan():
    return {
        "after_run": "source-run",
        "actions": [
            {
                "id": "train",
                "kind": "train",
                "algorithm": "ppo",
                "seconds": 60,
                "seeds": [101],
                "config": {},
                "experiment": "test",
                "benchmark_class": "cold_start",
            }
        ],
    }


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -1, 0, 3601, "60"])
def test_invalid_budget_is_rejected(value):
    candidate = plan()
    candidate["actions"][0]["seconds"] = value
    with pytest.raises(ValueError):
        MODULE.validate_plan(candidate)


@pytest.mark.parametrize("seeds", [[True], [False], [1, 1], [[1]], [float("nan")]])
def test_invalid_seed_is_rejected(seeds):
    candidate = plan()
    candidate["actions"][0]["seeds"] = seeds
    with pytest.raises(ValueError):
        MODULE.validate_plan(candidate)


def test_expanded_budgets_include_source_fanout():
    candidate = plan()
    candidate["actions"][0]["seeds"] = [101, 102, 103]
    child = {
        **candidate["actions"][0],
        "id": "adapt",
        "source": "action:train",
        "seeds": [1, 2],
        "benchmark_class": "fine_tuning",
    }
    candidate["actions"].append(child)
    assert MODULE.validate_plan(candidate) == {
        "expanded_jobs": 9,
        "requested_learning_seconds": 540,
    }
    candidate["total_requested_learning_seconds"] = 60
    with pytest.raises(ValueError):
        MODULE.validate_plan(candidate)


def test_evaluation_cannot_be_used_as_parent_model():
    candidate = plan()
    candidate["actions"].extend(
        [
            {
                "id": "evaluate",
                "kind": "generalization",
                "source": "action:train",
                "episodes": 20,
                "seed_start": 20000,
            },
            {
                "id": "invalid",
                "kind": "generalization",
                "source": "action:evaluate",
                "episodes": 20,
                "seed_start": 20000,
            },
        ]
    )
    with pytest.raises(ValueError):
        MODULE.validate_plan(candidate)


@pytest.mark.parametrize("field", ["seconds", "seed", "callback", "parent_checkpoint"])
def test_config_cannot_override_job_clock_or_lineage(field):
    candidate = plan()
    candidate["actions"][0]["config"][field] = 1
    with pytest.raises(ValueError):
        MODULE.validate_plan(candidate)


def test_resume_rejects_valid_but_unrelated_training_result():
    action = plan()["actions"][0]
    record = {
        "status": "completed",
        "algorithm": "ppo",
        "seed": 101,
        "experiment_id": "test",
        "parent_run": None,
        "configuration": {"seconds": 60, "benchmark_class": "cold_start"},
    }
    MODULE.validate_job_record(record, action, None, 101)
    record["seed"] = 102
    with pytest.raises(ValueError):
        MODULE.validate_job_record(record, action, None, 101)


def test_resume_rejects_wrong_generalization_checkpoint():
    action = {"kind": "generalization", "episodes": 20, "seed_start": 20000}
    record = {
        "status": "completed",
        "parent_checkpoint": "wrong",
        "configuration": {"episodes": 20, "seed_start": 20000, "generalization": True},
    }
    with pytest.raises(ValueError):
        MODULE.validate_job_record(record, action, {"checkpoint_hash": "expected"}, None)
