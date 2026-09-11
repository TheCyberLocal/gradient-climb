"""Scientific pairing uses explicit provenance and never silently drops episodes."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "report_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_research.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evaluation():
    return {
        "run_id": "before",
        "checkpoint_hash": "parent-hash",
        "results": {
            "scope": "uncalibrated_simulator",
            "simulator_version": "test-1",
            "calibration_version": "uncalibrated",
            "profile": "default",
            "terrain": "train",
            "max_steps": 1000,
            "deterministic": True,
            "seeds": [10000, 10001, 10002],
            "episodes": [
                {"seed": seed, "distance": value}
                for seed, value in zip([10000, 10001, 10002], [10, 20, 30], strict=True)
            ],
        },
    }


def test_pairing_preserves_direction_and_episode_ids():
    before = evaluation()
    after = copy.deepcopy(before)
    after.update(run_id="after", checkpoint_hash="child-hash")
    after["results"]["episodes"].reverse()
    for episode in after["results"]["episodes"]:
        episode["distance"] += 5
    paired = MODULE.paired_evaluations(before, after)
    assert paired["difference_after_minus_before"]["distance_difference"]["mean"] == 5
    assert paired["seeds"] == [10000, 10001, 10002]
    assert paired["before_checkpoint_hash"] == "parent-hash"


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_steps", 999),
        ("simulator_version", "changed"),
        ("calibration_version", "calibrated"),
        ("deterministic", False),
        ("terrain", "rough"),
    ],
)
def test_pairing_rejects_scope_changes(field, value):
    before, after = evaluation(), evaluation()
    after["results"][field] = value
    with pytest.raises(ValueError):
        MODULE.paired_evaluations(before, after)


def test_pairing_rejects_duplicate_episode_seed():
    before, after = evaluation(), evaluation()
    after["results"]["episodes"][0]["seed"] = 10001
    with pytest.raises(ValueError):
        MODULE.paired_evaluations(before, after)


def test_missing_distance_is_not_silently_dropped():
    measured = evaluation()
    measured["results"]["episodes"][0]["distance"] = float("nan")
    with pytest.raises(ValueError):
        MODULE.distance_result(measured)
