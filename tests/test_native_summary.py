import pytest

from gradientclimb.evaluation.native_summary import describe_native_attempts


def episode(control, score, classification="success_natural_scored"):
    return {"distance": score, "attempt": {"policy": control, "classification": classification}}


def test_mixed_controls_keep_missing_scores_and_have_no_trained_policy_interval():
    result = describe_native_attempts(
        [episode("gas", 200), episode("random", 4), episode("gas", None, "success_unscored")],
        4,
    )
    assert result["not_attempted"] == 1
    assert result["observed_scores"] == 2 and result["unknown_scores"] == 1
    assert result["by_control"]["gas"]["attempts"] == 2
    assert result["by_control"]["gas"]["observed_result_distance"]["mean"] == 200
    assert result["by_control"]["gas"]["unknown_scores"] == 1
    assert result["by_control"]["random"]["observed_result_distance"]["mean"] == 4
    assert result["observed_result_distance"]["ci95_low"] is None
    assert "trained policy" not in result["observed_result_distance"]["ci_method"]
    assert not result["qualification_inferred"] and not result["learning_speed_inferred"]


def test_observed_score_survives_readiness_failure_without_becoming_success():
    row = episode("gas", 205, "failure")
    row["park_error"] = "Synthetic failure after accepted terminal observation"
    result = describe_native_attempts([row], 1)
    assert result["observed_result_distance"]["median"] == 205
    assert result["classifications"] == {"failure": 1}


def test_empty_and_all_unknown_attempts_preserve_missingness():
    assert describe_native_attempts([], 12)["not_attempted"] == 12
    result = describe_native_attempts([episode("gas", None, "failure")], 12)
    assert result["observed_result_distance"] is None
    assert result["unknown_scores"] == 1 and result["not_attempted"] == 11


@pytest.mark.parametrize("score", [True, -1, float("nan"), float("inf"), "200"])
def test_invalid_observed_distances_are_not_silently_cast(score):
    with pytest.raises(ValueError, match="Observed scores"):
        describe_native_attempts([episode("gas", score)], 1)


def test_cannot_hide_an_observed_attempt_by_reducing_requested_count():
    with pytest.raises(ValueError, match="Requested count"):
        describe_native_attempts([episode("gas", 10)], 0)


def test_attempt_with_failed_episode_artifact_write_is_not_called_unattempted():
    first = episode("gas", 10)
    first["attempt"]["index"] = 0
    second = {"index": 1, "policy": "random", "classification": "failure"}
    result = describe_native_attempts([first], 3, attempt_outcomes=[first["attempt"], second])
    assert result["attempts"] == 2 and result["not_attempted"] == 1
    assert result["unknown_scores"] == result["attempts_missing_episode_summary"] == 1
    assert result["by_control"]["random"]["classifications"] == {"failure": 1}


def test_conflicting_or_duplicate_attempt_records_cannot_be_pooled():
    row = episode("gas", 10)
    row["attempt"]["index"] = 0
    with pytest.raises(ValueError, match="unique"):
        describe_native_attempts([row, row], 2, attempt_outcomes=[row["attempt"]])
    with pytest.raises(ValueError, match="disagree"):
        describe_native_attempts(
            [row], 1, attempt_outcomes=[{**row["attempt"], "policy": "random"}]
        )
