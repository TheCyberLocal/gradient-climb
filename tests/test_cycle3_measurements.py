"""Adversarial contracts, without live input or scientific qualification claims."""

import itertools

import pytest

from gradientclimb.experiments.cycle3_measurements import (
    EligibilityPolicy,
    EpisodeEndpoint,
    GameplayOutcome,
    Readiness,
    RecoveryWindow,
    SafetyEvent,
    ScoreIncrement,
    ScoreLedger,
    SessionSafety,
    assess_eligibility,
    lexicographic_key,
    recovery_score_credit,
)


def endpoint(reason="driver_down", **changes):
    return EpisodeEndpoint(
        **{
            "reason": reason,
            "horizon_seconds": 60,
            "elapsed_seconds": 40,
            "natural_terminal_observed": reason
            in {"driver_down", "out_of_fuel", "natural_unlabeled"},
            **changes,
        }
    )


def gameplay(**changes):
    return GameplayOutcome(
        **{
            "episode_id": "e1",
            "endpoint": endpoint(),
            "started_ns": 100,
            "boundary_ns": 500,
            "distance_m": 243,
            "distance_source": "stable_terminal_distance",
            "distance_evidence_id": "sealed-terminal-pair",
            "observation_coverage": 0.9,
            **changes,
        }
    )


def safety(**changes):
    return SessionSafety(
        **{
            "monitored_from_ns": 0,
            "monitored_through_ns": 1000,
            "assessed_at_ns": 1000,
            "inputs_released": True,
            **changes,
        }
    )


def eligibility_policy(**changes):
    return EligibilityPolicy(
        **{
            "policy_id": "learning-test-3.0",
            "allow_censored_learning": False,
            "require_readiness_for_learning": False,
            "allow_unlabeled_natural_learning": False,
            "minimum_observation_coverage": 0.8,
            **changes,
        }
    )


def readiness(**changes):
    return Readiness(**{"ready": True, "state": "tune", "checked_at_ns": 900, **changes})


@pytest.mark.parametrize("reason", ["driver_down", "out_of_fuel", "natural_unlabeled"])
def test_natural_failure_at_horizon_is_termination_not_censoring(reason):
    # Evidence of a natural failure dominates the fact that the clock also hit 60 s.
    end = endpoint(reason, elapsed_seconds=60)
    assert end.terminated is True and end.right_censored is False
    assert not end.truncated and not end.interrupted
    assert end.outcome_flags()["terminated"] is True


@pytest.mark.parametrize("reason", ["horizon", "stall_limit", "frame_limit"])
def test_governed_stops_are_censored_not_crashes(reason):
    end = endpoint(reason, elapsed_seconds=60)
    assert end.truncated and end.right_censored and not end.terminated
    assert not end.interrupted


@pytest.mark.parametrize("reason", ["administrative_interrupt", "instrumentation_failure"])
def test_early_administrative_stops_are_separate(reason):
    end = endpoint(reason, elapsed_seconds=7)
    assert end.interrupted and end.right_censored and not end.terminated and not end.truncated


@pytest.mark.parametrize("reason", ["unknown", "not_started"])
def test_unknown_endpoints_do_not_manufacture_survival_or_failure(reason):
    end = endpoint(reason, elapsed_seconds=None)
    assert end.terminated is None and end.right_censored is None


def test_endpoint_rejects_false_horizon_and_unfounded_natural_cause():
    with pytest.raises(ValueError, match="early administrative"):
        endpoint("horizon", elapsed_seconds=12)
    with pytest.raises(ValueError, match="terminal evidence"):
        endpoint(natural_terminal_observed=False)
    with pytest.raises(ValueError, match="terminal evidence"):
        endpoint("horizon", elapsed_seconds=60, natural_terminal_observed=True)
    with pytest.raises(ValueError, match="unstarted"):
        endpoint("not_started", elapsed_seconds=1)
    with pytest.raises(ValueError, match="elapsed time"):
        endpoint(elapsed_seconds=None)


def test_park_failure_preserves_verified_result_without_permitting_next_input():
    result = assess_eligibility(
        gameplay(),
        Readiness(ready=False, failure="unknown advertisement"),
        safety(),
        eligibility_policy(),
    )
    assert result.measurement_valid and result.evaluation_eligible and result.learning_eligible
    assert result.session_safe and not result.next_episode_eligible
    # Requiring parking for learning remains an explicit prospective choice.
    stricter = assess_eligibility(
        gameplay(),
        Readiness(ready=False),
        safety(),
        eligibility_policy(require_readiness_for_learning=True),
    )
    assert stricter.evaluation_eligible and not stricter.learning_eligible


@pytest.mark.parametrize(
    "kind", ["unintended_action", "guard_fault", "release_failure", "manual_intervention"]
)
@pytest.mark.parametrize("time,eligible", [(50, False), (300, False), (500, False), (501, True)])
def test_safety_fault_timing_does_not_erase_measurement_or_conceal_session_fault(
    kind, time, eligible
):
    fault = SafetyEvent(timestamp_ns=time, kind=kind, evidence_id="fault-trace")
    result = assess_eligibility(
        gameplay(), readiness(), safety(events=(fault,)), eligibility_policy()
    )
    assert result.measurement_valid
    assert result.evaluation_eligible is eligible and result.learning_eligible is eligible
    assert not result.session_safe and not result.next_episode_eligible


@pytest.mark.parametrize(
    "changes",
    [
        {"monitored_from_ns": 101},
        {"monitored_through_ns": 499},
    ],
)
def test_incomplete_gameplay_safety_monitoring_blocks_evaluation(changes):
    result = assess_eligibility(gameplay(), readiness(), safety(**changes), eligibility_policy())
    assert (
        result.measurement_valid and not result.evaluation_eligible and not result.learning_eligible
    )


@pytest.mark.parametrize(
    "changes",
    [{"inputs_released": None}, {"inputs_released": False}, {"monitored_through_ns": 999}],
)
def test_release_unknown_or_stale_safety_assessment_blocks_next_episode(changes):
    result = assess_eligibility(gameplay(), readiness(), safety(**changes), eligibility_policy())
    assert result.evaluation_eligible and not result.next_episode_eligible


def test_missing_distance_and_coverage_remain_missing():
    unknown = gameplay(distance_m=None, distance_source="unknown", distance_evidence_id=None)
    result = assess_eligibility(unknown, readiness(), safety(), eligibility_policy())
    assert not result.measurement_valid and not result.evaluation_eligible
    low = assess_eligibility(
        gameplay(observation_coverage=None), readiness(), safety(), eligibility_policy()
    )
    assert low.evaluation_eligible and not low.learning_eligible
    with pytest.raises(ValueError, match="source"):
        gameplay(distance_source="unknown")
    with pytest.raises(ValueError, match="evidence"):
        gameplay(distance_evidence_id=None)
    with pytest.raises(ValueError, match="natural endpoint"):
        gameplay(endpoint=endpoint("horizon", elapsed_seconds=60))


def test_censored_result_is_usable_evaluation_with_explicit_learning_policy():
    row = gameplay(
        endpoint=endpoint("horizon", elapsed_seconds=60), distance_source="verified_paused_frame"
    )
    no = assess_eligibility(row, readiness(state="paused"), safety(), eligibility_policy())
    yes = assess_eligibility(
        row, readiness(state="paused"), safety(), eligibility_policy(allow_censored_learning=True)
    )
    assert no.evaluation_eligible and not no.learning_eligible
    assert yes.learning_eligible


def test_ready_and_safety_timestamps_are_validated():
    with pytest.raises(ValueError, match="Ready requires"):
        Readiness(ready=True)
    with pytest.raises(ValueError, match="ordered"):
        safety(monitored_through_ns=1001)
    with pytest.raises(ValueError, match="future events"):
        safety(events=(SafetyEvent(timestamp_ns=1001, kind="guard_fault", evidence_id="f"),))
    assert not assess_eligibility(
        gameplay(), readiness(checked_at_ns=499), safety(), eligibility_policy()
    ).next_episode_eligible
    assert not assess_eligibility(
        gameplay(), readiness(checked_at_ns=1001), safety(), eligibility_policy()
    ).next_episode_eligible


def ledger():
    return ScoreLedger(
        coins_start=100,
        coins_end=160,
        start_seconds=0,
        end_seconds=3,
        increments=tuple(
            ScoreIncrement(
                increment_id=f"i{i}",
                start_seconds=i,
                end_seconds=i + 1,
                coins=20,
                evidence_id=f"counter-pair-{i}",
            )
            for i in range(3)
        ),
    )


def window(event_id="w1", outcome="recovered", start=0, end=2):
    return RecoveryWindow(
        event_id=event_id,
        start_seconds=start,
        end_seconds=end,
        outcome=outcome,
        fully_observed=outcome != "censored",
        natural_terminal_observed=outcome == "fatal",
        evidence_id=f"labels-{event_id}",
    )


def test_duplicate_overlap_conserves_credit_and_cannot_censor_failed_windows_away():
    windows = (window("recovered"), window("fatal", "fatal", 1, 3))
    for order in itertools.permutations(windows):
        result = recovery_score_credit(ledger(), order, unrecovered_credit_fraction=0.5)
        assert result.raw_gain == 60 and result.credited_gain == 20 and result.withheld_gain == 40
        assert sum(row["raw_coins"] for row in result.allocations) == 60
        assert result.credited_gain + result.withheld_gain == result.raw_gain
        assert result.allocations[1]["overlapping_event_ids"] == ("fatal", "recovered")


@pytest.mark.parametrize(
    "outcome,fraction,expected",
    [
        ("recovered", 0, 60),
        ("failed", 0, 20),
        ("failed", 0.5, 40),
        ("fatal", 1, 20),
        ("censored", 1, 20),
    ],
)
def test_recovery_factors_and_base_score(outcome, fraction, expected):
    result = recovery_score_credit(
        ledger(), (window(outcome=outcome),), unrecovered_credit_fraction=fraction
    )
    assert result.credited_gain == expected


def test_uncertain_increment_straddling_window_boundary_is_conservatively_withheld():
    result = recovery_score_credit(
        ledger(), (window(outcome="fatal", start=0.5, end=1.5),), unrecovered_credit_fraction=0
    )
    assert result.credited_gain == 20  # Both overlapping intervals count once and earn zero.
    absent = recovery_score_credit(ledger(), (), unrecovered_credit_fraction=0)
    assert absent.credited_gain == 60


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_credit_fraction_refused(bad):
    with pytest.raises(ValueError, match="within"):
        recovery_score_credit(ledger(), (), unrecovered_credit_fraction=bad)


def test_unknown_score_or_recovery_is_ineligible_not_zero():
    for data, windows in [(None, ()), (ledger(), None)]:
        result = recovery_score_credit(data, windows, unrecovered_credit_fraction=0)
        assert not result.eligible and result.credited_gain is None and result.raw_gain is None


def test_recovery_cannot_claim_fully_observed_continuation_past_episode_boundary():
    with pytest.raises(ValueError, match="censored or fatal"):
        recovery_score_credit(ledger(), (window(end=4),), unrecovered_credit_fraction=0)
    with pytest.raises(ValueError, match="starts after"):
        recovery_score_credit(ledger(), (window(start=3, end=4),), unrecovered_credit_fraction=0)
    result = recovery_score_credit(
        ledger(),
        (window(outcome="censored", start=2, end=4),),
        unrecovered_credit_fraction=1,
    )
    assert result.credited_gain == 40


@pytest.mark.parametrize(
    "change,message",
    [
        ({"coins_end": 159}, "conserve"),
        ({"coins_end": 99}, "decrease"),
        ({"end_seconds": 4}, "boundary"),
        ({"increments": ()}, "partition"),
    ],
)
def test_invalid_score_partitions_refused(change, message):
    with pytest.raises(ValueError, match=message):
        ScoreLedger(**{**ledger().model_dump(), **change})


def test_duplicate_and_overlapping_increment_identifiers_refused():
    data = ledger().model_dump()
    data["increments"][1]["increment_id"] = "i0"
    with pytest.raises(ValueError, match="Duplicate"):
        ScoreLedger(**data)
    data = ledger().model_dump()
    data["increments"][1]["start_seconds"] = 0.5
    with pytest.raises(ValueError, match="nonoverlapping"):
        ScoreLedger(**data)
    with pytest.raises(ValueError, match="Duplicate recovery"):
        recovery_score_credit(ledger(), (window(), window()), unrecovered_credit_fraction=0)


def test_fatal_and_missing_recovery_evidence_cannot_be_labeled_recovered():
    with pytest.raises(ValueError, match="terminal evidence"):
        RecoveryWindow(
            **{**window(outcome="fatal").model_dump(), "natural_terminal_observed": False}
        )
    with pytest.raises(ValueError, match="natural failure"):
        RecoveryWindow(**{**window().model_dump(), "natural_terminal_observed": True})
    with pytest.raises(ValueError, match="censored"):
        RecoveryWindow(**{**window().model_dump(), "fully_observed": False})


def test_lexicographic_tiers_dominate_unbounded_scores_and_use_score_only_for_ties():
    def key(distance, survived, score):
        return lexicographic_key(
            distance_m=distance,
            survived_horizon=survived,
            useful_score=score,
            minimum_distance_m=250,
        )

    huge = 10**300
    assert key(250, False, 0) > key(249, True, huge)
    assert key(250, True, 0) > key(100000, False, huge)
    assert key(251, True, 0) > key(250, True, huge)
    assert key(251, True, 2) > key(251, True, 1)
    assert key(None, True, 0) is None and key(250, None, 0) is None and key(250, True, None) is None
    with pytest.raises(ValueError, match="nonnegative integers"):
        key(True, True, 0)


def test_historical_objectives_keep_original_version_and_scalar_contract():
    from gradientclimb.experiments.objectives import (
        METRIC_SCHEMA_VERSION,
        EpisodeOutcome,
        compute_fitness,
    )

    old = EpisodeOutcome(
        episode_id="old",
        policy_id="p",
        horizon_seconds=60,
        distance_m=243,
        distance_source="stable_terminal_distance",
    )
    assert METRIC_SCHEMA_VERSION == "behavioral-metrics-2.0"
    assert compute_fitness("distance-only-2.0", {}, old).value == 243
