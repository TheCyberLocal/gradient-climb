"""Cycle 2 objective versioning: unknown stays unknown, and every arm is explicit."""

import pytest

from gradientclimb.experiments.objectives import (
    METRIC_SCHEMA_VERSION,
    OBJECTIVE_FAMILIES,
    EpisodeOutcome,
    ExperimentVersions,
    RecoveryEvent,
    TimeToReach,
    compute_fitness,
    derived_metrics,
    detect_airtime_events,
    hacking_diagnostics,
    hud_progress_trace,
    recovery_events_from_windows,
    stagnation_seconds,
    stamp_versions,
    time_to_reach,
)


def outcome(**overrides):
    base = {
        "episode_id": "e0",
        "policy_id": "p",
        "horizon_seconds": 60.0,
        "distance_m": 300,
        "distance_source": "stable_terminal_distance",
        "terminal_cause": "driver_down",
        "survived_horizon": False,
        "gameplay_seconds": 40.0,
        "coins_start": 1000,
        "coins_end": 1250,
        "score_gain": 250,
        "score_source": "hud_coin_counter",
    }
    base.update(overrides)
    return EpisodeOutcome(**base)


def test_versions_require_registered_objective_and_stamp_once():
    versions = ExperimentVersions(
        observation_schema="hcr-screen-relative-1",
        reward_fitness="distance-only-2.0",
        benchmark="native-reliability-2.0",
    )
    stamped = stamp_versions({"a": 1}, versions)
    assert stamped["cycle2_versions"]["metric_schema"] == METRIC_SCHEMA_VERSION
    assert stamped["a"] == 1
    with pytest.raises(ValueError, match="already declares"):
        stamp_versions(stamped, versions)
    with pytest.raises(ValueError, match="registered"):
        ExperimentVersions(
            observation_schema="x", reward_fitness="distance-only-9.9", benchmark="b"
        )
    ExperimentVersions(
        observation_schema="x", reward_fitness="historical:ppo-shaped", benchmark="b"
    )


def test_outcome_coherence_rules():
    with pytest.raises(ValueError, match="declare its source"):
        outcome(distance_source="unknown")
    with pytest.raises(ValueError, match="cannot claim"):
        outcome(distance_m=None)
    with pytest.raises(ValueError, match="coin boundary"):
        outcome(coins_start=None)
    with pytest.raises(ValueError, match="must equal"):
        outcome(score_gain=1)
    unknown = outcome(
        distance_m=None,
        distance_source="unknown",
        coins_start=None,
        coins_end=None,
        score_gain=None,
    )
    assert unknown.distance_m is None and unknown.score_gain is None


def test_distance_only_and_unknown_inputs_are_ineligible_not_zero():
    decision = compute_fitness("distance-only-2.0", {}, outcome())
    assert decision.eligible and decision.value == 300.0
    missing = compute_fitness(
        "distance-only-2.0", {}, outcome(distance_m=None, distance_source="unknown")
    )
    assert not missing.eligible and missing.value is None and "distance_m" in missing.reason
    no_score = compute_fitness(
        "distance-score-2.0",
        {"score_weight_m_per_coin": 0.5},
        outcome(coins_start=None, coins_end=None, score_gain=None),
    )
    assert not no_score.eligible and no_score.value is None


def test_parameter_sets_are_exact_and_numeric():
    with pytest.raises(ValueError, match="exactly the parameters"):
        compute_fitness("distance-score-2.0", {}, outcome())
    with pytest.raises(TypeError, match="numeric"):
        compute_fitness("distance-score-2.0", {"score_weight_m_per_coin": True}, outcome())
    with pytest.raises(ValueError, match="Unknown objective"):
        compute_fitness("score-only-1.0", {}, outcome())
    for name, family in OBJECTIVE_FAMILIES.items():
        assert family["arm"] in "ABCDE" and name.endswith("-2.0")


def test_score_arm_and_time_arm_credit_only_observed_goal_attainment():
    score = compute_fitness("distance-score-2.0", {"score_weight_m_per_coin": 0.2}, outcome())
    assert score.value == pytest.approx(300 + 0.2 * 250)
    parameters = {
        "score_weight_m_per_coin": 0.2,
        "goal_distance_m": 200,
        "time_weight_m_per_second": 2.0,
    }
    unmeasured = compute_fitness("distance-score-time-2.0", parameters, outcome())
    assert not unmeasured.eligible and "not measured" in unmeasured.reason
    reached = outcome(time_to_reach=[TimeToReach(goal_m=200, seconds=20.0, reached=True)])
    decision = compute_fitness("distance-score-time-2.0", parameters, reached)
    assert decision.eligible and decision.value == pytest.approx(300 + 50 + 2.0 * 40)
    not_reached = outcome(time_to_reach=[TimeToReach(goal_m=200, seconds=None, reached=False)])
    assert compute_fitness("distance-score-time-2.0", parameters, not_reached).value == (
        pytest.approx(350.0)
    )
    unknown = outcome(time_to_reach=[TimeToReach(goal_m=200, seconds=None, reached=None)])
    assert not compute_fitness("distance-score-time-2.0", parameters, unknown).eligible


def test_recovery_conditioned_credit_withholds_failed_fatal_and_censored_windows():
    def event(result, gain):
        return RecoveryEvent(
            kind="airtime",
            start_seconds=5.0,
            end_seconds=6.0,
            frames=3,
            window_seconds=3.0,
            window_observed=result != "censored",
            terminal_within_window=result == "fatal",
            score_gain_in_window=gain,
            outcome=result,
        )

    events = [
        event("recovered", 100),
        event("failed", 60),
        event("fatal", 50),
        event("censored", 40),
    ]
    rows = outcome(
        airtime_events=4,
        recovery_events=events,
        recoveries_success=1,
        recoveries_failed=1,
    )
    parameters = {"score_weight_m_per_coin": 1.0, "unrecovered_credit_fraction": 0.0}
    decision = compute_fitness("recovery-conditioned-2.0", parameters, rows)
    assert decision.eligible
    # base gain is 250 - 250 = 0; only the recovered window's 100 is credited.
    assert decision.value == pytest.approx(300 + 100)
    partial = compute_fitness(
        "recovery-conditioned-2.0",
        {"score_weight_m_per_coin": 1.0, "unrecovered_credit_fraction": 0.5},
        rows,
    )
    assert partial.value == pytest.approx(300 + 100 + 30)
    unmeasured = compute_fitness("recovery-conditioned-2.0", parameters, outcome())
    assert not unmeasured.eligible and "not measured" in unmeasured.reason
    with pytest.raises(ValueError, match="disagrees"):
        outcome(airtime_events=4, recovery_events=events, recoveries_success=2)


def test_lexicographic_arm_orders_survival_and_minimum_distance_before_score():
    parameters = {
        "minimum_distance_m": 250,
        "score_weight_m_per_coin": 1.0,
        "survival_bonus_m": 100,
    }
    short_rich = outcome(distance_m=100, score_gain=900, coins_end=1900)
    long_poor = outcome(distance_m=260, score_gain=0, coins_end=1000, survived_horizon=True)
    assert compute_fitness("lexicographic-2.0", parameters, long_poor).value > (
        compute_fitness("lexicographic-2.0", parameters, short_rich).value
    )
    assert not compute_fitness(
        "lexicographic-2.0", parameters, outcome(survived_horizon=None)
    ).eligible


def test_derived_ratios_publish_denominators_and_unknown_on_invalid_denominator():
    metrics = derived_metrics(outcome())
    assert metrics["score_per_metre"]["value"] == pytest.approx(250 / 300)
    assert metrics["score_per_metre"]["numerator"] == 250
    zero = derived_metrics(
        outcome(distance_m=0, gameplay_seconds=0.0, hud_max_progress_m=0, hud_read_fraction=1.0)
    )
    assert zero["score_per_metre"]["value"] is None and not zero["score_per_metre"]["eligible"]
    assert zero["hud_progress_per_gameplay_second"]["value"] is None
    assert zero["hud_progress_per_gameplay_second"]["semantics"].startswith("HUD")


def rows_from(values_by_step, hud_by_step, dt=0.5, radius=0.17, radius_valid=True):
    names = [
        "left_wheel_to_terrain_axles",
        "right_wheel_to_terrain_axles",
        "left_radius_axles",
        "right_radius_axles",
    ]
    rows = []
    for step, (clearance, hud) in enumerate(zip(values_by_step, hud_by_step, strict=True)):
        valid = clearance is not None
        rows.append(
            {
                "step": step,
                "elapsed_seconds": step * dt,
                "screen": {
                    "feature_names": names,
                    "values": [clearance or 0.0, clearance or 0.0, radius, radius],
                    "valid": [valid, valid, radius_valid, radius_valid],
                    "hud": {"valid": hud is not None, "hud_displayed_progress_meters": hud},
                },
            }
        )
    return rows


def test_airtime_detection_requires_valid_clearance_on_consecutive_frames():
    clearance = [0.0, 0.3, 0.4, None, 0.5, 0.0, 0.2, 0.0]
    hud = [0, 5, 10, 15, 20, 25, 30, 35]
    rows = rows_from(clearance, hud)
    events = detect_airtime_events(rows, radius_multiple=None, minimum_frames=2)
    assert len(events) == 1 and events[0]["frames"] == 2
    assert events[0]["start_seconds"] == 0.5 and events[0]["end_seconds"] == 1.0
    assert detect_airtime_events(rows, radius_multiple=None, minimum_frames=1)[1]["frames"] == 1
    with pytest.raises(ValueError):
        detect_airtime_events([], clearance_threshold_axles=0)
    with pytest.raises(ValueError, match="radius_multiple"):
        detect_airtime_events([], radius_multiple=1.0)


def test_airtime_detection_is_radius_relative_by_default():
    # A resting wheel's centre sits one radius (0.17) above the terrain: not airborne.
    resting = [0.17, 0.18, 0.19, 0.2, 0.2]
    hud = [0, 1, 2, 3, 4]
    assert detect_airtime_events(rows_from(resting, hud)) == []
    # Clear of the terrain by more than 1.6 radii on consecutive frames: one event.
    airborne = [0.17, 0.4, 0.45, 0.5, 0.17]
    events = detect_airtime_events(rows_from(airborne, hud))
    assert len(events) == 1 and events[0]["frames"] == 3
    # Without valid radii the radius-relative detector cannot decide: no event.
    assert detect_airtime_events(rows_from(airborne, hud, radius_valid=False)) == []
    # The absolute floor still applies when radii are tiny.
    assert detect_airtime_events(rows_from([0.1, 0.12, 0.12, 0.1, 0.1], hud, radius=0.02)) == []


def test_hud_trace_rejects_decreases_and_time_to_reach_reports_unknown_coverage():
    rows = rows_from([0.0] * 6, [0, 20, 15, 40, None, 80])
    trace = hud_progress_trace(rows)
    assert [v for _, v in trace] == [0, 20, 40, 80]
    reached = time_to_reach(trace, [30, 100], horizon_seconds=60, coverage=0.8)
    assert reached[0].reached is True and reached[0].seconds == 1.5
    assert reached[1].reached is False and reached[1].seconds is None
    sparse = time_to_reach(trace, [100], horizon_seconds=60, coverage=0.2)
    assert sparse[0].reached is None
    assert stagnation_seconds(trace) == pytest.approx(1.0)
    assert stagnation_seconds(trace[:1]) is None


def test_recovery_windows_apply_registered_progress_criterion():
    trace = [(0.0, 0), (1.0, 10), (2.0, 20), (3.0, 20), (4.0, 20), (6.0, 60)]
    events = [
        {"start_seconds": 0.5, "end_seconds": 1.0, "frames": 2},
        {"start_seconds": 2.5, "end_seconds": 3.0, "frames": 2},
    ]
    recovered = recovery_events_from_windows(
        events,
        trace,
        window_seconds=3.0,
        recovery_progress_m=10,
        gameplay_seconds=10.0,
        terminal_cause="truncated_horizon",
        coin_trace=[(0.0, 100), (2.0, 150), (5.0, 180)],
    )
    assert recovered[0].outcome == "recovered" and recovered[0].score_gain_in_window == 50
    assert recovered[1].outcome == "recovered"
    fatal = recovery_events_from_windows(
        events[:1],
        trace[:3],
        window_seconds=3.0,
        recovery_progress_m=10,
        gameplay_seconds=2.5,
        terminal_cause="driver_down",
    )
    assert fatal[0].outcome == "fatal" and fatal[0].score_gain_in_window is None
    censored = recovery_events_from_windows(
        events[:1],
        trace[:3],
        window_seconds=3.0,
        recovery_progress_m=10,
        gameplay_seconds=2.5,
        terminal_cause="aborted",
    )
    assert censored[0].outcome == "censored" and not censored[0].window_observed
    failed = recovery_events_from_windows(
        events[1:],
        trace,
        window_seconds=1.0,
        recovery_progress_m=10,
        gameplay_seconds=10.0,
        terminal_cause="truncated_horizon",
    )
    assert failed[0].outcome == "failed"


def test_hacking_diagnostics_mark_unmeasured_indicators_explicitly():
    report = hacking_diagnostics([outcome(), outcome(episode_id="e1", survived_horizon=True)])
    assert report["episodes"] == 2
    assert report["flips_then_death"]["measurable"] is False
    assert report["backward_loops"]["value"] is None and not report["backward_loops"]["measurable"]
    assert report["coin_chasing"]["measurable"] is True
    assert report["slow_but_safe"]["survival_fraction"] == 0.5
    with pytest.raises(ValueError):
        hacking_diagnostics([])
