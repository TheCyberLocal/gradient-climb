"""Cycle 2 versioned behavioral metrics, objective families and version stamping.

Nothing here rewrites a Cycle 1 record. Every identifier is explicit and frozen by
the experiment definition that uses it. An unknown measurement stays ``None``; any
objective that needs an unknown quantity is ineligible, never zero. Training may
optimize one scalar, but the full outcome vector is always retained beside it.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

METRIC_SCHEMA_VERSION = "behavioral-metrics-2.0"
RECOVERY_DEFINITION_VERSION = "recovery-progress-2.0"
AIRTIME_DETECTOR_VERSION = "airtime-wheel-clearance-2.0"
VERSIONS_KEY = "cycle2_versions"

TerminalCause = Literal[
    "driver_down", "out_of_fuel", "truncated_horizon", "aborted", "unknown", "natural_unlabeled"
]
DistanceSource = Literal["stable_terminal_distance", "verified_paused_frame", "unknown"]


class ExperimentVersions(BaseModel):
    """Every dimension a serious Cycle 2 run must declare before collecting data."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_schema: Literal["behavioral-metrics-2.0"] = METRIC_SCHEMA_VERSION
    observation_schema: str = Field(min_length=1)
    reward_fitness: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    simulator: str | None = None
    calibration: str | None = None
    ui_profile: str | None = None
    score_reader: str | None = None
    recovery_definition: str | None = None

    @model_validator(mode="after")
    def known_objective(self):
        if self.reward_fitness not in OBJECTIVE_FAMILIES and not self.reward_fitness.startswith(
            "historical:"
        ):
            raise ValueError(
                "reward_fitness must be a registered Cycle 2 objective or historical:*"
            )
        return self


def stamp_versions(config: dict[str, Any], versions: ExperimentVersions) -> dict[str, Any]:
    """Return a copy of a run configuration carrying the declared versions."""
    if VERSIONS_KEY in config:
        raise ValueError("Configuration already declares Cycle 2 versions")
    return {**config, VERSIONS_KEY: versions.model_dump(mode="json")}


class RecoveryEvent(BaseModel):
    """One maneuver window and its independently checkable outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["airtime", "rotation"]
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    frames: int = Field(ge=1)
    progress_at_end_m: int | None = None
    progress_after_window_m: int | None = None
    window_seconds: FiniteFloat = Field(gt=0)
    window_observed: bool
    terminal_within_window: bool
    score_gain_in_window: int | None = None
    outcome: Literal["recovered", "failed", "fatal", "censored"]

    @model_validator(mode="after")
    def ordered(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("An event cannot end before it starts")
        return self


class TimeToReach(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    goal_m: int = Field(gt=0)
    seconds: FiniteFloat | None = Field(default=None, ge=0)
    reached: bool | None = None


class EpisodeOutcome(BaseModel):
    """The independent per-episode outcome vector; every field keeps its own missingness."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_schema_version: Literal["behavioral-metrics-2.0"] = METRIC_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    # Endpoint
    distance_m: int | None = Field(default=None, ge=0)
    distance_source: DistanceSource = "unknown"
    terminal_cause: TerminalCause = "unknown"
    survived_horizon: bool | None = None
    horizon_seconds: FiniteFloat = Field(gt=0)
    # Time
    gameplay_seconds: FiniteFloat | None = Field(default=None, ge=0)
    session_seconds: FiniteFloat | None = Field(default=None, ge=0)
    reset_seconds: FiniteFloat | None = Field(default=None, ge=0)
    advertisement_seconds: FiniteFloat | None = Field(default=None, ge=0)
    advertisement_encounters: int = Field(default=0, ge=0)
    # Score
    coins_start: int | None = Field(default=None, ge=0)
    coins_end: int | None = Field(default=None, ge=0)
    score_gain: int | None = None
    score_source: str = "unknown"
    # Progress proxies from the HUD progress display
    hud_max_progress_m: int | None = Field(default=None, ge=0)
    hud_read_fraction: FiniteFloat | None = Field(default=None, ge=0, le=1)
    time_to_reach: list[TimeToReach] = Field(default_factory=list)
    stagnation_seconds: FiniteFloat | None = Field(default=None, ge=0)
    backward_motion_measured: bool = False
    # Stunt / recovery
    airtime_events: int | None = Field(default=None, ge=0)
    airtime_seconds: FiniteFloat | None = Field(default=None, ge=0)
    rotation_events: int | None = Field(default=None, ge=0)
    recovery_events: list[RecoveryEvent] = Field(default_factory=list)
    recoveries_success: int | None = Field(default=None, ge=0)
    recoveries_failed: int | None = Field(default=None, ge=0)
    fatal_stunt: bool | None = None
    # Measurement quality and control accounting
    frames: int = Field(default=0, ge=0)
    dispatched_actions: int = Field(default=0, ge=0)
    stale_captures: int = Field(default=0, ge=0)
    missing_required_feature_frames: int = Field(default=0, ge=0)
    unknown_state_frames: int = Field(default=0, ge=0)
    manual_interventions: int = Field(default=0, ge=0)
    unintended_actions: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if self.distance_m is not None and self.distance_source == "unknown":
            raise ValueError("A known distance must declare its source")
        if self.distance_m is None and self.distance_source != "unknown":
            raise ValueError("A missing distance cannot claim a source")
        if self.score_gain is not None and (self.coins_start is None or self.coins_end is None):
            raise ValueError("Score gain requires both coin boundary readings")
        if (
            self.coins_start is not None
            and self.coins_end is not None
            and self.score_gain is not None
            and self.score_gain != self.coins_end - self.coins_start
        ):
            raise ValueError("Score gain must equal the coin boundary difference")
        for name in ("recoveries_success", "recoveries_failed"):
            count = getattr(self, name)
            if count is not None and self.recovery_events:
                outcome = "recovered" if name == "recoveries_success" else "failed"
                observed = sum(1 for e in self.recovery_events if e.outcome == outcome)
                if count != observed:
                    raise ValueError(f"{name} disagrees with the recorded events")
        return self


def _ratio(name, numerator, denominator, *, minimum_denominator=0.0):
    eligible = (
        numerator is not None
        and denominator is not None
        and math.isfinite(denominator)
        and denominator > minimum_denominator
    )
    return {
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "eligible": bool(eligible),
        "value": (numerator / denominator) if eligible else None,
    }


def derived_metrics(outcome: EpisodeOutcome) -> dict[str, Any]:
    """Ratios publish numerator, denominator and eligibility; an invalid denominator is unknown."""
    return {
        "score_per_metre": _ratio("score_per_metre", outcome.score_gain, outcome.distance_m),
        "score_per_gameplay_second": _ratio(
            "score_per_gameplay_second", outcome.score_gain, outcome.gameplay_seconds
        ),
        "score_per_session_second": _ratio(
            "score_per_session_second", outcome.score_gain, outcome.session_seconds
        ),
        "hud_progress_per_gameplay_second": {
            **_ratio(
                "hud_progress_per_gameplay_second",
                outcome.hud_max_progress_m,
                outcome.gameplay_seconds,
            ),
            "semantics": "HUD maximum-progress proxy; not signed world velocity",
        },
        "distance_per_session_second": _ratio(
            "distance_per_session_second", outcome.distance_m, outcome.session_seconds
        ),
        "hud_coverage": outcome.hud_read_fraction,
        "survival": outcome.survived_horizon,
        "terminal_cause": outcome.terminal_cause,
    }


OBJECTIVE_FAMILIES: dict[str, dict[str, Any]] = {
    "distance-only-2.0": {
        "arm": "A",
        "summary": "Qualified terminal or full-horizon paused distance in metres.",
        "parameters": (),
        "requires": ("distance_m",),
    },
    "distance-score-2.0": {
        "arm": "B",
        "summary": "Distance plus a frozen linear coin-gain term.",
        "parameters": ("score_weight_m_per_coin",),
        "requires": ("distance_m", "score_gain"),
    },
    "distance-score-time-2.0": {
        "arm": "C",
        "summary": "Arm B plus credit for reaching a frozen goal distance early, "
        "only when the goal was observably reached.",
        "parameters": ("score_weight_m_per_coin", "goal_distance_m", "time_weight_m_per_second"),
        "requires": ("distance_m", "score_gain"),
    },
    "recovery-conditioned-2.0": {
        "arm": "D",
        "summary": "Distance plus base coin gain; coin gain inside maneuver windows is credited "
        "only when the registered recovery criterion succeeds.",
        "parameters": ("score_weight_m_per_coin", "unrecovered_credit_fraction"),
        "requires": ("distance_m", "score_gain"),
    },
    "lexicographic-2.0": {
        "arm": "E",
        "summary": "Optional: survival and a minimum distance dominate; score breaks ties.",
        "parameters": ("minimum_distance_m", "score_weight_m_per_coin", "survival_bonus_m"),
        "requires": ("distance_m", "score_gain"),
    },
}


class FitnessDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    objective: str
    eligible: bool
    value: FiniteFloat | None = None
    reason: str
    components: dict[str, Any] = Field(default_factory=dict)


def _check_parameters(objective_id: str, parameters: dict[str, Any]) -> dict[str, float]:
    family = OBJECTIVE_FAMILIES.get(objective_id)
    if family is None:
        raise ValueError(f"Unknown objective version: {objective_id}")
    expected = set(family["parameters"])
    if set(parameters) != expected:
        raise ValueError(f"{objective_id} requires exactly the parameters {sorted(expected)}")
    frozen = {}
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Parameter {name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Parameter {name} must be finite and nonnegative")
        frozen[name] = float(value)
    return frozen


def compute_fitness(
    objective_id: str, parameters: dict[str, Any], outcome: EpisodeOutcome
) -> FitnessDecision:
    """Scalar training fitness for one episode; ineligible whenever a required input is unknown."""
    frozen = _check_parameters(objective_id, parameters)
    family = OBJECTIVE_FAMILIES[objective_id]
    missing = [name for name in family["requires"] if getattr(outcome, name) is None]
    if missing:
        return FitnessDecision(
            objective=objective_id,
            eligible=False,
            reason=f"Unknown required measurement: {', '.join(missing)}",
        )
    distance = float(outcome.distance_m)
    components: dict[str, Any] = {"distance_m": distance}
    if objective_id == "distance-only-2.0":
        return FitnessDecision(
            objective=objective_id,
            eligible=True,
            value=distance,
            reason="Qualified distance",
            components=components,
        )
    weight = frozen["score_weight_m_per_coin"]
    score_term = weight * float(outcome.score_gain)
    components["score_gain"] = outcome.score_gain
    components["score_term_m"] = score_term
    if objective_id == "distance-score-2.0":
        return FitnessDecision(
            objective=objective_id,
            eligible=True,
            value=distance + score_term,
            reason="Distance plus frozen score term",
            components=components,
        )
    if objective_id == "distance-score-time-2.0":
        goal = int(frozen["goal_distance_m"])
        matching = [row for row in outcome.time_to_reach if row.goal_m == goal]
        if len(matching) != 1:
            return FitnessDecision(
                objective=objective_id,
                eligible=False,
                reason=f"Time to reach the frozen goal {goal} m was not measured",
                components=components,
            )
        row = matching[0]
        if row.reached is None:
            return FitnessDecision(
                objective=objective_id,
                eligible=False,
                reason="Goal attainment is unknown because HUD coverage was insufficient",
                components=components,
            )
        saved = max(0.0, outcome.horizon_seconds - row.seconds) if row.reached else 0.0
        time_term = frozen["time_weight_m_per_second"] * saved
        components.update(goal_reached=row.reached, seconds_saved=saved, time_term_m=time_term)
        return FitnessDecision(
            objective=objective_id,
            eligible=True,
            value=distance + score_term + time_term,
            reason="Distance, score and goal-time credit",
            components=components,
        )
    if objective_id == "recovery-conditioned-2.0":
        fraction = frozen["unrecovered_credit_fraction"]
        if fraction > 1:
            raise ValueError("unrecovered_credit_fraction must not exceed 1")
        if outcome.airtime_events is None:
            return FitnessDecision(
                objective=objective_id,
                eligible=False,
                reason="Maneuver events were not measured; recovery credit cannot be assigned",
                components=components,
            )
        event_gain = 0
        credited = 0.0
        for event in outcome.recovery_events:
            if event.score_gain_in_window is None:
                return FitnessDecision(
                    objective=objective_id,
                    eligible=False,
                    reason="An event window has unknown score gain",
                    components=components,
                )
            event_gain += event.score_gain_in_window
            if event.outcome == "recovered":
                credited += event.score_gain_in_window
            elif event.outcome == "failed":
                credited += fraction * event.score_gain_in_window
            # fatal and censored windows earn nothing.
        base_gain = float(outcome.score_gain) - event_gain
        value = distance + weight * (base_gain + credited)
        components.update(
            base_score_gain=base_gain,
            event_score_gain=event_gain,
            credited_event_gain=credited,
            recovery_definition=RECOVERY_DEFINITION_VERSION,
        )
        return FitnessDecision(
            objective=objective_id,
            eligible=True,
            value=value,
            reason="Distance plus recovery-conditioned score",
            components=components,
        )
    if objective_id == "lexicographic-2.0":
        if outcome.survived_horizon is None:
            return FitnessDecision(
                objective=objective_id,
                eligible=False,
                reason="Survival is unknown",
                components=components,
            )
        minimum = frozen["minimum_distance_m"]
        tier = min(distance, minimum) * 1000.0
        survival = frozen["survival_bonus_m"] if outcome.survived_horizon else 0.0
        value = tier + distance + survival + score_term
        components.update(tier_m=tier, survival_bonus_m=survival)
        return FitnessDecision(
            objective=objective_id,
            eligible=True,
            value=value,
            reason="Lexicographic survival/minimum-distance tiers before score",
            components=components,
        )
    raise AssertionError("unreachable")  # pragma: no cover


def hacking_diagnostics(outcomes: list[EpisodeOutcome]) -> dict[str, Any]:
    """Eight required reward-hacking indicators over one policy's evaluation episodes.

    Every indicator states whether it was measurable. Unmeasured indicators are
    ``None`` rather than a favorable number.
    """
    if not outcomes:
        raise ValueError("At least one outcome is required")
    n = len(outcomes)

    def fraction(predicate, population):
        rows = [o for o in outcomes if population(o)]
        if not rows:
            return None
        return sum(1 for o in rows if predicate(o)) / len(rows)

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    rotation_known = [o for o in outcomes if o.rotation_events is not None]
    airtime_known = [o for o in outcomes if o.airtime_events is not None]
    score_known = [o for o in outcomes if o.score_gain is not None and o.distance_m is not None]
    return {
        "episodes": n,
        "flips_then_death": {
            "measurable": bool(rotation_known),
            "rotation_events_per_episode": mean([o.rotation_events for o in rotation_known]),
            "fatal_after_rotation_fraction": fraction(
                lambda o: bool(o.fatal_stunt),
                lambda o: o.rotation_events is not None and o.rotation_events > 0,
            ),
        },
        "airtime_without_progress": {
            "measurable": bool(airtime_known),
            "airtime_seconds_per_episode": mean([o.airtime_seconds for o in airtime_known]),
            "unrecovered_event_fraction": fraction(
                lambda o: any(e.outcome in {"failed", "fatal"} for e in o.recovery_events),
                lambda o: bool(o.recovery_events),
            ),
        },
        "slow_but_safe": {
            "measurable": any(o.stagnation_seconds is not None for o in outcomes),
            "survival_fraction": fraction(
                lambda o: bool(o.survived_horizon), lambda o: o.survived_horizon is not None
            ),
            "mean_stagnation_seconds": mean([o.stagnation_seconds for o in outcomes]),
            "mean_distance_m": mean([o.distance_m for o in outcomes]),
        },
        "reckless_speed": {
            "measurable": any(o.time_to_reach for o in outcomes),
            "early_death_fraction": fraction(
                lambda o: (
                    o.gameplay_seconds is not None and o.gameplay_seconds < 0.5 * o.horizon_seconds
                ),
                lambda o: o.terminal_cause in {"driver_down", "out_of_fuel", "natural_unlabeled"},
            ),
            "mean_hud_progress_per_second": mean(
                [derived_metrics(o)["hud_progress_per_gameplay_second"]["value"] for o in outcomes]
            ),
        },
        "coin_chasing": {
            "measurable": bool(score_known),
            "mean_score_per_metre": mean(
                [derived_metrics(o)["score_per_metre"]["value"] for o in score_known]
            ),
            "distance_note": "compare against the distance-only arm at matched budget",
        },
        "backward_loops": {
            "measurable": any(o.backward_motion_measured for o in outcomes),
            "value": None,
            "note": "Signed world motion is not measured by the current image-relative bridge",
        },
        "stagnation": {
            "measurable": any(o.stagnation_seconds is not None for o in outcomes),
            "max_stagnation_seconds": max(
                [o.stagnation_seconds for o in outcomes if o.stagnation_seconds is not None],
                default=None,
            ),
        },
        "local_score_farming": {
            "measurable": bool(score_known)
            and any(o.stagnation_seconds is not None for o in score_known),
            "score_during_stagnation_note": "requires coin trace alignment; see event windows",
            "mean_score_gain": mean([o.score_gain for o in score_known]),
        },
    }


def wheels_clear(
    names,
    values,
    valid,
    *,
    clearance_threshold_axles: float = 0.15,
    radius_multiple: float | None = 1.6,
) -> bool:
    """True only when both wheels are measured clear of the terrain on this frame.

    Unknown wheel or radius measurements never count as airborne. The same rule
    feeds ``detect_airtime_events`` and any scripted baseline that must react to
    airtime, so the two can never disagree about what "airborne" means.
    """
    index = {name: position for position, name in enumerate(names)}
    wheels = ("left_wheel_to_terrain_axles", "right_wheel_to_terrain_axles")
    radii = ("left_radius_axles", "right_radius_axles")
    needed = wheels + (radii if radius_multiple is not None else ())
    if not all(name in index and bool(valid[index[name]]) for name in needed):
        return False
    thresholds = [clearance_threshold_axles, clearance_threshold_axles]
    if radius_multiple is not None:
        thresholds = [
            max(floor, radius_multiple * float(values[index[radius]]))
            for floor, radius in zip(thresholds, radii, strict=True)
        ]
    return all(
        float(values[index[wheel]]) > threshold
        for wheel, threshold in zip(wheels, thresholds, strict=True)
    )


def body_pitch_radians(names, values, valid) -> float | None:
    """Body axis angle from the doubled-angle encoding, in (-pi/2, pi/2]; None if unknown.

    The axis is modulo pi, so this cannot distinguish upright from inverted; it is
    a level-versus-tilted magnitude, not a signed nose-up/nose-down measurement.
    """
    index = {name: position for position, name in enumerate(names)}
    sin_name, cos_name = "body_sin_2angle", "body_cos_2angle"
    if not all(name in index and bool(valid[index[name]]) for name in (sin_name, cos_name)):
        return None
    return 0.5 * math.atan2(float(values[index[sin_name]]), float(values[index[cos_name]]))


def detect_airtime_events(
    rows: list[dict[str, Any]],
    *,
    clearance_threshold_axles: float = 0.15,
    radius_multiple: float | None = 1.6,
    minimum_frames: int = 2,
) -> list[dict[str, Any]]:
    """Segment windows where both wheels are measured clear of the terrain.

    Rows are the runner's per-frame observation records. A wheel counts as clear
    when its centre-to-terrain distance exceeds both the absolute floor and
    ``radius_multiple`` times its own measured radius: a grounded wheel's centre
    already sits about one radius above the surface, so an absolute threshold alone
    reads resting wheels as airborne. Both wheels, both clearances and (when a
    radius multiple is used) both radii must be valid on ``minimum_frames``
    consecutive frames. Missing measurements end a window rather than extend it.
    Detector version: ``airtime-wheel-clearance-2.0``. Thresholds are image-relative
    hypotheses until validated against independent labels.
    """
    if not math.isfinite(clearance_threshold_axles) or clearance_threshold_axles <= 0:
        raise ValueError("Clearance threshold must be positive")
    if radius_multiple is not None and (
        not math.isfinite(radius_multiple) or radius_multiple <= 1.0
    ):
        raise ValueError("radius_multiple must exceed 1.0 (a resting wheel sits at one radius)")
    if type(minimum_frames) is not int or minimum_frames < 1:
        raise ValueError("minimum_frames must be a positive integer")
    events, current = [], None
    for row in rows:
        screen = row.get("screen") or {}
        names = screen.get("feature_names") or []
        values, valid = screen.get("values") or [], screen.get("valid") or []
        airborne = False
        if names and len(values) == len(names) == len(valid):
            airborne = wheels_clear(
                names,
                values,
                valid,
                clearance_threshold_axles=clearance_threshold_axles,
                radius_multiple=radius_multiple,
            )
        elapsed = row.get("elapsed_seconds")
        if airborne and elapsed is not None:
            if current is None:
                current = {"start_seconds": elapsed, "end_seconds": elapsed, "frames": 1}
            else:
                current["end_seconds"] = elapsed
                current["frames"] += 1
        elif current is not None:
            if current["frames"] >= minimum_frames:
                events.append({**current, "detector": AIRTIME_DETECTOR_VERSION})
            current = None
    if current is not None and current["frames"] >= minimum_frames:
        events.append({**current, "detector": AIRTIME_DETECTOR_VERSION})
    return events


def hud_progress_trace(rows: list[dict[str, Any]]) -> list[tuple[float, int]]:
    """(elapsed_seconds, displayed metres) for accepted, nondecreasing HUD reads."""
    trace, maximum = [], None
    for row in rows:
        hud = (row.get("screen") or {}).get("hud") or {}
        value = hud.get("hud_displayed_progress_meters") if hud.get("valid") else None
        elapsed = row.get("elapsed_seconds")
        if value is None or elapsed is None or (maximum is not None and value < maximum):
            continue
        maximum = value
        trace.append((float(elapsed), int(value)))
    return trace


def time_to_reach(trace: list[tuple[float, int]], goals, *, horizon_seconds, coverage) -> list:
    """First accepted HUD time at or beyond each goal; unknown when coverage is insufficient."""
    result = []
    for goal in goals:
        goal = int(goal)
        hit = next((t for t, v in trace if v >= goal), None)
        if hit is not None:
            result.append(TimeToReach(goal_m=goal, seconds=hit, reached=True))
        elif coverage is not None and coverage >= 0.5 and trace and trace[-1][1] < goal:
            result.append(TimeToReach(goal_m=goal, seconds=None, reached=False))
        else:
            result.append(TimeToReach(goal_m=goal, seconds=None, reached=None))
    return result


def stagnation_seconds(trace: list[tuple[float, int]], *, minimum_increment_m: int = 1):
    """Longest gap between accepted HUD increases; None without at least two reads."""
    if len(trace) < 2:
        return None
    longest, last_time = 0.0, trace[0][0]
    last_value = trace[0][1]
    for elapsed, value in trace[1:]:
        if value - last_value >= minimum_increment_m:
            longest = max(longest, elapsed - last_time)
            last_time, last_value = elapsed, value
    return max(longest, trace[-1][0] - last_time)


def recovery_events_from_windows(
    events: list[dict[str, Any]],
    trace: list[tuple[float, int]],
    *,
    window_seconds: float,
    recovery_progress_m: int,
    gameplay_seconds: float,
    terminal_cause: str,
    coin_trace: list[tuple[float, int]] | None = None,
) -> list[RecoveryEvent]:
    """Apply ``recovery-progress-2.0``: after an event, progress must continue by the
    registered increment within the window, with no terminal failure inside it, and
    the whole window must be observed. Otherwise the event is failed, fatal or censored.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0 or recovery_progress_m < 1:
        raise ValueError("Recovery window and progress increment must be positive")
    fatal_causes = {"driver_down", "out_of_fuel", "natural_unlabeled"}
    result = []
    for event in events:
        end = float(event["end_seconds"])
        window_end = end + window_seconds
        at_end = next((v for t, v in reversed(trace) if t <= end), None)
        later = [v for t, v in trace if end < t <= window_end]
        after = max(later) if later else None
        gain = None
        if coin_trace is not None:
            before = next((c for t, c in reversed(coin_trace) if t <= event["start_seconds"]), None)
            coins_after = next((c for t, c in reversed(coin_trace) if t <= window_end), None)
            if before is not None and coins_after is not None:
                gain = coins_after - before
        terminal_inside = terminal_cause in fatal_causes and gameplay_seconds <= window_end
        observed = gameplay_seconds >= window_end or terminal_inside
        if terminal_inside:
            outcome = "fatal"
        elif not observed or at_end is None or after is None:
            outcome = "censored"
        elif after - at_end >= recovery_progress_m:
            outcome = "recovered"
        else:
            outcome = "failed"
        result.append(
            RecoveryEvent(
                kind=event.get("kind", "airtime"),
                start_seconds=float(event["start_seconds"]),
                end_seconds=end,
                frames=int(event["frames"]),
                progress_at_end_m=at_end,
                progress_after_window_m=after,
                window_seconds=float(window_seconds),
                window_observed=bool(observed),
                terminal_within_window=bool(terminal_inside),
                score_gain_in_window=gain,
                outcome=outcome,
            )
        )
    return result
