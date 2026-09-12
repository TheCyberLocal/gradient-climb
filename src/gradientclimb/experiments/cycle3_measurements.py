"""Prospective measurement contracts; Cycle 1/2 objectives remain untouched.

These records describe evidence, not a live input authorization mechanism. Native
guards must independently recheck the host immediately before delivering input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

METRIC_VERSION = "behavioral-metrics-3.0"
RECOVERY_VERSION = "recovery-score-ledger-3.0"
LEXICOGRAPHIC_VERSION = "lexicographic-3.0"

EndpointReason = Literal[
    "driver_down",
    "out_of_fuel",
    "natural_unlabeled",
    "horizon",
    "stall_limit",
    "frame_limit",
    "administrative_interrupt",
    "instrumentation_failure",
    "unknown",
    "not_started",
]
NATURAL_REASONS = frozenset({"driver_down", "out_of_fuel", "natural_unlabeled"})
CENSOR_REASONS = frozenset(
    {
        "horizon",
        "stall_limit",
        "frame_limit",
        "administrative_interrupt",
        "instrumentation_failure",
    }
)


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EpisodeEndpoint(FrozenRecord):
    """Why observation stopped; an operational stall limit is a censored stop."""

    reason: EndpointReason
    horizon_seconds: FiniteFloat = Field(gt=0)
    elapsed_seconds: FiniteFloat | None = Field(default=None, ge=0)
    natural_terminal_observed: bool = False

    @model_validator(mode="after")
    def coherent(self):
        natural = self.reason in NATURAL_REASONS
        if natural != self.natural_terminal_observed:
            raise ValueError("A natural termination requires explicit natural terminal evidence")
        if self.reason == "not_started":
            if self.elapsed_seconds not in (None, 0):
                raise ValueError("An unstarted episode cannot contain gameplay time")
        elif self.reason != "unknown" and self.elapsed_seconds is None:
            raise ValueError("A classified endpoint requires measured elapsed time")
        if self.reason == "horizon" and self.elapsed_seconds < self.horizon_seconds:
            raise ValueError("An early administrative deadline is not the episode horizon")
        return self

    @property
    def terminated(self) -> bool | None:
        if self.reason in {"unknown", "not_started"}:
            return None
        return self.reason in NATURAL_REASONS

    @property
    def truncated(self) -> bool:
        return self.reason in {"horizon", "stall_limit", "frame_limit"}

    @property
    def interrupted(self) -> bool:
        return self.reason in {"administrative_interrupt", "instrumentation_failure"}

    @property
    def right_censored(self) -> bool | None:
        if self.reason in {"unknown", "not_started"}:
            return None
        return self.reason in CENSOR_REASONS

    def outcome_flags(self) -> dict:
        """Serialize derived flags explicitly beside the validated endpoint."""
        return {
            **self.model_dump(mode="json"),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "interrupted": self.interrupted,
            "right_censored": self.right_censored,
        }


class GameplayOutcome(FrozenRecord):
    metric_schema_version: Literal["behavioral-metrics-3.0"] = METRIC_VERSION
    episode_id: str = Field(min_length=1)
    endpoint: EpisodeEndpoint
    started_ns: int = Field(ge=0, strict=True)
    boundary_ns: int = Field(ge=0, strict=True)
    distance_m: int | None = Field(default=None, ge=0, strict=True)
    distance_source: Literal[
        "stable_terminal_distance", "verified_paused_frame", "verified_diagnostic_frame", "unknown"
    ] = "unknown"
    distance_evidence_id: str | None = Field(default=None, min_length=1)
    observation_coverage: FiniteFloat | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def coherent(self):
        if self.boundary_ns < self.started_ns:
            raise ValueError("Gameplay boundary precedes its start")
        if (self.distance_m is None) != (self.distance_source == "unknown"):
            raise ValueError("Distance value and known source must occur together")
        if (self.distance_m is None) != (self.distance_evidence_id is None):
            raise ValueError("Known distance requires a retained evidence identifier")
        if self.endpoint.reason == "not_started" and self.distance_m is not None:
            raise ValueError("An unstarted episode has no episode distance")
        if self.distance_source == "stable_terminal_distance" and not self.endpoint.terminated:
            raise ValueError("Terminal distance requires a natural endpoint")
        return self


class Readiness(FrozenRecord):
    ready: bool | None = None
    state: Literal["tune", "paused", "unknown"] = "unknown"
    checked_at_ns: int | None = Field(default=None, ge=0, strict=True)
    failure: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.ready is True and (
            self.state == "unknown" or self.checked_at_ns is None or self.failure is not None
        ):
            raise ValueError("Ready requires a checked stationary state without a failure")
        return self


class SafetyEvent(FrozenRecord):
    timestamp_ns: int = Field(ge=0, strict=True)
    kind: Literal["unintended_action", "manual_intervention", "guard_fault", "release_failure"]
    evidence_id: str = Field(min_length=1)


class SessionSafety(FrozenRecord):
    monitored_from_ns: int = Field(ge=0, strict=True)
    monitored_through_ns: int = Field(ge=0, strict=True)
    assessed_at_ns: int = Field(ge=0, strict=True)
    inputs_released: bool | None = None
    events: tuple[SafetyEvent, ...] = ()

    @model_validator(mode="after")
    def coherent(self):
        if not self.monitored_from_ns <= self.monitored_through_ns <= self.assessed_at_ns:
            raise ValueError("Safety monitoring and assessment times are not ordered")
        if any(e.timestamp_ns > self.assessed_at_ns for e in self.events):
            raise ValueError("A safety assessment cannot contain future events")
        return self


class EligibilityPolicy(FrozenRecord):
    """All learning choices must be supplied by the prospective protocol."""

    policy_id: str = Field(min_length=1)
    allow_censored_learning: bool
    require_readiness_for_learning: bool
    allow_unlabeled_natural_learning: bool
    minimum_observation_coverage: FiniteFloat = Field(ge=0, le=1)


class EligibilityDecision(FrozenRecord):
    policy_id: str
    measurement_valid: bool
    evaluation_eligible: bool
    learning_eligible: bool
    next_episode_eligible: bool
    session_safe: bool
    reasons: tuple[str, ...]


def assess_eligibility(
    outcome: GameplayOutcome, readiness: Readiness, safety: SessionSafety, policy: EligibilityPolicy
) -> EligibilityDecision:
    """Retain valid results after parking failure, while blocking further input.

    Safety contamination during or before the episode blocks evaluation and learning.
    A later fault does not erase completed, verified gameplay evidence. The session
    remains unsafe and next-episode eligibility remains false after any fault.
    """
    reasons = []
    measured = outcome.distance_m is not None
    covered = (
        safety.monitored_from_ns <= outcome.started_ns
        and safety.monitored_through_ns >= outcome.boundary_ns
    )
    contaminated = any(e.timestamp_ns <= outcome.boundary_ns for e in safety.events)
    endpoint_known = outcome.endpoint.reason in NATURAL_REASONS | CENSOR_REASONS
    evaluation = measured and covered and not contaminated and endpoint_known
    ready = readiness.ready is True and (
        outcome.boundary_ns <= readiness.checked_at_ns <= safety.assessed_at_ns
    )
    session_safe = (
        not safety.events
        and safety.inputs_released is True
        and safety.monitored_through_ns == safety.assessed_at_ns
    )
    next_episode = ready and session_safe
    learning = evaluation
    if not measured:
        reasons.append("distance_unknown")
    if not covered:
        reasons.append("gameplay_safety_coverage_unknown")
    if contaminated:
        reasons.append("gameplay_contaminated_by_safety_or_manual_event")
    if not endpoint_known:
        reasons.append("endpoint_unknown_or_not_started")
    if outcome.endpoint.right_censored and not policy.allow_censored_learning:
        learning = False
        reasons.append("censored_learning_excluded_by_protocol")
    if (
        outcome.endpoint.reason == "natural_unlabeled"
        and not policy.allow_unlabeled_natural_learning
    ):
        learning = False
        reasons.append("unlabeled_natural_learning_excluded_by_protocol")
    if (
        outcome.observation_coverage is None
        or outcome.observation_coverage < policy.minimum_observation_coverage
    ):
        learning = False
        reasons.append("learning_observation_coverage_insufficient_or_unknown")
    if not ready:
        reasons.append("next_episode_not_ready")
        if policy.require_readiness_for_learning:
            learning = False
    if not session_safe:
        reasons.append("session_safety_not_established")
    return EligibilityDecision(
        policy_id=policy.policy_id,
        measurement_valid=measured,
        evaluation_eligible=evaluation,
        learning_eligible=learning,
        next_episode_eligible=next_episode,
        session_safe=session_safe,
        reasons=tuple(reasons),
    )


class ScoreIncrement(FrozenRecord):
    """One counter difference over (start_seconds, end_seconds], counted once."""

    increment_id: str = Field(min_length=1)
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(gt=0)
    coins: int = Field(ge=0, strict=True)
    evidence_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Score increments require a positive measured interval")
        return self


class ScoreLedger(FrozenRecord):
    """Complete episode counter partition; missing boundaries remain None."""

    coins_start: int = Field(ge=0, strict=True)
    coins_end: int = Field(ge=0, strict=True)
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(gt=0)
    increments: tuple[ScoreIncrement, ...]

    @model_validator(mode="after")
    def conserved(self):
        if self.coins_end < self.coins_start:
            raise ValueError("Counter decrease requires rejected evidence or a new score segment")
        if not self.increments:
            raise ValueError("A known ledger requires a complete measured interval partition")
        if len({i.increment_id for i in self.increments}) != len(self.increments):
            raise ValueError("Duplicate score increment identifier")
        cursor = self.start_seconds
        for item in self.increments:
            if item.start_seconds != cursor:
                raise ValueError("Score intervals must be ordered, contiguous and nonoverlapping")
            cursor = item.end_seconds
        if cursor != self.end_seconds:
            raise ValueError("Score intervals must reach the episode score boundary")
        if sum(i.coins for i in self.increments) != self.coins_end - self.coins_start:
            raise ValueError("Score increments must conserve the episode boundary difference")
        return self


class RecoveryWindow(FrozenRecord):
    event_id: str = Field(min_length=1)
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(gt=0)
    outcome: Literal["recovered", "failed", "fatal", "censored"]
    fully_observed: bool
    natural_terminal_observed: bool
    evidence_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Recovery window must have positive duration")
        if self.outcome == "fatal":
            if not self.natural_terminal_observed:
                raise ValueError("Fatal recovery requires natural terminal evidence")
        elif self.natural_terminal_observed:
            raise ValueError("Observed natural failure cannot be called recovered or censored")
        if self.outcome in {"recovered", "failed"} and not self.fully_observed:
            raise ValueError("Unobserved recovery windows must be censored")
        return self


class ScoreCredit(FrozenRecord):
    version: Literal["recovery-score-ledger-3.0"] = RECOVERY_VERSION
    eligible: bool
    raw_gain: int | None = None
    credited_gain: FiniteFloat | None = None
    withheld_gain: FiniteFloat | None = None
    allocations: tuple[dict, ...] = ()
    reason: str


def recovery_score_credit(
    ledger: ScoreLedger | None,
    windows: tuple[RecoveryWindow, ...] | None,
    *,
    unrecovered_credit_fraction: float,
) -> ScoreCredit:
    """Count each counter increment once with the strictest overlapping window.

    An interval crossing a window boundary is wholly subject to that window. This
    conservatively handles uncertain collection times without inventing timestamps.
    None means unmeasured windows; an empty tuple means measured absence of events.
    """
    import math

    if (
        isinstance(unrecovered_credit_fraction, bool)
        or not math.isfinite(unrecovered_credit_fraction)
        or not 0 <= unrecovered_credit_fraction <= 1
    ):
        raise ValueError("unrecovered_credit_fraction must be finite and within [0, 1]")
    if ledger is None or windows is None:
        return ScoreCredit(eligible=False, reason="Score boundaries or recovery coverage unknown")
    if len({w.event_id for w in windows}) != len(windows):
        raise ValueError("Duplicate recovery event identifier")
    for window in windows:
        if window.start_seconds >= ledger.end_seconds:
            raise ValueError("Recovery event starts after the episode score boundary")
        if window.end_seconds > ledger.end_seconds and window.outcome in {"recovered", "failed"}:
            raise ValueError(
                "A recovery window beyond the episode boundary must be censored or fatal"
            )
    factors = {
        "recovered": 1.0,
        "failed": unrecovered_credit_fraction,
        "fatal": 0.0,
        "censored": 0.0,
    }
    allocations = []
    credited = 0.0
    for item in ledger.increments:
        overlaps = tuple(
            w
            for w in windows
            if (item.start_seconds < w.end_seconds and item.end_seconds > w.start_seconds)
        )
        factor = min((factors[w.outcome] for w in overlaps), default=1.0)
        amount = item.coins * factor
        credited += amount
        allocations.append(
            {
                "increment_id": item.increment_id,
                "raw_coins": item.coins,
                "overlapping_event_ids": tuple(sorted(w.event_id for w in overlaps)),
                "credit_fraction": factor,
                "credited_coins": amount,
                "evidence_id": item.evidence_id,
            }
        )
    raw = ledger.coins_end - ledger.coins_start
    return ScoreCredit(
        eligible=True,
        raw_gain=raw,
        credited_gain=credited,
        withheld_gain=raw - credited,
        allocations=tuple(allocations),
        reason="Unique interval attribution; raw = credited + withheld",
    )


def lexicographic_key(
    *,
    distance_m: int | None,
    survived_horizon: bool | None,
    useful_score: int | None,
    minimum_distance_m: int,
) -> tuple[int, bool, int, int] | None:
    """Maximize (progress to minimum, survival, distance, score), in that order.

    No scalar conversion or large finite multiplier preserves arbitrary tier
    dominance. Unknown required measurements return no eligible ordering key.
    """
    if type(minimum_distance_m) is not int or minimum_distance_m <= 0:
        raise ValueError("minimum_distance_m must be a positive integer")
    for value in (distance_m, useful_score):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Distance and useful score must be nonnegative integers or unknown")
    if survived_horizon is not None and type(survived_horizon) is not bool:
        raise ValueError("Survival must be a boolean or unknown")
    if distance_m is None or useful_score is None or survived_horizon is None:
        return None
    return min(distance_m, minimum_distance_m), survived_horizon, distance_m, useful_score
