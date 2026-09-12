"""Prospective real-competence accounting. Historical clocks are never inferred.

Episodes measure experience. Compute measures cost. Wall-clock time measures
rapidity. Real-game capability determines whether the learning mattered.
"""

from __future__ import annotations

import json
import stat
import statistics
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from gradientclimb.artifacts import hash_config, sha256_file


def _linked(path: Path) -> bool:
    """Reject Windows junctions too, including on supported Python 3.11."""
    return path.is_symlink() or bool(
        getattr(path.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


Nonnegative = Annotated[FiniteFloat, Field(ge=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PriorClass = Literal[
    "cold-start",
    "demonstration-assisted",
    "simulator-pretrained",
    "generalist-adaptation",
    "fine-tuning",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CostVector(Contract):
    """Additive costs only. Unknown is null; no exchange rate between dimensions.

    GPU utilization-equivalent seconds must use the same device/scope for a
    comparison. Device-wide telemetry cannot establish exclusive learner cost.
    Memory is recorded separately as peaks/means; it must never be summed here.
    """

    cpu_core_seconds: Nonnegative | None = None
    gpu_utilization_equivalent_seconds: Nonnegative | None = None
    simulator_transitions: Annotated[int, Field(ge=0)] | None = None
    simulator_episodes: Annotated[int, Field(ge=0)] | None = None
    simulator_seconds: Nonnegative | None = None
    physics_steps: Annotated[int, Field(ge=0)] | None = None
    rendered_frames: Annotated[int, Field(ge=0)] | None = None
    policy_decisions: Annotated[int, Field(ge=0)] | None = None
    optimizer_updates: Annotated[int, Field(ge=0)] | None = None
    real_game_interaction_seconds: Nonnegative | None = None
    real_game_episodes: Annotated[int, Field(ge=0)] | None = None
    # These are subsets of total CPU/GPU above, not additional additive charges.
    data_generation_cpu_core_seconds: Nonnegative | None = None
    data_generation_gpu_utilization_equivalent_seconds: Nonnegative | None = None


class EvidenceRef(Contract):
    """Artifact-root-relative path; sealed loader verifies the bytes."""

    path: str = Field(min_length=1)
    sha256: Digest


class PriorCost(Contract):
    cost_id: str = Field(min_length=1)
    kind: Literal[
        "human-demonstration",
        "simulator-fitting",
        "teacher-training",
        "distillation",
        "hyperparameter-search",
        "pretraining",
        "generalist-training",
        "fine-tuning",
        "data-generation",
    ]
    parents: list[str] = Field(default_factory=list)
    cost: CostVector
    elapsed_seconds: Nonnegative | None = None
    human_seconds: Nonnegative | None = None
    evidence: list[EvidenceRef] = Field(min_length=1)


class EfficiencyProtocol(Contract):
    schema_version: Literal["learning-efficiency-3.0"] = "learning-efficiency-3.0"
    protocol_id: str = Field(min_length=1)
    registered_at: datetime
    thresholds_m: list[Nonnegative] = Field(default_factory=lambda: [500, 1000, 2000], min_length=1)
    statistic: Literal["median_real_distance_m"] = "median_real_distance_m"
    evaluation_episodes: int = Field(default=20, ge=1)
    evaluation_horizon_seconds: Nonnegative = 900
    training_budget_seconds: Nonnegative
    wall_clock_boundary: Literal["command_start_to_frozen_checkpoint"] = (
        "command_start_to_frozen_checkpoint"
    )
    evaluation_procedure: str = Field(min_length=1)
    resource_accounting: str = Field(min_length=1)
    experience_accounting: str = Field(min_length=1)
    prior_class: PriorClass
    # Explicit compatibility key describes hardware AND measurement coverage/scope.
    compute_comparison_scope: str = Field(min_length=1)
    evidence: EvidenceRef

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        if self.registered_at.utcoffset() is None:
            raise ValueError("registration needs an explicit timezone")
        if sorted(set(self.thresholds_m)) != self.thresholds_m:
            raise ValueError("thresholds must be increasing and unique")
        if self.training_budget_seconds <= 0 or self.evaluation_horizon_seconds <= 0:
            raise ValueError("training budget and evaluation horizon must be positive")
        return self


class CheckpointCost(Contract):
    checkpoint_sha256: Digest
    checkpoint_evidence: EvidenceRef
    elapsed_seconds: Nonnegative
    # This receipt includes imports/setup and checkpoint publication; it is not
    # the old learner-only timer relabeled as command-start time.
    command_clock_evidence: EvidenceRef
    cost: CostVector
    resource_evidence: list[EvidenceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_checkpoint(self) -> Self:
        if self.checkpoint_sha256 != self.checkpoint_evidence.sha256:
            raise ValueError("checkpoint evidence must identify the evaluated weights")
        return self


class RealEpisode(Contract):
    episode_id: str = Field(min_length=1)
    distance_m: Nonnegative | None
    eligible: bool
    endpoint: Literal[
        "natural", "horizon", "stall_limit", "frame_limit", "interrupted", "instrumentation_failure"
    ]


class FrozenRealEvaluation(Contract):
    evaluation_run_id: str = Field(min_length=1)
    checkpoint_sha256: Digest
    protocol_id: str
    profile_id: str
    scenario_sha256: Digest
    policy_kind: Literal["learned"]
    domain: Literal["real-game"] = "real-game"
    frozen: Literal[True] = True
    independent: Literal[True] = True
    learner_updates_during_evaluation: Literal[0] = 0
    split: Literal["development", "qualification"]
    horizon_seconds: Nonnegative
    verification_elapsed_seconds: Nonnegative
    evaluation_elapsed_seconds: Nonnegative
    episodes: list[RealEpisode]
    evaluation_cost: CostVector
    evidence: list[EvidenceRef] = Field(min_length=1)


class LearningEfficiencyStudy(Contract):
    schema_version: Literal["learning-efficiency-study-3.0"] = "learning-efficiency-study-3.0"
    study_id: str = Field(min_length=1)
    training_run_id: str = Field(min_length=1)
    training_seed: int
    started_at: datetime
    protocol: EfficiencyProtocol
    profile_id: str = Field(min_length=1)
    real_scenario: EvidenceRef
    simulator_fidelity_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    evaluation_split: Literal["development", "qualification"]
    prior_roots: list[str]
    policy_prior_roots: list[str]
    prior_costs: list[PriorCost]
    lineage_declared_complete: bool
    checkpoints: list[CheckpointCost]
    evaluations: list[FrozenRealEvaluation]

    @model_validator(mode="after")
    def validate_study(self) -> Self:
        if self.started_at.utcoffset() is None or self.started_at <= self.protocol.registered_at:
            raise ValueError("registration must precede data collection in timezone-aware time")
        if self.protocol.prior_class == "cold-start" and self.policy_prior_roots:
            raise ValueError("cold-start cannot hide inherited training behind a prior root")
        if self.protocol.prior_class != "cold-start" and not self.policy_prior_roots:
            raise ValueError("assisted/adaptation classes require explicit prior lineage")
        if len({x.cost_id for x in self.prior_costs}) != len(self.prior_costs):
            raise ValueError("duplicate prior cost identity")
        if not set(self.policy_prior_roots).issubset(self.prior_roots):
            raise ValueError("policy priors must be included in the full system lineage")
        digests = [x.checkpoint_sha256 for x in self.checkpoints]
        times = [x.elapsed_seconds for x in self.checkpoints]
        if len(set(digests)) != len(digests) or times != sorted(set(times)):
            raise ValueError("checkpoint identities and ordered times must be unique")
        if times and times[-1] > self.protocol.training_budget_seconds:
            raise ValueError("checkpoint exceeds preregistered command-start budget")
        previous: dict = {}
        for point in self.checkpoints:
            for key, value in point.cost.model_dump().items():
                if value is not None and key in previous and value < previous[key]:
                    raise ValueError(f"cumulative checkpoint cost decreased: {key}")
                if value is not None:
                    previous[key] = value
        seen_episodes: set[str] = set()
        seen_evaluations: set[str] = set()
        for evaluation in self.evaluations:
            if evaluation.checkpoint_sha256 not in digests:
                raise ValueError("evaluation must link to a measured checkpoint")
            if evaluation.evaluation_run_id == self.training_run_id:
                raise ValueError("independent evaluation needs a separate run")
            checkpoint = next(
                x for x in self.checkpoints if x.checkpoint_sha256 == evaluation.checkpoint_sha256
            )
            if (
                evaluation.verification_elapsed_seconds
                < checkpoint.elapsed_seconds + evaluation.evaluation_elapsed_seconds
            ):
                raise ValueError("evaluation completion cannot precede checkpoint production")
            if evaluation.evaluation_run_id in seen_evaluations:
                raise ValueError("duplicate evaluation run")
            seen_evaluations.add(evaluation.evaluation_run_id)
            for episode in evaluation.episodes:
                if episode.episode_id in seen_episodes:
                    raise ValueError("real executions cannot be counted twice")
                seen_episodes.add(episode.episode_id)
        _lineage(self)  # cycles are invalid; missing ancestors remain explicitly unknown.
        return self


def _sum_costs(costs: list[CostVector], *, complete: bool = True) -> dict:
    return {
        key: sum(values) if complete and all(x is not None for x in values) else None
        for key in CostVector.model_fields
        for values in [[getattr(cost, key) for cost in costs]]
    }


def _lineage(study: LearningEfficiencyStudy) -> tuple[list[PriorCost], list[str]]:
    nodes = {node.cost_id: node for node in study.prior_costs}
    visited: set[str] = set()
    active: set[str] = set()
    missing: set[str] = set()

    def visit(identity: str) -> None:
        if identity in active:
            raise ValueError("prior lineage contains a cycle")
        if identity in visited:
            return
        if identity not in nodes:
            missing.add(identity)
            return
        active.add(identity)
        for parent in nodes[identity].parents:
            visit(parent)
        active.remove(identity)
        visited.add(identity)

    for root in study.prior_roots:
        visit(root)
    # Unreferenced supplied costs must not silently disappear from a claimed lineage.
    if set(nodes) - visited:
        raise ValueError("prior cost entries must be reachable from declared roots")
    return [nodes[key] for key in sorted(visited)], sorted(missing)


def analyze_study(study: LearningEfficiencyStudy | dict) -> dict:
    """Analyze supplied attestations; use load_studies to verify artifact bytes.

    First observed passing checkpoint is an upper bound on acquisition time.
    The interval between evaluated checkpoints does not prove monotonic learning.
    No eligible independent evaluation means unknown, not zero or a failed policy.
    """
    study = LearningEfficiencyStudy.model_validate(study)
    protocol = study.protocol
    priors, missing = _lineage(study)
    complete = study.lineage_declared_complete and not missing
    prior_cost = _sum_costs([node.cost for node in priors], complete=complete)
    warnings = []
    if not complete:
        warnings.append(f"Inherited costs incomplete; missing ancestors: {missing}")
    curve = []
    for point in study.checkpoints:
        matched = [e for e in study.evaluations if e.checkpoint_sha256 == point.checkpoint_sha256]
        eligible = [
            e
            for e in matched
            if (
                e.protocol_id == protocol.protocol_id
                and e.profile_id == study.profile_id
                and e.scenario_sha256 == study.real_scenario.sha256
                and e.split == study.evaluation_split
                and e.horizon_seconds == protocol.evaluation_horizon_seconds
                and len(e.episodes) == protocol.evaluation_episodes
                and all(
                    x.eligible and x.distance_m is not None and x.endpoint in {"natural", "horizon"}
                    for x in e.episodes
                )
            )
        ]
        # A single declared set prevents selecting the luckiest repeated evaluation.
        if len(matched) != 1 or len(eligible) != 1:
            warnings.append(
                f"Checkpoint {point.checkpoint_sha256}: missing, ambiguous or ineligible evaluation"
            )
            continue
        evaluation = eligible[0]
        distances = sorted(x.distance_m for x in evaluation.episodes)
        curve.append(
            {
                "time_seconds": point.elapsed_seconds,
                "checkpoint_sha256": point.checkpoint_sha256,
                "median_distance_m": statistics.median(distances),
                "minimum_distance_m": min(distances),
                "quartiles_distance_m": statistics.quantiles(distances, n=4, method="inclusive")
                if len(distances) > 1
                else [distances[0]] * 3,
                "distance_distribution_m": distances,
                "episodes": len(distances),
                "training_seeds": 1,
                "evaluation_run_id": evaluation.evaluation_run_id,
                "verification_elapsed_seconds": evaluation.verification_elapsed_seconds,
                "evaluation_elapsed_seconds": evaluation.evaluation_elapsed_seconds,
                "own_cost": point.cost.model_dump(),
                "evaluation_cost": evaluation.evaluation_cost.model_dump(),
                "diagnostic_rates_per_command_second": {
                    key: value / point.elapsed_seconds
                    if value is not None and point.elapsed_seconds > 0
                    else None
                    for key, value in point.cost.model_dump().items()
                    if key
                    in {
                        "physics_steps",
                        "rendered_frames",
                        "policy_decisions",
                        "optimizer_updates",
                        "simulator_transitions",
                    }
                },
            }
        )
    thresholds = []
    for threshold in protocol.thresholds_m:
        passing = next((x for x in curve if x["median_distance_m"] >= threshold), None)
        earlier = [x for x in curve if passing and x["time_seconds"] < passing["time_seconds"]]
        selected = passing or (curve[-1] if curve else None)
        own = selected["own_cost"] if selected else CostVector().model_dump()
        thresholds.append(
            {
                "threshold_m": threshold,
                "status": "reached" if passing else "right_censored" if curve else "not_evaluable",
                "time_seconds": passing["time_seconds"] if passing else None,
                "censor_seconds": curve[-1]["time_seconds"] if curve and not passing else None,
                "interval_lower_seconds": earlier[-1]["time_seconds"] if earlier else None,
                "budget_seconds": protocol.training_budget_seconds,
                "checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
                "own_cost": own,
                "prior_cost": prior_cost,
                "combined_cost": _sum_costs([CostVector(**own), CostVector(**prior_cost)]),
            }
        )
    return {
        "schema_version": "efficiency-analysis-3.0",
        "study_id": study.study_id,
        "training_run_id": study.training_run_id,
        "training_seed": study.training_seed,
        "method": study.method,
        "prior_class": protocol.prior_class,
        "profile_id": study.profile_id,
        "scenario_sha256": study.real_scenario.sha256,
        "simulator_fidelity_id": study.simulator_fidelity_id,
        "protocol_id": protocol.protocol_id,
        "evaluation_split": study.evaluation_split,
        "compute_comparison_scope": protocol.compute_comparison_scope,
        "comparison_contract_sha256": hash_config(
            {
                "statistic": protocol.statistic,
                "training_budget_seconds": protocol.training_budget_seconds,
                "evaluation_episodes": protocol.evaluation_episodes,
                "evaluation_horizon_seconds": protocol.evaluation_horizon_seconds,
                "evaluation_procedure": protocol.evaluation_procedure,
                "wall_clock_boundary": protocol.wall_clock_boundary,
                "resource_accounting": protocol.resource_accounting,
                "experience_accounting": protocol.experience_accounting,
            }
        ),
        "wall_clock_boundary": protocol.wall_clock_boundary,
        "thresholds": thresholds,
        "checkpoint_curve": curve,
        "lineage_complete": complete,
        "prior_ledger": [x.model_dump(mode="json") for x in priors],
        "policy_prior_roots": study.policy_prior_roots,
        "evaluation_cost": _sum_costs([e.evaluation_cost for e in study.evaluations]),
        "provenance_status": "supplied_attestations_not_file_verified",
        "warnings": warnings,
        "interpretation": "First observed passing frozen checkpoint, not exact acquisition time; censor at last eligible evaluation. Prior elapsed times are separate and are not summed into a fictitious serial clock.",
    }


def compare_studies(
    reports: list[dict], threshold_m: float, *, axes: list[str] | None = None
) -> dict:
    """Pareto sets inside comparable cohorts; missing/censored costs cannot dominate."""
    axes = axes or ["time_seconds", "cpu_core_seconds", "real_game_interaction_seconds"]
    if (
        not axes
        or len(set(axes)) != len(axes)
        or any(key != "time_seconds" and key not in CostVector.model_fields for key in axes)
    ):
        raise ValueError("declare unique measured Pareto dimensions")
    cohorts: dict[tuple, list[tuple[dict, list[float]]]] = {}
    excluded = []
    for report in reports:
        if "error" in report:
            excluded.append({"study_id": report["study_id"], "reason": report["error"]})
            continue
        point = next((x for x in report["thresholds"] if x["threshold_m"] == threshold_m), None)
        values = (
            [
                point["time_seconds"] if key == "time_seconds" else point["combined_cost"].get(key)
                for key in axes
            ]
            if point
            else []
        )
        if (
            not point
            or point["status"] != "reached"
            or not report["lineage_complete"]
            or any(x is None for x in values)
        ):
            excluded.append(
                {
                    "study_id": report["study_id"],
                    "reason": "unreached threshold or incomplete measured costs/lineage",
                }
            )
            continue
        cohort = tuple(
            report[k]
            for k in (
                "protocol_id",
                "profile_id",
                "scenario_sha256",
                "prior_class",
                "evaluation_split",
                "compute_comparison_scope",
                "comparison_contract_sha256",
            )
        )
        cohorts.setdefault(cohort, []).append((report, values))
    result = []
    for cohort, entries in cohorts.items():
        for index, (report, values) in enumerate(entries):
            dominated_by = [
                other["study_id"]
                for j, (other, costs) in enumerate(entries)
                if j != index
                and all(a <= b for a, b in zip(costs, values, strict=True))
                and any(a < b for a, b in zip(costs, values, strict=True))
            ]
            result.append(
                {
                    "study_id": report["study_id"],
                    "cohort": list(cohort),
                    "values": dict(zip(axes, values, strict=True)),
                    "pareto_frontier": not dominated_by,
                    "dominated_by": dominated_by,
                }
            )
    return {
        "threshold_m": threshold_m,
        "axes": axes,
        "comparisons": result,
        "excluded": excluded,
        "note": "Descriptive per-run Pareto set; repeated training seeds and uncertainty remain separate. No opaque score.",
    }


def _evidence_refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield EvidenceRef(**value)
        else:
            for child in value.values():
                yield from _evidence_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _evidence_refs(child)


def _verify_references(root: Path, study: LearningEfficiencyStudy) -> None:
    for reference in _evidence_refs(study.model_dump(mode="json")):
        relative = Path(reference.path)
        if relative.is_absolute() or ".." in relative.parts or ":" in reference.path:
            raise ValueError("evidence reference must stay within artifact root")
        candidate = root / relative
        if any(
            _linked(p)
            for p in [candidate, *candidate.parents]
            if p != root and p.is_relative_to(root)
        ):
            raise ValueError("evidence references cannot traverse links")
        if (
            not candidate.resolve().is_relative_to(root)
            or sha256_file(candidate) != reference.sha256
        ):
            raise ValueError(f"evidence hash mismatch: {reference.path}")


def publish_study(root: str | Path, study: LearningEfficiencyStudy | dict) -> dict:
    """Seal a new analysis input and report, without modifying source experiments."""
    from gradientclimb.experiments.recorder import RunRecorder, load_run

    root = Path(root).resolve()
    study = LearningEfficiencyStudy.model_validate(study)
    _verify_references(root, study)
    report = analyze_study(study)
    with RunRecorder(
        root,
        "real-learning-efficiency-analysis",
        study.model_dump(mode="json"),
        study.training_seed,
        algorithm="analysis",
        environment="real-game-evidence",
        parent_run=study.training_run_id,
        telemetry_interval_seconds=0,
    ) as run:
        for name, value, kind in [
            (
                "learning-efficiency.json",
                study.model_dump(mode="json"),
                "learning_efficiency_study",
            ),
            ("learning-efficiency-analysis.json", report, "learning_efficiency_analysis"),
        ]:
            path = run.directory / name
            with path.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            run.register_artifact(path, kind)
        run.finalize(
            analysis_version="efficiency-analysis-3.0",
            study_id=study.study_id,
            new_training_episodes=0,
            new_evaluation_episodes=0,
            source_independence="requires_protocol_audit",
        )
    return load_run(root, run.run_id)


def load_studies(root: str | Path) -> list[dict]:
    """Read only explicit new sealed study artifacts; never reinterpret old runs.

    Hash checks establish byte identity, not the truth of independence assertions.
    Scan canonical run directories only; do not follow worktrees or junctions.
    Invalid envelopes are returned as visible errors rather than silently omitted.
    """
    from gradientclimb.experiments.recorder import verify_run

    root = Path(root).resolve()
    reports = []
    runs = root / "runs"
    if not runs.is_dir() or _linked(runs):
        return reports
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir() or _linked(directory):
            continue
        envelope = directory / "learning-efficiency.json"
        if not envelope.is_file():
            continue
        try:
            if _linked(envelope):
                raise ValueError("study envelope cannot be a link")
            study = LearningEfficiencyStudy.model_validate_json(
                envelope.read_text(encoding="utf-8")
            )
            integrity = verify_run(root, directory.name)
            if not integrity["valid"]:
                raise ValueError("study's canonical run seal failed")
            record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            if not any(
                a["sha256"] == sha256_file(envelope) and a["kind"] == "learning_efficiency_study"
                for a in record["artifact_manifest"]
            ):
                raise ValueError("study envelope must be a registered sealed artifact")
            _verify_references(root, study)
            report = analyze_study(study)
            report["provenance_status"] = "sealed_envelope_and_referenced_bytes_verified"
            report["warnings"].append(
                "Artifact identity verified; independence and measurement validity still require protocol audit."
            )
            reports.append(report)
        except (ValueError, OSError, KeyError, TypeError) as error:
            reports.append(
                {
                    "schema_version": "efficiency-analysis-3.0",
                    "study_id": directory.name,
                    "error": str(error),
                    "thresholds": [],
                    "checkpoint_curve": [],
                }
            )
    return reports
