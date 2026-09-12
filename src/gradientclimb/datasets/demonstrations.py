"""Assemble reviewed human-observation windows without inventing action delivery.

All references are portable paths relative to the project root. A frozen ledger
assigns entire recording sessions. Review documents attest visual coverage; hash
verification establishes byte identity, not the truth of the review. This module
never extracts features, trains, captures the desktop, or modifies source runs.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from gradientclimb.artifacts import canonical_json, hash_config, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative, _walk
from gradientclimb.capture.demonstrations import causal_action_pairs, read_complete_journal
from gradientclimb.environments.profiles import UpgradeLevel

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0, strict=True)]
Seconds = Annotated[FiniteFloat, Field(gt=0, strict=True)]
Purpose = Literal["imitation", "development", "human-benchmark", "construction", "qualification"]
Split = Literal["train", "development", "human-benchmark", "construction", "qualification"]


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EvidenceRef(FrozenContract):
    path: str
    sha256: Digest

    @model_validator(mode="after")
    def portable(self) -> Self:
        _relative(self.path)
        return self


class SessionAssignment(FrozenContract):
    run_id: Name
    session_id: Name
    source_purpose: Purpose
    split: Split

    @model_validator(mode="after")
    def valid_assignment(self) -> Self:
        if "/" in self.run_id or _relative(self.run_id) != self.run_id:
            raise ValueError("Run identity must be a single portable path component")
        if self.split == "train" and self.source_purpose != "imitation":
            raise ValueError("Only an imitation-purpose session may enter training")
        if self.split == "development" and self.source_purpose not in {"imitation", "development"}:
            raise ValueError(
                "Benchmark, qualification and construction purposes cannot enter development"
            )
        if (
            self.source_purpose in {"human-benchmark", "qualification"}
            and self.split != self.source_purpose
        ):
            raise ValueError("Protected source purposes cannot be relabeled")
        return self


class PartitionLedger(FrozenContract):
    schema_version: Literal["demonstration-partitions-3.0"] = "demonstration-partitions-3.0"
    ledger_id: Name
    created_at: datetime
    sessions: tuple[SessionAssignment, ...] = Field(min_length=1, max_length=10000)
    note: Name

    @model_validator(mode="after")
    def whole_sessions(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError("Partition creation time requires a timezone")
        if len({s.run_id for s in self.sessions}) != len(self.sessions) or len(
            {s.session_id for s in self.sessions}
        ) != len(self.sessions):
            raise ValueError("Whole sessions and recording runs must have one partition assignment")
        return self


class DemonstrationSource(FrozenContract):
    run_id: Name
    session_id: Name
    run_sha256: Digest
    configuration_sha256: Digest
    frames: EvidenceRef
    controls: EvidenceRef


class ReviewedProfile(FrozenContract):
    profile_id: Name
    vehicle_name: Name | None = None
    map_name: Name | None = None
    game_build: Name | None = None
    upgrades: tuple[UpgradeLevel, ...] = ()
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    field_evidence: dict[
        Literal["vehicle_name", "map_name", "game_build", "upgrades"], tuple[EvidenceRef, ...]
    ] = Field(default_factory=dict)
    limitations: Name

    @model_validator(mode="after")
    def known_fields_are_supported(self) -> Self:
        available = {(ref.path, ref.sha256) for ref in self.evidence}
        for name in ("vehicle_name", "map_name", "game_build", "upgrades"):
            references = self.field_evidence.get(name, ())
            if getattr(self, name) and not references:
                raise ValueError(f"Known profile field needs explicit evidence: {name}")
            if any((ref.path, ref.sha256) not in available for ref in references):
                raise ValueError("Profile field evidence must be included in profile evidence")
        if len({upgrade.name for upgrade in self.upgrades}) != len(self.upgrades):
            raise ValueError("Reviewed upgrade names must be unique")
        return self

    @property
    def identity_complete(self) -> bool:
        return bool(
            self.vehicle_name
            and self.map_name
            and self.game_build
            and self.upgrades
            and all(u.observed_level is not None for u in self.upgrades)
        )


class ReviewedSegment(FrozenContract):
    segment_id: Name
    episode_id: Name | None = None
    first_frame_index: Count
    end_frame_index_exclusive: Count
    started_ns: Count
    end_ns_exclusive: Count
    state: Literal["playing"] = "playing"
    start_context: Literal["episode_start", "continuation", "unknown"]
    end_context: Literal["natural_terminal", "pause", "menu", "capture_end", "unknown"]
    # Every frame in the index span must be individually represented here.
    frame_evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=10000)
    note: Name

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if (
            self.first_frame_index >= self.end_frame_index_exclusive
            or self.started_ns >= self.end_ns_exclusive
        ):
            raise ValueError("Reviewed segment boundaries must increase")
        if len({e.path for e in self.frame_evidence}) != len(self.frame_evidence):
            raise ValueError("Duplicate reviewed frame evidence")
        return self


class SegmentationReview(FrozenContract):
    schema_version: Literal["demonstration-segmentation-3.0"] = "demonstration-segmentation-3.0"
    review_id: Name
    reviewer: Name
    reviewed_at: datetime
    source: DemonstrationSource
    accepted_capture_status: Literal["completed", "failed", "cancelled"]
    capture_failure_review_note: Name | None = None
    profile: ReviewedProfile
    segments: tuple[ReviewedSegment, ...] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def review_is_explicit(self) -> Self:
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("Review time requires a timezone")
        if self.accepted_capture_status != "completed" and not self.capture_failure_review_note:
            raise ValueError("Failed/cancelled capture needs an explicit review acceptance note")
        if len({s.segment_id for s in self.segments}) != len(self.segments):
            raise ValueError("Segment identities must be unique within a session")
        previous = None
        for segment in self.segments:
            if previous and (
                segment.first_frame_index < previous.end_frame_index_exclusive
                or segment.started_ns < previous.end_ns_exclusive
            ):
                raise ValueError("Reviewed segments must be ordered and nonoverlapping")
            previous = segment
        return self


class WindowPlan(FrozenContract):
    schema_version: Literal["imitation-window-plan-3.0"] = "imitation-window-plan-3.0"
    dataset_id: Name
    artifact_root: str = "artifacts"
    partition_ledger: EvidenceRef
    reviews: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=64)
    split: Literal["train", "development"]
    history_frames: Annotated[int, Field(ge=1, le=16, strict=True)] = 4
    maximum_frame_gap_seconds: Seconds = 0.25
    maximum_frame_age_seconds: Seconds = 0.45
    maximum_target_lag_seconds: Seconds = 0.1
    maximum_poll_gap_seconds: Seconds = 0.05
    maximum_source_frames: Annotated[int, Field(ge=1, le=100000, strict=True)] = 10000
    maximum_source_control_samples: Annotated[int, Field(ge=1, le=1000000, strict=True)] = 200000
    maximum_journal_bytes: Annotated[int, Field(ge=1, le=1024**3, strict=True)] = 128 * 1024**2
    prior_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    note: Name

    @model_validator(mode="after")
    def portable_root(self) -> Self:
        _relative(self.artifact_root)
        if len({r.path for r in self.reviews}) != len(self.reviews):
            raise ValueError("Review references must be unique")
        return self


class _Frame(FrozenContract):
    frame_index: Count
    started_ns: Count
    completed_ns: Count
    observation_ready_ns: Count
    timestamp_ns: Count
    path: str
    file_sha256: Digest

    @model_validator(mode="after")
    def ordered(self) -> Self:
        _relative(self.path)
        if (
            not self.started_ns
            <= self.timestamp_ns
            <= self.completed_ns
            <= self.observation_ready_ns
        ):
            raise ValueError("Frame timestamps are not causally ordered")
        return self


def _read_ref(root: Path, reference: EvidenceRef) -> Path:
    path = _inside(root, reference.path)
    if not path.is_file() or sha256_file(path) != reference.sha256:
        raise ValueError(f"Evidence hash mismatch: {reference.path}")
    return path


def _load_ref(root: Path, reference: EvidenceRef, model):
    return model.model_validate_json(_read_ref(root, reference).read_bytes())


def _source(
    root: Path, plan: WindowPlan, review: SegmentationReview, assignment: SessionAssignment
):
    from gradientclimb.experiments import verify_run

    source = review.source
    prefix = f"{plan.artifact_root}/runs/{assignment.run_id}"
    run_ref = EvidenceRef(path=f"{prefix}/run.json", sha256=source.run_sha256)
    config_ref = EvidenceRef(path=f"{prefix}/config.json", sha256=source.configuration_sha256)
    record = json.loads(_read_ref(root, run_ref).read_bytes())
    config = json.loads(_read_ref(root, config_ref).read_bytes())
    if source.run_id != assignment.run_id or source.session_id != assignment.session_id:
        raise ValueError("Review source does not match the whole-session partition")
    if (
        config.get("session_id") != assignment.session_id
        or config.get("purpose") != assignment.source_purpose
        or config.get("source_partition") != assignment.source_purpose
        or config.get("version") != "human-demonstration-3.0"
        or config.get("input_injection") is not False
        or config.get("controls") != {"gas": "VK_RIGHT", "brake": "VK_LEFT"}
        or record.get("algorithm") != "human"
        or config != record.get("configuration")
        or record.get("run_id") != assignment.run_id
    ):
        raise ValueError("Original session identity, purpose or recorder configuration disagrees")
    if assignment.source_purpose in {
        "human-benchmark",
        "qualification",
        "construction",
    } or config.get("benchmark_excluded_from_training"):
        raise ValueError("Protected/non-imitation source purpose cannot supply dataset windows")
    if record.get("status") != review.accepted_capture_status:
        raise ValueError("Review must explicitly accept the actual capture status")
    # Validate every seal reference before verify_run follows it, including reparse points.
    registered = set()
    for artifact in record["artifact_manifest"]:
        _inside(root, f"{prefix}/{_relative(artifact['path'])}")
        registered.add(artifact["sha256"])
    for journal in ("metrics.jsonl", "telemetry.jsonl", "progress.jsonl", "seal.json"):
        _inside(root, f"{prefix}/{journal}")
    for _ in _walk(_inside(root, prefix)):
        pass  # Only this explicit source run; reject links before canonical recursive checking.
    integrity = verify_run(_inside(root, plan.artifact_root), assignment.run_id)
    if not integrity["valid"]:
        raise ValueError(f"Source run seal failed: {integrity['errors']}")
    journals = []
    for reference, name in ((source.frames, "frames.jsonl"), (source.controls, "controls.jsonl")):
        if reference.path != f"{prefix}/demonstration/{name}" or reference.sha256 not in registered:
            raise ValueError("Source journals must identify the registered demonstration payload")
        path = _read_ref(root, reference)
        if path.stat().st_size > plan.maximum_journal_bytes:
            raise ValueError("Source journal exceeds declared byte bound")
        journals.append(read_complete_journal(path))
    if (
        len(journals[0]["rows"]) > plan.maximum_source_frames
        or len(journals[1]["rows"]) > plan.maximum_source_control_samples
    ):
        raise ValueError("Source journal exceeds declared row bound")
    frames = [
        _Frame.model_validate({key: row[key] for key in _Frame.model_fields})
        for row in journals[0]["rows"]
    ]
    if [f.frame_index for f in frames] != list(range(len(frames))) or any(
        b.started_ns <= a.completed_ns for a, b in pairwise(frames)
    ):
        raise ValueError("Source frames must be unique and strictly ordered")
    controls = journals[1]["rows"]
    if any(
        type(row.get("sample_index")) is not int or row["sample_index"] != index
        for index, row in enumerate(controls)
    ):
        raise ValueError("Control sample identities must match their unique journal positions")
    # Validate control ordering/types even when every source frame is outside review.
    causal_action_pairs([], controls)
    for reference in review.profile.evidence:
        _read_ref(root, reference)
    return frames, controls, journals, record, config, registered


def assemble_windows(project_root: str | Path, plan: WindowPlan | dict) -> dict:
    """Pure derived result; no output or source file is written by this function."""
    root = Path(project_root).absolute()
    _no_links(root)
    plan = WindowPlan.model_validate(plan)
    ledger = _load_ref(root, plan.partition_ledger, PartitionLedger)
    assignments = {row.run_id: row for row in ledger.sessions}
    for reference in plan.prior_evidence:
        _read_ref(root, reference)
    windows, exclusions, source_reports = [], [], []
    seen_sessions = set()
    total_frames = total_controls = 0
    for review_ref in plan.reviews:
        review = _load_ref(root, review_ref, SegmentationReview)
        assignment = assignments.get(review.source.run_id)
        if assignment is None or assignment.split != plan.split:
            raise ValueError("Every reviewed source must belong to the selected ledger split")
        if assignment.session_id in seen_sessions:
            raise ValueError("A session cannot be supplied by multiple reviews")
        seen_sessions.add(assignment.session_id)
        frames, controls, journals, record, config, registered = _source(
            root, plan, review, assignment
        )
        total_frames += len(frames)
        total_controls += len(controls)
        if (
            total_frames > plan.maximum_source_frames
            or total_controls > plan.maximum_source_control_samples
        ):
            raise ValueError("Combined source journals exceed the declared dataset row bound")
        by_index = {}
        for segment in review.segments:
            if segment.end_frame_index_exclusive > len(frames):
                raise ValueError("Reviewed frame span exceeds the complete source journal")
            evidence = {ref.path: ref.sha256 for ref in segment.frame_evidence}
            expected = {}
            for frame in frames[segment.first_frame_index : segment.end_frame_index_exclusive]:
                path = f"{Path(review.source.frames.path).parent.as_posix()}/{frame.path}"
                expected[path] = frame.file_sha256
                if frame.file_sha256 not in registered:
                    raise ValueError(
                        "Reviewed frame is absent from the registered source artifacts"
                    )
                _read_ref(root, EvidenceRef(path=path, sha256=frame.file_sha256))
                by_index[frame.frame_index] = segment
            if evidence != expected:
                raise ValueError(
                    "Every frame in the reviewed span needs its exact evidence reference"
                )
        pairs = causal_action_pairs(
            [{**f.model_dump(), "state_evidence": {"state": "playing"}} for f in frames],
            controls,
            maximum_lag_seconds=plan.maximum_target_lag_seconds,
            maximum_poll_gap_seconds=plan.maximum_poll_gap_seconds,
        )
        history = deque(maxlen=plan.history_frames)
        previous_segment = previous_frame = None
        history_reset = None
        resets = Counter()
        for frame, pair in zip(frames, pairs, strict=True):
            segment = by_index.get(frame.frame_index)
            reason = None
            if segment is None:
                reason = "outside_reviewed_playing_segment"
            elif (
                not segment.started_ns
                <= frame.started_ns
                <= frame.completed_ns
                <= frame.observation_ready_ns
                < segment.end_ns_exclusive
            ):
                reason = "observation_outside_reviewed_time_boundary"
            elif (
                frame.observation_ready_ns - frame.started_ns > plan.maximum_frame_age_seconds * 1e9
            ):
                reason = "observation_age_exceeds_bound"
            if reason:
                history.clear()
                previous_segment = previous_frame = None
            else:
                if segment.segment_id != previous_segment:
                    history.clear()
                    resets["segment_boundary"] += 1
                    history_reset = {
                        "reason": "segment_boundary_or_excluded_observation",
                        "frame_index": frame.frame_index,
                        "timestamp_ns": frame.timestamp_ns,
                    }
                elif (
                    frame.timestamp_ns - previous_frame.timestamp_ns
                    > plan.maximum_frame_gap_seconds * 1e9
                ):
                    history.clear()
                    resets["frame_gap"] += 1
                    history_reset = {
                        "reason": "frame_gap",
                        "frame_index": frame.frame_index,
                        "timestamp_ns": frame.timestamp_ns,
                    }
                history.append(frame)
                previous_segment, previous_frame = segment.segment_id, frame
                if not pair["eligible"]:
                    reason = "control_target_unbracketed_overlapping_or_late"
                else:
                    index = pair["target_control_sample_index"]
                    target, before = controls[index], controls[index - 1]
                    if not (
                        segment.started_ns <= before["started_ns"]
                        and target["completed_ns"] < segment.end_ns_exclusive
                        and all(f.observation_ready_ns <= target["started_ns"] for f in history)
                    ):
                        reason = "control_bracket_or_history_outside_segment"
                    elif len(history) < plan.history_frames:
                        reason = "incomplete_history_no_padding"
            if reason:
                exclusions.append(
                    {
                        "run_id": assignment.run_id,
                        "frame_index": frame.frame_index,
                        "reason": reason,
                    }
                )
                continue
            target = controls[pair["target_control_sample_index"]]
            content = {
                "run_id": assignment.run_id,
                "session_id": assignment.session_id,
                "source_purpose": assignment.source_purpose,
                "split": assignment.split,
                "review": review_ref.model_dump(),
                "segment_id": segment.segment_id,
                "episode_id": segment.episode_id,
                "profile": review.profile.model_dump(mode="json"),
                "training_eligible": assignment.split == "train"
                and review.profile.identity_complete,
                "eligibility": "construction_only_profile_incomplete"
                if not review.profile.identity_complete
                else "training_candidate"
                if assignment.split == "train"
                else "development_only",
                "history_reset": history_reset,
                "frames": [
                    {
                        **f.model_dump(),
                        "path": f"{Path(review.source.frames.path).parent.as_posix()}/{f.path}",
                    }
                    for f in history
                ],
                "target": {
                    "control_sample_index": pair["target_control_sample_index"],
                    "started_ns": target["started_ns"],
                    "completed_ns": target["completed_ns"],
                    "gas": target["gas"],
                    "brake": target["brake"],
                    "code": int(target["gas"]) + 2 * int(target["brake"]),
                    "semantics": "observed_focused_key_state_poll; not_intent_or_OS_delivery",
                    "action_duration_seconds": None,
                    "target_lag_seconds": (target["completed_ns"] - frame.observation_ready_ns)
                    / 1e9,
                    "preceding_bracket_sample_index": pair["target_control_sample_index"] - 1,
                    "simultaneous_change_order": target.get("simultaneous_change_order"),
                },
                "previous_os_action": None,
            }
            windows.append({"window_id": hash_config(content), **content})
        source_reports.append(
            {
                "source": review.source.model_dump(),
                "assignment": assignment.model_dump(),
                "capture_status": record["status"],
                "capture_failure_review_note": review.capture_failure_review_note,
                "source_git_sha": record.get("git_sha"),
                "original_operator_configuration_note": config.get("operator_configuration_note"),
                "original_configuration_independently_verified": config.get(
                    "configuration_independently_verified"
                ),
                "reviewed_profile_identity_complete": review.profile.identity_complete,
                "source_frames": len(frames),
                "source_control_samples": len(controls),
                "trailing_partial_bytes": {
                    "frames": journals[0]["trailing_partial_bytes"],
                    "controls": journals[1]["trailing_partial_bytes"],
                },
                "history_resets": dict(sorted(resets.items())),
                "prior_cost": {
                    "cost_id": f"human-demonstration:{assignment.run_id}",
                    "kind": "human-demonstration",
                    "capture_wall_seconds": record.get("summary", {})
                    .get("demonstration", {})
                    .get("capture_wall_clock_seconds"),
                    "recorder_wall_seconds": record.get("wall_clock_seconds"),
                    "resource_accounting": record.get("summary", {}).get("resources"),
                    "human_practice_seconds": config.get("prior_knowledge", {}).get(
                        "human_practice_seconds"
                    ),
                    "source_prior_declaration": config.get("prior_knowledge"),
                    "scope": "source_recording_and_declared_priors; unknown_costs_remain_unknown",
                },
            }
        )
    result = {
        "schema_version": "imitation-windows-3.0",
        "dataset_id": plan.dataset_id,
        "plan": plan.model_dump(mode="json"),
        "plan_sha256": hash_config(plan.model_dump(mode="json")),
        "partition_ledger_sha256": plan.partition_ledger.sha256,
        "sources": source_reports,
        "windows": windows,
        "exclusions": exclusions,
        "summary": {
            "windows": len(windows),
            "unique_source_frames": len(
                {(w["run_id"], f["frame_index"]) for w in windows for f in w["frames"]}
            ),
            "excluded_target_frames": len(exclusions),
            "exclusion_reasons": dict(sorted(Counter(e["reason"] for e in exclusions).items())),
            "target_state_counts": dict(
                sorted(Counter(str(w["target"]["code"]) for w in windows).items())
            ),
            "independent_sessions": len(seen_sessions),
            "training_eligible": bool(windows) and all(w["training_eligible"] for w in windows),
            "training_eligible_windows": sum(w["training_eligible"] for w in windows),
            "construction_only_profile_incomplete_windows": sum(
                w["eligibility"] == "construction_only_profile_incomplete" for w in windows
            ),
            "new_gameplay_episodes": 0,
            "reviewed_complete_episode_count": None,
            "feature_extraction_performed": False,
            "training_performed": False,
            "qualification_evidence": False,
        },
        "limitations": [
            "Review is an attestation, not machine-verified visual truth.",
            "Recorded ordering does not establish human reaction or delivered game actions.",
            "No frame padding, inferred episode duration, fitted feature cache or training targets from actionless video.",
            "One frozen whole-session ledger must govern every consumer; new ledgers must not reassign consumed sessions.",
        ],
    }
    result["dataset_sha256"] = hash_config(result)
    return result


def publish_dataset(project_root: str | Path, plan_path: str) -> dict:
    """Seal a new derived run; sampled resources cover assembly inside the recorder."""
    from gradientclimb.datasets.partitions import partition_publication
    from gradientclimb.experiments import RunRecorder, verify_run

    root = Path(project_root).absolute()
    _no_links(root)
    path = _inside(root, plan_path)
    plan = WindowPlan.model_validate_json(path.read_bytes())
    with (
        partition_publication(root, plan) as binding,
        RunRecorder(
            _inside(root, plan.artifact_root),
            "reviewed-human-imitation-windows",
            binding.configuration(plan.model_dump(mode="json")),
            algorithm="offline-window-assembly-3.0",
            environment="saved-human-demonstration",
            evidence_domain="derived_real_demonstration_references",
            qualifies_real_game=False,
            source_root=root,
        ) as run,
    ):
        binding.register(run)
        run.register_artifact(path, "imitation_window_plan")
        dataset = assemble_windows(root, plan)
        for reference, kind in [
            *((ref, "segmentation_review") for ref in plan.reviews),
            *((ref, "prior_evidence") for ref in plan.prior_evidence),
        ]:
            run.register_artifact(_read_ref(root, reference), kind)
        output = run.directory / "imitation-windows.json"
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(dataset) + "\n")
        artifact = run.register_artifact(output, "imitation_window_dataset")
        run.finalize(
            **dataset["summary"],
            dataset_sha256=dataset["dataset_sha256"],
            dataset_artifact_sha256=artifact["sha256"],
            dataset_resource_scope="recorder_resource_window_includes_source_verification_and_assembly; excludes_imports_initial_plan_parse_partition_authority_preflight_and_post_sample_sealing",
            prior_costs=[source["prior_cost"] for source in dataset["sources"]],
            environment_steps=0,
            training_steps=0,
            episodes=0,
            optimizer_updates=0,
        )
    return {
        "run_id": run.run_id,
        "directory": str(run.directory),
        "dataset_sha256": dataset["dataset_sha256"],
        "verification": verify_run(_inside(root, plan.artifact_root), run.run_id),
        **dataset["summary"],
    }
