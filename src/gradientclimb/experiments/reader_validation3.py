"""Additive, prospective result-reader construction and blinded qualification.

Only stored images are read. No capture or input backend is instantiated. Hashes
establish byte identity; session identity and exposure audits remain attestations
whose truth must be reviewed. Historical reader tools and records are untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative, _walk
from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.control.game_adapter import GameUIProfile, GameUIRecognizer
from gradientclimb.perception.hud import segment_digits
from gradientclimb.perception.scoring import ResultDistanceReader

from .reader_provenance import (
    ConstructionSource,
    PredictionExposureStatus,
    ReaderSession,
    ReaderSplitManifest,
    construction_closure,
    resolve_prediction_exposure,
    validate_blinded_split,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EvidenceRef(Contract):
    path: str
    sha256: Digest

    @model_validator(mode="after")
    def portable(self):
        _relative(self.path)
        return self


class ReaderProtocol3(Contract):
    schema_version: Literal["reader-validation-3.0"] = "reader-validation-3.0"
    protocol_id: Name
    status: Literal["draft", "registered"]
    registered_at: AwareDatetime | None
    environment: Literal["actual_hill_climb_racing", "synthetic"]
    base_result_reader_sha256: Digest
    construction_session_ids: tuple[Name, ...] = Field(min_length=1)
    development_session_ids: tuple[Name, ...] = ()
    minimum_positive_endpoints: Annotated[int, Field(ge=20)] = 20
    minimum_accepted_positive_endpoints: Annotated[int, Field(ge=20)] = 20
    minimum_availability: Annotated[float, Field(ge=0.90, le=1)] = 0.90
    minimum_heldout_sessions: Annotated[int, Field(ge=3)] = 3
    minimum_non_result_frames: Annotated[int, Field(ge=10)] = 10
    minimum_unreadable_result_frames: Annotated[int, Field(ge=5)] = 5
    required_positive_digits: str = "0123456789"
    maximum_labels: Annotated[int, Field(ge=35, le=1000)] = 200
    maximum_source_bytes: Annotated[int, Field(ge=1024, le=2 * 1024**3)] = 512 * 1024**2
    qualification_note: Name

    @model_validator(mode="after")
    def declared(self):
        if self.status == "registered" and self.registered_at is None:
            raise ValueError("Registered protocol needs its prospective registration time")
        if set(self.required_positive_digits) != set("0123456789"):
            raise ValueError("Positive qualification coverage requires the complete alphabet")
        if len(set(self.construction_session_ids)) != len(self.construction_session_ids):
            raise ValueError("Duplicate construction session")
        if set(self.construction_session_ids) & set(self.development_session_ids):
            raise ValueError("A session has one declared purpose")
        return self


class ExposureAudit(Contract):
    session_id: Name
    prediction_exposure_status: PredictionExposureStatus | None = None
    first_prediction_exposure_at: AwareDatetime | None
    exposure_reviewed_through: AwareDatetime
    reviewer: Name
    note: Name

    @model_validator(mode="after")
    def coherent(self):
        resolve_prediction_exposure(
            self.prediction_exposure_status, self.first_prediction_exposure_at
        )
        if (
            self.first_prediction_exposure_at is not None
            and self.first_prediction_exposure_at > self.exposure_reviewed_through
        ):
            raise ValueError("Prediction exposure follows the audit cutoff")
        return self

    @property
    def exposure_status(self):
        return resolve_prediction_exposure(
            self.prediction_exposure_status, self.first_prediction_exposure_at
        )


class ResultAnnotation(Contract):
    label_id: Name
    session_id: Name
    run_id: Name
    # Same endpoint in different animation frames cannot inflate the denominator.
    sampling_unit_id: Name
    frame: EvidenceRef
    source_hashes: tuple[Digest, ...] = Field(min_length=1)
    field: Literal["result_distance"] = "result_distance"
    is_result: bool = Field(strict=True)
    text: str | None
    labeled_at: AwareDatetime
    reviewer: Name
    note: Name

    @model_validator(mode="after")
    def truth(self):
        if self.text is not None and (
            not self.is_result or not self.text or any(c not in "0123456789" for c in self.text)
        ):
            raise ValueError("Only a readable result has a digits-only text label")
        if self.frame.sha256 not in self.source_hashes:
            raise ValueError("Original frame hash must be included in annotation ancestry")
        return self

    @property
    def value(self):
        return (
            self.text
            if self.text is not None
            else ("unreadable_result" if self.is_result else "non_result")
        )


class ReaderGraph3(Contract):
    sources: tuple[ConstructionSource, ...] = Field(min_length=1, max_length=2000)
    reader_roots: tuple[Name, ...] = Field(min_length=1)
    anchor_roots: tuple[Name, ...] = Field(min_length=1)
    ui_roots: tuple[Name, ...] = Field(min_length=1)
    references: dict[str, EvidenceRef]


class CommonPlan(Contract):
    protocol: EvidenceRef
    artifact_root: str = "artifacts"
    result_reader: EvidenceRef
    ui_profile: EvidenceRef
    ui_reference_root: str = "artifacts"
    graph: ReaderGraph3
    annotations: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=1000)
    annotation_run_id: Name
    session_declarations: dict[str, Name]
    exposure_audits: dict[str, EvidenceRef]

    @model_validator(mode="after")
    def safe_roots(self):
        _relative(self.artifact_root)
        _relative(self.ui_reference_root)
        return self


class ConstructionPlan3(CommonPlan):
    schema_version: Literal["reader-construction-plan-3.0"] = "reader-construction-plan-3.0"
    sessions: tuple[ReaderSession, ...] = Field(min_length=1)


class EvaluationPlan3(CommonPlan):
    schema_version: Literal["reader-evaluation-plan-3.0"] = "reader-evaluation-plan-3.0"
    split: ReaderSplitManifest
    reader_frozen_at: AwareDatetime
    reader_construction_run_id: Name
    reader_freeze_run_id: Name


class ReaderFreezePlan3(Contract):
    schema_version: Literal["reader-freeze-plan-3.0"] = "reader-freeze-plan-3.0"
    artifact_root: str = "artifacts"
    protocol: EvidenceRef
    result_reader: EvidenceRef
    ui_profile: EvidenceRef
    reader_construction_run_id: Name


def _bank_member(record, directory, path, digest):
    # Dependency manifests need their original relative directory structure.
    # Both the original payload and its registered private copy are seal-checked.
    return path.is_relative_to(directory) and any(
        artifact["sha256"] == digest for artifact in record["artifact_manifest"]
    )


def _read(root, reference):
    path = _inside(root, reference.path)
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("Evidence missing or larger than the per-file bound")
    if sha256_file(path) != reference.sha256:
        raise ValueError(f"Evidence hash mismatch: {reference.path}")
    return path


def _load(root, reference, model):
    # Parse the bytes whose digest was checked, even if the mutable original
    # changes after path validation. In particular, an exposure status cannot
    # differ from the audit identified by the recorded hash.
    payload = _read(root, reference).read_bytes()
    if len(payload) > 32 * 1024**2 or sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("Evidence changed before parsing its exact payload")
    return model.model_validate_json(payload)


def _graph(root, graph):
    # construction_closure consumes this same explicit graph interface; no dummy
    # held-out labels or model_construct bypass is used for construction plans.
    hashes, sessions = construction_closure(graph)
    if set(graph.references) != {node.source_id for node in graph.sources}:
        raise ValueError("Every ancestor requires an exact file reference")
    for node in graph.sources:
        reference = graph.references[node.source_id]
        if reference.sha256 != node.sha256:
            raise ValueError("Ancestor node and file hashes disagree")
        _read(root, reference)
    return hashes, sessions


def _dependencies(root, plan, hashes, excluded_sessions):
    """Bind embedded reader/UI dependencies to the independently declared closure."""
    paths = []

    def require(path, digest):
        if digest not in hashes:
            raise ValueError("Embedded reader/UI ancestor omitted from source closure")
        reference = EvidenceRef(path=path.relative_to(root).as_posix(), sha256=digest)
        paths.append(_read(root, reference))
        return json.loads(path.read_bytes()) if path.suffix == ".json" else None

    manifest_path = _read(root, plan.result_reader)
    data = require(manifest_path, plan.result_reader.sha256)
    numeric_path = _inside(manifest_path.parent, data["numeric_manifest"]["file"])
    numeric = require(numeric_path, data["numeric_manifest"]["sha256"])
    anchor_path = _inside(manifest_path.parent, data["anchor"]["file"])
    require(anchor_path, data["anchor"]["sha256"])
    for glyph in numeric["glyphs"]:
        require(_inside(numeric_path.parent, glyph["file"]), glyph["sha256"])
        if glyph["source_sha256"] not in hashes:
            raise ValueError("Glyph raw-frame ancestry missing from closure")
    # Parent manifests carry their own bank, anchor and raw-frame dependencies.
    # Scan declared ancestor JSONs rather than trusting legacy parent path syntax.
    for reference in plan.graph.references.values():
        path = _read(root, reference)
        if path.suffix != ".json":
            continue
        ancestor = json.loads(path.read_bytes())
        required = []
        if isinstance(ancestor, dict):
            for key in ("numeric_manifest", "anchor", "extended_from"):
                if isinstance(ancestor.get(key), dict) and "sha256" in ancestor[key]:
                    required.append(ancestor[key]["sha256"])
            if ancestor.get("anchor_source_sha256"):
                required.append(ancestor["anchor_source_sha256"])
            for glyph in ancestor.get("glyphs", []):
                required.extend((glyph["sha256"], glyph["source_sha256"]))
            if not set(ancestor.get("construction_sessions", [])) <= excluded_sessions:
                raise ValueError("Parent reader construction sessions omitted from closure")
        if not set(required) <= hashes:
            raise ValueError("Parent reader ancestry is incomplete")
    profile_path = _read(root, plan.ui_profile)
    profile = GameUIProfile.model_validate(require(profile_path, plan.ui_profile.sha256))
    for variant in profile.variants:
        require(_inside(_inside(root, plan.ui_reference_root), variant.file), variant.sha256)
    nodes = {node.source_id: node for node in plan.graph.sources}
    for roots, digest in (
        (plan.graph.reader_roots, plan.result_reader.sha256),
        (plan.graph.anchor_roots, data["anchor"]["sha256"]),
        (plan.graph.ui_roots, plan.ui_profile.sha256),
    ):
        if digest not in {nodes[node].sha256 for node in roots}:
            raise ValueError("Declared source roots must bind the actual reader, anchor and UI")
    return (
        ResultDistanceReader.from_manifest(manifest_path),
        GameUIRecognizer(profile, _inside(root, plan.ui_reference_root)),
        paths,
    )


def _verified_run(root, artifact_root, run_id):
    from gradientclimb.experiments import verify_run

    if "/" in run_id or _relative(run_id) != run_id:
        raise ValueError("Run identity must be one safe path component")
    directory = _inside(root, f"{artifact_root}/runs/{run_id}")
    for _ in _walk(directory):
        pass  # Reject junctions before canonical seal verification follows paths.
    record = json.loads((directory / "run.json").read_bytes())
    if record["run_id"] != run_id:
        raise ValueError("Canonical run identity mismatch")
    for artifact in record["artifact_manifest"]:
        _inside(directory, artifact["path"])
    if not verify_run(_inside(root, artifact_root), run_id)["valid"]:
        raise ValueError("Source run seal failed")
    return record, directory


def _sealed_at(directory):
    return datetime.fromisoformat(json.loads((directory / "seal.json").read_bytes())["created_at"])


def _registered(record, directory, path, digest):
    return any(
        artifact["sha256"] == digest and directory / artifact["path"] == path
        for artifact in record["artifact_manifest"]
    )


def _exposure_attestation(attempt, session, status, first_view, origin, audit_ref=None):
    if attempt is None or status == "none_reported":
        return
    attempt.exposure_attestations.append(
        {
            "session_id": session.session_id,
            "run_ids": list(session.run_ids),
            "prediction_exposure_status": status,
            "first_prediction_exposure_at": first_view.isoformat() if first_view else None,
            "attestation_origin": origin,
            "exposure_audit": audit_ref.model_dump() if audit_ref is not None else None,
        }
    )
    attempt.run.annotate(prediction_exposure_attestations=list(attempt.exposure_attestations))
    for run_id in session.run_ids:
        attempt.event(
            "prediction_exposure_not_ruled_out",
            source_run_id=run_id,
            session_id=session.session_id,
            prediction_exposure_status=status,
            first_prediction_exposure_at=first_view.isoformat() if first_view else None,
            attestation_origin=origin,
            exposure_audit=audit_ref.model_dump() if audit_ref is not None else None,
            note="Prior exposure attestation only; this event executes no new inference.",
        )


def _session_evidence(root, plan, sessions, protocol, attempt=None):
    from .reader_receipts3 import verify_session_declaration3

    by_id = {session.session_id: session for session in sessions}
    if len(by_id) != len(sessions):
        raise ValueError("Duplicate session identity")
    if set(plan.session_declarations) != set(by_id) or set(plan.exposure_audits) != set(by_id):
        raise ValueError("Every supplied session needs its declaration and exposure audit")
    runs, declarations = {}, {}
    for session in sessions:
        audit_ref = plan.exposure_audits[session.session_id]
        audit = _load(root, audit_ref, ExposureAudit)
        if audit.session_id != session.session_id:
            raise ValueError("Exposure audit belongs to another session")
        _exposure_attestation(
            attempt,
            session,
            audit.exposure_status,
            audit.first_prediction_exposure_at,
            "audit",
            audit_ref,
        )
        if audit_ref.sha256 != session.exposure_evidence_sha256 or (
            audit.session_id != session.session_id
            or audit.exposure_status != session.exposure_status
            or audit.first_prediction_exposure_at != session.first_prediction_exposure_at
            or audit.exposure_reviewed_through != session.exposure_reviewed_through
        ):
            raise ValueError("Exposure audit bytes disagree with session attestation")
        if (
            session.purpose == "heldout"
            and audit.exposure_status != "none_reported"
            and audit.first_prediction_exposure_at is None
        ):
            raise ValueError("Unknown first-view timing cannot establish unexposed heldout labels")
        for run_id in session.run_ids:
            if "/" in run_id or _relative(run_id) != run_id or run_id in runs:
                raise ValueError("Run must have one safe explicit session assignment")
            record, directory = _verified_run(root, plan.artifact_root, run_id)
            start = datetime.fromisoformat(record["start_time"])
            end = _sealed_at(directory)
            if not session.acquired_at <= start <= end <= session.sealed_at:
                raise ValueError("Session bounds must cover the sealed run's actual times")
            if record["environment"] != protocol.environment:
                raise ValueError("Source environment does not match the registered protocol")
            runs[run_id] = (session, record, directory)
        declaration = verify_session_declaration3(
            root,
            plan.artifact_root,
            plan.session_declarations[session.session_id],
            plan.protocol.model_dump(),
            session.session_id,
            session.purpose,
            session.acquired_at,
            [runs[run_id][1] for run_id in session.run_ids],
        )
        if (
            isinstance(plan, EvaluationPlan3)
            and session.purpose == "heldout"
            and (declaration["candidate_freeze_run_id"] != plan.reader_freeze_run_id)
        ):
            raise ValueError("Session declaration binds another frozen candidate")
        declarations[session.session_id] = declaration
    return runs, declarations


def _annotations(root, plan, runs, declarations, unit_receipts):
    from .reader_units3 import (
        ReaderUnitResolver3,
        expected_negative_unit_ids,
        result_selection_coverage,
    )

    resolver = ReaderUnitResolver3(root, plan.artifact_root)
    annotation_record, annotation_directory = _verified_run(
        root, plan.artifact_root, plan.annotation_run_id
    )
    if annotation_record["status"] != "completed":
        raise ValueError("Annotations require a completed sealed label publication")
    annotation_sealed_at = _sealed_at(annotation_directory)
    annotations, ids, units, frames = [], set(), set(), set()
    for reference in plan.annotations:
        label = _load(root, reference, ResultAnnotation)
        if not _registered(
            annotation_record, annotation_directory, _read(root, reference), reference.sha256
        ):
            raise ValueError("Annotation must be an exact artifact in a sealed label run")
        if label.labeled_at > annotation_sealed_at:
            raise ValueError("Annotation timestamp follows its immutable label seal")
        if label.label_id in ids or label.frame.sha256 in frames:
            raise ValueError("Duplicate annotation/frame cannot inflate evidence")
        ids.add(label.label_id)
        frames.add(label.frame.sha256)
        if label.run_id not in runs or runs[label.run_id][0].session_id != label.session_id:
            raise ValueError("Annotation run/session was not declared")
        session, record, directory = runs[label.run_id]
        if label.labeled_at < session.sealed_at:
            raise ValueError("Annotation must follow source sealing")
        path = _read(root, label.frame)
        if not path.is_relative_to(directory):
            raise ValueError("Annotation frame lies outside its declared source run")
        if not _registered(record, directory, path, label.frame.sha256):
            raise ValueError("Annotation frame must be the exact registered source artifact")
        resolved = resolver.resolve(
            label.frame.model_dump(),
            source_run_id=label.run_id,
            label_is_result=label.is_result,
            purpose=session.purpose,
            selection_document=declarations[label.session_id]["selection_document"],
            construction_fallback_reason="Legacy construction image lacks prospective native endpoint metadata; never qualification evidence"
            if session.purpose == "construction"
            else None,
        )
        unit = resolved["canonical_unit_id"]
        if unit in units:
            raise ValueError("Repeated canonical sampling unit cannot inflate endpoint evidence")
        units.add(unit)
        unit_receipts[label.label_id] = resolved
        if isinstance(plan, EvaluationPlan3) and (
            annotation_sealed_at > plan.split.frozen_at
            or (
                session.first_prediction_exposure_at is not None
                and annotation_sealed_at >= session.first_prediction_exposure_at
            )
        ):
            raise ValueError("All label bytes must be sealed before viewed session predictions")
        annotations.append((label, reference, path))
    selection_coverage = {}
    if isinstance(plan, EvaluationPlan3):
        expected = set()
        for session_id, declaration in declarations.items():
            if declaration["purpose"] != "heldout":
                continue
            records = [
                record for session, record, _ in runs.values() if session.session_id == session_id
            ]
            expected.update(expected_negative_unit_ids(declaration["selection_document"], records))
        if not expected <= units:
            raise ValueError("Every preregistered negative sampling unit must be labeled")
        selection_coverage = result_selection_coverage(
            [record for session, record, _ in runs.values() if session.purpose == "heldout"]
        )
        if not set(selection_coverage["expected_result_unit_ids"]) <= units:
            raise ValueError(
                "Every retained result endpoint must be labeled under the registered selection rule"
            )
        selection_coverage["expected_negative_unit_ids"] = sorted(expected)
    return annotations, selection_coverage


def _prepare(project_root, plan):
    root = Path(project_root).absolute()
    _no_links(root)
    protocol = _load(root, plan.protocol, ReaderProtocol3)
    if protocol.status != "registered":
        raise ValueError("Draft preregistration cannot authorize construction or evaluation")
    if protocol.registered_at > datetime.now(UTC):
        raise ValueError("Protocol registration cannot be in the future")
    if len(plan.annotations) > protocol.maximum_labels:
        raise ValueError("Registered label bound exceeded")
    hashes, excluded_sessions = _graph(root, plan.graph)
    reader, ui, dependencies = _dependencies(root, plan, hashes, excluded_sessions)
    return root, protocol, hashes, excluded_sessions, reader, ui, dependencies


def _state(ui, rgb):
    # Offline placeholders serve only the pure pixel recognizer. No time, capture
    # freshness, control authorization or live action claim is derived from them.
    frame = CapturedFrame(
        rgb, 0, 0, "offline", ClientRect(0, 0, rgb.shape[1], rgb.shape[0]), "stored_png", 0
    )
    observation = ui.observe(frame)
    return observation.state


def _rgb(path, expected_sha256):
    payload = path.read_bytes()
    if len(payload) > 32 * 1024**2 or sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frame changed before image decoding")
    with Image.open(BytesIO(payload)) as image:
        if image.size != (1034, 581):
            raise ValueError("Reader frame dimensions must match the frozen native geometry")
        return np.asarray(image.convert("RGB"))


def _all_sources(root, plan, dependencies, labels):
    references = [
        plan.protocol,
        plan.result_reader,
        plan.ui_profile,
        *plan.annotations,
        *plan.graph.references.values(),
        *plan.exposure_audits.values(),
    ]
    run_ids = {
        plan.annotation_run_id,
        *plan.session_declarations.values(),
        *(label.run_id for label, _, _ in labels),
    }
    if isinstance(plan, EvaluationPlan3):
        run_ids.update((plan.reader_construction_run_id, plan.reader_freeze_run_id))
    receipts = [
        _inside(root, f"{plan.artifact_root}/runs/{run_id}/{name}")
        for run_id in run_ids
        for name in ("run.json", "seal.json")
    ]
    return sorted(
        {
            *dependencies,
            *[_read(root, ref) for ref in references],
            *[path for _, _, path in labels],
            *receipts,
        }
    )


def _bounded_sources(root, plan, protocol, dependencies, labels):
    sources = _all_sources(root, plan, dependencies, labels)
    size = sum(path.stat().st_size for path in sources)
    if size > protocol.maximum_source_bytes:
        raise ValueError("Registered total source-byte bound exceeded")
    if shutil.disk_usage(_inside(root, plan.artifact_root)).free < size + 64 * 1024**2:
        raise ValueError("Insufficient free disk for bounded source copies and report overhead")
    return [(path, sha256_file(path)) for path in sources]


def _copy_sources(run, sources, kind):
    for path, digest in sources:
        artifact = run.register_artifact(path, kind)
        if artifact["sha256"] != digest:
            raise ValueError("Source evidence changed during publication")


def _work_cost(started, cpu_started):
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_core_seconds": time.process_time() - cpu_started,
        "gpu_utilization_equivalent_seconds": None,
        "scope": "canonical attempt setup through this journal event, including validation, fitting or inference and publication; Python startup/imports excluded; overlaps canonical resource sampling and must not be summed with it",
    }


class _ReaderAttempt:
    """Durable operation starts conservatively count exposure even on interruption."""

    def __init__(self, run, operation, started, cpu_started):
        self.run, self.operation = run, operation
        self.started, self.cpu_started = started, cpu_started
        self.path = run.directory / "reader-operations.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.counts = Counter()
        self.units = {}
        self.report = None
        self.exposure_attestations = []
        self.sequence = -1

    def event(self, phase, **details):
        self.counts[phase] += 1
        self.sequence += 1
        cost = _work_cost(self.started, self.cpu_started)
        row = {
            "schema_version": "reader-operation-event-3.0",
            "sequence": self.sequence,
            "at": datetime.now(UTC).isoformat(),
            "phase": phase,
            "operation": self.operation,
            "counters": dict(self.counts),
            "work": cost,
            **details,
        }
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.run.annotate(
            operation_counters=dict(self.counts),
            operation_work=cost,
            last_operation_phase=phase,
            qualification_evidence=False,
        )

    def frame_event(self, phase, label, reference, **details):
        self.event(
            phase,
            source_run_id=label.run_id,
            session_id=label.session_id,
            label_id=label.label_id,
            frame_sha256=label.frame.sha256,
            annotation_sha256=reference.sha256,
            canonical_unit=self.units.get(label.label_id),
            **details,
        )

    def close(self):
        self.stream.close()
        self.run.register_artifact(self.path, "reader_operation_journal")


def _run_operation(project_root, plan, worker, operation):
    from gradientclimb.experiments import RunRecorder

    from .reader_exposure3 import reader_publication_lease

    root = Path(project_root).absolute()
    _no_links(root)
    started, cpu_started = time.perf_counter(), time.process_time()
    config = plan.model_dump(mode="json")
    with (
        reader_publication_lease(root, plan.artifact_root),
        RunRecorder(
            _inside(root, plan.artifact_root),
            "result-reader-construction-3.0"
            if operation == "construction"
            else "reader-validation-3.0",
            config,
            algorithm="stored-pixel-reader",
            environment="stored_real_frames",
            source_root=root,
            telemetry_interval_seconds=0,
            qualifies_real_game=False,
        ) as run,
    ):
        attempt = _ReaderAttempt(run, operation, started, cpu_started)
        try:
            sessions = plan.sessions if operation == "construction" else plan.split.sessions
            for session in sessions:
                _exposure_attestation(
                    attempt,
                    session,
                    session.exposure_status,
                    session.first_prediction_exposure_at,
                    "session",
                )
            attempt.event(
                "validation_started", source_run_ids=[r for s in sessions for r in s.run_ids]
            )
            for reference in (plan.protocol, plan.result_reader, plan.ui_profile):
                artifact = run.register_artifact(_read(root, reference), "frozen_reader_input")
                if artifact["sha256"] != reference.sha256:
                    raise ValueError("Frozen input changed during publication")
            result = worker(root, plan, attempt)
            attempt.event("operation_completed")
            attempt.close()
            status = result.pop("_status", "completed")
            if attempt.report is not None:
                run.annotate(
                    report={**attempt.report, "publication_state": "owning_completed_seal_required"}
                )
            run.finalize(status=status, episode_count=0, qualification_evidence=False)
        except BaseException as error:
            if attempt.report is not None:
                downgraded = {
                    **attempt.report,
                    "reader_accuracy_qualified": False,
                    "publication_complete": False,
                    "publication_state": "failed",
                }
                try:
                    run.annotate(report=downgraded, partial_report=downgraded)
                except RuntimeError:
                    # Recorder storage failure can already have closed the run.
                    # Its unsealed envelope cannot authorize qualification.
                    error.add_note(
                        "Recorder already closed during failed publication; no completed seal authority is available"
                    )
            try:
                if not attempt.stream.closed:
                    attempt.event(
                        "operation_failed",
                        error_type=type(error).__name__,
                        error=str(error),
                        error_evidence=getattr(error, "evidence", None),
                    )
                    attempt.close()
            except BaseException as storage_error:  # noqa: BLE001 - retain original fault and journal
                error.add_note(f"Operation journal finalization failed: {storage_error}")
            raise
    return result


def build_reader3(project_root, plan: ConstructionPlan3 | dict):
    return _run_operation(
        project_root, ConstructionPlan3.model_validate(plan), _build_reader3, "construction"
    )


def evaluate_reader3(project_root, plan: EvaluationPlan3 | dict):
    plan = EvaluationPlan3.model_validate(plan)
    result = _run_operation(project_root, plan, _evaluate_reader3, "evaluation")
    return load_reader_report3(project_root, plan.artifact_root, result["run_id"])


def load_reader_report3(project_root, artifact_root, run_id):
    """Reader qualification authority requires the completed, verified envelope."""
    root = Path(project_root).absolute()
    _no_links(root)
    record, directory = _verified_run(root, artifact_root, run_id)
    if record["experiment_id"] != "reader-validation-3.0":
        raise ValueError("Reader report must belong to a reader-validation-3.0 run")
    report = record["summary"].get("report", record["summary"].get("partial_report", {}))
    return {
        **report,
        "run_id": run_id,
        "publication_state": record["status"],
        "reader_accuracy_qualified": record["status"] == "completed"
        and bool(report.get("reader_accuracy_qualified", False)),
        "canonical_seal": {
            "path": (directory / "seal.json").relative_to(root).as_posix(),
            "sha256": sha256_file(directory / "seal.json"),
        },
    }


def freeze_reader3(project_root, plan: ReaderFreezePlan3 | dict):
    """Seal immutable candidate hashes before acquisition; no prediction or fitting."""
    from gradientclimb.experiments import RunRecorder

    plan = ReaderFreezePlan3.model_validate(plan)
    root = Path(project_root).absolute()
    _no_links(root)
    _relative(plan.artifact_root)
    protocol = _load(root, plan.protocol, ReaderProtocol3)
    if protocol.status != "registered" or protocol.registered_at > datetime.now(UTC):
        raise ValueError("Reader freeze requires a registered protocol")
    record, directory = _verified_run(root, plan.artifact_root, plan.reader_construction_run_id)
    manifest = _read(root, plan.result_reader)
    if record["status"] != "completed" or not _bank_member(
        record, directory, manifest, plan.result_reader.sha256
    ):
        raise ValueError("Candidate must be an artifact of successful sealed construction")
    if (
        record["experiment_id"] != "result-reader-construction-3.0"
        or record["configuration"].get("protocol") != plan.protocol.model_dump()
        or json.loads(manifest.read_bytes()).get("construction_protocol_sha256")
        != plan.protocol.sha256
    ):
        raise ValueError("Candidate construction must belong to this registered protocol")
    config = plan.model_dump(mode="json")
    config["freeze_requested_at"] = datetime.now(UTC).isoformat()
    with RunRecorder(
        _inside(root, plan.artifact_root),
        "reader-candidate-freeze-3.0",
        config,
        algorithm="receipt",
        environment="stored_real_frames",
        source_root=root,
        telemetry_interval_seconds=0,
        qualifies_real_game=False,
    ) as run:
        for ref in (plan.protocol, plan.result_reader, plan.ui_profile):
            run.register_artifact(_read(root, ref), "frozen_reader_source")
        run.finalize(status="completed", episode_count=0, qualification_evidence=False)
    return {
        "run_id": run.run_id,
        "frozen_at": _sealed_at(run.directory).isoformat(),
        "reader_sha256": plan.result_reader.sha256,
        "qualification_evidence": False,
    }


def _build_reader3(root, plan, attempt):
    """Build a new bank only after byte/protocol/purpose checks; no old mutation."""
    run = attempt.run
    root, protocol, _, _, base_reader, ui, dependencies = _prepare(root, plan)
    if plan.result_reader.sha256 != protocol.base_result_reader_sha256:
        raise ValueError("Construction must extend the preregistered base reader")
    runs, declarations = _session_evidence(root, plan, plan.sessions, protocol, attempt)
    labels, _ = _annotations(root, plan, runs, declarations, attempt.units)
    sources = _bounded_sources(root, plan, protocol, dependencies, labels)
    now = datetime.now(UTC)
    for label, reference, path in labels:
        session = runs[label.run_id][0]
        if (
            session.purpose != "construction"
            or label.session_id not in protocol.construction_session_ids
        ):
            raise ValueError("Construction label is outside the declared construction sessions")
        if not label.is_result or label.text is None or label.labeled_at > now:
            raise ValueError("Construction requires prior readable result-distance labels")
        attempt.frame_event("image_processing_started", label, reference)
        rgb = _rgb(path, label.frame.sha256)
        if _state(ui, rgb) != "result":
            raise ValueError("Frozen UI recognizer does not support this construction result")
        if len(segment_digits(rgb, base_reader.reader.roi)) != len(label.text):
            raise ValueError("Declared result text does not match digit segmentation")
        attempt.frame_event("image_processing_completed", label, reference)
    # Every source has been checked before a single template is added.
    reader = base_reader.reader
    before = len(reader.glyphs)
    for label, reference, path in labels:
        attempt.frame_event("fitting_started", label, reference)
        reader.add_labeled_region(
            _rgb(path, label.frame.sha256),
            label.text,
            reader.roi,
            source_sha256=label.frame.sha256,
            annotation_id=label.label_id,
        )
        attempt.frame_event("fitting_completed", label, reference, glyphs_added=len(label.text))
    _copy_sources(run, sources, "reader_construction_source")
    numeric = reader.save(run.directory / "result-glyphs")
    old = json.loads(_read(root, plan.result_reader).read_bytes())
    anchor = _inside(_read(root, plan.result_reader).parent, old["anchor"]["file"])
    copied_anchor = run.directory / "distance-label-anchor.png"
    shutil.copyfile(anchor, copied_anchor)
    manifest = {
        **old,
        "numeric_manifest": {
            "file": "result-glyphs/hud-glyphs.json",
            "sha256": sha256_file(numeric),
        },
        "anchor": {"file": copied_anchor.name, "sha256": sha256_file(copied_anchor)},
        "extended_from": {
            "manifest": plan.result_reader.path,
            "sha256": plan.result_reader.sha256,
        },
        "construction_protocol_sha256": plan.protocol.sha256,
        "construction_sessions": sorted({label.session_id for label, _, _ in labels}),
    }
    manifest_path = run.directory / "result-reader.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in [manifest_path, copied_anchor, *numeric.parent.iterdir()]:
        run.register_artifact(path, "result_reader_dependency")
    rebuilt = ResultDistanceReader.from_manifest(manifest_path)
    checks = []
    for label, reference, path in labels:
        attempt.frame_event(
            "prediction_started", label, reference, purpose="construction_selfcheck"
        )
        check = {
            "label_id": label.label_id,
            "expected": int(label.text),
            "reading": rebuilt.read(_rgb(path, label.frame.sha256), result_state_confirmed=True),
        }
        checks.append(check)
        attempt.frame_event("prediction_completed", label, reference, row=check)
    matched = all(
        c["reading"]["valid"] and c["reading"]["distance_meters"] == c["expected"] for c in checks
    )
    run.annotate(
        episode_count=0,
        glyphs_before=before,
        glyphs_added=len(reader.glyphs) - before,
        construction_selfcheck_all_match=matched,
        checks=checks,
        qualification_evidence=False,
    )
    return {
        "_status": "completed" if matched else "failed",
        "run_id": run.run_id,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "selfcheck_all_match": matched,
        "qualification_evidence": False,
    }


def _evaluate_reader3(root, plan, attempt):
    """Evaluate exactly the blinded annotations; preserve positive and negative gaps."""
    from .reader_exposure3 import assert_no_prior_reader_exposure

    run = attempt.run
    root, protocol, _, excluded_sessions, reader, ui, dependencies = _prepare(root, plan)
    split = plan.split
    if (
        split.protocol_id != protocol.protocol_id
        or split.protocol_sha256 != plan.protocol.sha256
        or split.registered_at != protocol.registered_at
    ):
        raise ValueError("Split must bind the actual registered protocol")
    if any(
        getattr(split, key) != getattr(plan.graph, key)
        for key in ("sources", "reader_roots", "anchor_roots", "ui_roots")
    ):
        raise ValueError("Blinded split must use the actual verified source graph")
    provenance = validate_blinded_split(split)
    bank_record, bank_directory = _verified_run(
        root, plan.artifact_root, plan.reader_construction_run_id
    )
    if not _bank_member(
        bank_record, bank_directory, _read(root, plan.result_reader), plan.result_reader.sha256
    ):
        raise ValueError("Frozen reader must belong to its declared sealed construction run")
    bank_sealed_at = _sealed_at(bank_directory)
    if bank_sealed_at > plan.reader_frozen_at or bank_record["status"] != "completed":
        raise ValueError("Reader freeze must follow successful sealed construction")
    if (
        not protocol.registered_at
        <= plan.reader_frozen_at
        < min(s.acquired_at for s in split.sessions if s.purpose == "heldout")
    ):
        raise ValueError("Reader must be frozen before every held-out acquisition")
    freeze_record, freeze_directory = _verified_run(
        root, plan.artifact_root, plan.reader_freeze_run_id
    )
    freeze = freeze_record["configuration"]
    if (
        freeze_record["status"] != "completed"
        or freeze_record["experiment_id"] != "reader-candidate-freeze-3.0"
        or freeze.get("result_reader") != plan.result_reader.model_dump()
        or freeze.get("ui_profile") != plan.ui_profile.model_dump()
        or freeze.get("protocol") != plan.protocol.model_dump()
        or freeze.get("reader_construction_run_id") != plan.reader_construction_run_id
        or _sealed_at(freeze_directory) != plan.reader_frozen_at
        or _sealed_at(freeze_directory)
        >= min(s.acquired_at for s in split.sessions if s.purpose == "heldout")
    ):
        raise ValueError(
            "Immutable reader freeze must predate acquisition and bind candidate hashes"
        )
    forbidden = (
        set(protocol.construction_session_ids)
        | set(protocol.development_session_ids)
        | excluded_sessions
    )
    if any(s.session_id in forbidden for s in split.sessions if s.purpose == "heldout"):
        raise ValueError("Protocol excludes exposed construction/development sessions")
    runs, declarations = _session_evidence(root, plan, split.sessions, protocol, attempt)
    prior_attempts = assert_no_prior_reader_exposure(
        root,
        plan.artifact_root,
        tuple(run_id for run_id, (session, _, _) in runs.items() if session.purpose == "heldout"),
        current_run_id=run.run_id,
    )
    run.annotate(prior_reader_attempts=prior_attempts)
    labels, source_selection = _annotations(root, plan, runs, declarations, attempt.units)
    sources = _bounded_sources(root, plan, protocol, dependencies, labels)
    by_id = {label.label_id: label for label in split.labels}
    if set(by_id) != {label.label_id for label, _, _ in labels}:
        raise ValueError("Every blinded label must have its exact annotation bytes")
    for label, ref, _ in labels:
        blinded = by_id[label.label_id]
        if (
            blinded.annotation_sha256 != ref.sha256
            or blinded.session_id != label.session_id
            or blinded.run_id != label.run_id
            or blinded.frame_sha256 != label.frame.sha256
            or blinded.field != label.field
            or blinded.value != label.value
            or blinded.source_hashes != label.source_hashes
            or blinded.labeled_at != label.labeled_at
        ):
            raise ValueError("Blinded provenance disagrees with annotation bytes")
    rows = []
    for label, reference, path in labels:
        attempt.frame_event("prediction_started", label, reference, purpose="heldout_qualification")
        rgb = _rgb(path, label.frame.sha256)
        state = _state(ui, rgb)
        out = reader.read(rgb, result_state_confirmed=state == "result")
        positive = label.text is not None
        exact = out["valid"] and positive and out["distance_meters"] == int(label.text)
        rows.append(
            {
                "label_id": label.label_id,
                "session_id": label.session_id,
                "sampling_unit_id": label.sampling_unit_id,
                "canonical_unit": attempt.units[label.label_id],
                "frame_sha256": label.frame.sha256,
                "kind": "positive" if positive else label.value,
                "expected": int(label.text) if positive else None,
                "ui_state": state,
                "reading": out,
                "exact": bool(exact),
                "wrong_accept": bool(out["valid"] and not exact),
            }
        )
        attempt.frame_event("prediction_completed", label, reference, row=rows[-1])
    positives = [r for r in rows if r["kind"] == "positive"]
    negatives = [r for r in rows if r["kind"] != "positive"]
    available = sum(r["reading"]["valid"] for r in positives)
    counts = Counter(r["kind"] for r in rows)
    observed_digits = set("".join(label.text or "" for label, _, _ in labels))
    exact_ids = {r["label_id"] for r in positives if r["exact"]}
    accepted_digits = set(
        "".join(label.text or "" for label, _, _ in labels if label.label_id in exact_ids)
    )
    availability = available / len(positives) if positives else None
    coverage = {
        "positive_endpoints": len(positives) >= protocol.minimum_positive_endpoints,
        "positive_sessions": len({r["session_id"] for r in positives})
        >= protocol.minimum_heldout_sessions,
        "accepted_positive_endpoints": available >= protocol.minimum_accepted_positive_endpoints,
        "availability": availability is not None and availability >= protocol.minimum_availability,
        "accepted_positive_sessions": len(
            {r["session_id"] for r in positives if r["reading"]["valid"]}
        )
        >= protocol.minimum_heldout_sessions,
        "non_result_negatives": counts["non_result"] >= protocol.minimum_non_result_frames,
        "unreadable_result_negatives": counts["unreadable_result"]
        >= protocol.minimum_unreadable_result_frames,
        "negative_sessions": len({r["session_id"] for r in negatives})
        >= protocol.minimum_heldout_sessions,
        "alphabet": set(protocol.required_positive_digits) <= observed_digits,
        "accepted_correct_alphabet": set(protocol.required_positive_digits) <= accepted_digits,
    }
    report = {
        "schema_version": "reader-validation-report-3.0",
        "run_id": run.run_id,
        "protocol_id": protocol.protocol_id,
        "reader_sha256": plan.result_reader.sha256,
        "provenance": provenance,
        "source_selection": source_selection,
        "positive_endpoints": len(positives),
        "positive_available": available,
        "availability": availability,
        "positive_sessions": len({r["session_id"] for r in positives}),
        "accepted_positive_sessions": len(
            {r["session_id"] for r in positives if r["reading"]["valid"]}
        ),
        "positive_exact": sum(r["exact"] for r in positives),
        "exact_accuracy_when_available": sum(r["exact"] for r in positives) / available
        if available
        else None,
        "wrong_accepts": sum(r["wrong_accept"] for r in rows),
        "positive_wrong_accepts": sum(r["wrong_accept"] for r in positives),
        "negative_wrong_accepts": sum(r["wrong_accept"] for r in negatives),
        "wrong_accept_rate_when_positive_available": sum(r["wrong_accept"] for r in positives)
        / available
        if available
        else None,
        "false_accept_rate_on_negatives": sum(r["wrong_accept"] for r in negatives) / len(negatives)
        if negatives
        else None,
        "negative_counts": {key: counts[key] for key in ("non_result", "unreadable_result")},
        "missing_positive_digits": sorted(set(protocol.required_positive_digits) - observed_digits),
        "missing_accepted_correct_digits": sorted(
            set(protocol.required_positive_digits) - accepted_digits
        ),
        "coverage": coverage,
        "rows": rows,
        "episode_count": 0,
        "publication_contract": "Reader qualification requires this owning run to be completed with a valid canonical seal; failed, cancelled or unsealed publications are ineligible regardless of finite-set criteria.",
        "note": "Observed finite-set agreement only; zero observed wrong accepts does not imply zero population error. Session identity/exposure assertions require audit.",
    }
    report["acceptance_criteria_met"] = all(coverage.values()) and report["wrong_accepts"] == 0
    report["reader_accuracy_qualified"] = (
        report["acceptance_criteria_met"]
        and protocol.environment == "actual_hill_climb_racing"
        and all(row["canonical_unit"]["qualification_eligible"] for row in rows)
    )
    attempt.report = report
    # Retain completed rows and denominators even if a later source copy fails.
    run.annotate(
        partial_report={**report, "reader_accuracy_qualified": False, "publication_complete": False}
    )
    _copy_sources(run, sources, "reader_validation_source")
    path = run.directory / "reader-validation-report.json"
    pending = {
        **report,
        "reader_accuracy_qualified": False,
        "publication_state": "pending_owning_seal",
    }
    path.write_text(json.dumps(pending, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    run.register_artifact(path, "reader_validation_report")
    run.annotate(episode_count=0, report=pending, qualification_evidence=False)
    return {"run_id": run.run_id, **report}
