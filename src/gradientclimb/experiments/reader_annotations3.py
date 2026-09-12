"""Seal explicit reviewed reader annotations without decoding or assigning truth."""

from __future__ import annotations

import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative

from .reader_exposure3 import assert_no_prior_reader_exposure, reader_publication_lease
from .reader_receipts3 import _run_component, _verified, verify_session_declaration3
from .reader_units3 import (
    ReaderUnitResolver3,
    expected_negative_unit_ids,
    result_selection_coverage,
)
from .reader_validation3 import (
    Contract,
    EvidenceRef,
    ExposureAudit,
    Name,
    ReaderProtocol3,
    ResultAnnotation,
)

EXPERIMENT = "reader-annotation-publication-3.0"


class AnnotationSession3(Contract):
    session_id: Name
    purpose: Literal["construction", "development", "heldout"]
    run_ids: tuple[Name, ...] = Field(min_length=1, max_length=1000)
    declaration_run_id: Name
    exposure_audit: EvidenceRef

    @model_validator(mode="after")
    def identities(self):
        for identity in (*self.run_ids, self.declaration_run_id):
            _run_component(identity)
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("Repeated source run in annotation session")
        return self


class ReaderAnnotationPublicationPlan3(Contract):
    schema_version: Literal["reader-annotation-publication-3.0"] = EXPERIMENT
    artifact_root: str = "artifacts"
    protocol: EvidenceRef
    sessions: tuple[AnnotationSession3, ...] = Field(min_length=1, max_length=1000)
    annotations: tuple[ResultAnnotation, ...] = Field(min_length=1, max_length=1000)
    note: Name

    @model_validator(mode="after")
    def identities(self):
        _relative(self.artifact_root)
        if len({s.session_id for s in self.sessions}) != len(self.sessions):
            raise ValueError("Repeated annotation session identity")
        run_ids = [r for s in self.sessions for r in s.run_ids]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("A source run cannot belong to multiple annotation sessions")
        return self


def _read(root, ref):
    path = _inside(root, ref.path)
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("Annotation evidence is missing or exceeds the per-file bound")
    if sha256_file(path) != ref.sha256:
        raise ValueError(f"Annotation evidence hash mismatch: {ref.path}")
    return path


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class _Publication:
    def __init__(self, run, started, cpu_started):
        self.run, self.started, self.cpu_started = run, started, cpu_started
        self.path = run.directory / "reader-operations.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.sequence = 0
        self.counts = Counter()

    def event(self, phase, **details):
        self.counts[phase] += 1
        work = {
            "elapsed_seconds": time.perf_counter() - self.started,
            "cpu_core_seconds": time.process_time() - self.cpu_started,
            "scope": "Publisher entry through this event, including validation/hash reads/JSON publication; excludes imports, initial JSON/root parsing and later sealing; overlaps recorder resources, do not add",
            "gpu_utilization_equivalent_seconds": None,
        }
        row = {
            "schema_version": "reader-operation-event-3.0",
            "sequence": self.sequence,
            "at": datetime.now(UTC).isoformat(),
            "operation": "annotation_publication",
            "phase": phase,
            "counters": dict(self.counts),
            "work": work,
            **details,
        }
        self.stream.write(canonical_json(row) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1
        self.run.annotate(operation_counters=dict(self.counts), operation_work=work)

    def close(self):
        self.stream.close()
        self.run.register_artifact(self.path, "reader_operation_journal")


def _validate(root, plan, publication):
    run = publication.run
    protocol_path = _read(root, plan.protocol)
    artifact = run.register_artifact(protocol_path, "reader_annotation_source")
    if artifact["sha256"] != plan.protocol.sha256:
        raise ValueError("Protocol changed during publication")
    protocol = ReaderProtocol3.model_validate_json(
        _inside(run.directory, artifact["path"]).read_bytes()
    )
    now = datetime.now(UTC)
    if protocol.status != "registered" or protocol.registered_at > now:
        raise ValueError("Annotation publication requires a registered, nonfuture protocol")
    if len(plan.annotations) > protocol.maximum_labels:
        raise ValueError("Registered annotation bound exceeded")
    dependencies = {protocol_path: plan.protocol.sha256}
    sources, declarations, audits, session_metadata = {}, {}, {}, []
    exposure_attestations = []
    copied = {protocol_path}
    for session in plan.sessions:
        permitted = {
            "construction": session.session_id in protocol.construction_session_ids,
            "development": session.session_id in protocol.development_session_ids,
            "heldout": session.session_id
            not in (*protocol.construction_session_ids, *protocol.development_session_ids),
        }
        if not permitted[session.purpose]:
            raise ValueError("Annotation session purpose disagrees with registered protocol")
        records, seals = [], []
        for run_id in session.run_ids:
            record, directory, sealed_at = _verified(root, plan.artifact_root, run_id)
            if record["environment"] != protocol.environment:
                raise ValueError("Annotation source environment disagrees with protocol")
            records.append(record)
            seals.append(sealed_at)
            sources[run_id] = (session, record, directory, sealed_at)
            for filename in ("run.json", "seal.json", "config.json"):
                path = directory / filename
                dependencies[path] = sha256_file(path)
            publication.event(
                "source_verified", source_run_id=run_id, session_id=session.session_id
            )
        acquired_at = min(datetime.fromisoformat(record["start_time"]) for record in records)
        declaration = verify_session_declaration3(
            root,
            plan.artifact_root,
            session.declaration_run_id,
            plan.protocol.model_dump(),
            session.session_id,
            session.purpose,
            acquired_at,
            records,
        )
        declarations[session.session_id] = declaration
        receipt_dir = _inside(root, f"{plan.artifact_root}/runs/{session.declaration_run_id}")
        for path in (
            receipt_dir / "run.json",
            receipt_dir / "seal.json",
            root / declaration["selection_file"],
        ):
            dependencies[path] = sha256_file(path)
        audit_path = _read(root, session.exposure_audit)
        artifact = run.register_artifact(audit_path, "reader_annotation_source")
        if artifact["sha256"] != session.exposure_audit.sha256:
            raise ValueError("Exposure audit changed during publication")
        copied.add(audit_path)
        audit = ExposureAudit.model_validate_json(
            _inside(run.directory, artifact["path"]).read_bytes()
        )
        if audit.prediction_exposure_status is None:
            raise ValueError("New annotation publication needs an explicit exposure status")
        if audit.session_id != session.session_id:
            raise ValueError("Annotation exposure audit session mismatch")
        dependencies[audit_path] = session.exposure_audit.sha256
        exposure_attestations.append(
            {
                "session_id": session.session_id,
                "run_ids": list(session.run_ids),
                "prediction_exposure_status": audit.exposure_status,
                "first_prediction_exposure_at": audit.first_prediction_exposure_at.isoformat()
                if audit.first_prediction_exposure_at
                else None,
                "audit": session.exposure_audit.model_dump(),
            }
        )
        # An event-write failure must not erase an audit already validated and
        # copied. The failed canonical envelope also retains this attestation.
        run.annotate(prediction_exposure_attestations=list(exposure_attestations))
        # Preserve known or unresolved prior exposure before any later validation
        # failure. The event does not claim this publisher executed a prediction.
        if audit.exposure_status != "none_reported":
            for run_id in session.run_ids:
                publication.event(
                    "prediction_exposure_not_ruled_out",
                    source_run_id=run_id,
                    session_id=session.session_id,
                    exposure_status=audit.exposure_status,
                    audit_sha256=session.exposure_audit.sha256,
                    first_prediction_exposure_at=audit.first_prediction_exposure_at.isoformat()
                    if audit.first_prediction_exposure_at
                    else None,
                    meaning="Prior exposure attestation, not a new prediction or measured first-view time",
                )
        if not max(seals) <= audit.exposure_reviewed_through <= now:
            raise ValueError("Exposure audit cutoff must cover sealed sources and not be future")
        if audit.first_prediction_exposure_at is not None and not (
            acquired_at <= audit.first_prediction_exposure_at <= audit.exposure_reviewed_through
        ):
            raise ValueError("Exposure timestamp is outside the reviewed source interval")
        if session.purpose == "heldout" and audit.exposure_status != "none_reported":
            raise ValueError("Held-out annotation publication requires none_reported exposure")
        audits[session.session_id] = audit
        session_metadata.append(
            {
                "session_id": session.session_id,
                "purpose": session.purpose,
                "source_run_ids": list(session.run_ids),
                "declaration": declaration,
                "exposure_audit": session.exposure_audit.model_dump(),
                "prediction_exposure_status": audit.exposure_status,
                "original_sources": [
                    {
                        "run_id": record["run_id"],
                        "status": record["status"],
                        "original_session_id": record["configuration"].get("session_id"),
                    }
                    for record in records
                ],
            }
        )
    heldout = [
        r for session in plan.sessions if session.purpose == "heldout" for r in session.run_ids
    ]
    if heldout:
        prior = assert_no_prior_reader_exposure(
            root, plan.artifact_root, heldout, current_run_id=run.run_id
        )
        run.annotate(prior_exposure_check=prior)
    resolver = ReaderUnitResolver3(root, plan.artifact_root)
    units, ids, frame_hashes, resolved = set(), set(), set(), []
    ancestry = {
        a["sha256"] for _, record, _, _ in sources.values() for a in record["artifact_manifest"]
    }
    for label in plan.annotations:
        if label.run_id not in sources or sources[label.run_id][0].session_id != label.session_id:
            raise ValueError("Annotation does not belong to its declared source session")
        session, _, _, _ = sources[label.run_id]
        session_sealed = max(sources[r][3] for r in session.run_ids)
        if (
            not session_sealed
            <= label.labeled_at
            <= audits[label.session_id].exposure_reviewed_through
        ):
            raise ValueError(
                "Annotation time must follow whole-session sealing and precede audit cutoff"
            )
        if label.label_id in ids or label.frame.sha256 in frame_hashes:
            raise ValueError("Duplicate annotation ID or original frame")
        if not set(label.source_hashes) <= ancestry:
            raise ValueError("Annotation ancestry contains an unverified source hash")
        unit = resolver.resolve(
            label.frame.model_dump(),
            source_run_id=label.run_id,
            label_is_result=label.is_result,
            purpose=session.purpose,
            selection_document=declarations[label.session_id]["selection_document"],
            construction_fallback_reason="Reviewed legacy construction artifact; unresolved endpoint metadata cannot qualify"
            if session.purpose == "construction"
            else None,
        )
        if unit["canonical_unit_id"] in units:
            raise ValueError("Duplicate canonical endpoint cannot inflate annotation evidence")
        ids.add(label.label_id)
        frame_hashes.add(label.frame.sha256)
        units.add(unit["canonical_unit_id"])
        resolved.append(unit)
    selection = result_selection_coverage([sources[r][1] for r in heldout]) if heldout else {}
    negatives = set()
    for session in plan.sessions:
        if session.purpose == "heldout":
            negatives.update(
                expected_negative_unit_ids(
                    declarations[session.session_id]["selection_document"],
                    [sources[r][1] for r in session.run_ids],
                )
            )
    if heldout and not (set(selection["expected_result_unit_ids"]) | negatives) <= units:
        raise ValueError(
            "Held-out annotations must cover every retained endpoint and preregistered negative"
        )
    if heldout:
        selection["expected_negative_unit_ids"] = sorted(negatives)
    frames = {_read(root, label.frame) for label in plan.annotations}
    label_bytes = sum(
        len(canonical_json(label.model_dump(mode="json")).encode("utf-8"))
        for label in plan.annotations
    )
    if (
        sum(p.stat().st_size for p in set(dependencies) | frames) + label_bytes
        > protocol.maximum_source_bytes
    ):
        raise ValueError("Registered source byte bound exceeded")
    for path, digest in dependencies.items():
        if path in copied:
            continue
        artifact = run.register_artifact(path, "reader_annotation_source")
        if artifact["sha256"] != digest:
            raise ValueError("Annotation dependency changed during publication")
    return resolved, session_metadata, selection


def publish_reader_annotations3(project_root, plan: ReaderAnnotationPublicationPlan3 | dict):
    """Publish reviewed labels; sealing never establishes independent truth/blinding."""
    from gradientclimb.experiments import RunRecorder

    root = Path(project_root).absolute()
    _no_links(root)
    raw = (
        plan.model_dump(mode="json") if isinstance(plan, ReaderAnnotationPublicationPlan3) else plan
    )
    if not isinstance(raw, dict) or len(canonical_json(raw).encode("utf-8")) > 32 * 1024**2:
        raise ValueError("Annotation plan must be a JSON object within 32 MiB")
    artifact_root = raw.get("artifact_root", "artifacts")
    _relative(artifact_root)
    started, cpu_started = time.perf_counter(), time.process_time()
    with (
        reader_publication_lease(root, artifact_root),
        RunRecorder(
            _inside(root, artifact_root),
            EXPERIMENT,
            raw,
            algorithm="reviewed-label-publication",
            environment="stored_reader_annotations",
            source_root=root,
            telemetry_interval_seconds=0,
            qualifies_real_game=False,
        ) as run,
    ):
        publication = None
        try:
            run.annotate(
                episodes=0,
                environment_steps=0,
                training_steps=0,
                optimizer_updates=0,
                policy_decisions=0,
                image_decodes=0,
                predictions=0,
                labels_generated=0,
                new_real_interaction_seconds=0,
                qualification_evidence=False,
                independent_truth_established=False,
                blinding_established=False,
                publication_complete=False,
                inherited_label_review_wall_seconds=None,
                inherited_label_review_compute=None,
                resource_scope="Recorder setup through final resource sample; imports, minimal JSON/root parsing and later sealing excluded; overlaps operation journal clocks",
            )
            publication = _Publication(run, started, cpu_started)
            publication.event("annotation_validation_started")
            plan = ReaderAnnotationPublicationPlan3.model_validate(raw)
            resolved, sessions, selection = _validate(root, plan, publication)
            annotations = []
            for index, (label, unit) in enumerate(zip(plan.annotations, resolved, strict=True)):
                publication.event(
                    "annotation_write_started",
                    source_run_id=label.run_id,
                    session_id=label.session_id,
                    label_id=label.label_id,
                )
                path = run.directory / f"annotation-{index:04d}.json"
                _write(path, label.model_dump(mode="json"))
                artifact = run.register_artifact(path, "reader_result_annotation")
                annotations.append(
                    {
                        "path": (run.directory / artifact["path"]).relative_to(root).as_posix(),
                        "sha256": artifact["sha256"],
                    }
                )
                publication.event(
                    "annotation_published",
                    source_run_id=label.run_id,
                    session_id=label.session_id,
                    label_id=label.label_id,
                    annotation=annotations[-1],
                    canonical_unit=unit,
                )
                run.annotate(
                    published_annotations=annotations, published_annotation_count=len(annotations)
                )
            index = {
                "schema_version": EXPERIMENT,
                "annotation_run_id": run.run_id,
                "annotations": annotations,
                "canonical_units": resolved,
                "sessions": sessions,
                "source_selection_coverage": selection,
                "qualification_evidence": False,
                "independent_truth_established": False,
                "blinding_established": False,
                "publication_state": "owning_completed_seal_required",
                "meaning": "Explicit reviewed labels only. Human truth/exposure assertions remain review obligations; this operation neither decodes nor predicts nor qualifies.",
            }
            output = run.directory / "reader-annotations.json"
            _write(output, index)
            run.register_artifact(output, "reader_annotation_index")
            publication.event("annotation_publication_completed")
            publication.close()
            run.finalize(episodes=0, environment_steps=0, training_steps=0, optimizer_updates=0)
        except BaseException as error:
            try:
                if publication is not None and not publication.stream.closed:
                    publication.event(
                        "annotation_publication_failed",
                        error_type=type(error).__name__,
                        error=str(error),
                        error_evidence=getattr(error, "evidence", None),
                    )
                    publication.close()
            except BaseException as storage_error:  # noqa: BLE001 - preserve primary and partial journal
                error.add_note(f"Annotation journal finalization failed: {storage_error}")
                if publication is not None and not publication.stream.closed:
                    try:
                        publication.close()
                    except BaseException as close_error:  # noqa: BLE001 - independent cleanup
                        error.add_note(f"Annotation journal close failed: {close_error}")
            raise
    record, directory, sealed_at = _verified(root, artifact_root, run.run_id)
    if record["status"] != "completed":
        raise ValueError("Annotation publication lacks a completed envelope")
    return {
        **index,
        "publication_state": "completed",
        "seal_created_at": sealed_at.isoformat(),
        "seal": {
            "path": (directory / "seal.json").relative_to(root).as_posix(),
            "sha256": sha256_file(directory / "seal.json"),
        },
    }
