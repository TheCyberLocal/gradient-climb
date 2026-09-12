"""Prospective whole-session reader partitions and blinded-label evidence.

This validates supplied provenance records, not the truth of a human assertion.
Source seal/hash verification and frozen reader execution are separate gates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from .cycle3_measurements import FrozenRecord

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
READER_PROVENANCE_VERSION = "reader-session-provenance-3.0"
PredictionExposureStatus = Literal["none_reported", "known", "unknown"]


def resolve_prediction_exposure(status, first_exposure_at):
    """Preserve old audited-absence semantics while admitting unknown first-view time."""
    if first_exposure_at is not None and status in ("none_reported", "unknown"):
        raise ValueError("A recorded exposed timestamp contradicts the exposure status")
    return status or ("known" if first_exposure_at is not None else "none_reported")


class ConstructionSource(FrozenRecord):
    """Reader, anchor, UI, glyph, crop or parent; each ancestor must resolve."""

    source_id: str = Field(min_length=1)
    sha256: Sha256
    session_ids: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()


class ReaderSession(FrozenRecord):
    session_id: str = Field(min_length=1)
    run_ids: tuple[str, ...] = Field(min_length=1)
    acquired_at: AwareDatetime
    sealed_at: AwareDatetime
    purpose: Literal["construction", "development", "heldout"]
    # Older declarations omit status: their original null=audited absence meaning
    # stays intact. New construction can state known/unknown exposure with no
    # invented first-view time. Such an interval cannot establish heldout blinding.
    prediction_exposure_status: PredictionExposureStatus | None = None
    first_prediction_exposure_at: AwareDatetime | None
    exposure_reviewed_through: AwareDatetime
    exposure_evidence_sha256: Sha256
    seal_verified: bool

    @model_validator(mode="after")
    def coherent(self):
        resolve_prediction_exposure(
            self.prediction_exposure_status, self.first_prediction_exposure_at
        )
        if self.sealed_at < self.acquired_at:
            raise ValueError("Session must be acquired before sealing")
        if self.exposure_reviewed_through < self.acquired_at:
            raise ValueError("Exposure review predates session acquisition")
        if self.first_prediction_exposure_at is not None and not (
            self.acquired_at <= self.first_prediction_exposure_at <= self.exposure_reviewed_through
        ):
            raise ValueError("Prediction exposure is outside the audited interval")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("Duplicate run in session")
        return self

    @property
    def exposure_status(self):
        return resolve_prediction_exposure(
            self.prediction_exposure_status, self.first_prediction_exposure_at
        )


class BlindedLabel(FrozenRecord):
    label_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    frame_sha256: Sha256
    # A crop/augmentation must retain its raw-frame ancestry; a different crop
    # hash must never turn a construction image into held-out evidence.
    source_hashes: tuple[Sha256, ...] = Field(min_length=1)
    field: str = Field(min_length=1)
    value: str = Field(min_length=1)
    labeled_at: AwareDatetime
    annotation_sha256: Sha256


class ReaderSplitManifest(FrozenRecord):
    version: Literal["reader-session-provenance-3.0"] = READER_PROVENANCE_VERSION
    protocol_id: str = Field(min_length=1)
    registered_at: AwareDatetime
    frozen_at: AwareDatetime
    protocol_sha256: Sha256
    sources: tuple[ConstructionSource, ...]
    # Require all three source categories explicitly, even when a category
    # shares a node with another category. Empty categories are not sufficient.
    reader_roots: tuple[str, ...] = Field(min_length=1)
    anchor_roots: tuple[str, ...] = Field(min_length=1)
    ui_roots: tuple[str, ...] = Field(min_length=1)
    sessions: tuple[ReaderSession, ...]
    labels: tuple[BlindedLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def chronological(self):
        if self.frozen_at < self.registered_at:
            raise ValueError("Manifest cutoff precedes protocol registration")
        return self


def construction_closure(manifest: ReaderSplitManifest) -> tuple[set[str], set[str]]:
    """Resolve every parent before returning construction hashes and sessions."""
    nodes = {node.source_id: node for node in manifest.sources}
    if len(nodes) != len(manifest.sources):
        raise ValueError("Duplicate construction source identifier")
    completed, active, hashes, sessions = set(), set(), set(), set()

    def visit(source_id):
        if source_id in active:
            raise ValueError("Construction provenance contains a cycle")
        if source_id in completed:
            return
        if source_id not in nodes:
            raise ValueError(f"Unresolved construction source: {source_id}")
        active.add(source_id)
        node = nodes[source_id]
        for parent in node.parents:
            visit(parent)
        hashes.add(node.sha256)
        sessions.update(node.session_ids)
        active.remove(source_id)
        completed.add(source_id)

    for root in manifest.reader_roots + manifest.anchor_roots + manifest.ui_roots:
        visit(root)
    if completed != set(nodes):
        raise ValueError("Construction manifest contains sources outside declared root closure")
    return hashes, sessions


def validate_blinded_split(manifest: ReaderSplitManifest) -> dict:
    """Fail closed on missing, exposed, duplicated or construction-linked data.

    All labels from a held-out session must precede that session's first viewed
    prediction. A run is never treated as an independent session automatically.
    """
    construction_hashes, construction_sessions = construction_closure(manifest)
    sessions = {session.session_id: session for session in manifest.sessions}
    if len(sessions) != len(manifest.sessions):
        raise ValueError("Duplicate session identifier")
    run_sessions = {}
    for session in manifest.sessions:
        for run_id in session.run_ids:
            if run_id in run_sessions:
                raise ValueError("A run cannot belong to multiple sessions")
            run_sessions[run_id] = session.session_id
    label_ids, frame_fields, used_sessions = set(), set(), set()
    frame_sessions = {}
    field_frames: dict[str, set[str]] = {}
    field_sessions: dict[str, set[str]] = {}
    for label in manifest.labels:
        if label.label_id in label_ids:
            raise ValueError("Duplicate label identifier")
        label_ids.add(label.label_id)
        key = label.frame_sha256, label.field
        if key in frame_fields:
            raise ValueError("Duplicate frame/field labels cannot inflate reader evidence")
        frame_fields.add(key)
        if (
            label.frame_sha256 in frame_sessions
            and frame_sessions[label.frame_sha256] != label.session_id
        ):
            raise ValueError("Identical frames cannot count as independent sessions")
        frame_sessions[label.frame_sha256] = label.session_id
        session = sessions.get(label.session_id)
        if session is None or run_sessions.get(label.run_id) != label.session_id:
            raise ValueError("Label run/session is not explicitly registered")
        if session.purpose != "heldout" or label.session_id in construction_sessions:
            raise ValueError("Held-out label comes from a construction/development session")
        hashes = set(label.source_hashes) | {label.frame_sha256}
        if hashes & construction_hashes:
            raise ValueError("Held-out frame/crop ancestry overlaps construction sources")
        if label.frame_sha256 not in label.source_hashes:
            raise ValueError("Label ancestry must include the original frame hash")
        if not session.seal_verified:
            raise ValueError("Held-out session seal has not been verified")
        if not manifest.registered_at < session.acquired_at:
            raise ValueError("Held-out session was acquired before prospective registration")
        if not session.sealed_at <= label.labeled_at <= manifest.frozen_at:
            raise ValueError("Label must follow source sealing and precede the frozen cutoff")
        if session.exposure_reviewed_through < manifest.frozen_at:
            raise ValueError("Prediction exposure audit does not cover the frozen cutoff")
        exposed: datetime | None = session.first_prediction_exposure_at
        if exposed is None and session.exposure_status != "none_reported":
            raise ValueError("Unknown first-view timing cannot establish unexposed heldout labels")
        if exposed is not None and label.labeled_at >= exposed:
            raise ValueError("Session predictions were exposed before all independent labels")
        used_sessions.add(label.session_id)
        field_frames.setdefault(label.field, set()).add(label.frame_sha256)
        field_sessions.setdefault(label.field, set()).add(label.session_id)
    return {
        "version": READER_PROVENANCE_VERSION,
        "protocol": manifest.protocol_id,
        "protocol_sha256": manifest.protocol_sha256,
        "labels": len(manifest.labels),
        "unique_frames": len({label.frame_sha256 for label in manifest.labels}),
        "independent_sessions": len(used_sessions),
        "construction_sources": len(manifest.sources),
        "construction_unique_hashes": len(construction_hashes),
        "construction_sessions": sorted(construction_sessions),
        "fields": {
            field: {"unique_frames": len(frames), "sessions": len(field_sessions[field])}
            for field, frames in sorted(field_frames.items())
        },
        "episode_count": 0,
        "reader_accuracy_qualified": False,
        "scope": "Provenance validation only; frozen reader accuracy and availability unmeasured",
    }
