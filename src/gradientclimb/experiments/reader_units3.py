"""Resolve independent reader units from sealed source identity, never label names.

The resolver does not read pixels or judge labels. Qualification chronology and
the selection document's immutable receipt are verified by reader_receipts3.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative

from .reader_receipts3 import _verified

VERSION = "reader-canonical-source-unit-3.0"
Name = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0, strict=True)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NegativeSamplingUnit3(_Contract):
    source_config_id: Name
    unit_id: Name
    stage: Name
    capture_ordinal: Count


class ReaderSourceSelection3(_Contract):
    schema_version: Literal["reader-source-selection-3.0"] = "reader-source-selection-3.0"
    result_endpoint_rule: Literal["first_retained_terminal_per_attempt"] = (
        "first_retained_terminal_per_attempt"
    )
    non_result_units: tuple[NegativeSamplingUnit3, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def distinct(self):
        for field in ("unit_id", "capture_ordinal"):
            if len({(u.source_config_id, getattr(u, field)) for u in self.non_result_units}) != len(
                self.non_result_units
            ):
                raise ValueError(
                    "Negative selection units and capture ordinals must be distinct within each source"
                )
        return self


class UnitResolutionError(ValueError):
    """Source identity is intact, but its sampling unit cannot be established."""


def _timing(metadata):
    fields = ("started_ns", "timestamp_ns", "completed_ns")
    values = [metadata.get(key) for key in fields]
    if any(type(v) is not int or v < 0 for v in values) or values != sorted(values):
        raise UnitResolutionError("Source sampling unit lacks a valid capture timing triple")
    return dict(zip(fields, values, strict=True))


def _negative_id(run_id, ordinal):
    return f"registered-negative:{run_id}:capture:{ordinal}"


def expected_negative_unit_ids(selection_document, source_records):
    """Expected complete denominator from receipt-verified selection/source records.

    Callers must pass canonical records already verified by the session receipt
    API. This pure helper does not replace its chronology or source binding checks.
    """
    selection = ReaderSourceSelection3.model_validate(selection_document)
    records = list(source_records)
    result = []
    for unit in selection.non_result_units:
        sources = [
            r
            for r in records
            if r["configuration"].get("reader_sampling_source_id") == unit.source_config_id
        ]
        if len(sources) != 1:
            raise UnitResolutionError(
                "Every preregistered negative unit must bind exactly one source run"
            )
        result.append(_negative_id(sources[0]["run_id"], unit.capture_ordinal))
    return tuple(result)


def _endpoint(record, artifact):
    if artifact["kind"] != "terminal_frame":
        raise UnitResolutionError("Result endpoints require an original terminal_frame artifact")
    timing = _timing(artifact.get("metadata", {}))
    matches, indices = [], set()
    episodes = record.get("summary", {}).get("episode_summaries", [])
    for episode in episodes:
        index = episode.get("attempt", {}).get("index")
        if type(index) is not int or index < 0 or index in indices:
            raise UnitResolutionError("Canonical native attempt indices are missing or ambiguous")
        indices.add(index)
        for reading in episode.get("terminal_readings", []):
            if reading.get("timestamp_ns") == timing["timestamp_ns"]:
                matches.append((index, reading))
    if len(matches) != 1:
        raise UnitResolutionError(
            "Terminal capture timestamp has a missing or ambiguous endpoint mapping"
        )
    index, reading = matches[0]
    if _timing(reading) != timing:
        raise UnitResolutionError("Terminal artifact and endpoint reading timing metadata disagree")
    for field in ("backend", "width", "height"):
        if (
            field in artifact["metadata"]
            and field in reading
            and artifact["metadata"][field] != reading[field]
        ):
            raise UnitResolutionError("Terminal artifact and reading capture metadata disagree")
    return {
        "canonical_unit_id": f"native-endpoint:{record['run_id']}:attempt:{index}",
        "unit_kind": "native_attempt_endpoint",
        "attempt_index": index,
        "capture_timing": timing,
    }


def result_selection_coverage(source_records):
    """Enumerate retained endpoints and separately retain attempts with no image.

    Records must already be seal-verified. No score, validity or label is read.
    """
    expected, unavailable = set(), []
    for record in source_records:
        retained = set()
        for artifact in record["artifact_manifest"]:
            if artifact["kind"] == "terminal_frame":
                unit = _endpoint(record, artifact)
                expected.add(unit["canonical_unit_id"])
                retained.add(unit["attempt_index"])
        seen = set()
        for episode in record.get("summary", {}).get("episode_summaries", []):
            index = episode.get("attempt", {}).get("index")
            if type(index) is not int or index < 0 or index in seen:
                raise UnitResolutionError(
                    "Canonical native attempt indices are missing or ambiguous"
                )
            seen.add(index)
            if index not in retained:
                unavailable.append(
                    {
                        "source_run_id": record["run_id"],
                        "attempt_index": index,
                        "reason": "no_retained_terminal_frame",
                    }
                )
    return {
        "expected_result_unit_ids": sorted(expected),
        "attempts_without_retained_terminal": unavailable,
    }


def expected_result_unit_ids(source_records):
    return tuple(result_selection_coverage(source_records)["expected_result_unit_ids"])


def _first_retained_endpoint(record, artifact, unit):
    candidates = []
    for retained in record["artifact_manifest"]:
        if retained["kind"] == "terminal_frame":
            resolved = _endpoint(record, retained)
            if resolved["canonical_unit_id"] == unit["canonical_unit_id"]:
                candidates.append((resolved["capture_timing"]["timestamp_ns"], retained))
    first_timestamp = min(stamp for stamp, _ in candidates)
    first = [row for stamp, row in candidates if stamp == first_timestamp]
    if len(first) != 1 or first[0]["path"] != artifact["path"]:
        raise UnitResolutionError(
            "Held-out endpoint must use the first retained terminal capture per attempt"
        )


def _negative(record, artifact, selection_document):
    if selection_document is None:
        raise UnitResolutionError(
            "Non-result qualification needs preregistered source sampling units"
        )
    selection = ReaderSourceSelection3.model_validate(selection_document)
    config_id = record["configuration"].get("reader_sampling_source_id")
    declared = [u for u in selection.non_result_units if u.source_config_id == config_id]
    if not declared:
        raise UnitResolutionError("Negative source configuration was not preregistered")
    if artifact["kind"] != "reader_non_result_frame":
        raise UnitResolutionError(
            "Negative qualification requires original reader_non_result_frame metadata"
        )
    matches, intervals = [], []
    for unit in declared:
        expected = {
            "reader_sampling_unit_id": unit.unit_id,
            "reader_sampling_stage": unit.stage,
            "reader_sampling_ordinal": unit.capture_ordinal,
        }
        captures = [
            a
            for a in record["artifact_manifest"]
            if a["kind"] == "reader_non_result_frame"
            and all(
                type(a.get("metadata", {}).get(k)) is type(v) and a["metadata"][k] == v
                for k, v in expected.items()
            )
        ]
        if len(captures) != 1:
            raise UnitResolutionError(
                "Every preregistered negative selector needs one unique source artifact"
            )
        capture = captures[0]
        timing = _timing(capture["metadata"])
        intervals.append((timing["started_ns"], timing["completed_ns"]))
        if capture["path"] == artifact["path"] and capture["sha256"] == artifact["sha256"]:
            matches.append((unit, timing))
    intervals.sort()
    if any(current[0] <= previous[1] for previous, current in pairwise(intervals)):
        raise UnitResolutionError(
            "Declared negative sampling units have overlapping capture intervals"
        )
    if len(matches) != 1:
        raise UnitResolutionError(
            "Source artifact does not identify one preregistered negative unit"
        )
    unit, timing = matches[0]
    return {
        "canonical_unit_id": _negative_id(record["run_id"], unit.capture_ordinal),
        "unit_kind": "preregistered_non_result_capture",
        "selection_unit": unit.model_dump(),
        "capture_timing": timing,
    }


class ReaderUnitResolver3:
    """Verify each source seal once per operation; never cache across publications."""

    def __init__(self, project_root, artifact_root="artifacts"):
        self.root = Path(project_root).absolute()
        _no_links(self.root)
        _relative(artifact_root)
        self.artifact_root = artifact_root
        self._sources = {}

    def resolve(
        self,
        frame,
        *,
        source_run_id,
        label_is_result,
        purpose,
        selection_document=None,
        construction_fallback_reason=None,
    ):
        if (
            purpose not in {"construction", "development", "heldout"}
            or type(label_is_result) is not bool
        ):
            raise ValueError("An explicit reader purpose and result-label kind are required")
        if construction_fallback_reason is not None and (
            purpose != "construction" or not construction_fallback_reason.strip()
        ):
            raise ValueError("Fallback requires an explicit construction-only reason")
        if source_run_id not in self._sources:
            record, directory, _ = _verified(self.root, self.artifact_root, source_run_id)
            if record["run_id"] != source_run_id:
                raise ValueError("Canonical record identity disagrees with source run")
            self._sources[source_run_id] = (
                record,
                directory,
                sha256_file(directory / "run.json"),
                sha256_file(directory / "seal.json"),
            )
        record, directory, record_hash, seal_hash = self._sources[source_run_id]
        if (
            sha256_file(directory / "run.json") != record_hash
            or sha256_file(directory / "seal.json") != seal_hash
        ):
            raise ValueError("Canonical source record or seal changed during unit resolution")
        path = _inside(self.root, frame["path"])
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or path.stat().st_size > 32 * 1024**2
        ):
            raise ValueError("Reader unit frame must be a bounded file inside its source run")
        relative = path.relative_to(directory).as_posix()
        artifacts = [
            a
            for a in record["artifact_manifest"]
            if a["path"] == relative and a["sha256"] == frame["sha256"]
        ]
        if len(artifacts) != 1:
            raise ValueError("Frame path and hash must identify exactly one sealed artifact")
        artifact = artifacts[0]
        if sha256_file(path) != frame["sha256"]:
            raise ValueError("Reader unit frame bytes changed after source verification")
        fallback = None
        try:
            unit = (
                _endpoint(record, artifact)
                if label_is_result
                else _negative(record, artifact, selection_document)
            )
            if purpose == "heldout" and label_is_result:
                ReaderSourceSelection3.model_validate(selection_document or {})
                _first_retained_endpoint(record, artifact, unit)
        except UnitResolutionError as error:
            if construction_fallback_reason is None:
                raise
            fallback = {
                "reason": construction_fallback_reason,
                "unresolved_source_mapping": str(error),
            }
            unit = {
                "canonical_unit_id": f"construction-artifact:{source_run_id}:{frame['sha256']}",
                "unit_kind": "unresolved_construction_artifact",
                "capture_timing": None,
            }
        return {
            "schema_version": VERSION,
            **unit,
            "source_run_id": source_run_id,
            "source_record_sha256": record_hash,
            "source_seal_sha256": seal_hash,
            "artifact": {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "artifact_id": artifact["artifact_id"],
                "kind": artifact["kind"],
            },
            "qualification_eligible": purpose == "heldout"
            and fallback is None
            and record["environment"] == "actual_hill_climb_racing",
            "construction_fallback": fallback,
            "limitations": "Identity and timing resolution only; not label correctness, reader accuracy, independent sampling or competence. Selection receipt chronology is checked separately.",
        }


def resolve_reader_unit(project_root, artifact_root, frame, **kwargs):
    """Convenience wrapper; use one ReaderUnitResolver3 instance for a label batch."""
    return ReaderUnitResolver3(project_root, artifact_root).resolve(frame, **kwargs)
