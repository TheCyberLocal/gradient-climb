"""Opt-in native reader sampling; no native access or candidate inference here.

The public preflight uses the same sealed-receipt primitives as the post-source
verifier. The existing reader3 source-unit contract remains authoritative.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative
from gradientclimb.control.native_protocol import validate_protocol

from .reader_receipts3 import (
    EXPERIMENT,
    SessionReceiptPlan3,
    _binding,
    _frozen_json,
    _subset,
    _time,
    _verified,
)
from .reader_units3 import ReaderSourceSelection3, _timing

VERSION = "reader-native-collection-3.0"
POPULATION = "first_retained_terminal_callback_per_attempt_after_start_reset"
PRIOR_RUNS = (
    "915ac38b-ac38-45f1-bc16-7540f5aa2353",
    "1c263031-ac29-4dbc-8fa5-864a2473116a",
)
PARENT_HASH = "a74c9408a2e76802501c54e3ec35bcab23594b075370d18f172fc70f47545953"
SOURCE_FILES = (
    "scripts/run_screen_episodes.py",
    "src/gradientclimb/experiments/reader_collection3.py",
)


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NativeReaderAmendment(BaseModel):
    """A concrete amendment, with no authority while its status is draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["reader-native-collection-3.0"] = VERSION
    status: Literal["draft", "registered"]
    registered_at: AwareDatetime | None = None
    native_protocol: Reference
    reader_protocol: Reference | None = None
    implementation_files: tuple[Reference, ...] = ()
    prior_session_run_ids: tuple[str, ...] = PRIOR_RUNS
    maximum_reliability_sessions: Literal[3] = 3
    maximum_new_sessions: Literal[1] = 1
    reliability_session_index: Literal[3] = 3
    terminal_population: Literal[
        "first_retained_terminal_callback_per_attempt_after_start_reset"
    ] = POPULATION
    negative_opportunities: Literal["first_tune_in_declared_attempt_start_reset"] = (
        "first_tune_in_declared_attempt_start_reset"
    )
    console: Literal["counts_and_safety_only_until_all_labels_sealed"] = (
        "counts_and_safety_only_until_all_labels_sealed"
    )
    candidate_inference: Literal["offline_after_annotation_seal_only"] = (
        "offline_after_annotation_seal_only"
    )
    note: str = Field(min_length=1)


def _reference(root, reference):
    path = _inside(root, reference.path)
    if not path.is_file() or sha256_file(path) != reference.sha256:
        raise ValueError("Native reader evidence hash mismatch")
    return path


def validate_native_reader_amendment(project_root, document, args):
    """Validate the additive registration before any native object is created."""
    root = Path(project_root).absolute()
    _no_links(root)
    amendment = NativeReaderAmendment.model_validate(document)
    if amendment.status != "registered" or amendment.registered_at is None:
        raise ValueError("Native reader acquisition amendment is DRAFT; dispatch prohibited")
    if amendment.registered_at > datetime.now(UTC):
        raise ValueError("Native reader registration is in the future")
    if tuple(amendment.prior_session_run_ids) != PRIOR_RUNS:
        raise ValueError("Both prior reliability sessions must remain counted")
    if amendment.native_protocol.sha256 != PARENT_HASH:
        raise ValueError("Native reader amendment must preserve the original 2.3 protocol")
    parent = json.loads(_reference(root, amendment.native_protocol).read_bytes())
    if args.exploratory:
        raise ValueError("Exploratory mode cannot bypass held-out acquisition registration")
    validate_protocol(parent, args)
    if amendment.reader_protocol is None:
        raise ValueError("Registered amendment requires its exact reader3 protocol")
    reader_protocol = json.loads(_reference(root, amendment.reader_protocol).read_bytes())
    if (
        reader_protocol.get("schema_version") != "reader-validation-3.0"
        or reader_protocol.get("status") != "registered"
        or not _time(reader_protocol.get("registered_at")) <= amendment.registered_at
    ):
        raise ValueError("Reader protocol must be registered before native amendment")
    if {ref.path for ref in amendment.implementation_files} != set(SOURCE_FILES) or len(
        amendment.implementation_files
    ) != len(SOURCE_FILES):
        raise ValueError("Registered amendment must pin both acquisition implementation files")
    for ref in amendment.implementation_files:
        _reference(root, ref)
        imported_source = Path(__file__).resolve().parents[3] / ref.path
        if sha256_file(imported_source) != ref.sha256:
            raise ValueError("Imported acquisition implementation differs from registration")
    return amendment, parent


def assert_reliability_capacity(project_root, artifact_root):
    """One remaining session, counted across experiment names and amendments."""
    root = Path(project_root).absolute()
    for run_id in PRIOR_RUNS:
        _verified(root, artifact_root, run_id)
    runs = _inside(root, f"{artifact_root}/runs")
    for directory in runs.iterdir():
        _no_links(directory)
        if not directory.is_dir():
            continue
        path = directory / "run-start.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_bytes())
        config = record.get("configuration", {})
        if record.get("run_id") in PRIOR_RUNS:
            continue
        if (
            config.get("protocol_sha256") == PARENT_HASH
            or config.get("reader_native_collection_version") == VERSION
        ):
            raise ValueError("Remaining reliability session already consumed, including failures")


@dataclass(frozen=True)
class PreparedReaderCollection:
    binding: dict
    selection: ReaderSourceSelection3
    dependencies: tuple[tuple[Path, str], ...]
    preflight_wall_seconds: float
    preflight_cpu_core_seconds: float

    def bind(self, configuration):
        """Keep declared nested constraints, and reject override of native settings."""
        for key, value in self.binding.items():
            if key in configuration and not _subset(value, configuration[key]):
                raise ValueError(f"Reader declaration conflicts with native configuration: {key}")
        return {**self.binding, **configuration}


def prepare_native_reader_collection(
    project_root,
    artifact_root,
    receipt_run_id,
    source_config_id,
    *,
    expected_protocol,
    attempts,
    environment="actual_hill_climb_racing",
):
    """Public pre-acquisition receipt API; no pixels, native objects or predictions.

    Uses frozen selection/protocol copies, never a later mutable selection file.
    Full post-acquisition receipt and source-unit verification remains required.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    root = Path(project_root).absolute()
    _no_links(root)
    _relative(artifact_root)
    record, directory, sealed_at = _verified(root, artifact_root, receipt_run_id)
    plan = SessionReceiptPlan3.model_validate(record["configuration"])
    if record["status"] != "completed" or record["experiment_id"] != EXPERIMENT:
        raise ValueError("Native acquisition requires a completed declaration receipt")
    if plan.purpose != "heldout" or plan.protocol.model_dump() != expected_protocol:
        raise ValueError("Native declaration must bind this held-out reader protocol")
    if sealed_at >= datetime.now(UTC):
        raise ValueError("Actual declaration seal must precede native acquisition")
    artifacts = [
        a for a in record["artifact_manifest"] if a["kind"] == "reader_session_declaration"
    ]
    if len(artifacts) != 1:
        raise ValueError("Receipt requires exactly one declaration artifact")
    payload_path = _inside(directory, artifacts[0]["path"])
    payload = json.loads(payload_path.read_bytes())
    binding = _binding(plan, receipt_run_id)
    if (
        payload.get("receipt_run_id") != receipt_run_id
        or payload.get("required_source_configuration") != binding
        or any(payload.get(k) != v for k, v in plan.model_dump(mode="json").items())
    ):
        raise ValueError("Sealed receipt payload disagrees with its original configuration")
    if binding.get("reader_sampling_source_id") != source_config_id:
        raise ValueError("Source sampling role must be bound by the sealed declaration")
    protocol, protocol_path = _frozen_json(record, directory, plan.protocol)
    selection_data, selection_path = _frozen_json(record, directory, plan.selection)
    if (
        protocol.get("status") != "registered"
        or protocol.get("schema_version") != "reader-validation-3.0"
        or protocol.get("environment") != environment
        or _time(protocol.get("registered_at")) > _time(record["start_time"])
    ):
        raise ValueError("Receipt protocol registration/environment does not authorize acquisition")
    freeze, freeze_dir, freeze_sealed_at = _verified(
        root, artifact_root, plan.candidate_freeze_run_id
    )
    if (
        freeze["status"] != "completed"
        or freeze["experiment_id"] != "reader-candidate-freeze-3.0"
        or freeze["configuration"].get("protocol") != expected_protocol
        or freeze_sealed_at >= _time(record["start_time"])
    ):
        raise ValueError("Candidate must be sealed before declaration under the same protocol")
    selection = ReaderSourceSelection3.model_validate(selection_data)
    units = [u for u in selection.non_result_units if u.source_config_id == source_config_id]
    if not units:
        raise ValueError("Source role requires predetermined non-result opportunities")
    stages = set()
    for unit in units:
        match = re.fullmatch(r"attempt:(0|[1-9][0-9]*):start:tune", unit.stage)
        if not match or int(match[1]) >= attempts or unit.stage in stages:
            raise ValueError(
                "Negative opportunities require unique, in-budget attempt start Tune stages"
            )
        stages.add(unit.stage)
    dependencies = (
        directory / "run.json",
        directory / "seal.json",
        payload_path,
        protocol_path,
        selection_path,
        freeze_dir / "run.json",
        freeze_dir / "seal.json",
    )
    return PreparedReaderCollection(
        binding,
        selection,
        tuple((p, sha256_file(p)) for p in dependencies),
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )


class NativeReaderCollection:
    """Persist exact callback frames; never infer numeric truth or call capture."""

    def __init__(self, run, prepared):
        self.run, self.prepared = run, prepared
        self.directory = run.directory / "reader-collection"
        self.directory.mkdir()
        self.journal = self.directory / "sampling.jsonl"
        self.attempt_index = None
        self.phase = "initial_cleanup"
        self.completed = set()
        self.started = set()
        self.artifacts = []
        self.closed = False
        self.units = [
            u
            for u in prepared.selection.non_result_units
            if u.source_config_id == prepared.binding["reader_sampling_source_id"]
        ]
        self._event(
            "collection_opened",
            terminal_population=POPULATION,
            selection=prepared.selection.model_dump(mode="json"),
        )
        try:
            for path, digest in prepared.dependencies:
                artifact = run.register_artifact(path, "reader_collection_declaration_source")
                if artifact["sha256"] != digest:
                    raise ValueError("Reader receipt dependency changed before acquisition")
        except BaseException as error:
            try:
                self._event("setup_failed", error_type=type(error).__name__)
                run.register_artifact(self.journal, "reader_collection_sampling_journal")
            except BaseException as persistence_error:  # noqa: BLE001 - preserve setup failure
                error.add_note(
                    f"Partial reader setup journal could not be registered: {persistence_error}"
                )
            raise
        run.annotate(
            reader_collection_preflight={
                "wall_seconds": prepared.preflight_wall_seconds,
                "cpu_core_seconds": prepared.preflight_cpu_core_seconds,
                "scope": "Receipt and freeze integrity validation only; excludes imports, amendment/budget validation, recorder initialization and later publication. CPU covers this process including its threads, excluding children.",
            }
        )

    def _event(self, event, **data):
        with self.journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                canonical_json(
                    {
                        "event": event,
                        "monotonic_ns": time.perf_counter_ns(),
                        "attempt_index": self.attempt_index,
                        "phase": self.phase,
                        **data,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _save(self, observation, kind, metadata):
        _timing(metadata)
        path = self.directory / f"capture-{len(self.artifacts):04d}.png"
        self._event("capture_started", kind=kind, capture_metadata=metadata, file=path.name)
        with path.open("xb") as stream:
            Image.fromarray(observation.frame.rgb).save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        artifact = self.run.register_artifact(path, kind, metadata)
        self.artifacts.append(artifact)
        self._event("capture_completed", artifact=artifact)
        return artifact

    def start_attempt(self, index):
        self.attempt_index = None
        self.phase = f"attempt:{index}:start:tune"
        for unit in self.units:
            if unit.stage == self.phase:
                self.started.add(unit.unit_id)
                self._event("negative_opportunity_started", unit=unit.model_dump())

    def observe_start(self, observation):
        if observation.state != "tune":
            return
        for unit in self.units:
            if unit.stage == self.phase and unit.unit_id not in self.completed:
                self._save(
                    observation,
                    "reader_non_result_frame",
                    {
                        **observation.frame.metadata(),
                        "ui_state": observation.state,
                        "reader_sampling_unit_id": unit.unit_id,
                        "reader_sampling_stage": unit.stage,
                        "reader_sampling_ordinal": unit.capture_ordinal,
                    },
                )
                self.completed.add(unit.unit_id)

    def activate_endpoint(self, index):
        self.attempt_index, self.phase = index, "attempt_gameplay_and_park"

    def terminal(self, observation):
        self._save(
            observation,
            "terminal_frame"
            if self.attempt_index is not None
            else "reader_out_of_attempt_terminal_frame",
            {
                **observation.frame.metadata(),
                "ui_state": observation.state,
                "reader_attempt_index": self.attempt_index,
                "reader_collection_phase": self.phase,
                "terminal_population": POPULATION,
            },
        )

    def summary(self):
        return {
            "schema_version": VERSION,
            "terminal_population": POPULATION,
            "negative_opportunities_declared": len(self.units),
            "negative_opportunities_started": len(self.started),
            "negative_opportunities_completed": len(self.completed),
            "negative_opportunities_missing": [
                u.model_dump() for u in self.units if u.unit_id not in self.completed
            ],
            "terminal_frames": sum(a["kind"] == "terminal_frame" for a in self.artifacts),
            "out_of_attempt_terminal_frames": sum(
                a["kind"] == "reader_out_of_attempt_terminal_frame" for a in self.artifacts
            ),
            "qualification_evidence": False,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.closed:
            return
        self.closed = True
        try:
            self._event("collection_closed", failed=exc_type is not None, summary=self.summary())
        finally:
            self.run.register_artifact(self.journal, "reader_collection_sampling_journal")
            self.run.annotate(reader_collection=self.summary())


def prediction_safe_console(*, run_id=None, attempt_index=None, failed=False, completed=None):
    """Allowlisted fields only; exception text and score-derived classes are excluded."""
    return {
        "reader_collection": VERSION,
        "run_id": run_id,
        "attempt_index": attempt_index,
        "status": "halted" if failed else "recorded",
        "completed_attempts": completed,
        "predictions_withheld_until_annotations_sealed": True,
    }


class ReaderSetupOwnership:
    """Release both owners on setup failure; normal runtime keeps its own cleanup."""

    def __init__(self, *owners):
        self.owners = list(owners)
        self.transferred = False

    def own(self, owner):
        self.owners.insert(0, owner)

    def __enter__(self):
        return self

    def transfer(self):
        self.transferred = True

    def __exit__(self, exc_type, exc, traceback):
        if self.transferred:
            return
        failures = []
        for owner in self.owners:
            try:
                owner.close()
            except BaseException as failure:  # noqa: BLE001 - close every owner, preserve primary
                failures.append(failure)
        if failures:
            if exc is not None:
                for failure in failures:
                    exc.add_note(
                        f"Reader setup cleanup also failed: {type(failure).__name__}: {failure}"
                    )
            else:
                raise BaseExceptionGroup("Reader setup ownership cleanup failed", failures)
