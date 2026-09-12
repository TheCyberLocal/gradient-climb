"""Immutable reader-session declarations before acquisition or construction use.

Held-out acquisition must embed the returned receipt identity in its original
configuration. A timestamp typed into a later annotation is not a declaration.
Selection document bytes are frozen here; sampling semantics are checked by the
reader's source-endpoint resolver and independent selection audit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _relative, _walk

VERSION = "reader-session-declaration-3.0"
EXPERIMENT = "reader-session-declaration-3.0"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _Reference(_Contract):
    path: str
    sha256: Digest

    @model_validator(mode="after")
    def portable(self):
        _relative(self.path)
        return self


class SessionReceiptPlan3(_Contract):
    schema_version: Literal["reader-session-declaration-3.0"] = VERSION
    protocol: _Reference
    session_id: Name
    purpose: Literal["construction", "development", "heldout"]
    selection: _Reference
    candidate_freeze_run_id: str | None = None
    source_run_ids: tuple[str, ...] = ()
    source_run_config_binding: dict[str, JsonValue]
    note: Name

    @model_validator(mode="after")
    def identity(self):
        for run_id in self.source_run_ids:
            _run_component(run_id)
        if self.candidate_freeze_run_id is not None:
            _run_component(self.candidate_freeze_run_id)
        if len(set(self.source_run_ids)) != len(self.source_run_ids):
            raise ValueError("Duplicate source run in declaration")
        if {"session_id", "reader_session_declaration"} & self.source_run_config_binding.keys():
            raise ValueError("Caller configuration cannot override receipt identity fields")
        if self.purpose == "heldout":
            if self.candidate_freeze_run_id is None or self.source_run_ids:
                raise ValueError("Held-out declaration needs a frozen candidate before new sources")
        elif not self.source_run_ids:
            raise ValueError(
                "Construction/development declaration needs exact existing source runs"
            )
        return self


def _run_component(value):
    if "/" in value or _relative(value) != value:
        raise ValueError("Run identity must be one safe path component")
    return value


def _read(root, reference):
    reference = _Reference.model_validate(reference)
    path = _inside(root, reference.path)
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("Receipt evidence is missing or exceeds its file bound")
    if sha256_file(path) != reference.sha256:
        raise ValueError("Receipt evidence hash mismatch")
    return path


def _time(value):
    # Pydantic handles awareness explicitly rather than accepting naive wall time.
    class Stamp(_Contract):
        value: AwareDatetime

    return Stamp(value=value).value


def _verified(root, artifact_root, run_id):
    from gradientclimb.experiments import verify_run

    directory = _inside(root, f"{artifact_root}/runs/{_run_component(run_id)}")
    for _ in _walk(directory):
        pass
    record = json.loads((directory / "run.json").read_bytes())
    for artifact in record["artifact_manifest"]:
        _inside(directory, artifact["path"])
    seal_path = _inside(directory, "seal.json")
    if not seal_path.is_file():
        raise ValueError("Declaration dependency is not sealed")
    seal = json.loads(seal_path.read_bytes())
    for name in seal["files"]:
        _inside(directory, name)
    if not verify_run(_inside(root, artifact_root), run_id)["valid"]:
        raise ValueError("Declaration dependency seal failed")
    sealed_at = _time(seal["created_at"])
    if sealed_at < _time(record["end_time"]):
        raise ValueError("Actual seal timestamp precedes source finalization")
    return record, directory, sealed_at


def _subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items()
        )
    return type(expected) is type(actual) and expected == actual


def _binding(plan, receipt_run_id):
    return {
        **plan.source_run_config_binding,
        "session_id": plan.session_id,
        "reader_session_declaration": {
            "receipt_run_id": receipt_run_id,
            "protocol_sha256": plan.protocol.sha256,
            "candidate_freeze_run_id": plan.candidate_freeze_run_id,
            "selection_sha256": plan.selection.sha256,
            "session_id": plan.session_id,
            "purpose": plan.purpose,
        },
    }


def _protocol(root, reference):
    path = _read(root, reference)
    data = json.loads(path.read_bytes())
    if data.get("schema_version", VERSION) != "reader-validation-3.0":
        # Explicit schema is required for receipts; legacy records remain readable elsewhere.
        raise ValueError("Session declaration requires a reader-validation-3.0 protocol")
    if data.get("status") != "registered" or _time(data.get("registered_at")) > datetime.now(UTC):
        raise ValueError("Session declaration requires a registered protocol")
    return path, data


def _frozen_json(record, directory, reference):
    copies = [
        a
        for a in record["artifact_manifest"]
        if a["kind"] == "reader_session_declaration_source" and a["sha256"] == reference.sha256
    ]
    if not copies:
        raise ValueError("Receipt lacks its hash-pinned protocol or selection bytes")
    path = _inside(directory, copies[0]["path"])
    return json.loads(path.read_bytes()), path


def _heldout_source_closure(root, artifact_root, receipt_run_id, session_id):
    """Resolve original acquisition bindings, ignoring nested later-reader plans."""
    store = _inside(root, artifact_root)
    matches = set()
    for directory in sorted(_inside(store, "runs").iterdir()):
        _no_links(directory)
        if not directory.is_dir():
            continue
        record_path = _inside(directory, "run.json")
        if not record_path.is_file():
            record_path = _inside(directory, "run-start.json")
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_bytes())
        config = record.get("configuration", {})
        binding = config.get("reader_session_declaration")
        if not isinstance(binding, dict):
            continue
        if binding.get("receipt_run_id") != receipt_run_id:
            if config.get("session_id") == session_id:
                raise ValueError("One source session identity cannot belong to different receipts")
            continue
        if config.get("session_id") != session_id or binding.get("session_id") != session_id:
            raise ValueError("Source receipt binding aliases its declared session identity")
        if record.get("run_id") != directory.name:
            raise ValueError("Acquisition record identity disagrees with its canonical directory")
        if not _inside(directory, "seal.json").is_file():
            raise ValueError("Unsealed acquisition bound to this receipt requires recovery")
        _verified(root, artifact_root, directory.name)
        matches.add(directory.name)
    return matches


def publish_session_declaration3(
    project_root,
    artifact_root,
    protocol_ref_dict,
    session_id,
    purpose,
    source_run_config_binding_dict,
    *,
    candidate_freeze_run_id=None,
    selection_ref,
    source_run_ids=(),
    note="Immutable session purpose and selection; no predictions or fitting.",
):
    """Seal purpose/selection now; return configuration to bind future capture."""
    from gradientclimb.experiments import RunRecorder

    root = Path(project_root).absolute()
    _no_links(root)
    _relative(artifact_root)
    plan = SessionReceiptPlan3(
        protocol=protocol_ref_dict,
        session_id=session_id,
        purpose=purpose,
        selection=selection_ref,
        candidate_freeze_run_id=candidate_freeze_run_id,
        source_run_ids=source_run_ids,
        source_run_config_binding=source_run_config_binding_dict,
        note=note,
    )
    protocol_path, protocol = _protocol(root, plan.protocol)
    selection_path = _read(root, plan.selection)
    selection = json.loads(selection_path.read_bytes())
    if not isinstance(selection, dict) or not selection:
        raise ValueError("Selection document must be a nonempty JSON object")
    dependencies = [protocol_path, selection_path]
    if plan.candidate_freeze_run_id is not None:
        freeze, directory, _ = _verified(root, artifact_root, plan.candidate_freeze_run_id)
        if (
            freeze["status"] != "completed"
            or freeze["experiment_id"] != "reader-candidate-freeze-3.0"
            or freeze["configuration"].get("protocol") != plan.protocol.model_dump()
        ):
            raise ValueError("Candidate freeze does not bind this registered protocol")
        dependencies.extend((directory / "run.json", directory / "seal.json"))
    for run_id in plan.source_run_ids:
        record, directory, _ = _verified(root, artifact_root, run_id)
        original_declaration = record["configuration"].get("reader_session_declaration", {})
        if original_declaration.get("purpose") == "heldout":
            raise ValueError("Original held-out declaration cannot be repurposed for construction")
        if not _subset(plan.source_run_config_binding, record["configuration"]):
            raise ValueError("Existing source configuration disagrees with declaration")
        if record["environment"] != protocol["environment"]:
            raise ValueError("Source environment disagrees with protocol")
        dependencies.extend((directory / "run.json", directory / "seal.json"))
    with RunRecorder(
        _inside(root, artifact_root),
        EXPERIMENT,
        plan.model_dump(mode="json"),
        algorithm="immutable-purpose-selection-receipt",
        environment="stored_reader_declaration",
        source_root=root,
        telemetry_interval_seconds=0,
        qualifies_real_game=False,
    ) as run:
        run.annotate(
            declaration_resource_scope="recorder publication window; excludes imports, initial validation and dependency seal preflight; no capture, predictions or fitting"
        )
        for path in dependencies:
            artifact = run.register_artifact(path, "reader_session_declaration_source")
            expected = {
                protocol_path: plan.protocol.sha256,
                selection_path: plan.selection.sha256,
            }.get(path)
            if expected is not None and artifact["sha256"] != expected:
                raise ValueError("Declaration evidence changed during publication")
        declaration = {
            **plan.model_dump(mode="json"),
            "receipt_run_id": run.run_id,
            "required_source_configuration": _binding(plan, run.run_id),
            "chronology_scope": "Actual receipt seal.created_at is the immutable declaration time; no backdated declared_at is accepted.",
            "selection_semantics": "Pinned document only; endpoint mappings and completeness require independent source-journal verification.",
        }
        output = run.directory / "reader-session-declaration.json"
        output.write_text(canonical_json(declaration) + "\n", encoding="utf-8", newline="\n")
        artifact = run.register_artifact(output, "reader_session_declaration")
        run.finalize(
            episodes=0, environment_steps=0, training_steps=0, qualification_evidence=False
        )
    _, _, sealed_at = _verified(root, artifact_root, run.run_id)
    return {
        "run_id": run.run_id,
        "receipt_run_id": run.run_id,
        "declared_at": sealed_at.isoformat(),
        "seal_created_at": sealed_at.isoformat(),
        "declaration_sha256": artifact["sha256"],
        "required_source_configuration": declaration["required_source_configuration"],
        "selection": plan.selection.model_dump(),
    }


def verify_session_declaration3(
    root,
    artifact_root,
    receipt_run_id,
    expected_protocol_ref,
    session_id,
    purpose,
    acquired_at,
    source_records,
) -> dict[str, Any]:
    """Verify actual receipt chronology and original, sealed source bindings."""
    root = Path(root).absolute()
    _no_links(root)
    _relative(artifact_root)
    record, directory, sealed_at = _verified(root, artifact_root, receipt_run_id)
    if record["status"] != "completed" or record["experiment_id"] != EXPERIMENT:
        raise ValueError("Session declaration receipt is not completed")
    plan = SessionReceiptPlan3.model_validate(record["configuration"])
    expected_protocol = _Reference.model_validate(expected_protocol_ref)
    if (
        plan.protocol != expected_protocol
        or plan.session_id != session_id
        or plan.purpose != purpose
    ):
        raise ValueError("Immutable declaration disagrees with protocol, session or purpose")
    artifacts = [
        a for a in record["artifact_manifest"] if a["kind"] == "reader_session_declaration"
    ]
    if len(artifacts) != 1:
        raise ValueError("Receipt requires exactly one immutable declaration artifact")
    declaration = json.loads(_inside(directory, artifacts[0]["path"]).read_bytes())
    if declaration.get("receipt_run_id") != receipt_run_id or any(
        declaration.get(key) != value for key, value in plan.model_dump(mode="json").items()
    ):
        raise ValueError("Declaration artifact disagrees with its sealed configuration")
    if declaration.get("required_source_configuration") != _binding(plan, receipt_run_id):
        raise ValueError("Declaration source binding disagrees with receipt identity")
    protocol, _ = _frozen_json(record, directory, plan.protocol)
    selection, selection_path = _frozen_json(record, directory, plan.selection)
    acquired_at = _time(acquired_at)
    if purpose == "heldout" and not sealed_at < acquired_at:
        raise ValueError("Actual declaration seal must precede held-out acquisition")
    if plan.candidate_freeze_run_id is not None:
        freeze, _, freeze_sealed_at = _verified(root, artifact_root, plan.candidate_freeze_run_id)
        if (
            freeze["status"] != "completed"
            or freeze["experiment_id"] != "reader-candidate-freeze-3.0"
            or freeze["configuration"].get("protocol") != plan.protocol.model_dump()
            or freeze_sealed_at >= _time(record["start_time"])
        ):
            raise ValueError("Candidate must be sealed before its session declaration")
    supplied = list(source_records)
    run_ids = [r["run_id"] for r in supplied]
    if not supplied or len(set(run_ids)) != len(run_ids):
        raise ValueError("Declaration verification needs distinct canonical source records")
    if purpose != "heldout" and set(run_ids) != set(plan.source_run_ids):
        raise ValueError("Existing-source declaration must retain its exact source run set")
    for supplied_record in supplied:
        source, _, source_sealed_at = _verified(root, artifact_root, supplied_record["run_id"])
        if source != supplied_record:
            raise ValueError("Supplied source record disagrees with canonical sealed bytes")
        if source["environment"] != protocol["environment"]:
            raise ValueError("Source environment disagrees with declaration protocol")
        if acquired_at > _time(source["start_time"]):
            raise ValueError("Session acquisition must precede each actual source run")
        expected = (
            _binding(plan, receipt_run_id)
            if purpose == "heldout"
            else plan.source_run_config_binding
        )
        if not _subset(expected, source["configuration"]):
            raise ValueError("Original source configuration does not bind its declaration")
        if purpose != "heldout" and source_sealed_at >= _time(record["start_time"]):
            raise ValueError("Existing construction source must be sealed before declaration")
    if purpose == "heldout" and set(run_ids) != _heldout_source_closure(
        root, artifact_root, receipt_run_id, session_id
    ):
        raise ValueError(
            "Held-out session must include every canonical source run bound to its receipt"
        )
    return {
        "receipt_run_id": receipt_run_id,
        "session_id": plan.session_id,
        "purpose": plan.purpose,
        "protocol_sha256": plan.protocol.sha256,
        "declared_at": sealed_at.isoformat(),
        "seal_created_at": sealed_at.isoformat(),
        "declaration_sha256": artifacts[0]["sha256"],
        "selection": plan.selection.model_dump(),
        "selection_document": selection,
        "selection_file": selection_path.relative_to(root).as_posix(),
        "candidate_freeze_run_id": plan.candidate_freeze_run_id,
        "source_run_ids": run_ids,
        "required_source_configuration": declaration["required_source_configuration"],
        "qualification_evidence": False,
    }
