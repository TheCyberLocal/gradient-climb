"""Monotonic known reader exposure for cooperating canonical-store publishers.

Fresh human attestations cannot erase already recorded source use. This is local
store authority, not a claim to discover unrecorded or external predictions.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links, _walk
from gradientclimb.capture.demonstrations import read_complete_journal

from .reader_receipts3 import _run_component, _verified

OPERATIONS = {
    "result-reader-construction-3.0",
    "reader-validation-3.0",
}
EXPOSURE_PHASES = {"image_processing_started", "fitting_started", "prediction_started"}
JOURNAL = "reader-operations.jsonl"


class KnownReaderExposure(ValueError):
    """Retain exact conflicting run references for the failed current attempt."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@contextmanager
def reader_publication_lease(project_root, artifact_root="artifacts"):
    """Hold before recorder construction through final seal; never native input."""
    from gradientclimb.datasets.partitions import _publication_lease

    root = Path(project_root).absolute()
    _no_links(root)
    # Share the existing canonical-store publication lease and its narrowly
    # exempted operational marker; reader work adds no new lock-file inventory.
    with _publication_lease(_inside(root, artifact_root)):
        yield


def _identities(record):
    """Use source-record bindings, never newly supplied annotation aliases."""
    config = record.get("configuration", {})
    identities = {("run", record["run_id"])}
    declaration = config.get("reader_session_declaration")
    if isinstance(declaration, dict) and declaration.get("receipt_run_id"):
        identities.add(("declaration", declaration["receipt_run_id"]))
    if config.get("session_id"):
        identities.add(("source_session", config["session_id"]))
    return identities


def _declared_sources(record):
    config = record.get("configuration", {})
    plan = config.get("plan", config)
    sessions = plan.get("sessions", plan.get("split", {}).get("sessions", []))
    groups = [set(s.get("run_ids", [])) for s in sessions]
    direct = set(config.get("source_run_ids", [])) | set(
        record.get("summary", {}).get("source_run_ids", [])
    )
    declared = set().union(direct, *groups)
    for run_id in declared:
        _run_component(run_id)
    return declared, groups


def assert_no_prior_reader_exposure(
    project_root,
    artifact_root,
    source_run_ids,
    *,
    current_run_id=None,
):
    """Call under reader_publication_lease before consuming evaluation labels.

    Known image processing, fitting or prediction attempts consume the whole source
    session for future blinded qualification, including failures. A retry must use a
    separately governed development analysis, not a new purportedly untouched set.
    """
    from gradientclimb.experiments import verify_run

    root = Path(project_root).absolute()
    _no_links(root)
    store = _inside(root, artifact_root)
    requested = list(source_run_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Exposure check needs distinct immutable source run identities")
    cache = {}

    def identities(run_id):
        if run_id not in cache:
            record, _, _ = _verified(root, artifact_root, _run_component(run_id))
            cache[run_id] = _identities(record)
        return cache[run_id]

    target = set().union(*(identities(run_id) for run_id in requested))
    previous_attempts = []
    runs = _inside(store, "runs")
    for directory in sorted(runs.iterdir()):
        _no_links(directory)
        if not directory.is_dir() or directory.name == current_run_id:
            continue
        record_path = directory / "run.json"
        journal_path = directory / JOURNAL
        _no_links(record_path)
        _no_links(journal_path)
        if not record_path.is_file():
            if journal_path.exists():
                raise KnownReaderExposure(
                    "Orphan reader journal requires recovery",
                    {
                        "run_id": directory.name,
                        "journal": journal_path.relative_to(root).as_posix(),
                    },
                )
            continue
        record = json.loads(record_path.read_bytes())
        if record.get("experiment_id") not in OPERATIONS:
            continue
        declared, groups = _declared_sources(record)
        declared_match = any(target & identities(run_id) for run_id in declared)
        receipt = {
            "run_id": directory.name,
            "run": {
                "path": record_path.relative_to(root).as_posix(),
                "sha256": sha256_file(record_path),
            },
            "status": record.get("status"),
        }
        if not declared and not journal_path.exists():
            raise KnownReaderExposure(
                "Prior reader attempt has unresolved source identity", receipt
            )
        if not (directory / "seal.json").is_file():
            if declared_match or not declared:
                raise KnownReaderExposure(
                    "Unfinished prior reader attempt requires exposure recovery", receipt
                )
            continue
        for _ in _walk(directory):
            pass
        seal = json.loads((directory / "seal.json").read_bytes())
        for name in seal["files"]:
            _inside(directory, name)
        if not verify_run(store, directory.name)["valid"]:
            raise KnownReaderExposure("Prior reader operation seal failed", receipt)
        receipt["seal"] = {
            "path": (directory / "seal.json").relative_to(root).as_posix(),
            "sha256": sha256_file(directory / "seal.json"),
        }
        if not journal_path.is_file():
            if declared_match:
                raise KnownReaderExposure(
                    "Prior reader attempt lacks durable exposure detail", receipt
                )
            continue
        if journal_path.stat().st_size > 32 * 1024**2:
            raise KnownReaderExposure("Prior reader journal exceeds the audit bound", receipt)
        journal = read_complete_journal(journal_path)
        events = journal["rows"]
        if journal["trailing_partial_bytes"]:
            raise KnownReaderExposure(
                "Sealed reader journal has an incomplete exposure record", receipt
            )
        exposed = set()
        for index, event in enumerate(events):
            if (
                event.get("sequence") != index
                or event.get("schema_version") != "reader-operation-event-3.0"
            ):
                raise KnownReaderExposure(
                    "Prior reader journal sequence/schema is invalid", receipt
                )
            if event.get("phase") not in EXPOSURE_PHASES:
                continue
            run_id = event.get("source_run_id")
            if not run_id or run_id not in declared:
                raise KnownReaderExposure(
                    "Prior reader exposure has no declared source identity", receipt
                )
            exposed.add(run_id)
        for group in groups:
            if exposed & group:
                exposed.update(group)
        if any(target & identities(run_id) for run_id in exposed):
            receipt["exposed_source_run_ids"] = sorted(exposed)
            receipt["journal"] = {
                "path": journal_path.relative_to(root).as_posix(),
                "sha256": sha256_file(journal_path),
            }
            raise KnownReaderExposure(
                f"Known reader exposure in {directory.name}; fresh audits or replacement labels cannot restore blinding",
                receipt,
            )
        if declared_match:
            previous_attempts.append(receipt)
    return {
        "scope": "known canonical reader operations in this store; external/unrecorded exposure still requires audit",
        "source_run_ids": requested,
        "prior_related_attempts_without_recorded_image_processing": previous_attempts,
        "prior_exposure_detected": False,
    }
