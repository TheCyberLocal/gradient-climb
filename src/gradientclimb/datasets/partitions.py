"""Local canonical-store authority for whole-session data partitions.

Supported publishers serialize compatibility checking through final sealing.
The OS-held byte/file lock is unrelated to native input ownership, is nonblocking,
and is released when the handle or process closes. Its one-byte file is operational
metadata; sealed ledger artifacts, including failed attempts, remain the authority.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gradientclimb.artifacts.archive import _inside, _no_links, _relative, _walk

from .demonstrations import EvidenceRef, PartitionLedger, WindowPlan, _load_ref, _read_ref

LOCK_NAME = ".imitation-partitions.lock"
BINDING_VERSION = "whole-session-partition-publication-3.0"
_active_roots: set[str] = set()
_active_guard = threading.Lock()


@contextmanager
def _publication_lease(artifact_root: Path):
    """Exclude threads and processes for this store without waiting or stale PIDs."""
    _no_links(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    identity = os.path.normcase(str(artifact_root.resolve()))
    with _active_guard:
        if identity in _active_roots:
            raise RuntimeError("Another partition publisher owns this artifact store")
        _active_roots.add(identity)
    stream = None
    try:
        path = artifact_root / LOCK_NAME
        _no_links(path)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise OSError("Partition publication locking is unsupported on this OS")
        except OSError as error:
            raise RuntimeError(
                "Cannot acquire exclusive partition publication ownership"
            ) from error
        stream.seek(0)
        if stream.read(2) != b"\0":
            raise ValueError("Partition lock path contains unexpected non-operational data")
        yield
    finally:
        try:
            if stream is not None:
                stream.close()  # Closing the OS handle releases the lock, including on failure.
        finally:
            with _active_guard:
                _active_roots.discard(identity)


def _historical_assignments(root: Path, artifact_root: Path):
    """Read canonical records only; verify payloads only for partition publications."""
    from gradientclimb.experiments import verify_run

    runs = artifact_root / "runs"
    _no_links(runs)
    if not runs.exists():
        return {}
    by_session, by_run = {}, {}
    for directory in sorted(runs.iterdir()):
        _no_links(directory)
        if not directory.is_dir():
            continue
        record_path = directory / "run.json"
        _no_links(record_path)
        if not record_path.is_file():
            record_path = directory / "run-start.json"
            _no_links(record_path)
        if not record_path.is_file():
            continue  # Unrelated recorder may be between directory creation and its first record.
        record = json.loads(record_path.read_bytes())
        config = record.get("configuration", {})
        marker = config.get("partition_publication")
        ledgers = [
            artifact
            for artifact in record.get("artifact_manifest", [])
            if artifact.get("kind") == "whole_session_partition_ledger"
        ]
        legacy_publisher = (
            config.get("schema_version") == "imitation-window-plan-3.0"
            or config.get("protocol_version") == "reviewed-geometry-diagnostic-3.0"
        )
        if not (marker or ledgers or legacy_publisher):
            continue
        _no_links(directory / "seal.json")
        if record.get("status") == "running" or not (directory / "seal.json").is_file():
            raise ValueError(f"Unfinished partition publication needs recovery: {directory.name}")
        for _ in _walk(directory):
            pass  # Reject links before canonical integrity verification follows the run.
        integrity = verify_run(artifact_root, directory.name)
        if not integrity["valid"]:
            raise ValueError(f"Prior partition publication seal failed: {directory.name}")
        if not ledgers:
            raise ValueError(
                f"Prior partition publication lacks its frozen ledger: {directory.name}"
            )
        if marker and (
            marker.get("schema_version") != BINDING_VERSION
            or not any(a["sha256"] == marker.get("ledger", {}).get("sha256") for a in ledgers)
        ):
            raise ValueError("Prior partition publication marker does not match its frozen ledger")
        for artifact in ledgers:
            reference = EvidenceRef(
                path=(directory.relative_to(root) / _relative(artifact["path"])).as_posix(),
                sha256=artifact["sha256"],
            )
            ledger = _load_ref(root, reference, PartitionLedger)
            for assignment in ledger.sessions:
                value = assignment.model_dump()
                previous = by_session.get(assignment.session_id) or by_run.get(assignment.run_id)
                if previous is not None and previous != value:
                    raise ValueError(
                        "Historical partition authorities conflict; explicit audit required"
                    )
                by_session[assignment.session_id] = value
                by_run[assignment.run_id] = value
    return by_session


@dataclass(frozen=True)
class PartitionPublication:
    project_root: Path
    artifact_root: Path
    reference: EvidenceRef
    ledger: PartitionLedger

    def configuration(self, config: dict) -> dict:
        return {
            **config,
            "partition_publication": {
                "schema_version": BINDING_VERSION,
                "ledger": self.reference.model_dump(),
                "scope": "this_canonical_artifact_store; additive_whole_session_assignments_only",
            },
        }

    def register(self, run) -> dict:
        """Freeze the accepted ledger before any source assembly or consumption."""
        if run.directory.parent.parent.resolve() != self.artifact_root.resolve():
            raise ValueError("Partition authority and recorder must use the same artifact store")
        artifact = run.register_artifact(
            _read_ref(self.project_root, self.reference),
            "whole_session_partition_ledger",
            {"ledger_id": self.ledger.ledger_id, "authority": BINDING_VERSION},
        )
        if artifact["sha256"] != self.reference.sha256:
            raise ValueError("Partition ledger changed during registration")
        return artifact


@contextmanager
def partition_publication(project_root: str | Path, plan: WindowPlan | dict):
    """Hold across RunRecorder creation, early register(), work, and final sealing.

    Never use this context for native input ownership. All future fitting/training
    publishers consuming these partitions must use the same context and marker.
    Pure assemble_windows() validates its supplied ledger without global claims.
    """
    root = Path(project_root).absolute()
    _no_links(root)
    plan = WindowPlan.model_validate(plan)
    artifact_root = _inside(root, plan.artifact_root)
    with _publication_lease(artifact_root):
        ledger = _load_ref(root, plan.partition_ledger, PartitionLedger)
        historical = _historical_assignments(root, artifact_root)
        current = {row.session_id: row.model_dump() for row in ledger.sessions}
        for session_id, assignment in historical.items():
            if current.get(session_id) != assignment:
                raise ValueError(
                    "Partition ledgers may only extend prior assignments; reassignment, aliases and omission are forbidden"
                )
        yield PartitionPublication(root, artifact_root, plan.partition_ledger, ledger)
