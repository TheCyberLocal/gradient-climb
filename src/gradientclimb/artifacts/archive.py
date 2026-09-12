"""Frozen private evidence inventories and verified, non-overwriting archive restoration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .integrity import canonical_json, sha256_file

MANIFEST_NAME = "evidence-manifest.json"
MANIFEST_VERSION = "evidence-archive-1"
CACHE_DIRECTORY = re.compile(
    r"(?:.*-test-tmp|test-.*|pytest-.*|scoring-tests-.*|stabilization-.*|"
    r"cycle3-(?:tests-.*|test-baseline.*|native-tests.*|full-foundation.*)|matplotlib-cache)"
)
EVIDENCE_DIRECTORIES = {
    "runs",
    "game-discovery",
    "post-benchmark",
    "diagnostics",
    "demos",
    "probe-analysis",
    "hud-independent-labels",
    "distillation",
    "drafts",
    "wip-screen-dynamics",
    "screen-dynamics",
    "headed-overhead",
}


def _relative(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name == "."
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 or character in '<>"|?*' for character in name)
    ):
        raise ValueError(f"Unsafe evidence path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name:
        raise ValueError(f"Non-canonical evidence path: {name!r}")
    for part in path.parts:
        stem = part.split(".")[0].upper()
        if (
            part in {".", ".."}
            or part.endswith((" ", "."))
            or stem
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
        ):
            raise ValueError(f"Unsafe evidence path component: {part!r}")
    return name


def _no_links(path: Path) -> None:
    """Reject symlinks and Windows junction/reparse points, including ancestors."""
    for item in (*reversed(path.parents), path):
        try:
            details = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or (
            getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Evidence paths must not traverse symlinks/junctions: {item}")


def _inside(root: Path, name: str) -> Path:
    candidate = root / _relative(name)
    _no_links(candidate)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes its root: {name}")
    return candidate


def _walk(path: Path):
    _no_links(path)
    if path.is_file():
        yield path
    elif path.is_dir():
        for child in sorted(path.iterdir()):
            yield from _walk(child)
    else:
        raise ValueError(f"Evidence is missing or is not a regular file/directory: {path}")


def _publish_new_bytes(path: Path, payload: bytes) -> None:
    _no_links(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication without overwriting a competing writer.
    finally:
        temporary.unlink(missing_ok=True)


def build_inventory(project_root: str | Path, output: str | Path, *, include=()) -> dict:
    """Freeze all canonical runs and non-cache auxiliary evidence; make no backup claim."""
    from gradientclimb.experiments import list_runs, verify_run

    root = Path(project_root).absolute()
    _no_links(root)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    artifacts = root / "artifacts"
    run_directory = artifacts / "runs"
    _no_links(run_directory)
    if not run_directory.is_dir():
        raise ValueError("The canonical artifacts/runs directory is unavailable")
    for _ in _walk(run_directory):
        pass  # Reject link traversal before reading any run record or seal.
    runs = list_runs(artifacts)
    recognized = {run["run_id"] for run in runs}
    for child in run_directory.iterdir():
        if not child.is_dir() or child.name not in recognized:
            raise ValueError(f"Uncatalogued canonical run-store entry: {child}")
    run_checks = []
    for run in runs:
        check = verify_run(artifacts, run["run_id"])
        if not check["valid"] or run["status"] == "running":
            raise ValueError(f"Cannot freeze an unfinished/invalid canonical run: {check}")
        run_checks.append(
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "seal_sha256": sha256_file(run_directory / run["run_id"] / "seal.json"),
            }
        )
    selections = []
    excluded = []
    unselected = []
    for child in sorted(artifacts.iterdir()):
        if child.name in EVIDENCE_DIRECTORIES:
            selections.append(child)
        elif CACHE_DIRECTORY.fullmatch(child.name):
            excluded.append(child.relative_to(root).as_posix())
        elif child.is_file():
            selections.append(child)
        else:
            unselected.append(child.relative_to(root).as_posix())
    selections.extend(_inside(root, name) for name in include)
    for name in ("configs", "research", "docs", "experiments/definitions"):
        selected = _inside(root, name)
        if selected.exists():
            selections.append(selected)
    if not (root / "configs/perception").is_dir():
        raise ValueError("Required configs/perception directory is unavailable")
    # Historical local inventories identify auxiliary evidence that must not disappear
    # behind a broad directory exclusion or a newly missing research folder.
    historical_differences = []
    for source in (root / "research/experiments").glob("*-local-evidence.json"):
        historical = json.loads(source.read_text(encoding="utf-8"))
        for folder in historical.get("folders", []):
            for item in folder.get("file_manifest", []):
                selected = _inside(root, item["path"])
                if not selected.is_file():
                    raise ValueError(f"Historically inventoried evidence is missing: {selected}")
                actual = sha256_file(selected)
                if item.get("sha256") and actual != item["sha256"]:
                    historical_differences.append(
                        {
                            "path": item["path"],
                            "historical_sha256": item["sha256"],
                            "current_sha256": actual,
                            "historical_inventory": source.relative_to(root).as_posix(),
                        }
                    )
                selections.append(selected)
        for item in historical.get("native_ui_references", []):
            selections.append(_inside(root, item["path"]))
    paths = {}
    for selected in selections:
        for path in _walk(selected):
            if path.absolute() == output:
                raise ValueError("An inventory cannot include its own output")
            name = _relative(path.relative_to(root).as_posix())
            alias = name.casefold()
            if alias in paths and paths[alias] != path:
                raise ValueError(f"Case-insensitive evidence path collision: {name}")
            paths[alias] = path
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths.values())
    ]
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(root),
        "source_git_sha": source.stdout.strip() if source.returncode == 0 else None,
        "scope": "Private canonical runs, non-cache auxiliary artifacts, configs, research, docs and protocols",
        "backup_status": "inventory_only; archive_and_verified_restore_required",
        "excluded_regenerable_directories": excluded,
        "canonical_runs": run_checks,
        "unselected_artifact_directories": unselected,
        "explicit_additional_selections": list(include),
        "historical_auxiliary_hash_differences": historical_differences,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    _publish_new_bytes(output, (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def _load_inventory(path: str | Path) -> tuple[dict, bytes]:
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported evidence inventory version")
    seen = set()
    for row in manifest["files"]:
        name = _relative(row["path"])
        if name.casefold() in seen:
            raise ValueError(f"Duplicate inventory path: {name}")
        seen.add(name.casefold())
        if (
            type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError(f"Invalid inventory size/hash: {name}")
    if manifest["file_count"] != len(seen) or manifest["total_bytes"] != sum(
        row["bytes"] for row in manifest["files"]
    ):
        raise ValueError("Inventory count/byte totals do not reconcile")
    return manifest, payload


def _copy_and_hash(reader, writer, expected: dict) -> None:
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: reader.read(1024 * 1024), b""):
        size += len(block)
        if size > expected["bytes"]:
            raise ValueError(f"Evidence exceeds inventoried bytes: {expected['path']}")
        digest.update(block)
        if writer is not None:
            writer.write(block)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Evidence hash/size mismatch: {expected['path']}")


def create_archive(project_root: str | Path, inventory: str | Path, output: str | Path) -> dict:
    """Copy exactly a frozen inventory to a new private ZIP; validate before publication."""
    root, output = Path(project_root).absolute(), Path(output).absolute()
    _no_links(root)
    _no_links(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, payload = _load_inventory(inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                archive.writestr(MANIFEST_NAME, payload)
                for row in manifest["files"]:
                    path = _inside(root, row["path"])
                    if not path.is_file():
                        raise ValueError(f"Inventoried evidence is missing: {path}")
                    with (
                        path.open("rb") as reader,
                        archive.open("files/" + row["path"], "w", force_zip64=True) as writer,
                    ):
                        _copy_and_hash(reader, writer, row)
            stream.flush()
            os.fsync(stream.fileno())
        verify_archive(temporary, inventory)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "backup_status": "archive_verified; restore_not_yet_tested",
    }


def verify_archive(path: str | Path, inventory: str | Path) -> dict:
    manifest, payload = _load_inventory(inventory)
    expected = {"files/" + row["path"]: row for row in manifest["files"]}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Duplicate/colliding archive members")
        if set(names) != {*expected, MANIFEST_NAME}:
            raise ValueError("Archive members do not exactly match the frozen inventory")
        for entry in entries:
            _relative(entry.filename)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in {0, stat.S_IFREG}:
                raise ValueError(f"Non-regular archive member: {entry.filename}")
            if entry.filename == MANIFEST_NAME:
                if entry.file_size != len(payload) or archive.read(entry) != payload:
                    raise ValueError("Archive manifest differs from the external frozen inventory")
            else:
                row = expected[entry.filename]
                if entry.file_size != row["bytes"]:
                    raise ValueError(f"Archive member size differs: {entry.filename}")
                with archive.open(entry) as reader:
                    _copy_and_hash(reader, None, row)
    return {
        "valid": True,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def restore_archive(path: str | Path, inventory: str | Path, destination: str | Path) -> dict:
    """Restore into a new directory only; failed restores remain visibly incomplete."""
    from gradientclimb.experiments import verify_run

    destination = Path(destination).absolute()
    _no_links(destination)
    if destination.exists():
        raise FileExistsError("Restore destination must not already exist")
    verified = verify_archive(path, inventory)
    manifest, payload = _load_inventory(inventory)
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".restore-incomplete.json"
    _publish_new_bytes(marker, canonical_json({"archive": str(Path(path).absolute())}).encode())
    with zipfile.ZipFile(path) as archive:
        for row in manifest["files"]:
            target = _inside(destination, row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("files/" + row["path"]) as reader, target.open("xb") as writer:
                _copy_and_hash(reader, writer, row)
                writer.flush()
                os.fsync(writer.fileno())
    run_checks = []
    for run in manifest.get("canonical_runs", []):
        checked = verify_run(destination / "artifacts", run["run_id"])
        if not checked["valid"]:
            raise ValueError(f"Restored canonical seal failed: {checked}")
        run_checks.append(checked)
    receipt = {
        **verified,
        "restored_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(path),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "destination": str(destination),
        "canonical_runs": run_checks,
        "backup_status": "archive_and_restore_verified",
        "durability_limit": "Verification does not establish independent hardware or offsite durability",
    }
    _publish_new_bytes(
        destination / "restore-receipt.json", (canonical_json(receipt) + "\n").encode()
    )
    marker.unlink()
    return receipt
