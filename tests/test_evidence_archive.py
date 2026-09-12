"""Real bytes, immutable failed runs, and hostile archive/path boundaries."""

import json
import stat
import zipfile

import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import (
    MANIFEST_NAME,
    build_inventory,
    create_archive,
    restore_archive,
    verify_archive,
)
from gradientclimb.experiments import RunRecorder, verify_run


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "source"
    (root / "configs/perception").mkdir(parents=True)
    (root / "configs/perception/profile.json").write_text('{"private":true}')
    (root / "artifacts/demos").mkdir(parents=True)
    (root / "artifacts/demos/demo.mp4").write_bytes(b"private recording fixture")
    (root / "artifacts/test-tmp").mkdir()
    (root / "artifacts/test-tmp/regenerable.bin").write_bytes(b"cache")
    (root / "artifacts/cycle3-native-source").mkdir()
    (root / "artifacts/cycle3-native-source/source.py").write_text("isolated source")
    (root / "research/experiments").mkdir(parents=True)
    with (
        pytest.raises(RuntimeError),
        RunRecorder(
            root / "artifacts",
            "negative-fixture",
            {},
            telemetry_interval_seconds=0,
            source_root=root,
        ) as run,
    ):
        run.metric("observed_work", 4)
        raise RuntimeError("preserve this failed run")
    manifest = tmp_path / "frozen.json"
    inventory = build_inventory(root, manifest)
    return root, manifest, inventory, run.run_id


def test_archive_restore_retains_all_bytes_and_failed_run_seal(tmp_path, evidence):
    root, manifest, inventory, run_id = evidence
    assert "artifacts/test-tmp" in inventory["excluded_regenerable_directories"]
    assert "artifacts/cycle3-native-source" in inventory["unselected_artifact_directories"]
    assert not any("source.py" in row["path"] for row in inventory["files"])
    seal = root / "artifacts/runs" / run_id / "seal.json"
    before = seal.read_bytes()
    path = tmp_path / "private-evidence.zip"
    archived = create_archive(root, manifest, path)
    assert archived["backup_status"] == "archive_verified; restore_not_yet_tested"
    receipt = restore_archive(path, manifest, tmp_path / "restored")
    assert receipt["backup_status"] == "archive_and_restore_verified"
    assert verify_run(tmp_path / "restored/artifacts", run_id)["valid"]
    assert seal.read_bytes() == before
    assert (tmp_path / "restored/artifacts/runs" / run_id / "seal.json").read_bytes() == before
    for row in inventory["files"]:
        assert sha256_file(tmp_path / "restored" / row["path"]) == row["sha256"]
    assert not (tmp_path / "restored/.restore-incomplete.json").exists()


def test_all_publication_destinations_refuse_overwrite(tmp_path, evidence):
    root, manifest, _, _ = evidence
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        build_inventory(root, manifest)
    assert manifest.read_bytes() == before
    path = tmp_path / "archive.zip"
    create_archive(root, manifest, path)
    digest = sha256_file(path)
    with pytest.raises(FileExistsError):
        create_archive(root, manifest, path)
    assert sha256_file(path) == digest
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        restore_archive(path, manifest, destination)
    assert list(destination.iterdir()) == []


def test_source_mutation_after_inventory_prevents_archive_publication(tmp_path, evidence):
    root, manifest, _, _ = evidence
    (root / "artifacts/demos/demo.mp4").write_bytes(b"changed after cutoff")
    with pytest.raises(ValueError, match="mismatch"):
        create_archive(root, manifest, tmp_path / "must-not-exist.zip")
    assert not (tmp_path / "must-not-exist.zip").exists()


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "C:stream", "a\\..\\x", "a/../x", "a//b", "CON", "x."]
)
def test_hostile_inventory_path_rejected_before_restore(tmp_path, evidence, name):
    _, manifest, inventory, _ = evidence
    inventory["files"][0]["path"] = name
    malicious = tmp_path / "malicious.json"
    malicious.write_text(json.dumps(inventory))
    with pytest.raises(ValueError):
        restore_archive(tmp_path / "unused.zip", malicious, tmp_path / "outside")
    assert not (tmp_path / "outside").exists()
    assert manifest.exists()


@pytest.mark.parametrize(
    "kind", ["extra", "symlink", "duplicate", "wrong_manifest", "tampered_bytes"]
)
def test_hostile_zip_members_rejected_before_destination_created(tmp_path, evidence, kind):
    root, manifest, _, _ = evidence
    original = tmp_path / "original.zip"
    create_archive(root, manifest, original)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(bad, "w") as target:
        for index, info in enumerate(source.infolist()):
            payload = source.read(info)
            if kind == "symlink" and index == 1:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            if kind == "wrong_manifest" and info.filename == MANIFEST_NAME:
                payload = b"{}"
            if kind == "tampered_bytes" and index == 1:
                payload = b"corrupt"
            target.writestr(info, payload)
        if kind == "extra":
            target.writestr("../escape.txt", b"bad")
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                target.writestr(MANIFEST_NAME, manifest.read_bytes())
    with pytest.raises(ValueError):
        restore_archive(bad, manifest, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_symlinked_source_and_restore_parent_are_rejected(tmp_path, evidence):
    root, manifest, _, _ = evidence
    link = tmp_path / "linked-parent"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="symlinks"):
        create_archive(link, manifest, tmp_path / "archive.zip")
    with pytest.raises(ValueError, match="symlinks"):
        restore_archive(tmp_path / "unused.zip", manifest, link / "restored")


def test_unfinished_run_blocks_inventory(tmp_path):
    root = tmp_path / "source"
    (root / "configs/perception").mkdir(parents=True)
    run = RunRecorder(root / "artifacts", "unfinished", {}, telemetry_interval_seconds=0)
    try:
        with pytest.raises(ValueError, match="unfinished"):
            build_inventory(root, tmp_path / "frozen.json")
    finally:
        run._abandon()
    assert not (tmp_path / "frozen.json").exists()


def test_windows_reparse_point_flag_is_rejected(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from gradientclimb.artifacts.archive import _no_links

    target = tmp_path / "junction"
    target.mkdir()
    original = Path.lstat

    def reparse_details(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if self == target:
            return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=0x400)
        return value

    monkeypatch.setattr(Path, "lstat", reparse_details)
    with pytest.raises(ValueError, match="junctions"):
        _no_links(target / "inside.bin")


def test_restore_disk_failure_remains_marked_incomplete(tmp_path, evidence, monkeypatch):
    from gradientclimb.artifacts import archive

    root, manifest, _, _ = evidence
    path = tmp_path / "archive.zip"
    create_archive(root, manifest, path)
    original = archive._copy_and_hash

    def disk_full(reader, writer, expected):
        if writer is not None:
            raise OSError("disk full while restoring")
        return original(reader, writer, expected)

    monkeypatch.setattr(archive, "_copy_and_hash", disk_full)
    destination = tmp_path / "partial-restore"
    with pytest.raises(OSError, match="disk full"):
        restore_archive(path, manifest, destination)
    assert (destination / ".restore-incomplete.json").exists()
    assert not (destination / "restore-receipt.json").exists()
    assert verify_archive(path, manifest)["valid"]
