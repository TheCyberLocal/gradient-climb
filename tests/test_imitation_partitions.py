"""Canonical partition extensions, failed-attempt binding, and OS ownership."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from demo_dataset_fixtures import make_demo_source, write_json

from gradientclimb.datasets.demonstrations import WindowPlan, publish_dataset
from gradientclimb.datasets.partitions import (
    LOCK_NAME,
    _publication_lease,
    partition_publication,
)


def test_sealed_ledgers_allow_reuse_and_additive_extension(tmp_path):
    root, _, _, ledger, freeze, *_ = make_demo_source(tmp_path)
    first = publish_dataset(root, "research/plan.json")
    second = publish_dataset(root, "research/plan.json")
    assert first["dataset_sha256"] == second["dataset_sha256"]
    ledger["sessions"].append(
        {
            "run_id": "reserved-run",
            "session_id": "reserved-session",
            "source_purpose": "qualification",
            "split": "qualification",
        }
    )
    freeze()
    assert publish_dataset(root, "research/plan.json")["verification"]["valid"]


@pytest.mark.parametrize("change", ["reassignment", "session_alias", "run_alias", "omission"])
def test_prior_assignments_cannot_be_reassigned_aliased_or_forgotten(tmp_path, change):
    root, plan, _, ledger, freeze, *_ = make_demo_source(tmp_path)
    ledger["sessions"].append(
        {
            "run_id": "reserved-run",
            "session_id": "reserved-session",
            "source_purpose": "qualification",
            "split": "qualification",
        }
    )
    freeze()
    publish_dataset(root, "research/plan.json")
    before = {p.name for p in (root / "artifacts/runs").iterdir()}
    if change == "reassignment":
        ledger["sessions"][0]["split"] = "development"
        plan["split"] = "development"
    elif change == "session_alias":
        ledger["sessions"][0]["session_id"] = "new-alias"
    elif change == "run_alias":
        ledger["sessions"][0]["run_id"] = "new-alias"
    else:
        ledger["sessions"].pop()
    freeze()
    with pytest.raises(ValueError, match="only extend"):
        publish_dataset(root, "research/plan.json")
    assert {p.name for p in (root / "artifacts/runs").iterdir()} == before


def test_failed_sealed_assembly_keeps_early_partition_binding(tmp_path, monkeypatch):
    import gradientclimb.datasets.demonstrations as module

    root, plan, _, ledger, freeze, *_ = make_demo_source(tmp_path)

    def fail(*_):
        raise RuntimeError("injected assembly failure after ledger registration")

    with monkeypatch.context() as patch:
        patch.setattr(module, "assemble_windows", fail)
        with pytest.raises(RuntimeError, match="injected"):
            publish_dataset(root, "research/plan.json")
    records = [json.loads(p.read_bytes()) for p in (root / "artifacts/runs").glob("*/run.json")]
    failed = next(r for r in records if r["algorithm"] == "offline-window-assembly-3.0")
    assert failed["status"] == "failed"
    assert any(a["kind"] == "whole_session_partition_ledger" for a in failed["artifact_manifest"])
    ledger["sessions"][0]["split"] = plan["split"] = "development"
    freeze()
    with pytest.raises(ValueError, match="only extend"):
        publish_dataset(root, "research/plan.json")


@pytest.mark.parametrize("identity_file", ["run-start.json", "run.json"])
def test_unfinished_publication_fails_closed_after_os_lock_is_gone(tmp_path, identity_file):
    root, plan, *_ = make_demo_source(tmp_path)
    with partition_publication(root, plan) as binding:
        unfinished = root / "artifacts/runs" / str(uuid.uuid4()) / identity_file
        write_json(
            unfinished,
            {
                "status": "running",
                "configuration": binding.configuration(plan),
                "artifact_manifest": [],
            },
        )
    with pytest.raises(ValueError, match="Unfinished partition publication"):
        publish_dataset(root, "research/plan.json")


def test_tampered_historical_ledger_or_seal_cannot_clear_assignment(tmp_path):
    root, plan, *_ = make_demo_source(tmp_path)
    published = publish_dataset(root, "research/plan.json")
    directory = Path(published["directory"])
    record = json.loads((directory / "run.json").read_bytes())
    artifact = next(
        a for a in record["artifact_manifest"] if a["kind"] == "whole_session_partition_ledger"
    )
    (directory / artifact["path"]).write_text("{}", encoding="utf-8")
    with (
        pytest.raises(ValueError, match="seal failed"),
        partition_publication(root, WindowPlan.model_validate(plan)),
    ):
        pass


def test_lease_is_nonblocking_per_store_and_releases_on_exception(tmp_path):
    root = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="primary failure"), _publication_lease(root):
        with (
            pytest.raises(RuntimeError, match="Another partition publisher"),
            _publication_lease(root),
        ):
            pass
        with _publication_lease(tmp_path / "other-store"):
            pass
        raise RuntimeError("primary failure")
    with _publication_lease(root):
        pass
    assert (root / LOCK_NAME).read_bytes() == b"\0"


def child(script, root):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_other_process_cannot_enter_same_publication_transaction(tmp_path):
    root = tmp_path / "artifacts"
    script = """import sys
from pathlib import Path
from gradientclimb.datasets.partitions import _publication_lease
try:
    with _publication_lease(Path(sys.argv[1])):
        print('acquired')
except RuntimeError:
    print('blocked')
"""
    with _publication_lease(root):
        blocked = child(script, root)
        assert blocked.returncode == 0 and blocked.stdout.strip() == "blocked", blocked.stderr
    acquired = child(script, root)
    assert acquired.returncode == 0 and acquired.stdout.strip() == "acquired", acquired.stderr


def test_process_exit_releases_os_lock_without_deleting_metadata(tmp_path):
    root = tmp_path / "artifacts"
    crashed = child(
        """import os,sys
from pathlib import Path
from gradientclimb.datasets.partitions import _publication_lease
with _publication_lease(Path(sys.argv[1])):
    os._exit(7)
""",
        root,
    )
    assert crashed.returncode == 7
    with _publication_lease(root):
        assert (root / LOCK_NAME).is_file()


def test_reserved_lock_path_does_not_overwrite_unexpected_bytes(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / LOCK_NAME).write_bytes(b"unrelated evidence")
    with pytest.raises(ValueError, match="unexpected"), _publication_lease(root):
        pass
    assert (root / LOCK_NAME).read_bytes() == b"unrelated evidence"
