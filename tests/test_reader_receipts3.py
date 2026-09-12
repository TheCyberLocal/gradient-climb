"""Synthetic immutable declarations; no reader fitting, predictions or capture."""

import json
from datetime import UTC, datetime

import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.experiments.reader_receipts3 import (
    SessionReceiptPlan3,
    publish_session_declaration3,
    verify_session_declaration3,
)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def sealed(root, experiment, config, *, status="completed"):
    with RunRecorder(
        root / "artifacts",
        experiment,
        config,
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        run.finalize(status=status, episodes=0, environment_steps=0, training_steps=0)
    return load_run(root / "artifacts", run.run_id)


@pytest.fixture
def prepared(tmp_path):
    protocol = write(
        tmp_path,
        "protocol.json",
        {
            "schema_version": "reader-validation-3.0",
            "status": "registered",
            "registered_at": "2020-01-01T00:00:00+00:00",
            "environment": "synthetic",
        },
    )
    selection = write(
        tmp_path,
        "selection.json",
        {"rule": "first terminal frame for each declared attempt", "attempt_indices": [0, 1]},
    )
    freeze = sealed(tmp_path, "reader-candidate-freeze-3.0", {"protocol": protocol})
    return tmp_path, protocol, selection, freeze


def declare(prepared, **overrides):
    root, protocol, selection, freeze = prepared
    arguments = {
        "project_root": root,
        "artifact_root": "artifacts",
        "protocol_ref_dict": protocol,
        "session_id": "new-session",
        "purpose": "heldout",
        "source_run_config_binding_dict": {"acquisition": {"mode": "synthetic"}},
        "candidate_freeze_run_id": freeze["run_id"],
        "selection_ref": selection,
    }
    return publish_session_declaration3(**{**arguments, **overrides})


def verify(prepared, receipt, source, **overrides):
    root, protocol, *_ = prepared
    arguments = {
        "root": root,
        "artifact_root": "artifacts",
        "receipt_run_id": receipt["run_id"],
        "expected_protocol_ref": protocol,
        "session_id": "new-session",
        "purpose": "heldout",
        "acquired_at": source["start_time"],
        "source_records": [source],
    }
    return verify_session_declaration3(**{**arguments, **overrides})


def test_heldout_receipt_freezes_selection_and_original_capture_binding(prepared):
    root, _, selection, freeze = prepared
    receipt = declare(prepared)
    source = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    result = verify(prepared, receipt, source)
    assert result["candidate_freeze_run_id"] == freeze["run_id"]
    assert result["selection"] == selection
    assert result["declared_at"] == receipt["seal_created_at"]
    assert result["source_run_ids"] == [source["run_id"]]
    assert not result["qualification_evidence"]
    record = load_run(root / "artifacts", receipt["run_id"])
    assert record["episodes"] == record["environment_steps"] == record["training_steps"] == 0
    assert verify_run(root / "artifacts", receipt["run_id"])["valid"]


def test_backdating_declaration_cannot_use_run_end_as_seal_time(prepared):
    receipt = declare(prepared)
    root = prepared[0]
    record = load_run(root / "artifacts", receipt["run_id"])
    assert datetime.fromisoformat(record["end_time"]) < datetime.fromisoformat(
        receipt["declared_at"]
    )
    source = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    with pytest.raises(ValueError, match="Actual declaration seal"):
        verify(prepared, receipt, source, acquired_at=record["end_time"])


@pytest.mark.parametrize(
    "mutation", ["missing", "other_receipt", "session_alias", "selection", "config"]
)
def test_source_must_bind_immutable_declaration_in_original_configuration(prepared, mutation):
    receipt = declare(prepared)
    config = json.loads(json.dumps(receipt["required_source_configuration"]))
    if mutation == "missing":
        del config["reader_session_declaration"]
    elif mutation == "other_receipt":
        config["reader_session_declaration"]["receipt_run_id"] = "another-receipt"
    elif mutation == "session_alias":
        config["session_id"] = "alias"
    elif mutation == "selection":
        config["reader_session_declaration"]["selection_sha256"] = "0" * 64
    else:
        config["acquisition"]["mode"] = "different"
    source = sealed(prepared[0], "synthetic-acquisition", config)
    with pytest.raises(ValueError, match="Original source configuration"):
        verify(prepared, receipt, source)


def test_existing_source_cannot_be_retroactively_declared_heldout(prepared):
    source = sealed(prepared[0], "existing-source", {})
    receipt = declare(prepared)
    with pytest.raises(ValueError, match="Actual declaration seal"):
        verify(prepared, receipt, source)


def test_construction_declaration_postdates_exact_existing_source_set(prepared):
    root = prepared[0]
    source = sealed(root, "existing-construction-source", {"mode": "construction"})
    receipt = declare(
        prepared,
        purpose="construction",
        candidate_freeze_run_id=None,
        source_run_ids=[source["run_id"]],
        source_run_config_binding_dict={"mode": "construction"},
    )
    result = verify(prepared, receipt, source, purpose="construction")
    assert result["purpose"] == "construction"
    unrelated = sealed(root, "another-source", {"mode": "construction"})
    with pytest.raises(ValueError, match="exact source run set"):
        verify(prepared, receipt, unrelated, purpose="construction")


def test_in_memory_record_cannot_override_actual_source_configuration(prepared):
    receipt = declare(prepared)
    source = sealed(prepared[0], "synthetic-acquisition", {})
    source["configuration"] = receipt["required_source_configuration"]
    with pytest.raises(ValueError, match="canonical sealed bytes"):
        verify(prepared, receipt, source)


def test_changed_external_selection_does_not_rewrite_sealed_declaration(prepared):
    root, _, selection, _ = prepared
    receipt = declare(prepared)
    (root / selection["path"]).write_text('{"rule":"changed after acquisition"}', encoding="utf-8")
    source = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    result = verify(prepared, receipt, source)
    assert result["selection"]["sha256"] == selection["sha256"]
    assert result["selection_document"]["attempt_indices"] == [0, 1]
    assert sha256_file(root / result["selection_file"]) == selection["sha256"]
    assert verify_run(root / "artifacts", receipt["run_id"])["valid"]


def test_modified_private_receipt_payload_fails_seal(prepared):
    receipt = declare(prepared)
    root = prepared[0]
    source = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    directory = root / "artifacts/runs" / receipt["run_id"]
    record = json.loads((directory / "run.json").read_bytes())
    artifact = next(
        a for a in record["artifact_manifest"] if a["kind"] == "reader_session_declaration"
    )
    (directory / artifact["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="seal failed"):
        verify(prepared, receipt, source)


def test_failed_or_wrong_protocol_candidate_freeze_is_not_accepted(prepared):
    root, protocol, *_ = prepared
    failed = sealed(root, "reader-candidate-freeze-3.0", {"protocol": protocol}, status="failed")
    with pytest.raises(ValueError, match="Candidate freeze"):
        declare(prepared, candidate_freeze_run_id=failed["run_id"])
    wrong = sealed(
        root,
        "reader-candidate-freeze-3.0",
        {"protocol": {"path": "other.json", "sha256": "0" * 64}},
    )
    with pytest.raises(ValueError, match="Candidate freeze"):
        declare(prepared, candidate_freeze_run_id=wrong["run_id"])


@pytest.mark.parametrize("run_id", ["../other", "nested/run", "C:/run", "folder\\run"])
def test_receipt_run_identity_rejects_escape(run_id):
    with pytest.raises(ValueError):
        SessionReceiptPlan3(
            protocol={"path": "protocol.json", "sha256": "0" * 64},
            session_id="session",
            purpose="heldout",
            selection={"path": "selection.json", "sha256": "1" * 64},
            candidate_freeze_run_id=run_id,
            source_run_config_binding={},
            note="Synthetic path check",
        )


def test_publisher_refuses_changed_selection_and_identity_override(prepared):
    root, _, selection, _ = prepared
    with pytest.raises(ValueError, match="override"):
        declare(prepared, source_run_config_binding_dict={"session_id": "alias"})
    (root / selection["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        declare(prepared)


def test_naive_acquisition_time_is_rejected(prepared):
    receipt = declare(prepared)
    source = sealed(prepared[0], "synthetic-acquisition", receipt["required_source_configuration"])
    with pytest.raises(ValueError):
        verify(prepared, receipt, source, acquired_at=datetime.now(UTC).replace(tzinfo=None))


def test_selection_changed_during_copy_seals_failed_attempt(prepared, monkeypatch):
    root, _, selection, _ = prepared
    original = RunRecorder.register_artifact

    def change(run, path, kind, *args, **kwargs):
        if path == root / selection["path"]:
            path.write_text('{"rule":"changed during publication"}', encoding="utf-8")
        return original(run, path, kind, *args, **kwargs)

    monkeypatch.setattr(RunRecorder, "register_artifact", change)
    with pytest.raises(ValueError, match="changed during publication"):
        declare(prepared)
    records = [json.loads(p.read_bytes()) for p in (root / "artifacts/runs").glob("*/run.json")]
    receipt = next(r for r in records if r["experiment_id"] == "reader-session-declaration-3.0")
    assert receipt["status"] == "failed"
    assert verify_run(root / "artifacts", receipt["run_id"])["valid"]


def test_heldout_original_source_cannot_be_redeclared_for_construction(prepared):
    receipt = declare(prepared)
    source = sealed(prepared[0], "synthetic-acquisition", receipt["required_source_configuration"])
    with pytest.raises(ValueError, match="cannot be repurposed"):
        declare(
            prepared,
            purpose="construction",
            candidate_freeze_run_id=None,
            source_run_ids=[source["run_id"]],
            source_run_config_binding_dict={},
        )


def test_heldout_session_cannot_omit_an_entire_bound_acquisition_run(prepared):
    receipt = declare(prepared)
    root = prepared[0]
    first = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    second = sealed(
        root, "synthetic-acquisition", receipt["required_source_configuration"], status="failed"
    )
    with pytest.raises(ValueError, match="every canonical source run"):
        verify(prepared, receipt, first)
    result = verify(prepared, receipt, first, source_records=[first, second])
    assert result["source_run_ids"] == [first["run_id"], second["run_id"]]


def test_unfinished_acquisition_bound_to_receipt_cannot_disappear(prepared):
    receipt = declare(prepared)
    root = prepared[0]
    first = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    write(
        root,
        "artifacts/runs/pending-source/run-start.json",
        {
            "run_id": "pending-source",
            "status": "running",
            "configuration": receipt["required_source_configuration"],
        },
    )
    with pytest.raises(ValueError, match="Unsealed acquisition"):
        verify(prepared, receipt, first)


def test_nested_later_reader_plan_does_not_create_false_acquisition_membership(prepared):
    receipt = declare(prepared)
    root = prepared[0]
    first = sealed(root, "synthetic-acquisition", receipt["required_source_configuration"])
    sealed(
        root,
        "reader-validation-3.0",
        {"plan": {"copied_source_config": receipt["required_source_configuration"]}},
    )
    assert verify(prepared, receipt, first)["source_run_ids"] == [first["run_id"]]
