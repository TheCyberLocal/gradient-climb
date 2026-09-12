"""Opaque synthetic bytes exercise publication; no pixel truth or reader execution."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.experiments import reader_annotations3 as annotations
from gradientclimb.experiments.reader_exposure3 import KnownReaderExposure
from gradientclimb.experiments.reader_receipts3 import publish_session_declaration3
from gradientclimb.experiments.reader_validation3 import (
    EvidenceRef,
    ReaderProtocol3,
    ResultAnnotation,
    _annotations,
)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def source(root, config, *, same_attempt=False, negative=False):
    frames, timings = [], []
    with RunRecorder(
        root / "artifacts",
        "opaque-source",
        config,
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        for index in range(2):
            path = run.directory / f"frame-{index}.png"
            path.write_bytes(f"Not decodable image bytes {index}".encode())
            timing = {
                "started_ns": 100 * index + 1,
                "timestamp_ns": 100 * index + 2,
                "completed_ns": 100 * index + 3,
            }
            if negative and index == 1:
                timing.update(
                    reader_sampling_unit_id="negative-1",
                    reader_sampling_stage="pause",
                    reader_sampling_ordinal=1,
                )
            a = run.register_artifact(
                path,
                "reader_non_result_frame" if negative and index == 1 else "terminal_frame",
                timing,
            )
            frames.append(
                {
                    "path": (run.directory / a["path"]).relative_to(root).as_posix(),
                    "sha256": a["sha256"],
                }
            )
            timings.append(timing)
        episodes = [
            {"attempt": {"index": 0}, "terminal_readings": timings if same_attempt else timings[:1]}
        ]
        if not same_attempt and not negative:
            episodes.append({"attempt": {"index": 1}, "terminal_readings": timings[1:]})
        episodes.append({"attempt": {"index": 2}, "terminal_readings": []})
        run.finalize(status="failed", episodes=0, episode_summaries=episodes)
    return load_run(root / "artifacts", run.run_id), frames


def prepared(root, *, purpose="construction", same_attempt=False, negative=False):
    protocol = ReaderProtocol3(
        protocol_id="synthetic-annotation-test",
        status="registered",
        registered_at=datetime(2020, 1, 1, tzinfo=UTC),
        environment="synthetic",
        base_result_reader_sha256="a" * 64,
        construction_session_ids=("construction-session",),
        qualification_note="Synthetic contract only",
    )
    protocol_ref = write(root, "protocol.json", protocol.model_dump(mode="json"))
    selection_ref = write(
        root,
        "selection.json",
        {
            "schema_version": "reader-source-selection-3.0",
            "non_result_units": [
                {
                    "source_config_id": "source",
                    "unit_id": "negative-1",
                    "stage": "pause",
                    "capture_ordinal": 1,
                }
            ]
            if negative
            else [],
        },
    )
    sid = purpose + "-session"
    binding = {"reader_sampling_source_id": "source"}
    args = {
        "project_root": root,
        "artifact_root": "artifacts",
        "protocol_ref_dict": protocol_ref,
        "session_id": sid,
        "purpose": purpose,
        "source_run_config_binding_dict": binding,
        "selection_ref": selection_ref,
    }
    if purpose == "heldout":
        with RunRecorder(
            root / "artifacts",
            "reader-candidate-freeze-3.0",
            {"protocol": protocol_ref},
            source_root=root,
            telemetry_interval_seconds=0,
        ) as freeze:
            freeze.finalize(episodes=0)
        receipt = publish_session_declaration3(**args, candidate_freeze_run_id=freeze.run_id)
        record, frames = source(
            root,
            receipt["required_source_configuration"],
            same_attempt=same_attempt,
            negative=negative,
        )
    else:
        record, frames = source(root, binding, same_attempt=same_attempt, negative=negative)
        receipt = publish_session_declaration3(**args, source_run_ids=[record["run_id"]])
    labels = [
        {
            "label_id": f"label-{i}",
            "session_id": sid,
            "run_id": record["run_id"],
            "sampling_unit_id": f"untrusted-name-{i}",
            "frame": frame,
            "source_hashes": [frame["sha256"]],
            "is_result": not (negative and i == 1),
            "text": None if negative and i == 1 else "269",
            "labeled_at": datetime.now(UTC).isoformat(),
            "reviewer": "synthetic reviewer",
            "note": "Explicit fixture label; no pixels inspected",
        }
        for i, frame in enumerate(frames)
    ]
    audit = {
        "session_id": sid,
        "prediction_exposure_status": "none_reported" if purpose == "heldout" else "known",
        "first_prediction_exposure_at": None,
        "exposure_reviewed_through": datetime.now(UTC).isoformat(),
        "reviewer": "synthetic reviewer",
        "note": "Fixture attestation; known first-view time unresolved",
    }
    audit_ref = write(root, "audit.json", audit)
    plan = {
        "protocol": protocol_ref,
        "sessions": [
            {
                "session_id": sid,
                "purpose": purpose,
                "run_ids": [record["run_id"]],
                "declaration_run_id": receipt["run_id"],
                "exposure_audit": audit_ref,
            }
        ],
        "annotations": labels,
        "note": "Explicit offline synthetic publication",
    }
    return plan, record, frames, receipt, audit


def publication_runs(root):
    return [
        json.loads(p.read_text())
        for p in (root / "artifacts/runs").glob("*/run.json")
        if json.loads(p.read_text())["experiment_id"] == annotations.EXPERIMENT
    ]


def change_audit(root, plan, audit, **fields):
    audit = {**audit, **fields}
    plan["sessions"][0]["exposure_audit"] = write(root, "audit-successor.json", audit)


def test_publishes_exact_consumer_ready_labels_without_decoding_failed_source(
    tmp_path, monkeypatch
):
    plan, original, _, receipt, _ = prepared(tmp_path)
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Publisher decoded pixels"))
    result = annotations.publish_reader_annotations3(tmp_path, plan)
    assert result["publication_state"] == "completed"
    assert not result["qualification_evidence"] and not result["blinding_established"]
    assert result["sessions"][0]["prediction_exposure_status"] == "known"
    assert result["sessions"][0]["original_sources"][0]["original_session_id"] is None
    assert result["sessions"][0]["original_sources"][0]["status"] == "failed"
    record = load_run(tmp_path / "artifacts", result["annotation_run_id"])
    assert record["episodes"] == record["training_steps"] == record["optimizer_updates"] == 0
    assert record["summary"]["published_annotation_count"] == 2
    assert record["summary"]["image_decodes"] == record["summary"]["predictions"] == 0
    assert record["summary"]["resources"]["version"] == "resources-3.0"
    assert verify_run(tmp_path / "artifacts", result["annotation_run_id"])["valid"]
    original_seal = json.loads(
        (tmp_path / "artifacts/runs" / original["run_id"] / "seal.json").read_text()
    )
    session = SimpleNamespace(
        session_id="construction-session",
        purpose="construction",
        sealed_at=datetime.fromisoformat(original_seal["created_at"]),
    )
    consumer = SimpleNamespace(
        artifact_root="artifacts",
        annotation_run_id=result["annotation_run_id"],
        annotations=[EvidenceRef(**r) for r in result["annotations"]],
    )
    labels, _ = _annotations(
        tmp_path,
        consumer,
        {original["run_id"]: (session, original, tmp_path / "artifacts/runs" / original["run_id"])},
        {session.session_id: result["sessions"][0]["declaration"]},
        {},
    )
    assert [r[0] for r in labels] == [ResultAnnotation(**r) for r in plan["annotations"]]
    assert verify_run(tmp_path / "artifacts", receipt["run_id"])["valid"]


@pytest.mark.parametrize("purpose", ["construction", "heldout"])
def test_duplicate_animation_frames_cannot_create_distinct_endpoint_labels(tmp_path, purpose):
    plan, *_ = prepared(tmp_path, purpose=purpose, same_attempt=True)
    with pytest.raises(ValueError, match="canonical endpoint|first retained"):
        annotations.publish_reader_annotations3(tmp_path, plan)
    assert publication_runs(tmp_path)[0]["status"] == "failed"


def test_heldout_keeps_unavailable_source_attempts_and_all_registered_negatives(tmp_path):
    plan, *_ = prepared(tmp_path, purpose="heldout", negative=True)
    result = annotations.publish_reader_annotations3(tmp_path, plan)
    coverage = result["source_selection_coverage"]
    assert len(coverage["expected_negative_unit_ids"]) == 1
    assert len(coverage["expected_result_unit_ids"]) == 1
    assert coverage["attempts_without_retained_terminal"][0]["attempt_index"] == 2
    assert not result["independent_truth_established"]


@pytest.mark.parametrize("negative", [False, True])
def test_heldout_cannot_omit_retained_endpoint_or_preregistered_negative(tmp_path, negative):
    plan, *_ = prepared(tmp_path, purpose="heldout", negative=negative)
    plan["annotations"].pop()
    with pytest.raises(ValueError, match="every retained endpoint"):
        annotations.publish_reader_annotations3(tmp_path, plan)


@pytest.mark.parametrize("status", ["known", "unknown"])
def test_failed_publication_keeps_exposure_against_fresh_none_reported_audit(tmp_path, status):
    plan, _, _, _, audit = prepared(tmp_path, purpose="heldout")
    change_audit(tmp_path, plan, audit, prediction_exposure_status=status)
    with pytest.raises(ValueError, match="requires none_reported"):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert failed["summary"]["operation_counters"]["prediction_exposure_not_ruled_out"] == 1
    change_audit(tmp_path, plan, audit, prediction_exposure_status="none_reported")
    with pytest.raises(KnownReaderExposure, match="fresh audits"):
        annotations.publish_reader_annotations3(tmp_path, plan)


def test_exposure_event_write_fault_cannot_erase_already_read_audit(tmp_path, monkeypatch):
    plan, _, _, _, audit = prepared(tmp_path, purpose="heldout")
    change_audit(tmp_path, plan, audit, prediction_exposure_status="known")
    original = annotations._Publication.event

    def fault(self, phase, **details):
        if phase == "prediction_exposure_not_ruled_out":
            raise OSError("injected attestation event write fault")
        return original(self, phase, **details)

    monkeypatch.setattr(annotations._Publication, "event", fault)
    with pytest.raises(OSError, match="attestation event"):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert (
        failed["summary"]["prediction_exposure_attestations"][0]["prediction_exposure_status"]
        == "known"
    )
    monkeypatch.setattr(annotations._Publication, "event", original)
    change_audit(tmp_path, plan, audit, prediction_exposure_status="none_reported")
    with pytest.raises(KnownReaderExposure):
        annotations.publish_reader_annotations3(tmp_path, plan)


def test_draft_protocol_failure_retains_exact_input(tmp_path):
    plan, *_ = prepared(tmp_path)
    protocol = json.loads((tmp_path / plan["protocol"]["path"]).read_text())
    protocol["status"] = "draft"
    plan["protocol"] = write(tmp_path, "draft.json", protocol)
    with pytest.raises(ValueError, match="registered"):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert any(a["sha256"] == plan["protocol"]["sha256"] for a in failed["artifact_manifest"])


@pytest.mark.parametrize("input_name", ["audit", "protocol"])
def test_mutable_input_swap_after_registration_cannot_change_parsed_evidence(
    tmp_path, monkeypatch, input_name
):
    plan, _, _, _, audit = prepared(tmp_path, purpose="heldout")
    if input_name == "audit":
        change_audit(tmp_path, plan, audit, prediction_exposure_status="known")
        ref = plan["sessions"][0]["exposure_audit"]
        replacement = audit
        match = "requires none_reported"
    else:
        protocol = json.loads((tmp_path / plan["protocol"]["path"]).read_text())
        draft = {**protocol, "status": "draft"}
        plan["protocol"] = write(tmp_path, "draft.json", draft)
        ref = plan["protocol"]
        replacement = protocol
        match = "requires a registered"
    target = tmp_path / ref["path"]
    original = RunRecorder.register_artifact

    def swap(self, source_path, *args, **kwargs):
        result = original(self, source_path, *args, **kwargs)
        if source_path == target:
            target.write_text(json.dumps(replacement), encoding="utf-8")
        return result

    monkeypatch.setattr(RunRecorder, "register_artifact", swap)
    with pytest.raises(ValueError, match=match):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert failed["status"] == "failed"
    assert any(a["sha256"] == ref["sha256"] for a in failed["artifact_manifest"])
    if input_name == "audit":
        assert (
            failed["summary"]["prediction_exposure_attestations"][0]["prediction_exposure_status"]
            == "known"
        )


@pytest.mark.parametrize(
    "fault",
    [
        "future_label",
        "pre_source_label",
        "audit_cutoff",
        "implicit_exposure",
        "ancestry",
        "wrong_session",
        "wrong_purpose",
        "extra",
    ],
)
def test_invalid_contract_or_chronology_retains_failed_attempt(tmp_path, fault):
    plan, _, _, _, audit = prepared(tmp_path)
    if fault == "future_label":
        plan["annotations"][0]["labeled_at"] = "2099-01-01T00:00:00Z"
    elif fault == "pre_source_label":
        plan["annotations"][0]["labeled_at"] = "2020-01-01T00:00:00Z"
    elif fault == "audit_cutoff":
        change_audit(tmp_path, plan, audit, exposure_reviewed_through="2099-01-01T00:00:00Z")
    elif fault == "implicit_exposure":
        change_audit(tmp_path, plan, audit, prediction_exposure_status=None)
    elif fault == "ancestry":
        plan["annotations"][0]["source_hashes"].append("0" * 64)
    elif fault == "wrong_session":
        plan["annotations"][0]["session_id"] = "undeclared"
    elif fault == "wrong_purpose":
        plan["sessions"][0]["purpose"] = "heldout"
    else:
        plan["annotations"][0]["automatic_truth"] = True
    with pytest.raises(ValueError):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert failed["status"] == "failed"
    assert failed["summary"]["operation_counters"]["annotation_publication_failed"] == 1
    assert not failed["summary"]["qualification_evidence"]
    assert verify_run(tmp_path / "artifacts", failed["run_id"])["valid"]


@pytest.mark.parametrize("fault", ["hash", "unregistered", "path"])
def test_original_registered_frame_identity_required(tmp_path, fault):
    plan, _, frames, _, _ = prepared(tmp_path)
    if fault == "hash":
        (tmp_path / frames[0]["path"]).write_bytes(b"changed synthetic source")
    elif fault == "unregistered":
        path = tmp_path / frames[0]["path"]
        plan["annotations"][0]["frame"]["path"] = (
            (path.parent.parent / "frame-0.png").relative_to(tmp_path).as_posix()
        )
    else:
        plan["annotations"][0]["frame"]["path"] = "../outside.png"
    with pytest.raises(ValueError):
        annotations.publish_reader_annotations3(tmp_path, plan)


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_partial_write_retains_prior_labels_journal_and_primary_fault(
    tmp_path, monkeypatch, failure_type
):
    plan, *_ = prepared(tmp_path)
    original = annotations._write

    def failing(path, value):
        if path.name == "annotation-0001.json":
            path.write_bytes(b'{"incomplete":')
            raise failure_type("injected label write failure")
        return original(path, value)

    monkeypatch.setattr(annotations, "_write", failing)
    with pytest.raises(failure_type, match="injected label write"):
        annotations.publish_reader_annotations3(tmp_path, plan)
    failed = publication_runs(tmp_path)[0]
    assert failed["summary"]["published_annotation_count"] == 1
    assert failed["summary"]["operation_counters"]["annotation_write_started"] == 2
    assert failed["summary"]["operation_counters"]["annotation_published"] == 1
    assert not failed["summary"]["publication_complete"]
    directory = tmp_path / "artifacts/runs" / failed["run_id"]
    assert (directory / "annotation-0001.json").read_bytes() == b'{"incomplete":'
    assert verify_run(tmp_path / "artifacts", failed["run_id"])["valid"]


def test_journal_cleanup_does_not_replace_primary_validation_error(tmp_path, monkeypatch):
    plan, *_ = prepared(tmp_path)
    plan["annotations"][0]["session_id"] = "wrong"

    def bad_close(self):
        self.stream.close()
        raise OSError("journal registration failed")

    monkeypatch.setattr(annotations._Publication, "close", bad_close)
    with pytest.raises(ValueError, match="declared source session") as failure:
        annotations.publish_reader_annotations3(tmp_path, plan)
    assert any("journal registration failed" in n for n in failure.value.__notes__)
    assert publication_runs(tmp_path)[0]["status"] == "failed"


def test_repeated_journal_event_fault_still_closes_its_stream(tmp_path, monkeypatch):
    plan, *_ = prepared(tmp_path)
    observed = []

    def fail_event(self, phase, **details):
        observed.append(self.stream)
        raise OSError("journal event unavailable")

    monkeypatch.setattr(annotations._Publication, "event", fail_event)
    with pytest.raises(OSError, match="journal event unavailable") as failure:
        annotations.publish_reader_annotations3(tmp_path, plan)
    assert observed and all(stream.closed for stream in observed)
    assert any("journal finalization failed" in note for note in failure.value.__notes__)
    assert publication_runs(tmp_path)[0]["status"] == "failed"


def test_heldout_session_cannot_exclude_a_second_bound_source_run(tmp_path):
    plan, _, _, receipt, _ = prepared(tmp_path, purpose="heldout")
    source(tmp_path, receipt["required_source_configuration"])
    with pytest.raises(ValueError, match="every canonical source run"):
        annotations.publish_reader_annotations3(tmp_path, plan)
