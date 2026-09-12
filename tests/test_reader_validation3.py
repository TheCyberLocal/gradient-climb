"""Prospective reader tools use synthetic pixels and sealed synthetic records only."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.experiments import reader_validation3 as api
from gradientclimb.experiments.reader_receipts3 import publish_session_declaration3
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.scoring import _white_text

REGISTERED = "2020-01-01T00:00:00+00:00"
ROI = (738 / 1034, 151 / 581, 916 / 1034, 195 / 581)
ANCHOR = (602 / 1034, 152 / 581, 736 / 1034, 196 / 581)


def now():
    return datetime.now(UTC).isoformat()


def ref(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ref(root, path)


def pixels(text="269", *, result=True, nonce=0):
    rgb = np.full((581, 1034, 3), 40, np.uint8)
    if result:
        for x in range(0, 80, 8):
            rgb[4:36, x : x + 4] = (255, 255, 255)
    cv2.putText(rgb, "DISTANCE:", (606, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if text:
        cv2.putText(rgb, text, (765, 188), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    rgb[500:505, 5:10] = (nonce % 256, nonce // 256, 17)
    return rgb


def record_frames(root, images, configuration=None, specs=None):
    frames = []
    episodes = []
    with RunRecorder(
        root / "artifacts",
        "synthetic-reader-source",
        configuration or {},
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        for index, image in enumerate(images):
            path = run.directory / f"frame-{index}.png"
            Image.fromarray(image).save(path)
            timing = {
                "started_ns": index * 100 + 1,
                "timestamp_ns": index * 100 + 2,
                "completed_ns": index * 100 + 3,
            }
            is_result = specs is None or specs[index][1]
            if is_result:
                artifact = run.register_artifact(path, "terminal_frame", timing)
                episodes.append({"attempt": {"index": index}, "terminal_readings": [timing]})
            else:
                artifact = run.register_artifact(
                    path,
                    "reader_non_result_frame",
                    {
                        **timing,
                        "reader_sampling_unit_id": f"negative-{index}",
                        "reader_sampling_stage": "initial_tune",
                        "reader_sampling_ordinal": index,
                    },
                )
            frames.append(ref(root, run.directory / artifact["path"]))
        run.finalize(status="completed", episode_count=0, episode_summaries=episodes)
    return load_run(root / "artifacts", run.run_id), frames


def seal_labels(root, rows):
    references = []
    with RunRecorder(
        root / "artifacts",
        "synthetic-blind-labels",
        {},
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        for i, row in enumerate(rows):
            path = run.directory / f"annotation-{i}.json"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = run.register_artifact(path, "reader_annotation")
            references.append(ref(root, run.directory / artifact["path"]))
        run.finalize(status="completed", episode_count=0)
    return run.run_id, references


def annotation(frame, run_id, session_id, index, *, text="269", is_result=True):
    return {
        "label_id": f"{session_id}-label-{index}",
        "session_id": session_id,
        "run_id": run_id,
        "sampling_unit_id": f"endpoint-{index}",
        "frame": frame,
        "source_hashes": [frame["sha256"]],
        "field": "result_distance",
        "is_result": is_result,
        "text": text,
        "labeled_at": now(),
        "reviewer": "synthetic-fixture",
        "note": "Synthetic truth; no real measurements.",
    }


def declaration(root, session_id, purpose, protocol_ref, *, record=None, freeze=None, specs=None):
    selection = write(
        root,
        f"selection/{session_id}.json",
        {
            "schema_version": "reader-source-selection-3.0",
            "non_result_units": [
                {
                    "source_config_id": session_id,
                    "unit_id": f"negative-{j}",
                    "stage": "initial_tune",
                    "capture_ordinal": j,
                }
                for j, (_, is_result) in enumerate(specs or [])
                if not is_result
            ],
        },
    )
    return publish_session_declaration3(
        root,
        "artifacts",
        protocol_ref,
        session_id,
        purpose,
        {"reader_sampling_source_id": session_id} if purpose == "heldout" else {},
        candidate_freeze_run_id=freeze,
        source_run_ids=() if record is None else (record["run_id"],),
        selection_ref=selection,
        note="Synthetic receipt; no real source or prediction.",
    )


def session(root, record, session_id, purpose, protocol_ref, cutoff, receipt=None):
    audit = {
        "session_id": session_id,
        "first_prediction_exposure_at": None,
        "exposure_reviewed_through": cutoff,
        "reviewer": "synthetic",
        "note": "Synthetic chronology fixture.",
    }
    audit_ref = write(root, f"audit/{session_id}.json", audit)
    receipt = receipt or declaration(root, session_id, purpose, protocol_ref, record=record)
    actual_seal = json.loads(
        (root / "artifacts/runs" / record["run_id"] / "seal.json").read_bytes()
    )
    return (
        {
            "session_id": session_id,
            "run_ids": [record["run_id"]],
            "acquired_at": record["start_time"],
            "sealed_at": actual_seal["created_at"],
            "purpose": purpose,
            "first_prediction_exposure_at": None,
            "exposure_reviewed_through": cutoff,
            "exposure_evidence_sha256": audit_ref["sha256"],
            "seal_verified": True,
        },
        receipt["run_id"],
        audit_ref,
    )


def graph(root, reader_ref, ui_ref, paths, excluded=("base-session",)):
    bindings = {sha256_file(path): ref(root, path) for path in paths}
    reader_id = reader_ref["sha256"]
    manifest = json.loads((root / reader_ref["path"]).read_bytes())
    return {
        "sources": [
            {
                "source_id": digest,
                "sha256": digest,
                "session_ids": list(excluded),
                "parents": [other for other in bindings if other != digest]
                if digest == reader_id
                else [],
            }
            for digest in bindings
        ],
        "reader_roots": [reader_id],
        "anchor_roots": [manifest["anchor"]["sha256"]],
        "ui_roots": [ui_ref["sha256"]],
        "references": bindings,
    }


@pytest.fixture
def construction(tmp_path):
    root = tmp_path
    bank = root / "base"
    bank.mkdir()
    reader = HUDDigitReader(expected_size=(1034, 581), roi=ROI, alignment_pixels=1)
    for digit in "0123456789":
        path = bank / f"source-{digit}.png"
        Image.fromarray(pixels(digit)).save(path)
        reader.add_labeled_region(
            pixels(digit),
            digit,
            ROI,
            source_sha256=sha256_file(path),
            annotation_id=f"base-{digit}",
        )
    numeric = reader.save(bank / "result-glyphs")
    anchor = bank / "anchor.png"
    Image.fromarray(_white_text(pixels()[152:196, 602:736]).astype(np.uint8) * 255).save(anchor)
    base_ref = write(
        root,
        "base/result-reader.json",
        {
            "version": 1,
            "numeric_manifest": {
                "file": "result-glyphs/hud-glyphs.json",
                "sha256": sha256_file(numeric),
            },
            "anchor": {"file": "anchor.png", "sha256": sha256_file(anchor)},
            "anchor_roi": list(ANCHOR),
            "anchor_threshold": 0.9,
            "field": "right_side_result_distance",
        },
    )
    (root / "references").mkdir()
    ui_frame = root / "references/ui.png"
    Image.fromarray(pixels()).save(ui_frame)
    ui_ref = write(
        root,
        "ui.json",
        {
            "version": 1,
            "profile_id": "synthetic-ui",
            "expected_size": [1034, 581],
            "threshold": 0.99,
            "margin_pixels": 0,
            "variants": [
                {
                    "label": "result",
                    "state": "result",
                    "file": "ui.png",
                    "sha256": sha256_file(ui_frame),
                    "anchors": [[0, 0, 40, 40], [40, 0, 80, 40]],
                }
            ],
        },
    )
    protocol_ref = write(
        root,
        "protocol.json",
        {
            "schema_version": "reader-validation-3.0",
            "protocol_id": "reader-validation-3.0",
            "status": "registered",
            "registered_at": REGISTERED,
            "environment": "synthetic",
            "base_result_reader_sha256": base_ref["sha256"],
            "construction_session_ids": ["construction-session"],
            "qualification_note": "Synthetic protocol fixture.",
        },
    )
    record, frames = record_frames(root, [pixels("269", nonce=20)])
    rows = [annotation(frames[0], record["run_id"], "construction-session", 0)]
    label_run_id, labels = seal_labels(root, rows)
    s, decl, audit = session(
        root, record, "construction-session", "construction", protocol_ref, now()
    )
    paths = [p for p in bank.rglob("*") if p.is_file()] + [root / ui_ref["path"], ui_frame]
    plan = {
        "protocol": protocol_ref,
        "result_reader": base_ref,
        "ui_profile": ui_ref,
        "ui_reference_root": "references",
        "graph": graph(root, base_ref, ui_ref, paths),
        "annotations": labels,
        "annotation_run_id": label_run_id,
        "sessions": [s],
        "session_declarations": {s["session_id"]: decl},
        "exposure_audits": {s["session_id"]: audit},
    }
    return root, plan, paths, frames[0]


@pytest.fixture
def evaluation(construction, request):
    root, construction_plan, paths, construction_frame = construction
    built = api.build_reader3(root, construction_plan)
    candidate = {"path": built["manifest"], "sha256": built["manifest_sha256"]}
    freeze = api.freeze_reader3(
        root,
        {
            "protocol": construction_plan["protocol"],
            "result_reader": candidate,
            "ui_profile": construction_plan["ui_profile"],
            "reader_construction_run_id": built["run_id"],
        },
    )
    records, annotations, receipts = [], [], []
    for i in range(3):
        specs = [("269", True), (None, False), (None, True)]
        if getattr(request, "param", False):
            count = request.param if type(request.param) is int else 20
            specs = [(str(n), True) for n in range(count) if n % 3 == i]
            specs += [(None, False) for n in range(10) if n % 3 == i]
            specs += [(None, True) for n in range(5) if n % 3 == i]
        images = [
            pixels(
                text if text is not None else ("269" if not is_result else ""),
                result=is_result,
                nonce=30 + i * 40 + j,
            )
            for j, (text, is_result) in enumerate(specs)
        ]
        receipt = declaration(
            root,
            f"heldout-{i}",
            "heldout",
            construction_plan["protocol"],
            freeze=freeze["run_id"],
            specs=specs,
        )
        receipts.append(receipt)
        record, frames = record_frames(
            root, images, receipt["required_source_configuration"], specs
        )
        records.append(record)
        annotations.extend(
            annotation(frame, record["run_id"], f"heldout-{i}", j, text=text, is_result=is_result)
            for j, (frame, (text, is_result)) in enumerate(zip(frames, specs, strict=True))
        )
    label_run_id, label_refs = seal_labels(root, annotations)
    cutoff = now()
    sessions, declarations, audits = [], {}, {}
    for i, record in enumerate(records):
        s, d, a = session(
            root,
            record,
            f"heldout-{i}",
            "heldout",
            construction_plan["protocol"],
            cutoff,
            receipts[i],
        )
        sessions.append(s)
        declarations[s["session_id"]] = d
        audits[s["session_id"]] = a
    candidate_dir = root / "artifacts/runs" / built["run_id"]
    paths = [
        *paths,
        root / construction_frame["path"],
        candidate_dir / "result-reader.json",
        candidate_dir / "distance-label-anchor.png",
        *list((candidate_dir / "result-glyphs").iterdir()),
    ]
    source_graph = graph(
        root,
        candidate,
        construction_plan["ui_profile"],
        paths,
        ("base-session", "construction-session"),
    )
    split = {
        "protocol_id": "reader-validation-3.0",
        "protocol_sha256": construction_plan["protocol"]["sha256"],
        "registered_at": REGISTERED,
        "frozen_at": cutoff,
        **{
            key: source_graph[key]
            for key in ("sources", "reader_roots", "anchor_roots", "ui_roots")
        },
        "sessions": sessions,
        "labels": [],
    }
    for row, reference in zip(annotations, label_refs, strict=True):
        a = api.ResultAnnotation.model_validate(row)
        split["labels"].append(
            {
                "label_id": a.label_id,
                "session_id": a.session_id,
                "run_id": a.run_id,
                "frame_sha256": a.frame.sha256,
                "source_hashes": list(a.source_hashes),
                "field": a.field,
                "value": a.value,
                "labeled_at": a.labeled_at.isoformat(),
                "annotation_sha256": reference["sha256"],
            }
        )
    return root, {
        "protocol": construction_plan["protocol"],
        "result_reader": candidate,
        "ui_profile": construction_plan["ui_profile"],
        "ui_reference_root": "references",
        "graph": source_graph,
        "annotations": label_refs,
        "annotation_run_id": label_run_id,
        "session_declarations": declarations,
        "exposure_audits": audits,
        "split": split,
        "reader_frozen_at": freeze["frozen_at"],
        "reader_freeze_run_id": freeze["run_id"],
        "reader_construction_run_id": built["run_id"],
    }


def test_builder_preserves_base_and_guard_parameters(construction):
    root, plan, _, _ = construction
    before = sha256_file(root / plan["result_reader"]["path"])
    result = api.build_reader3(root, plan)
    assert result["selfcheck_all_match"] and not result["qualification_evidence"]
    assert sha256_file(root / plan["result_reader"]["path"]) == before
    manifest = json.loads((root / result["manifest"]).read_bytes())
    numeric = json.loads(
        (root / result["manifest"])
        .parent.joinpath(manifest["numeric_manifest"]["file"])
        .read_bytes()
    )
    assert numeric["threshold"] == 0.84 and numeric["ambiguity_margin"] == 0.04
    assert numeric["alignment_pixels"] == 1 and len(numeric["glyphs"]) == 13
    assert verify_run(root / "artifacts", result["run_id"])["valid"]


def test_evaluation_uses_pixel_ui_for_negatives_and_reports_missing_coverage(evaluation):
    root, plan = evaluation
    result = api.evaluate_reader3(root, plan)
    assert result["positive_available"] == result["positive_exact"] == 3
    assert result["negative_counts"] == {"non_result": 3, "unreadable_result": 3}
    assert result["wrong_accepts"] == 0
    assert {r["ui_state"] for r in result["rows"] if r["kind"] == "non_result"} == {"unknown"}
    assert not result["coverage"]["positive_endpoints"]
    assert not result["reader_accuracy_qualified"]
    assert verify_run(root / "artifacts", result["run_id"])["valid"]


@pytest.mark.parametrize("evaluation", [True], indirect=True)
def test_complete_synthetic_coverage_is_not_real_game_qualification(evaluation):
    root, plan = evaluation
    result = api.evaluate_reader3(root, plan)
    assert result["positive_available"] == result["positive_exact"] == 20
    assert result["negative_counts"] == {"non_result": 10, "unreadable_result": 5}
    assert all(result["coverage"].values())
    assert result["acceptance_criteria_met"]
    assert not result["reader_accuracy_qualified"]


def test_wrong_ui_acceptance_is_counted_instead_of_hidden_by_negative_truth(
    evaluation, monkeypatch
):
    root, plan = evaluation
    monkeypatch.setattr(api, "_state", lambda _ui, _rgb: "result")
    result = api.evaluate_reader3(root, plan)
    assert result["negative_wrong_accepts"] == result["wrong_accepts"] == 3
    assert result["false_accept_rate_on_negatives"] == 0.5
    assert not result["acceptance_criteria_met"]


def test_wrong_numeric_acceptance_is_counted(evaluation, monkeypatch):
    root, plan = evaluation
    original = api.ResultDistanceReader.read

    def wrong(reader, rgb, *, result_state_confirmed=False):
        out = original(reader, rgb, result_state_confirmed=result_state_confirmed)
        if out["valid"]:
            out["distance_meters"] = 299
        return out

    monkeypatch.setattr(api.ResultDistanceReader, "read", wrong)
    result = api.evaluate_reader3(root, plan)
    assert result["positive_wrong_accepts"] == result["wrong_accepts"] == 3
    assert result["exact_accuracy_when_available"] == 0
    assert not result["acceptance_criteria_met"]


def test_duplicate_sampling_units_rejected_before_execution(evaluation):
    root, plan = evaluation
    rows = [json.loads((root / r["path"]).read_bytes()) for r in plan["annotations"]]
    # Caller-selected aliases are audit text, not authoritative endpoint identity.
    rows[1]["sampling_unit_id"] = rows[0]["sampling_unit_id"]
    rows[1]["frame"] = rows[0]["frame"]
    rows[1]["source_hashes"] = rows[0]["source_hashes"]
    label_run_id, references = seal_labels(root, rows)
    plan["annotation_run_id"], plan["annotations"] = label_run_id, references
    for blinded, reference in zip(plan["split"]["labels"], references, strict=True):
        blinded["annotation_sha256"] = reference["sha256"]
    # The later replacement label seal remains before a newly audited cutoff.
    cutoff = now()
    plan["split"]["frozen_at"] = cutoff
    for s in plan["split"]["sessions"]:
        audit = json.loads((root / plan["exposure_audits"][s["session_id"]]["path"]).read_bytes())
        audit["exposure_reviewed_through"] = cutoff
        reference = write(root, f"replacement/{s['session_id']}.json", audit)
        plan["exposure_audits"][s["session_id"]] = reference
        s["exposure_evidence_sha256"] = reference["sha256"]
        s["exposure_reviewed_through"] = cutoff
    with pytest.raises(ValueError, match="Duplicate annotation/frame"):
        api.evaluate_reader3(root, plan)


@pytest.mark.parametrize("field", ["coins", "gameplay_progress", "terminal_cause", ""])
def test_strict_result_field_selection(field):
    row = annotation({"path": "frame.png", "sha256": "0" * 64}, "run", "session", 0)
    row["field"] = field
    with pytest.raises(ValueError):
        api.ResultAnnotation.model_validate(row)


@pytest.mark.parametrize("path", ["../outside.png", "C:/outside.png", "a\\b.png", "a/../b.png"])
def test_safe_paths(path):
    with pytest.raises(ValueError):
        api.EvidenceRef(path=path, sha256="0" * 64)


def test_draft_protocol_refuses_with_sealed_failed_attempt(construction):
    root, plan, _, _ = construction
    data = json.loads((root / plan["protocol"]["path"]).read_bytes())
    data.update(status="draft", registered_at=None)
    plan["protocol"] = write(root, "draft.json", data)
    before = set((root / "artifacts/runs").iterdir())
    with pytest.raises(ValueError, match="Draft"):
        api.build_reader3(root, plan)
    created = set((root / "artifacts/runs").iterdir()) - before
    assert len(created) == 1
    failed = load_run(root / "artifacts", created.pop().name)
    assert failed["status"] == "failed"
    assert failed["summary"]["operation_counters"]["validation_started"] == 1
    assert verify_run(root / "artifacts", failed["run_id"])["valid"]


def test_missing_raw_glyph_ancestor_is_rejected(construction):
    root, plan, _, _ = construction
    graph_data = plan["graph"]
    missing = next(
        k for k, r in graph_data["references"].items() if r["path"] == "base/source-6.png"
    )
    graph_data["sources"] = [s for s in graph_data["sources"] if s["source_id"] != missing]
    for source in graph_data["sources"]:
        source["parents"] = [p for p in source["parents"] if p != missing]
    del graph_data["references"][missing]
    with pytest.raises(ValueError, match="ancestry|closure"):
        api.build_reader3(root, plan)


def test_modified_ancestor_bytes_rejected(construction):
    root, plan, _, _ = construction
    (root / "base/source-6.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        api.build_reader3(root, plan)


def test_exposed_construction_session_cannot_be_heldout(evaluation):
    root, plan = evaluation
    plan["graph"]["sources"][0]["session_ids"].append("heldout-0")
    plan["split"]["sources"] = copy.deepcopy(plan["graph"]["sources"])
    with pytest.raises(ValueError, match="construction/development"):
        api.evaluate_reader3(root, plan)


def test_prediction_exposure_before_labels_is_rejected(evaluation):
    root, plan = evaluation
    session_row = plan["split"]["sessions"][0]
    session_row["first_prediction_exposure_at"] = session_row["sealed_at"]
    with pytest.raises(ValueError, match="exposed"):
        api.evaluate_reader3(root, plan)


def test_unsealed_or_changed_label_bytes_rejected(evaluation):
    root, plan = evaluation
    (root / plan["annotations"][0]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="seal failed"):
        api.evaluate_reader3(root, plan)


def test_annotation_and_blinded_value_disagree(evaluation):
    root, plan = evaluation
    plan["split"]["labels"][0]["value"] = "299"
    with pytest.raises(ValueError, match="annotation bytes"):
        api.evaluate_reader3(root, plan)


def test_reader_freeze_time_cannot_be_forged(evaluation):
    root, plan = evaluation
    plan["reader_frozen_at"] = REGISTERED
    with pytest.raises(ValueError, match="freeze"):
        api.evaluate_reader3(root, plan)


def test_session_purpose_must_precede_acquisition(evaluation):
    root, plan = evaluation
    late = declaration(
        root, "heldout-0", "heldout", plan["protocol"], freeze=plan["reader_freeze_run_id"]
    )
    plan["session_declarations"]["heldout-0"] = late["run_id"]
    with pytest.raises(ValueError, match="before|predate|bind|precede"):
        api.evaluate_reader3(root, plan)


def test_exposure_audit_bytes_must_match(evaluation):
    root, plan = evaluation
    a = plan["exposure_audits"]["heldout-0"]
    data = json.loads((root / a["path"]).read_bytes())
    data["exposure_reviewed_through"] = now()
    plan["exposure_audits"]["heldout-0"] = write(root, "changed-audit.json", data)
    with pytest.raises(ValueError, match="Exposure audit"):
        api.evaluate_reader3(root, plan)


@pytest.mark.parametrize("status", ["known", "unknown"])
def test_construction_preserves_unknown_first_view_time(construction, status):
    root, plan, _, _ = construction
    row = plan["sessions"][0]
    audit = json.loads((root / plan["exposure_audits"][row["session_id"]]["path"]).read_bytes())
    audit["prediction_exposure_status"] = status
    reference = write(root, "explicit-exposure.json", audit)
    plan["exposure_audits"][row["session_id"]] = reference
    row.update(prediction_exposure_status=status, exposure_evidence_sha256=reference["sha256"])
    built = api.build_reader3(root, plan)
    record = load_run(root / "artifacts", built["run_id"])
    saved = record["configuration"]["sessions"][0]
    assert saved["prediction_exposure_status"] == status
    assert saved["first_prediction_exposure_at"] is None
    assert record["episodes"] == 0


def test_exposure_status_must_match_exact_audit(construction):
    root, plan, _, _ = construction
    plan["sessions"][0]["prediction_exposure_status"] = "known"
    with pytest.raises(ValueError, match="Exposure audit bytes disagree"):
        api.build_reader3(root, plan)


def test_exposure_audit_rejects_timestamp_beyond_review_cutoff():
    with pytest.raises(ValueError, match="audit cutoff"):
        api.ExposureAudit(
            session_id="synthetic",
            prediction_exposure_status="known",
            first_prediction_exposure_at="2026-09-12T02:00:00Z",
            exposure_reviewed_through="2026-09-12T01:00:00Z",
            reviewer="test",
            note="Invalid synthetic chronology",
        )


def test_audit_status_swap_after_path_hash_check_is_rejected(tmp_path, monkeypatch):
    original = {
        "session_id": "synthetic",
        "prediction_exposure_status": "known",
        "first_prediction_exposure_at": None,
        "exposure_reviewed_through": now(),
        "reviewer": "test",
        "note": "Known exposure; no first-view timestamp",
    }
    reference = api.EvidenceRef.model_validate(write(tmp_path, "audit.json", original))
    read = api._read

    def swapped(root, ref):
        path = read(root, ref)
        path.write_text(
            json.dumps({**original, "prediction_exposure_status": "none_reported"}),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(api, "_read", swapped)
    with pytest.raises(ValueError, match="exact payload"):
        api._load(tmp_path, reference, api.ExposureAudit)


@pytest.mark.parametrize(
    "origin,status,event_fault",
    [("session", "known", False), ("audit", "unknown", False), ("audit", "known", True)],
)
def test_rejected_exposure_attestation_cannot_be_erased_on_retry(
    evaluation, monkeypatch, origin, status, event_fault
):
    from gradientclimb.experiments.reader_exposure3 import KnownReaderExposure

    root, plan = evaluation
    original = copy.deepcopy(plan)
    session = plan["split"]["sessions"][0]
    if origin == "session":
        session["prediction_exposure_status"] = status
    else:
        old_ref = plan["exposure_audits"][session["session_id"]]
        audit = json.loads((root / old_ref["path"]).read_bytes())
        audit["prediction_exposure_status"] = status
        plan["exposure_audits"][session["session_id"]] = write(root, "known-audit.json", audit)
    original_event = api._ReaderAttempt.event
    if event_fault:

        def broken_event(self, phase, **details):
            if phase == "prediction_exposure_not_ruled_out":
                raise OSError("Injected exposure journal failure before write")
            return original_event(self, phase, **details)

        monkeypatch.setattr(api._ReaderAttempt, "event", broken_event)
    with pytest.raises((ValueError, OSError)):
        api.evaluate_reader3(root, plan)
    failed = latest_attempt(root, "reader-validation-3.0")
    assert failed["status"] == "failed"
    events = operation_rows(root, failed)
    if not event_fault:
        assert any(
            e["phase"] == "prediction_exposure_not_ruled_out" and e["attestation_origin"] == origin
            for e in events
        )
    assert failed["summary"]["prediction_exposure_attestations"][0]["attestation_origin"] == origin
    assert not any(e["phase"] == "prediction_started" for e in events)
    monkeypatch.setattr(api._ReaderAttempt, "event", original_event)
    with pytest.raises(KnownReaderExposure) as error:
        api.evaluate_reader3(root, original)
    assert failed["run_id"] in json.dumps(error.value.evidence)


def test_wrong_session_audit_does_not_assign_exposure_to_requested_source(construction):
    from gradientclimb.experiments.reader_exposure3 import assert_no_prior_reader_exposure

    root, plan, _, _ = construction
    session = plan["sessions"][0]
    old_ref = plan["exposure_audits"][session["session_id"]]
    audit = json.loads((root / old_ref["path"]).read_bytes())
    audit.update(session_id="different-session", prediction_exposure_status="known")
    plan["exposure_audits"][session["session_id"]] = write(root, "wrong-session-audit.json", audit)
    with pytest.raises(ValueError, match="another session"):
        api.build_reader3(root, plan)
    failed = latest_attempt(root, "result-reader-construction-3.0")
    assert not any(
        e["phase"] == "prediction_exposure_not_ruled_out" for e in operation_rows(root, failed)
    )
    assert_no_prior_reader_exposure(root, "artifacts", session["run_ids"])


def test_partial_failed_label_publication_cannot_supply_construction(construction):
    root, plan, _, _ = construction
    rows = [json.loads((root / r["path"]).read_bytes()) for r in plan["annotations"]]
    references = []
    with RunRecorder(
        root / "artifacts",
        "synthetic-partial-labels",
        {},
        source_root=root,
        environment="synthetic",
        telemetry_interval_seconds=0,
    ) as run:
        for i, row in enumerate(rows):
            path = run.directory / f"annotation-{i}.json"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = run.register_artifact(path, "reader_annotation")
            references.append(ref(root, run.directory / artifact["path"]))
        run.finalize(status="failed", episode_count=0, reason="Injected after partial labels")
    plan["annotations"], plan["annotation_run_id"] = references, run.run_id
    with pytest.raises(ValueError, match="completed sealed label"):
        api.build_reader3(root, plan)


def latest_attempt(root, experiment):
    records = [json.loads(p.read_bytes()) for p in (root / "artifacts/runs").glob("*/run.json")]
    return max(
        (r for r in records if r["experiment_id"] == experiment), key=lambda r: r["start_time"]
    )


def operation_rows(root, record):
    path = root / "artifacts/runs" / record["run_id"] / "reader-operations.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_preflight_image_fault_retains_exposure_and_cost(construction, monkeypatch, failure):
    root, plan, _, frame = construction

    def broken(_ui, _rgb):
        raise failure("synthetic recognizer interruption")

    monkeypatch.setattr(api, "_state", broken)
    with pytest.raises(failure):
        api.build_reader3(root, plan)
    record = latest_attempt(root, "result-reader-construction-3.0")
    assert record["status"] == ("cancelled" if failure is KeyboardInterrupt else "failed")
    rows = operation_rows(root, record)
    starts = [r for r in rows if r["phase"] == "image_processing_started"]
    assert len(starts) == 1 and starts[0]["frame_sha256"] == frame["sha256"]
    assert starts[0]["source_run_id"] == plan["sessions"][0]["run_ids"][0]
    assert record["summary"]["operation_work"]["cpu_core_seconds"] >= 0
    assert not record["summary"]["qualification_evidence"]
    assert verify_run(root / "artifacts", record["run_id"])["valid"]


def test_fitting_fault_is_sealed_before_template_addition(construction, monkeypatch):
    root, plan, _, _ = construction

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic fitting failure")

    monkeypatch.setattr(HUDDigitReader, "add_labeled_region", broken)
    with pytest.raises(RuntimeError, match="fitting"):
        api.build_reader3(root, plan)
    record = latest_attempt(root, "result-reader-construction-3.0")
    assert record["summary"]["operation_counters"]["fitting_started"] == 1
    assert "fitting_completed" not in record["summary"]["operation_counters"]
    assert verify_run(root / "artifacts", record["run_id"])["valid"]


def test_prediction_fault_preserves_partial_rows_and_prevents_fresh_retry(evaluation, monkeypatch):
    root, plan = evaluation
    original = api.ResultDistanceReader.read
    calls = 0

    def broken(reader, rgb, *, result_state_confirmed=False):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second prediction fault")
        return original(reader, rgb, result_state_confirmed=result_state_confirmed)

    monkeypatch.setattr(api.ResultDistanceReader, "read", broken)
    with pytest.raises(RuntimeError, match="second prediction"):
        api.evaluate_reader3(root, plan)
    record = latest_attempt(root, "reader-validation-3.0")
    rows = operation_rows(root, record)
    assert sum(r["phase"] == "prediction_started" for r in rows) == 2
    completed = [r for r in rows if r["phase"] == "prediction_completed"]
    assert len(completed) == 1 and completed[0]["row"]["expected"] == 269
    assert verify_run(root / "artifacts", record["run_id"])["valid"]
    monkeypatch.setattr(api.ResultDistanceReader, "read", original)
    with pytest.raises(ValueError, match="Known reader exposure"):
        api.evaluate_reader3(root, plan)


def test_publication_failure_retains_all_predictions_with_no_qualification(evaluation, monkeypatch):
    root, plan = evaluation
    original = RunRecorder.register_artifact

    def broken(run, path, kind, metadata=None):
        if kind == "reader_validation_source":
            raise OSError("synthetic source-copy failure")
        return original(run, path, kind, metadata)

    monkeypatch.setattr(RunRecorder, "register_artifact", broken)
    with pytest.raises(OSError, match="source-copy"):
        api.evaluate_reader3(root, plan)
    record = latest_attempt(root, "reader-validation-3.0")
    partial = record["summary"]["partial_report"]
    assert len(partial["rows"]) == 9 and not partial["reader_accuracy_qualified"]
    assert not partial["publication_complete"]
    assert record["summary"]["operation_counters"]["prediction_completed"] == 9
    assert verify_run(root / "artifacts", record["run_id"])["valid"]


@pytest.mark.parametrize("failure_stage", ["completion_event", "journal_registration"])
def test_late_publication_failure_downgrades_provisional_qualification(
    evaluation, monkeypatch, failure_stage
):
    root, plan = evaluation
    original_worker = api._evaluate_reader3

    def candidate(root, plan, attempt):
        result = original_worker(root, plan, attempt)
        # Fault fixture exercises a true provisional flag without claiming these
        # synthetic frames could qualify a real reader.
        attempt.report["reader_accuracy_qualified"] = True
        return {**result, "reader_accuracy_qualified": True}

    monkeypatch.setattr(api, "_evaluate_reader3", candidate)
    if failure_stage == "completion_event":
        original = api._ReaderAttempt.event

        def broken(attempt, phase, **details):
            if phase == "operation_completed":
                raise OSError("synthetic late publication failure")
            return original(attempt, phase, **details)

        monkeypatch.setattr(api._ReaderAttempt, "event", broken)
    else:
        original = RunRecorder.register_artifact

        def broken(run, path, kind, metadata=None):
            if kind == "reader_operation_journal":
                raise OSError("synthetic late publication failure")
            return original(run, path, kind, metadata)

        monkeypatch.setattr(RunRecorder, "register_artifact", broken)
    with pytest.raises(OSError, match="late publication"):
        api.evaluate_reader3(root, plan)
    record = latest_attempt(root, "reader-validation-3.0")
    assert record["status"] == "failed"
    assert len(record["summary"]["report"]["rows"]) == 9
    assert not record["summary"]["report"]["reader_accuracy_qualified"]
    assert not record["summary"]["partial_report"]["reader_accuracy_qualified"]
    pending = json.loads(
        (root / "artifacts/runs" / record["run_id"] / "reader-validation-report.json").read_bytes()
    )
    assert not pending["reader_accuracy_qualified"]
    assert verify_run(root / "artifacts", record["run_id"])["valid"]
    loaded = api.load_reader_report3(root, "artifacts", record["run_id"])
    assert loaded["publication_state"] == "failed" and not loaded["reader_accuracy_qualified"]
    assert len(loaded["rows"]) == 9


@pytest.mark.parametrize("evaluation", [True, 23], indirect=True)
def test_availability_and_accepted_count_are_independent_gates(evaluation, monkeypatch):
    root, plan = evaluation
    positive_count = sum(row["value"].isdigit() for row in plan["split"]["labels"])
    original = api.ResultDistanceReader.read

    def sparse(reader, rgb, *, result_state_confirmed=False):
        out = original(reader, rgb, result_state_confirmed=result_state_confirmed)
        cutoff = 1 if positive_count == 20 else 20
        if out["valid"] and out["distance_meters"] >= cutoff:
            out = {**out, "valid": False, "distance_meters": None}
        return out

    monkeypatch.setattr(api.ResultDistanceReader, "read", sparse)
    report = api.evaluate_reader3(root, plan)
    assert report["wrong_accepts"] == 0
    assert not report["coverage"]["availability"]
    assert report["positive_available"] == (1 if positive_count == 20 else 20)
    assert report["coverage"]["accepted_positive_endpoints"] == (positive_count == 23)
    assert not report["acceptance_criteria_met"]


def test_omitted_preregistered_negative_rejected(evaluation):
    root, plan = evaluation
    omitted = next(
        row["label_id"] for row in plan["split"]["labels"] if row["value"] == "non_result"
    )
    plan["annotations"] = [
        ref
        for ref in plan["annotations"]
        if json.loads((root / ref["path"]).read_bytes())["label_id"] != omitted
    ]
    plan["split"]["labels"] = [row for row in plan["split"]["labels"] if row["label_id"] != omitted]
    with pytest.raises(ValueError, match="Every preregistered negative"):
        api.evaluate_reader3(root, plan)


def test_unused_construction_session_can_remain_in_full_provenance(evaluation, construction):
    root, plan = evaluation
    _, construction_plan, _, _ = construction
    plan["split"]["sessions"].extend(construction_plan["sessions"])
    plan["session_declarations"].update(construction_plan["session_declarations"])
    plan["exposure_audits"].update(construction_plan["exposure_audits"])
    report = api.evaluate_reader3(root, plan)
    assert report["positive_exact"] == 3
    assert len(report["source_selection"]["expected_result_unit_ids"]) == 6


def test_omitting_retained_result_endpoint_is_rejected(evaluation):
    root, plan = evaluation
    omitted = plan["split"]["labels"][0]["label_id"]
    plan["annotations"] = [
        ref
        for ref in plan["annotations"]
        if json.loads((root / ref["path"]).read_bytes())["label_id"] != omitted
    ]
    plan["split"]["labels"] = [row for row in plan["split"]["labels"] if row["label_id"] != omitted]
    with pytest.raises(ValueError, match="Every retained result endpoint"):
        api.evaluate_reader3(root, plan)


def test_report_loader_rejects_true_flag_in_failed_envelope(tmp_path):
    with RunRecorder(
        tmp_path / "artifacts",
        "reader-validation-3.0",
        {},
        source_root=tmp_path,
        telemetry_interval_seconds=0,
    ) as run:
        run.finalize(
            status="failed", report={"reader_accuracy_qualified": True, "positive_exact": 20}
        )
    report = api.load_reader_report3(tmp_path, "artifacts", run.run_id)
    assert report["positive_exact"] == 20
    assert not report["reader_accuracy_qualified"]
    assert report["publication_state"] == "failed"
