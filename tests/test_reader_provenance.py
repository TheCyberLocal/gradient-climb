"""Whole-session exclusion and chronology gates use synthetic provenance only."""

from datetime import UTC, datetime, timedelta

import pytest

from gradientclimb.experiments.reader_provenance import (
    ReaderSplitManifest,
    construction_closure,
    validate_blinded_split,
)


def stamp(hour):
    return datetime(2026, 9, 12, tzinfo=UTC) + timedelta(hours=hour)


def digest(number):
    return f"{number:064x}"


def manifest_data():
    return {
        "protocol_id": "reader-validation-3.0",
        "registered_at": stamp(0),
        "frozen_at": stamp(6),
        "protocol_sha256": digest(1),
        "sources": [
            {"source_id": "reader", "sha256": digest(2), "parents": ["parent-bank"]},
            {"source_id": "parent-bank", "sha256": digest(3), "session_ids": ["old-session"]},
            {"source_id": "anchor", "sha256": digest(4), "session_ids": ["anchor-session"]},
            {"source_id": "ui", "sha256": digest(5), "parents": ["ui-crop"]},
            {"source_id": "ui-crop", "sha256": digest(6), "session_ids": ["ui-session"]},
        ],
        "reader_roots": ["reader"],
        "anchor_roots": ["anchor"],
        "ui_roots": ["ui"],
        "sessions": [
            {
                "session_id": "new-session",
                "run_ids": ["run-1", "run-2"],
                "acquired_at": stamp(1),
                "sealed_at": stamp(2),
                "purpose": "heldout",
                "first_prediction_exposure_at": stamp(5),
                "exposure_reviewed_through": stamp(6),
                "exposure_evidence_sha256": digest(7),
                "seal_verified": True,
            }
        ],
        "labels": [
            {
                "label_id": "label-1",
                "session_id": "new-session",
                "run_id": "run-1",
                "frame_sha256": digest(8),
                "source_hashes": [digest(8)],
                "field": "distance",
                "value": "243",
                "labeled_at": stamp(3),
                "annotation_sha256": digest(9),
            }
        ],
    }


def validate(data):
    return validate_blinded_split(ReaderSplitManifest(**data))


def test_transitive_reader_anchor_and_ui_construction_closure():
    hashes, sessions = construction_closure(ReaderSplitManifest(**manifest_data()))
    assert hashes == {digest(i) for i in range(2, 7)}
    assert sessions == {"old-session", "anchor-session", "ui-session"}


def test_valid_blind_split_reports_real_session_count_and_zero_episodes():
    data = manifest_data()
    data["labels"].append(
        {
            **data["labels"][0],
            "label_id": "label-2",
            "run_id": "run-2",
            "frame_sha256": digest(10),
            "source_hashes": [digest(10)],
        }
    )
    report = validate(data)
    assert report["labels"] == 2 and report["unique_frames"] == 2
    assert report["independent_sessions"] == 1 and report["episode_count"] == 0
    assert not report["reader_accuracy_qualified"]
    assert report["fields"]["distance"] == {"unique_frames": 2, "sessions": 1}


@pytest.mark.parametrize("source", ["old-session", "anchor-session", "ui-session"])
def test_new_crop_in_construction_session_is_still_excluded(source):
    data = manifest_data()
    data["sessions"][0]["session_id"] = source
    data["labels"][0]["session_id"] = source
    with pytest.raises(ValueError, match="construction/development"):
        validate(data)


@pytest.mark.parametrize("source_hash", [digest(3), digest(4), digest(6)])
def test_crop_hash_cannot_hide_reader_anchor_or_ui_ancestry(source_hash):
    data = manifest_data()
    data["labels"][0]["source_hashes"].append(source_hash)
    with pytest.raises(ValueError, match="ancestry overlaps"):
        validate(data)


@pytest.mark.parametrize("label_hour", [5, 5.5])
def test_exposed_predictions_cannot_become_blinded_labels(label_hour):
    data = manifest_data()
    data["labels"][0]["labeled_at"] = stamp(label_hour)
    with pytest.raises(ValueError, match="exposed"):
        validate(data)


def test_a_later_label_in_the_same_session_cannot_evade_session_exposure():
    data = manifest_data()
    data["labels"].append(
        {
            **data["labels"][0],
            "label_id": "second",
            "run_id": "run-2",
            "frame_sha256": digest(10),
            "source_hashes": [digest(10)],
            "labeled_at": stamp(5.5),
        }
    )
    with pytest.raises(ValueError, match="exposed"):
        validate(data)


def test_audited_absence_of_exposure_is_allowed_but_absent_audit_is_not():
    data = manifest_data()
    data["sessions"][0]["first_prediction_exposure_at"] = None
    assert validate(data)["labels"] == 1
    data["sessions"][0].pop("exposure_evidence_sha256")
    with pytest.raises(ValueError):
        validate(data)


@pytest.mark.parametrize(
    "session_change,match",
    [
        ({"purpose": "development"}, "construction/development"),
        ({"seal_verified": False}, "seal"),
        ({"acquired_at": stamp(0)}, "prospective registration"),
        ({"exposure_reviewed_through": stamp(5.5)}, "frozen cutoff"),
    ],
)
def test_prospective_session_requirements(session_change, match):
    data = manifest_data()
    data["sessions"][0].update(session_change)
    with pytest.raises(ValueError, match=match):
        validate(data)


@pytest.mark.parametrize("label_hour", [1.5, 6.5])
def test_labels_must_follow_seal_and_precede_cutoff(label_hour):
    data = manifest_data()
    data["labels"][0]["labeled_at"] = stamp(label_hour)
    with pytest.raises(ValueError, match="source sealing"):
        validate(data)


@pytest.mark.parametrize("label_change", [{"run_id": "undeclared"}, {"session_id": "undeclared"}])
def test_explicit_run_and_session_membership_required(label_change):
    data = manifest_data()
    data["labels"][0].update(label_change)
    with pytest.raises(ValueError, match="explicitly registered"):
        validate(data)


def test_duplicate_frames_across_sessions_do_not_inflate_independence():
    data = manifest_data()
    data["sessions"].append({**data["sessions"][0], "session_id": "other", "run_ids": ["run-3"]})
    data["labels"].append(
        {**data["labels"][0], "label_id": "label-2", "session_id": "other", "run_id": "run-3"}
    )
    with pytest.raises(ValueError, match="Duplicate frame/field"):
        validate(data)


def test_different_fields_on_same_frame_are_one_frame_not_multiple_episodes():
    data = manifest_data()
    data["labels"].append(
        {**data["labels"][0], "label_id": "label-2", "field": "coins", "value": "200"}
    )
    assert validate(data)["unique_frames"] == 1 and validate(data)["labels"] == 2


def test_same_frame_in_different_fields_cannot_claim_two_sessions():
    data = manifest_data()
    data["sessions"].append({**data["sessions"][0], "session_id": "other", "run_ids": ["run-3"]})
    data["labels"].append(
        {
            **data["labels"][0],
            "label_id": "label-2",
            "session_id": "other",
            "run_id": "run-3",
            "field": "coins",
        }
    )
    with pytest.raises(ValueError, match="independent sessions"):
        validate(data)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d["sources"][0].update(parents=["missing"]), "Unresolved"),
        (lambda d: d["sources"][1].update(parents=["reader"]), "cycle"),
        (lambda d: d["sources"].append(dict(d["sources"][0])), "Duplicate construction"),
        (
            lambda d: d["sources"].append({"source_id": "orphan", "sha256": digest(20)}),
            "outside declared",
        ),
        (lambda d: d["sessions"].append(dict(d["sessions"][0])), "Duplicate session"),
        (
            lambda d: d["sessions"].append({**d["sessions"][0], "session_id": "other"}),
            "multiple sessions",
        ),
        (lambda d: d["labels"].append(dict(d["labels"][0])), "Duplicate label"),
        (lambda d: d["labels"][0].update(source_hashes=[digest(10)]), "original frame hash"),
    ],
)
def test_incomplete_or_ambiguous_provenance_refused(mutation, match):
    data = manifest_data()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        validate(data)


def test_timezone_hashes_and_explicit_root_categories_are_required():
    data = manifest_data()
    data["registered_at"] = datetime(2026, 9, 12)  # noqa: DTZ001 - deliberately invalid fixture
    with pytest.raises(ValueError):
        validate(data)
    data = manifest_data()
    data["protocol_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        validate(data)
    data = manifest_data()
    data["anchor_roots"] = []
    with pytest.raises(ValueError):
        validate(data)
