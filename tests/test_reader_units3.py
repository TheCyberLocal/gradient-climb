import json

import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run
from gradientclimb.experiments.reader_units3 import (
    ReaderUnitResolver3,
    expected_negative_unit_ids,
    expected_result_unit_ids,
    resolve_reader_unit,
    result_selection_coverage,
)


def source(tmp_path, *, fault=None, negative=False):
    frames, readings = [], []
    with RunRecorder(
        tmp_path / "artifacts",
        "synthetic-reader-units",
        {"reader_sampling_source_id": "fixture-source"},
        environment="synthetic",
        telemetry_interval_seconds=0,
        source_root=tmp_path,
    ) as run:
        for index in range(3):
            path = run.directory / f"source-{index}.png"
            path.write_bytes(f"Opaque source bytes {index}; no pixel decoding".encode())
            timing = {
                "started_ns": index * 100 + 1,
                "timestamp_ns": index * 100 + 2,
                "completed_ns": index * 100 + 3,
                "width": 10,
                "height": 10,
                "backend": "synthetic",
            }
            if negative:
                timing.update(
                    reader_sampling_unit_id=f"menu-{index}",
                    reader_sampling_stage="tune",
                    reader_sampling_ordinal=index,
                )
                if fault == "overlap" and index == 1:
                    timing.update(started_ns=2, timestamp_ns=3, completed_ns=4)
            metadata = {} if fault == "missing_metadata" else timing
            artifact = run.register_artifact(
                path, "reader_non_result_frame" if negative else "terminal_frame", metadata
            )
            if fault == "duplicate_artifact" and index == 0:
                run.register_artifact(path, artifact["kind"], metadata)
            frames.append(
                {
                    "path": (run.directory / artifact["path"]).relative_to(tmp_path).as_posix(),
                    "sha256": artifact["sha256"],
                }
            )
            readings.append({**timing, "valid": False, "distance_meters": None})
        if fault == "timing_mismatch":
            readings[0]["started_ns"] = 0
        if fault == "capture_mismatch":
            readings[0]["width"] = 11
        episodes = [
            {"attempt": {"index": 8}, "terminal_readings": readings[:2]},
            {"attempt": {"index": 9}, "terminal_readings": readings[2:]},
        ]
        if fault == "duplicate_timestamp":
            episodes[1]["terminal_readings"].append(readings[0])
        if fault == "duplicate_attempt":
            episodes[1]["attempt"]["index"] = 8
        if fault == "missing_reading":
            episodes[0]["terminal_readings"].pop(0)
        run.finalize(status="failed", episodes=0, episode_summaries=episodes)
    return run.run_id, frames, load_run(tmp_path / "artifacts", run.run_id)


def selection():
    return {
        "schema_version": "reader-source-selection-3.0",
        "non_result_units": [
            {
                "source_config_id": "fixture-source",
                "unit_id": f"menu-{i}",
                "stage": "tune",
                "capture_ordinal": i,
            }
            for i in range(3)
        ],
    }


def resolve(root, run_id, frame, **kwargs):
    return resolve_reader_unit(
        root,
        "artifacts",
        frame,
        source_run_id=run_id,
        label_is_result=True,
        purpose="heldout",
        **kwargs,
    )


def test_different_terminal_image_bytes_share_one_canonical_endpoint(tmp_path):
    run_id, frames, _ = source(tmp_path)
    resolver = ReaderUnitResolver3(tmp_path)
    rows = [
        resolver.resolve(frame, source_run_id=run_id, label_is_result=True, purpose="construction")
        for frame in frames
    ]
    assert frames[0]["sha256"] != frames[1]["sha256"]
    assert rows[0]["canonical_unit_id"] == rows[1]["canonical_unit_id"]
    assert rows[2]["canonical_unit_id"] != rows[0]["canonical_unit_id"]
    assert rows[0]["attempt_index"] == 8
    assert not rows[0][
        "qualification_eligible"
    ]  # Synthetic identity never becomes real qualification.
    assert rows[0]["capture_timing"] == {"started_ns": 1, "timestamp_ns": 2, "completed_ns": 3}
    assert rows[0]["artifact"]["sha256"] == frames[0]["sha256"]
    assert rows[0]["construction_fallback"] is None


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing_metadata", "timing triple"),
        ("timing_mismatch", "timing metadata disagree"),
        ("capture_mismatch", "capture metadata disagree"),
        ("duplicate_timestamp", "ambiguous endpoint"),
        ("duplicate_attempt", "attempt indices"),
        ("missing_reading", "missing or ambiguous"),
        ("duplicate_artifact", "exactly one sealed artifact"),
    ],
)
def test_result_and_unreadable_endpoints_fail_closed_on_missing_or_ambiguous_identity(
    tmp_path, fault, match
):
    run_id, frames, _ = source(tmp_path, fault=fault)
    # Result-label readability is deliberately not an input; both classes share this gate.
    with pytest.raises(ValueError, match=match):
        resolve(tmp_path, run_id, frames[0])


def test_frame_hash_cannot_alias_an_unregistered_original_or_another_path(tmp_path):
    run_id, frames, _ = source(tmp_path)
    frame = {**frames[0], "path": f"artifacts/runs/{run_id}/source-0.png"}
    assert sha256_file(tmp_path / frame["path"]) == frame["sha256"]
    with pytest.raises(ValueError, match="exactly one sealed artifact"):
        resolve(tmp_path, run_id, frame)
    with pytest.raises(ValueError, match="exactly one sealed artifact"):
        resolve(tmp_path, run_id, {**frames[0], "sha256": frames[1]["sha256"]})


def test_explicit_construction_fallback_never_qualifies_and_cannot_hide_bad_hash(tmp_path):
    run_id, frames, _ = source(tmp_path, fault="missing_metadata")
    resolver = ReaderUnitResolver3(tmp_path)
    with pytest.raises(ValueError, match="timing triple"):
        resolver.resolve(
            frames[0], source_run_id=run_id, label_is_result=True, purpose="construction"
        )
    row = resolver.resolve(
        frames[0],
        source_run_id=run_id,
        label_is_result=True,
        purpose="construction",
        construction_fallback_reason="Legacy artifact lacks canonical timing",
    )
    assert (
        row["unit_kind"] == "unresolved_construction_artifact" and not row["qualification_eligible"]
    )
    with pytest.raises(ValueError, match="construction-only"):
        resolve(tmp_path, run_id, frames[0], construction_fallback_reason="Cannot exempt heldout")
    with pytest.raises(ValueError, match="exactly one sealed artifact"):
        resolver.resolve(
            {**frames[0], "sha256": "a" * 64},
            source_run_id=run_id,
            label_is_result=True,
            purpose="construction",
            construction_fallback_reason="Cannot exempt invalid source",
        )


def test_cache_keeps_verified_identity_but_detects_subsequent_frame_tampering(tmp_path):
    run_id, frames, _ = source(tmp_path)
    resolver = ReaderUnitResolver3(tmp_path)
    resolver.resolve(frames[0], source_run_id=run_id, label_is_result=True, purpose="heldout")
    (tmp_path / frames[1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        resolver.resolve(frames[1], source_run_id=run_id, label_is_result=True, purpose="heldout")


def test_negative_units_come_from_predeclared_source_selectors_and_complete_expected_set(tmp_path):
    run_id, frames, record = source(tmp_path, negative=True)
    resolver = ReaderUnitResolver3(tmp_path)
    rows = [
        resolver.resolve(
            frame,
            source_run_id=run_id,
            label_is_result=False,
            purpose="heldout",
            selection_document=selection(),
        )
        for frame in frames
    ]
    expected = expected_negative_unit_ids(selection(), [record])
    assert tuple(row["canonical_unit_id"] for row in rows) == expected
    assert len(set(expected)) == 3
    assert all(row["unit_kind"] == "preregistered_non_result_capture" for row in rows)
    assert all("attempt_index" not in row for row in rows)
    assert all(not row["qualification_eligible"] for row in rows)
    # A subset can be detected independently of arbitrary annotation unit names.
    assert {rows[0]["canonical_unit_id"]} != set(expected)


@pytest.mark.parametrize(
    "fault", ["no_selection", "wrong_unit", "duplicate_ordinal", "missing_source", "overlap"]
)
def test_negative_qualification_rejects_invented_missing_or_overlapping_units(tmp_path, fault):
    run_id, frames, _record = source(tmp_path, negative=True, fault=fault)
    declared = selection()
    if fault == "no_selection":
        declared = None
    elif fault == "wrong_unit":
        declared["non_result_units"][0]["unit_id"] = "Invented after seeing pixels"
    elif fault == "duplicate_ordinal":
        declared["non_result_units"][1]["capture_ordinal"] = 0
    elif fault == "missing_source":
        with pytest.raises(ValueError, match="exactly one source run"):
            expected_negative_unit_ids(declared, [])
        return
    with pytest.raises(ValueError):
        ReaderUnitResolver3(tmp_path).resolve(
            frames[0],
            source_run_id=run_id,
            label_is_result=False,
            purpose="heldout",
            selection_document=declared,
        )


def test_mutated_canonical_record_is_rejected_before_unit_resolution(tmp_path):
    run_id, frames, record = source(tmp_path)
    record["summary"]["episode_summaries"][0]["attempt"]["index"] = 42
    (tmp_path / "artifacts/runs" / run_id / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="seal failed"):
        resolve(tmp_path, run_id, frames[0])


def test_cache_cannot_hide_later_canonical_metadata_tampering(tmp_path):
    run_id, frames, record = source(tmp_path)
    resolver = ReaderUnitResolver3(tmp_path)
    resolver.resolve(frames[0], source_run_id=run_id, label_is_result=True, purpose="heldout")
    record["summary"]["episode_summaries"][0]["attempt"]["index"] = 42
    (tmp_path / "artifacts/runs" / run_id / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="record or seal changed"):
        resolver.resolve(frames[1], source_run_id=run_id, label_is_result=True, purpose="heldout")


def test_heldout_cannot_choose_later_animation_frame(tmp_path):
    run_id, frames, record = source(tmp_path)
    assert resolve(tmp_path, run_id, frames[0])["attempt_index"] == 8
    with pytest.raises(ValueError, match="first retained terminal"):
        resolve(tmp_path, run_id, frames[1])
    expected = expected_result_unit_ids([record])
    assert expected == (
        f"native-endpoint:{run_id}:attempt:8",
        f"native-endpoint:{run_id}:attempt:9",
    )


def test_attempt_without_retained_terminal_stays_explicitly_unavailable(tmp_path):
    run_id, _, record = source(tmp_path)
    record["summary"]["episode_summaries"].append(
        {"attempt": {"index": 10}, "terminal_readings": []}
    )
    # Pure helper fixture, not a claim that this changed record is canonical.
    coverage = result_selection_coverage([record])
    assert len(coverage["expected_result_unit_ids"]) == 2
    assert coverage["attempts_without_retained_terminal"] == [
        {"source_run_id": run_id, "attempt_index": 10, "reason": "no_retained_terminal_frame"}
    ]
