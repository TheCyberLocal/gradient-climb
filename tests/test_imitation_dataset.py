"""Synthetic recorded bytes exercise splits, causal windows and immutable evidence."""

import copy
import json
import stat
from functools import partial
from pathlib import Path

import pytest
from demo_dataset_fixtures import make_demo_source, reference, write_json

from gradientclimb.artifacts import canonical_json
from gradientclimb.datasets.demonstrations import (
    EvidenceRef,
    PartitionLedger,
    SegmentationReview,
    SessionAssignment,
    WindowPlan,
    assemble_windows,
    publish_dataset,
)
from gradientclimb.experiments import verify_run


@pytest.fixture
def source(tmp_path):
    return partial(make_demo_source, tmp_path)


def test_deterministic_causal_references_keep_masks_and_missing_profile_honest(source):
    root, plan, review, _, _, frames, controls = source()
    before = (root / review["source"]["frames"]["path"]).read_bytes()
    result = assemble_windows(root, plan)
    assert canonical_json(result) == canonical_json(assemble_windows(root, plan))
    assert result["summary"]["windows"] == 5
    assert result["summary"]["excluded_target_frames"] == 3
    assert result["summary"]["unique_source_frames"] == 8
    assert result["summary"]["training_eligible"] is False
    assert result["sources"][0]["prior_cost"]["human_practice_seconds"] is None
    assert result["sources"][0]["capture_status"] == "failed"
    for window in result["windows"]:
        assert len(window["frames"]) == 4 and window["previous_os_action"] is None
        assert window["eligibility"] == "construction_only_profile_incomplete"
        target = window["target"]
        assert target["action_duration_seconds"] is None
        assert all(f["observation_ready_ns"] <= target["started_ns"] for f in window["frames"])
        assert target["gas"] == controls[target["control_sample_index"]]["gas"]
        assert window["history_reset"]["frame_index"] == frames[0]["frame_index"]
    assert before == (root / review["source"]["frames"]["path"]).read_bytes()


def test_gap_resets_history_without_padding(source):
    starts = [1_000_000_000 + i * 100_000_000 for i in (0, 1, 2, 3, 8, 9, 10, 11)]
    root, plan, *_ = source(starts=starts)
    result = assemble_windows(root, plan)
    assert [[f["frame_index"] for f in w["frames"]] for w in result["windows"]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert result["windows"][-1]["history_reset"]["reason"] == "frame_gap"


def test_adjacent_review_segments_never_share_history(source):
    root, plan, review, _, freeze, frames, _ = source()
    one = review["segments"][0]
    two = copy.deepcopy(one)
    one.update(
        end_frame_index_exclusive=4,
        end_ns_exclusive=frames[4]["started_ns"],
        frame_evidence=one["frame_evidence"][:4],
    )
    two.update(
        segment_id="playing-2",
        first_frame_index=4,
        started_ns=frames[4]["started_ns"],
        frame_evidence=two["frame_evidence"][4:],
    )
    review["segments"] = [one, two]
    freeze()
    windows = assemble_windows(root, plan)["windows"]
    assert [w["segment_id"] for w in windows] == ["playing-1", "playing-2"]
    assert [[f["frame_index"] for f in w["frames"]] for w in windows] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_exclusive_boundary_rejects_target_even_when_image_is_in_span(source):
    root, plan, review, _, freeze, _, _ = source()
    baseline = assemble_windows(root, plan)
    review["segments"][0]["end_ns_exclusive"] = baseline["windows"][-1]["target"]["completed_ns"]
    freeze()
    result = assemble_windows(root, plan)
    assert result["summary"]["windows"] == 4
    assert result["exclusions"][-1]["reason"] == "control_bracket_or_history_outside_segment"


def test_unreviewed_observation_ready_tail_is_not_used(source):
    root, plan, review, _, freeze, frames, _ = source()
    review["segments"][0]["end_ns_exclusive"] = frames[-1]["completed_ns"] + 1
    freeze()
    assert (
        assemble_windows(root, plan)["exclusions"][-1]["reason"]
        == "observation_outside_reviewed_time_boundary"
    )


def test_partial_journal_tails_are_retained_and_counted(source):
    root, plan, review, *_ = source(partial_frames=b'{"partial":', partial_controls=b"partial")
    before = (root / review["source"]["frames"]["path"]).read_bytes()
    result = assemble_windows(root, plan)
    assert result["sources"][0]["trailing_partial_bytes"] == {"frames": 11, "controls": 7}
    assert result["summary"]["windows"] == 5
    assert before == (root / review["source"]["frames"]["path"]).read_bytes()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_reviewed_frame",
        "wrong_frame_hash",
        "session",
        "purpose",
        "status",
        "unaccepted_failure",
    ],
)
def test_review_identity_coverage_and_capture_acceptance_are_required(source, fault):
    root, plan, review, ledger, freeze, *_ = source()
    if fault == "missing_reviewed_frame":
        review["segments"][0]["frame_evidence"].pop()
    elif fault == "wrong_frame_hash":
        review["segments"][0]["frame_evidence"][0]["sha256"] = "f" * 64
    elif fault == "session":
        review["source"]["session_id"] = "another-session"
    elif fault == "purpose":
        ledger["sessions"][0].update(source_purpose="development", split="development")
        plan["split"] = "development"
    elif fault == "status":
        review["accepted_capture_status"] = "completed"
    else:
        review["capture_failure_review_note"] = None
    freeze()
    with pytest.raises(ValueError):
        assemble_windows(root, plan)


def test_changed_source_bytes_and_unhashed_review_are_rejected(source):
    root, plan, review, *_ = source()
    path = root / plan["reviews"][0]["path"]
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        assemble_windows(root, plan)
    path.write_bytes(before)
    (root / review["segments"][0]["frame_evidence"][0]["path"]).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="seal failed"):
        assemble_windows(root, plan)


@pytest.mark.parametrize(
    "purpose", ["development", "human-benchmark", "qualification", "construction"]
)
def test_only_original_imitation_purpose_may_supply_training(purpose):
    with pytest.raises(ValueError):
        SessionAssignment(
            run_id="source", session_id="session", source_purpose=purpose, split="train"
        )


def test_protected_source_cannot_be_selected_via_development_plan(source):
    root, plan, *_ = source(purpose="human-benchmark")
    with pytest.raises(ValueError, match="selected ledger split"):
        assemble_windows(root, plan)


def test_session_alias_and_duplicate_review_cannot_leak_across_splits(source):
    root, plan, review, ledger, _freeze, *_ = source()
    ledger["sessions"].append(
        {**ledger["sessions"][0], "run_id": "different-run", "split": "development"}
    )
    with pytest.raises(ValueError, match="one partition"):
        PartitionLedger.model_validate(ledger)
    ledger["sessions"].pop()
    alias = root / "research/review-alias.json"
    write_json(alias, review)
    plan["reviews"].append(reference(root, alias))
    with pytest.raises(ValueError, match="multiple reviews"):
        assemble_windows(root, plan)


@pytest.mark.parametrize(
    "path",
    [
        "../private",
        "C:/private",
        "nested/../private",
        "a\\b",
        "a:stream",
        "NUL.txt",
        "a/",
        "/outside",
    ],
)
def test_nonportable_or_escaping_paths_rejected(path):
    with pytest.raises(ValueError):
        EvidenceRef(path=path, sha256="a" * 64)


def test_reparse_point_is_rejected_before_reading_evidence(source, monkeypatch):
    root, plan, *_ = source()
    target = root / plan["reviews"][0]["path"]
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == target:
            from types import SimpleNamespace

            return SimpleNamespace(
                st_mode=info.st_mode,
                st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
            )
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="symlinks/junctions"):
        assemble_windows(root, plan)


def test_known_profile_fields_require_per_field_evidence_and_enable_only_train(source):
    root, plan, review, _, freeze, *_ = source()
    profile = review["profile"]
    profile.update(
        vehicle_name="fixture vehicle",
        map_name="fixture map",
        game_build="fixture build",
        upgrades=[{"name": "slot-1", "observed_level": 2, "observed_maximum": 2}],
    )
    with pytest.raises(ValueError, match="explicit evidence"):
        SegmentationReview.model_validate(review)
    profile["field_evidence"] = {
        key: profile["evidence"] for key in ("vehicle_name", "map_name", "game_build", "upgrades")
    }
    freeze()
    result = assemble_windows(root, plan)
    assert result["summary"]["training_eligible"]
    assert not result["summary"]["qualification_evidence"]


def test_invalid_clock_order_and_future_observations_rejected(source):
    def invalid(frames):
        frames[3]["observation_ready_ns"] = frames[3]["completed_ns"] - 1

    root, plan, *_ = source(change_frames=invalid)
    with pytest.raises(ValueError, match="causally ordered"):
        assemble_windows(root, plan)


def test_missing_final_control_target_is_censored(source):
    def truncate(controls):
        controls[:] = [row for row in controls if row["started_ns"] < 1_720_000_000]

    root, plan, *_ = source(change_controls=truncate)
    result = assemble_windows(root, plan)
    assert result["summary"]["windows"] == 4
    assert result["exclusions"][-1]["reason"] == "control_target_unbracketed_overlapping_or_late"


def test_contracts_refuse_extra_fields_and_nonfinite_bounds(source):
    _, plan, *_ = source()
    with pytest.raises(ValueError):
        WindowPlan.model_validate({**plan, "qualification_override": True})
    for bound in (float("nan"), float("inf"), 0, -1, True):
        with pytest.raises(ValueError):
            WindowPlan.model_validate({**plan, "maximum_frame_gap_seconds": bound})


def test_poll_bracket_cannot_cross_review_start_even_if_target_is_inside(source):
    def change(controls):
        controls[:] = [row for row in controls if row["started_ns"] >= 1_021_000_000]
        controls.insert(
            0,
            {
                "started_ns": 995_000_000,
                "completed_ns": 1_019_000_000,
                "gas": False,
                "brake": False,
            },
        )
        for index, row in enumerate(controls):
            row["sample_index"] = index

    root, plan, *_ = source(change_controls=change)
    plan["history_frames"] = 1
    result = assemble_windows(root, plan)
    assert result["exclusions"][0] == {
        "run_id": result["sources"][0]["source"]["run_id"],
        "frame_index": 0,
        "reason": "control_bracket_or_history_outside_segment",
    }


def test_excessive_poll_gap_does_not_become_neutral_target(source):
    def change(controls):
        controls[:] = [
            row for row in controls if not 1_651_000_000 <= row["started_ns"] < 1_721_000_000
        ]
        for index, row in enumerate(controls):
            row["sample_index"] = index

    root, plan, *_ = source(change_controls=change)
    result = assemble_windows(root, plan)
    assert result["summary"]["windows"] == 4
    assert result["exclusions"][-1]["reason"] == "control_target_unbracketed_overlapping_or_late"
    assert all(w["frames"][-1]["frame_index"] != 7 for w in result["windows"])


def test_stale_observation_clears_history_before_new_windows(source):
    def change(frames):
        frames[3]["observation_ready_ns"] = frames[3]["started_ns"] + 200_000_000

    root, plan, *_ = source(change_frames=change)
    plan["maximum_frame_age_seconds"] = 0.1
    result = assemble_windows(root, plan)
    assert result["summary"]["windows"] == 1
    assert [f["frame_index"] for f in result["windows"][0]["frames"]] == [4, 5, 6, 7]
    assert any(
        e["frame_index"] == 3 and e["reason"] == "observation_age_exceeds_bound"
        for e in result["exclusions"]
    )


def test_malformed_complete_tail_is_an_error_not_partial_recovery(source):
    root, plan, *_ = source(partial_frames=b"not-json\n")
    with pytest.raises(ValueError):
        assemble_windows(root, plan)


def test_declared_row_and_byte_bounds_are_enforced(source):
    root, plan, *_ = source()
    for key in ("maximum_source_frames", "maximum_source_control_samples", "maximum_journal_bytes"):
        with pytest.raises(ValueError, match="bound"):
            assemble_windows(root, {**plan, key: 1})


def test_publication_seals_derived_references_and_cost_without_changing_sources(source):
    root, _plan, review, *_ = source()
    source_seal = root / "artifacts/runs" / review["source"]["run_id"] / "seal.json"
    before = source_seal.read_bytes()
    result = publish_dataset(root, "research/plan.json")
    assert result["verification"]["valid"] and result["windows"] == 5
    assert before == source_seal.read_bytes()
    record = json.loads((Path(result["directory"]) / "run.json").read_bytes())
    assert record["summary"]["resources"]["version"] == "resources-3.0"
    assert record["summary"]["prior_costs"][0]["human_practice_seconds"] is None
    assert record["episodes"] == record["training_steps"] == record["environment_steps"] == 0
    assert verify_run(root / "artifacts", review["source"]["run_id"])["valid"]
