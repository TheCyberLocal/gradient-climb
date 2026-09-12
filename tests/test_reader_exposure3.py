"""Known canonical reader use cannot be erased by replacement annotations."""

import json

import pytest

from gradientclimb.experiments import RunRecorder, load_run
from gradientclimb.experiments.reader_exposure3 import (
    KnownReaderExposure,
    assert_no_prior_reader_exposure,
    reader_publication_lease,
)


def source(root, config=None):
    with RunRecorder(
        root / "artifacts",
        "synthetic-source",
        config or {},
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        run.finalize(episodes=0)
    return run.run_id


def attempt(
    root,
    run_ids,
    *,
    phase="prediction_started",
    status="completed",
    journal=True,
    groups=None,
    experiment="reader-validation-3.0",
    exposure_fields=None,
):
    config = {
        "split": {
            "sessions": [
                {"session_id": f"session-{i}", "run_ids": group, **(exposure_fields or {})}
                for i, group in enumerate(groups or [run_ids])
            ]
        }
    }
    with RunRecorder(
        root / "artifacts",
        experiment,
        config,
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    ) as run:
        if journal:
            event = {
                "schema_version": "reader-operation-event-3.0",
                "sequence": 0,
                "phase": phase,
                "operation": "evaluation",
                "source_run_id": run_ids[0],
            }
            path = run.directory / "reader-operations.jsonl"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8", newline="\n")
            run.register_artifact(path, "reader_operation_journal")
        run.finalize(status=status, episodes=0)
    return run.run_id


@pytest.mark.parametrize(
    "phase", ["image_processing_started", "fitting_started", "prediction_started"]
)
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_known_processing_blocks_later_relabeling_and_fresh_audits(tmp_path, phase, status):
    run_id = source(tmp_path)
    prior = attempt(tmp_path, [run_id], phase=phase, status=status)
    with pytest.raises(KnownReaderExposure, match="fresh audits") as failure:
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])
    assert failure.value.evidence["run_id"] == prior
    assert failure.value.evidence["exposed_source_run_ids"] == [run_id]


@pytest.mark.parametrize("binding", ["session", "receipt"])
def test_original_source_identity_blocks_session_aliases_across_runs(tmp_path, binding):
    config = (
        {"session_id": "actual-session"}
        if binding == "session"
        else {"reader_session_declaration": {"receipt_run_id": "immutable-declaration"}}
    )
    first, second = source(tmp_path, config), source(tmp_path, config)
    attempt(tmp_path, [first])
    with pytest.raises(KnownReaderExposure, match="Known reader exposure"):
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [second])


def test_exposure_consumes_declared_whole_session_not_only_selected_frame_run(tmp_path):
    first, second = source(tmp_path), source(tmp_path)
    attempt(tmp_path, [first, second])
    with pytest.raises(KnownReaderExposure):
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [second])


def test_independent_session_is_not_contaminated_by_unrelated_exposure(tmp_path):
    first = source(tmp_path, {"session_id": "first"})
    second = source(tmp_path, {"session_id": "second"})
    attempt(tmp_path, [first])
    result = assert_no_prior_reader_exposure(tmp_path, "artifacts", [second])
    assert not result["prior_exposure_detected"]
    assert result["prior_related_attempts_without_recorded_image_processing"] == []


def test_failed_preflight_retains_cost_reference_without_inventing_exposure(tmp_path):
    run_id = source(tmp_path)
    prior = attempt(tmp_path, [run_id], phase="validation_started", status="failed")
    result = assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])
    assert result["prior_related_attempts_without_recorded_image_processing"][0]["run_id"] == prior


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_annotation_exposure_attestation_survives_failed_publication(tmp_path, status):
    run_id = source(tmp_path)
    prior = attempt(
        tmp_path,
        [run_id],
        experiment="reader-annotation-publication-3.0",
        phase="prediction_exposure_not_ruled_out",
        status=status,
    )
    with pytest.raises(KnownReaderExposure, match="fresh audits") as failure:
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])
    assert failure.value.evidence["run_id"] == prior


@pytest.mark.parametrize(
    "fields",
    [
        {"prediction_exposure_status": "known", "first_prediction_exposure_at": None},
        {"prediction_exposure_status": "unknown", "first_prediction_exposure_at": None},
        {"first_prediction_exposure_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_immutable_configured_exposure_survives_pre_event_failure(tmp_path, fields):
    first, second = source(tmp_path), source(tmp_path)
    prior = attempt(
        tmp_path,
        [first, second],
        phase="validation_started",
        status="failed",
        exposure_fields=fields,
    )
    with pytest.raises(KnownReaderExposure, match="fresh audits") as failure:
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [second])
    assert failure.value.evidence["run_id"] == prior
    assert set(failure.value.evidence["configured_prediction_exposure_source_run_ids"]) == {
        first,
        second,
    }


def test_label_publication_alone_does_not_claim_prediction_exposure(tmp_path):
    run_id = source(tmp_path)
    prior = attempt(
        tmp_path,
        [run_id],
        experiment="reader-annotation-publication-3.0",
        phase="annotation_published",
    )
    report = assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])
    assert not report["prior_exposure_detected"]
    assert report["prior_related_attempts_without_recorded_image_processing"][0]["run_id"] == prior


def test_missing_journal_does_not_erase_legacy_or_failed_reader_use(tmp_path):
    run_id = source(tmp_path)
    attempt(tmp_path, [run_id], status="failed", journal=False)
    with pytest.raises(KnownReaderExposure, match="lacks durable exposure"):
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])


def test_unfinished_prior_reader_attempt_requires_recovery(tmp_path):
    run_id = source(tmp_path)
    prior = attempt(tmp_path, [run_id], status="failed")
    directory = tmp_path / "artifacts/runs" / prior
    # Synthetic process-crash fixture: no source evidence is modified.
    (directory / "seal.json").unlink()
    with pytest.raises(KnownReaderExposure, match="Unfinished"):
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])


def test_corrupt_prior_journal_seal_is_not_ignored(tmp_path):
    run_id = source(tmp_path)
    prior = attempt(tmp_path, [run_id])
    path = tmp_path / "artifacts/runs" / prior / "reader-operations.jsonl"
    path.write_bytes(b"")
    with pytest.raises(KnownReaderExposure, match="seal failed"):
        assert_no_prior_reader_exposure(tmp_path, "artifacts", [run_id])


def test_current_recorder_is_excluded_but_lease_prevents_parallel_checks(tmp_path):
    run_id = source(tmp_path)
    with (
        reader_publication_lease(tmp_path, "artifacts"),
        RunRecorder(
            tmp_path / "artifacts",
            "reader-validation-3.0",
            {"source_run_ids": [run_id]},
            environment="synthetic",
            source_root=tmp_path,
            telemetry_interval_seconds=0,
        ) as run,
    ):
        with (
            pytest.raises(RuntimeError, match="Another partition publisher"),
            reader_publication_lease(tmp_path, "artifacts"),
        ):
            pass
        result = assert_no_prior_reader_exposure(
            tmp_path, "artifacts", [run_id], current_run_id=run.run_id
        )
        assert not result["prior_exposure_detected"]
        run.finalize(episodes=0)
    assert load_run(tmp_path / "artifacts", run.run_id)["status"] == "completed"
