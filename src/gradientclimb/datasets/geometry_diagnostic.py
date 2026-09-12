"""Construction-only geometry coverage on reviewed, causally assembled windows.

Detector validity is a hypothesis, not labeled accuracy. No policy, HUD reader,
privileged state or inferred human action enters this diagnostic.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links
from gradientclimb.datasets.demonstrations import WindowPlan, assemble_windows
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile


def measure_unique_frames(project_root, windows, measurer, *, maximum_frames=1000, on_event=None):
    """Read each image once and report attempted, completed and failed operations."""
    if type(maximum_frames) is not int or not 1 <= maximum_frames <= 10000:
        raise ValueError("Maximum diagnostic frames must be an integer in 1..10000")
    root = Path(project_root).absolute()
    _no_links(root)
    references = {}
    reuse_counts = {}
    for window in windows:
        for frame in window["frames"]:
            key = (window["run_id"], frame["frame_index"])
            if key in references and references[key] != frame:
                raise ValueError("Conflicting references for the same source frame")
            references[key] = dict(frame)
            reuse_counts[key] = reuse_counts.get(key, 0) + 1
    if not 1 <= len(references) <= maximum_frames:
        raise ValueError("Unique source frame count is outside the diagnostic bound")
    emit = on_event if on_event is not None else lambda _event: None
    emit({"event": "selection", "planned_source_frames": len(references)})
    rows = []
    for (run_id, frame_index), reference in sorted(references.items()):
        identity = {"run_id": run_id, "frame_index": frame_index, "source": reference}
        emit({"event": "frame_started", **identity})
        started = time.perf_counter()
        stage = "path_and_hash_validation"
        try:
            path = _inside(root, reference["path"])
            if sha256_file(path) != reference["file_sha256"]:
                raise ValueError("Source image hash changed before geometry measurement")
            stage = "image_decode"
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            decoded = time.perf_counter()
            stage = "geometry_measurement"
            measurement = measurer.measure(rgb)
            finished = time.perf_counter()
            row = {
                **identity,
                "window_reference_uses": reuse_counts[(run_id, frame_index)],
                "hash_and_decode_seconds": decoded - started,
                "measurement_seconds": finished - decoded,
                "measurement": measurement.as_dict(),
            }
        except BaseException as error:
            try:
                emit(
                    {
                        "event": "frame_failed",
                        **identity,
                        "failed_at_stage": stage,
                        "operation_elapsed_seconds": time.perf_counter() - started,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            except BaseException as journal_error:  # noqa: BLE001 - preserve the original failure
                error.add_note(f"Failure journal write also failed: {journal_error}")
            raise
        emit({"event": "frame_completed", "measurement": row})
        rows.append(row)
    costs = np.asarray([row["measurement_seconds"] for row in rows])
    return {
        "schema_version": "reviewed-geometry-diagnostic-3.0",
        "rows": rows,
        "summary": {
            "unique_source_frames": len(rows),
            "attempted_source_frames": len(rows),
            "completed_source_frames": len(rows),
            "failed_source_frames": 0,
            "window_frame_reference_uses": sum(reuse_counts.values()),
            "detector_valid_body_frames": sum(
                row["measurement"]["body"].get("valid", False) for row in rows
            ),
            "detector_valid_wheel_frames": sum(
                row["measurement"]["wheels"].get("valid", False) for row in rows
            ),
            "detector_valid_terrain_frames": sum(
                row["measurement"]["terrain"].get("valid", False) for row in rows
            ),
            "measured_geometry_accuracy": None,
            "measurement_seconds_total": float(costs.sum()),
            "measurement_seconds_median": float(np.median(costs)),
            "measurement_seconds_p95": float(np.quantile(costs, 0.95)),
            "hash_and_decode_seconds_total": sum(row["hash_and_decode_seconds"] for row in rows),
            "new_real_interaction_seconds": 0,
            "new_environment_transitions": 0,
            "new_gameplay_episodes": 0,
            "policy_decisions": 0,
            "optimizer_updates": 0,
            "qualifies_real_game": False,
        },
        "interpretation": "Construction detector coverage only. Validity is not independently labeled accuracy, successful control, or real competence. Per-image timings exclude assembly, provenance, journal writes and artifact publication; recorder costs retain their separate scope.",
    }


@contextmanager
def _measurement_journal(run):
    """Seal flushed operation evidence on both normal and exceptional exits."""
    path = run.directory / "geometry-measurements.jsonl"
    stream = path.open("x", encoding="utf-8", newline="\n")
    progress = {
        "planned_source_frames": None,
        "attempted_source_frames": 0,
        "completed_source_frames": 0,
        "failed_source_frames": 0,
        "persisted_journal_events": 0,
    }

    def record(event):
        entry = {
            "schema_version": "reviewed-geometry-operations-3.0",
            "event_index": progress["persisted_journal_events"],
            "diagnostic_elapsed_seconds": run.elapsed_seconds,
            **event,
        }
        stream.write(canonical_json(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        if event["event"] == "selection":
            progress["planned_source_frames"] = event["planned_source_frames"]
        for kind, counter in (
            ("frame_started", "attempted_source_frames"),
            ("frame_completed", "completed_source_frames"),
            ("frame_failed", "failed_source_frames"),
        ):
            if event["event"] == kind:
                progress[counter] += 1
        progress["persisted_journal_events"] += 1
        run.annotate(geometry_progress=progress)
        run.record_progress(
            {
                "environment_steps": 0,
                "training_steps": 0,
                "training_clock_seconds": 0,
                "episodes": 0,
                "optimizer_updates": 0,
                "counter_provenance": "flushed_geometry_operation_journal",
                "geometry_progress": progress,
            }
        )

    failure = None
    try:
        record({"event": "diagnostic_started"})
        yield record
    except BaseException as error:
        failure = error
        raise
    finally:
        cleanup_errors = []
        for operation in (
            stream.close,
            lambda: run.register_artifact(path, "reviewed_geometry_operation_journal"),
        ):
            try:
                operation()
            except BaseException as error:  # noqa: BLE001 - attempt registration after close failure
                cleanup_errors.append(error)
        if cleanup_errors:
            details = "; ".join(f"{type(e).__name__}: {e}" for e in cleanup_errors)
            if failure is not None:
                failure.add_note(f"Geometry journal cleanup also failed: {details}")
            else:
                raise RuntimeError(
                    f"Geometry journal cleanup failed: {details}"
                ) from cleanup_errors[0]


def publish_geometry_diagnostic(project_root, plan_path, profile_path, *, maximum_frames=1000):
    """Reconstruct the frozen window plan and seal a separately costed diagnostic."""
    from gradientclimb.datasets.partitions import partition_publication
    from gradientclimb.experiments import RunRecorder, verify_run

    root = Path(project_root).absolute()
    _no_links(root)
    plan_file, profile_file = _inside(root, plan_path), _inside(root, profile_path)
    plan = WindowPlan.model_validate_json(plan_file.read_bytes())
    profile = MeasurementProfile(**json.loads(profile_file.read_bytes()))
    config = {
        "protocol_version": "reviewed-geometry-diagnostic-3.0",
        "window_plan": {"path": plan_path, "sha256": sha256_file(plan_file)},
        "measurement_profile": {"path": profile_path, "sha256": sha256_file(profile_file)},
        "maximum_frames": maximum_frames,
        "selection": "Every unique frame referenced by every accepted window; no outcome-based subsampling",
        "fitting": False,
        "accuracy_labels": None,
        "prior_class": "human-demonstration and engineered-perception construction costs; not a learning run",
        "prior_evidence": [ref.model_dump() for ref in plan.prior_evidence],
        "source_partition": plan.split,
    }
    with (
        partition_publication(root, plan) as binding,
        RunRecorder(
            _inside(root, plan.artifact_root),
            "reviewed-human-geometry-diagnostic",
            binding.configuration(config),
            algorithm="fixed-pixel-measurement",
            environment="saved-human-demonstration",
            evidence_domain="construction_real_game_pixel_measurement",
            qualifies_real_game=False,
            source_root=root,
        ) as run,
    ):
        binding.register(run)
        run.annotate(
            geometry_resource_scope="recorder_resource_window_includes_source_verification_assembly_and_diagnostic; excludes_imports_initial_plan_and_profile_parse_partition_authority_preflight_and_post_sample_sealing",
            geometry_per_image_timing_scope="hash_decode_and_measurement_only; excludes_assembly_provenance_journal_writes_and_artifact_publication",
            policy_decisions=0,
            evaluation_episodes=0,
            real_game_evaluation_episodes=0,
            new_real_interaction_seconds=0,
            new_environment_transitions=0,
            new_gameplay_episodes=0,
            qualifies_real_game=False,
        )
        run.register_artifact(plan_file, "imitation_window_plan")
        run.register_artifact(profile_file, "measurement_profile")
        with _measurement_journal(run) as record_event:
            dataset = assemble_windows(root, plan)
            for ref, kind in [
                *((ref, "segmentation_review") for ref in plan.reviews),
                *((ref, "prior_evidence") for ref in plan.prior_evidence),
            ]:
                run.register_artifact(_inside(root, ref.path), kind)
            run.annotate(
                window_dataset_sha256=dataset["dataset_sha256"],
                window_assembly_summary=dataset["summary"],
                inherited_sources=dataset["sources"],
            )
            report = measure_unique_frames(
                root,
                dataset["windows"],
                HCRPixelMeasurer(profile),
                maximum_frames=maximum_frames,
                on_event=record_event,
            )
            report.update(
                window_dataset_sha256=dataset["dataset_sha256"],
                window_assembly_summary=dataset["summary"],
                inherited_sources=dataset["sources"],
                input_contract=config,
                geometry_profile=profile.profile_id,
                operation_journal="geometry-measurements.jsonl",
            )
            output = run.directory / "reviewed-geometry-diagnostic.json"
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_json(report) + "\n")
            run.register_artifact(output, "reviewed_geometry_diagnostic")
        run.finalize(episodes=0, environment_steps=0, training_steps=0, **report["summary"])
        run_id = run.run_id
    return {
        "run_id": run_id,
        "report": str(output),
        "summary": report["summary"],
        "verification": verify_run(_inside(root, plan.artifact_root), run_id),
    }
