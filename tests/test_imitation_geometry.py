import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from demo_dataset_fixtures import make_demo_source
from PIL import Image, UnidentifiedImageError

from gradientclimb.artifacts import sha256_file
from gradientclimb.datasets.geometry_diagnostic import (
    measure_unique_frames,
    publish_geometry_diagnostic,
)
from gradientclimb.experiments import verify_run


class Detector:
    def __init__(self):
        self.calls = 0

    def measure(self, rgb):
        self.calls += 1
        valid = bool(rgb[0, 0, 0])
        result = {
            name: {"valid": valid, "reason": "synthetic detector hypothesis"}
            for name in ("body", "wheels", "terrain")
        }
        return SimpleNamespace(as_dict=lambda: result)


def frame(tmp_path, index, value):
    path = tmp_path / f"frame{index}.png"
    Image.new("RGB", (8, 8), (value, 0, 0)).save(path)
    return {"path": path.name, "file_sha256": sha256_file(path), "frame_index": index}


def test_deduplicates_reused_images_but_preserves_invalid_measurements(tmp_path):
    one, two = frame(tmp_path, 1, 20), frame(tmp_path, 2, 0)
    detector = Detector()
    report = measure_unique_frames(
        tmp_path,
        [
            {"run_id": "recording", "frames": [one, two]},
            {"run_id": "recording", "frames": [two]},
        ],
        detector,
    )
    assert detector.calls == 2
    assert report["summary"]["window_frame_reference_uses"] == 3
    assert report["summary"]["detector_valid_body_frames"] == 1
    assert len(report["rows"]) == 2
    assert report["rows"][1]["window_reference_uses"] == 2
    assert report["summary"]["measured_geometry_accuracy"] is None
    assert report["summary"]["optimizer_updates"] == 0


def test_same_frame_identity_cannot_have_conflicting_bytes(tmp_path):
    reference = frame(tmp_path, 1, 20)
    with pytest.raises(ValueError, match="Conflicting references"):
        measure_unique_frames(
            tmp_path,
            [
                {"run_id": "recording", "frames": [reference]},
                {"run_id": "recording", "frames": [{**reference, "file_sha256": "a" * 64}]},
            ],
            Detector(),
        )


@pytest.mark.parametrize("change", ["bytes", "escape"])
def test_refuses_changed_or_escaping_image(tmp_path, change):
    reference = frame(tmp_path, 1, 20)
    if change == "bytes":
        Image.new("RGB", (8, 8)).save(tmp_path / reference["path"])
    else:
        reference["path"] = "../frame.png"
    detector = Detector()
    with pytest.raises(ValueError):
        measure_unique_frames(tmp_path, [{"run_id": "recording", "frames": [reference]}], detector)
    assert detector.calls == 0


def test_frame_limit_is_enforced_before_measurement(tmp_path):
    refs = [frame(tmp_path, i, 20) for i in range(2)]
    detector = Detector()
    with pytest.raises(ValueError, match="outside the diagnostic bound"):
        measure_unique_frames(
            tmp_path, [{"run_id": "recording", "frames": refs}], detector, maximum_frames=1
        )
    assert detector.calls == 0


def publication_source(tmp_path, *, invalid_image_index=None):
    def image_bytes(index):
        if index == invalid_image_index:
            return b"Sealed source bytes that are not a decodable image"
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), (index, 0, 0)).save(stream, format="PNG")
        return stream.getvalue()

    root, plan, review, *_ = make_demo_source(tmp_path, frame_bytes=image_bytes)
    profile = {
        "profile_id": "synthetic-geometry-mismatch",
        "expected_size": [16, 16],
        "scene_roi": [0, 0, 1, 1],
    }
    # A real measurer rejects these tiny fixture images without invoking OpenCV.
    (root / "research/profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return root, plan, review


def diagnostic_directory(root, source_run_id):
    directories = [p for p in (root / "artifacts/runs").iterdir() if p.name != source_run_id]
    assert len(directories) == 1
    return directories[0]


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_offline_accounting(record):
    for counter in ("episodes", "environment_steps", "training_steps", "optimizer_updates"):
        assert record[counter] == 0
    assert record["evaluation_results"] == []
    assert record["summary"]["evaluation_episodes"] == 0
    assert record["summary"]["real_game_evaluation_episodes"] == 0
    assert record["summary"]["policy_decisions"] == 0
    assert record["summary"]["training_clock_seconds"] == 0
    assert record["summary"]["resources"]["version"] == "resources-3.0"
    assert record["summary"]["inherited_sources"][0]["prior_cost"]["capture_wall_seconds"] == 1
    assert record["summary"]["inherited_sources"][0]["prior_cost"]["human_practice_seconds"] is None


def test_publisher_assembles_sealed_png_sources_and_keeps_diagnostic_costs_separate(tmp_path):
    root, _, review = publication_source(tmp_path)
    source_id = review["source"]["run_id"]
    source_seal = root / "artifacts/runs" / source_id / "seal.json"
    original_seal = source_seal.read_bytes()
    result = publish_geometry_diagnostic(root, "research/plan.json", "research/profile.json")
    directory = Path(result["report"]).parent
    record = json.loads((directory / "run.json").read_bytes())
    report = json.loads(Path(result["report"]).read_bytes())
    journal = json_rows(directory / "geometry-measurements.jsonl")
    completed = [event["measurement"] for event in journal if event["event"] == "frame_completed"]
    assert result["verification"]["valid"] and record["status"] == "completed"
    assert len(completed) == report["summary"]["unique_source_frames"] == 8
    assert completed == report["rows"]
    assert report["summary"]["window_frame_reference_uses"] == 20
    assert report["summary"]["detector_valid_body_frames"] == 0
    assert report["summary"]["measured_geometry_accuracy"] is None
    assert all(
        row["measurement"]["body"]["reason"] == "Profile geometry mismatch" for row in completed
    )
    progress = json_rows(directory / "progress.jsonl")[-1]["geometry_progress"]
    assert progress["planned_source_frames"] == progress["attempted_source_frames"] == 8
    assert progress["completed_source_frames"] == 8 and progress["failed_source_frames"] == 0
    assert progress == record["summary"]["geometry_progress"]
    assert any(
        a["kind"] == "reviewed_geometry_operation_journal" for a in record["artifact_manifest"]
    )
    assert (directory / "evaluations.jsonl").read_bytes() == b""
    assert_offline_accounting(record)
    assert source_seal.read_bytes() == original_seal
    assert verify_run(root / "artifacts", source_id)["valid"]


@pytest.mark.parametrize("failure", ["decode", "detector", "interrupt"])
def test_publisher_seals_partial_operations_and_counts_after_failure(
    tmp_path, monkeypatch, failure
):
    root, _, review = publication_source(
        tmp_path, invalid_image_index=2 if failure == "decode" else None
    )
    source_id = review["source"]["run_id"]
    source_seal = root / "artifacts/runs" / source_id / "seal.json"
    original_seal = source_seal.read_bytes()
    exception = {
        "decode": UnidentifiedImageError,
        "detector": RuntimeError,
        "interrupt": KeyboardInterrupt,
    }[failure]

    class FailingDetector(Detector):
        def measure(self, rgb):
            if self.calls == 2:
                # These rows must be visible before the third operation fails.
                directory = diagnostic_directory(root, source_id)
                events = json_rows(directory / "geometry-measurements.jsonl")
                assert sum(e["event"] == "frame_completed" for e in events) == 2
                assert events[-1]["event"] == "frame_started"
                counts = json_rows(directory / "progress.jsonl")[-1]["geometry_progress"]
                assert counts["attempted_source_frames"] == 3
                assert counts["completed_source_frames"] == 2
                raise exception("Synthetic third-frame interruption")
            return super().measure(rgb)

    if failure != "decode":
        monkeypatch.setattr(
            "gradientclimb.datasets.geometry_diagnostic.HCRPixelMeasurer",
            lambda _profile: FailingDetector(),
        )
    with pytest.raises(exception):
        publish_geometry_diagnostic(root, "research/plan.json", "research/profile.json")
    directory = diagnostic_directory(root, source_id)
    record = json.loads((directory / "run.json").read_bytes())
    events = json_rows(directory / "geometry-measurements.jsonl")
    assert record["status"] == ("cancelled" if failure == "interrupt" else "failed")
    assert sum(e["event"] == "frame_completed" for e in events) == 2
    assert events[-1]["event"] == "frame_failed" and events[-1]["frame_index"] == 2
    assert events[-1]["failed_at_stage"] == (
        "image_decode" if failure == "decode" else "geometry_measurement"
    )
    progress = record["summary"]["geometry_progress"]
    assert progress["planned_source_frames"] == 8
    assert progress["attempted_source_frames"] == 3
    assert progress["completed_source_frames"] == 2
    assert progress["failed_source_frames"] == 1
    assert not (directory / "reviewed-geometry-diagnostic.json").exists()
    assert any(
        a["kind"] == "reviewed_geometry_operation_journal" for a in record["artifact_manifest"]
    )
    assert verify_run(root / "artifacts", directory.name)["valid"]
    assert_offline_accounting(record)
    assert source_seal.read_bytes() == original_seal
    assert verify_run(root / "artifacts", source_id)["valid"]
