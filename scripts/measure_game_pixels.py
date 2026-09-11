"""Measure already saved, hashed frames; this utility has no live capture/input path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments.recorder import RunRecorder, verify_run
from gradientclimb.perception.annotations import FrameAnnotation, evaluate_measurements
from gradientclimb.perception.measurements import (
    HCRPixelMeasurer,
    MeasurementProfile,
    TerrainMotionTracker,
)


def load_profile(path: Path) -> MeasurementProfile:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["expected_size"] = tuple(config["expected_size"])
    return MeasurementProfile(**config)


def measure_manifest(
    frames_path: Path,
    profile_path: Path,
    artifact_root: Path,
    *,
    session_id: str,
    max_frames=500,
    annotations: Path | None = None,
    split="heldout",
) -> dict:
    if not session_id or not 1 <= max_frames <= 10000:
        raise ValueError("A session ID and a frame bound in 1..10000 are required")
    frames_path, profile_path = frames_path.resolve(), profile_path.resolve()
    profile = load_profile(profile_path)
    measurer, tracker = HCRPixelMeasurer(profile), TerrainMotionTracker(HCRPixelMeasurer(profile))
    rows = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > max_frames:
        raise ValueError("Manifest exceeds max_frames; select an explicit bounded subset")
    configuration = {
        "profile": json.loads(profile_path.read_text(encoding="utf-8")),
        "profile_sha256": sha256_file(profile_path),
        "frames_manifest_sha256": sha256_file(frames_path),
        "session_id": session_id,
        "max_frames": max_frames,
        "coordinate_system": "analyzed_screen_pixels",
        "generalization_accuracy": None,
    }
    outputs, previous_hash = [], None
    with RunRecorder(
        artifact_root,
        "offline-game-pixel-measurements",
        configuration,
        algorithm="hcr-profile-pixels-1",
        environment="saved-hcr-pixels",
        evidence_domain="real_game_pixel_measurement",
        telemetry_interval_seconds=0,
    ) as run:
        run.register_artifact(profile_path, "pixel_measurement_profile")
        run.register_artifact(frames_path, "source_frame_manifest")
        for index, row in enumerate(rows):
            if not row.get("path"):
                tracker.reset()
                previous_hash = None
                outputs.append(
                    {
                        "frame_index": index,
                        "valid_source": False,
                        "reason": "No recorded frame; temporal continuity broken",
                    }
                )
                continue
            source = (frames_path.parent / row["path"]).resolve()
            if not source.is_relative_to(frames_path.parent):
                raise ValueError("Frame paths must stay within the manifest directory")
            digest = sha256_file(source)
            if row.get("sha256") != digest:
                raise ValueError(f"Frame {index} has absent or mismatched recorded SHA-256")
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            if (rgb.shape[1], rgb.shape[0]) != profile.expected_size:
                raise ValueError(
                    "Frame does not match profile; explicitly derive/register resized images before measurement"
                )
            source_record = run.register_artifact(
                source,
                "real_game_measurement_source",
                {"session_id": session_id, "frame_index": index},
            )
            measurement = measurer.measure(rgb)
            timestamp = row.get("timestamp_ns")
            if timestamp is None:
                camera = {
                    "valid": False,
                    "quality": 0.0,
                    "reason": "No capture timestamp; camera interval not measured",
                }
                tracker.reset()
            else:
                camera = tracker.update(rgb, measurement, timestamp)
            outputs.append(
                {
                    "frame_index": index,
                    "session_id": session_id,
                    "valid_source": True,
                    "source_sha256": digest,
                    "source_artifact_id": source_record["artifact_id"],
                    "timestamp_ns": timestamp,
                    "previous_source_sha256": previous_hash,
                    "measurement": measurement.as_dict(),
                    "camera": camera,
                }
            )
            previous_hash = digest if timestamp is not None else None
        path = run.directory / "pixel-measurements.jsonl"
        path.write_text(
            "".join(json.dumps(output, allow_nan=False) + "\n" for output in outputs),
            encoding="utf-8",
        )
        run.register_artifact(path, "pixel_measurements", {"quality_is_probability": False})
        measured = [row for row in outputs if row["valid_source"]]
        summary = {
            "source_frames": len(measured),
            "wheel_supported_frames": sum(
                row["measurement"]["wheels"]["valid"] for row in measured
            ),
            "camera_supported_intervals": sum(row["camera"]["valid"] for row in measured),
            "heldout_accuracy_measured": False,
        }
        if annotations:
            labels = [
                FrameAnnotation.model_validate_json(line)
                for line in annotations.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = evaluate_measurements(
                {row["source_sha256"]: row for row in measured}, labels, split=split
            )
            report_path = run.directory / "pixel-label-evaluation.json"
            report_path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            run.register_artifact(annotations, "independent_pixel_labels")
            run.register_artifact(report_path, "pixel_label_evaluation", {"split": split})
            summary["heldout_accuracy_measured"] = split == "heldout"
        for name in ("source_frames", "wheel_supported_frames", "camera_supported_intervals"):
            run.metric(name, summary[name])
        run.finalize(episodes=0, environment_steps=0, training_steps=0, **summary)
        result = {"run_id": run.run_id, "measurements": str(path), **summary}
    result["verification"] = verify_run(artifact_root, result["run_id"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/perception/hcr-discovery-wrapper.json")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--session-id")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--split", choices=["train", "heldout"], default="heldout")
    parser.add_argument("--write-annotation-schema", type=Path)
    args = parser.parse_args()
    if args.write_annotation_schema:
        args.write_annotation_schema.write_text(
            json.dumps(FrameAnnotation.model_json_schema(), indent=2) + "\n", encoding="utf-8"
        )
    else:
        if not args.frames or not args.session_id:
            parser.error("--frames and --session-id are required for measurement")
        print(
            json.dumps(
                measure_manifest(
                    args.frames,
                    args.profile,
                    args.artifacts,
                    session_id=args.session_id,
                    max_frames=args.max_frames,
                    annotations=args.annotations,
                    split=args.split,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
