"""Evaluate frozen local readers against predeclared labels and saved frames only."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments.recorder import RunRecorder, verify_run
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.scoring import ResultDistanceReader


def evaluate(args):
    gameplay_path = args.bank / "gameplay-glyphs/hud-glyphs.json"
    result_path = args.bank / "result-reader.json"
    gameplay = HUDDigitReader.from_manifest(gameplay_path)
    terminal = ResultDistanceReader.from_manifest(result_path)
    labels = json.loads(args.labels.read_text())["labels"]
    train_hashes = {g["source_sha256"] for g in terminal.reader.glyphs + gameplay.glyphs}
    if train_hashes & {label["source_sha256"] for label in labels}:
        raise ValueError("Heldout sources overlap glyph construction")
    rows = [json.loads(line) for line in args.frames.read_text().splitlines() if line.strip()]
    if len(rows) > args.max_frames:
        raise ValueError("Frame bound exceeded")
    if train_hashes & {row["sha256"] for row in rows}:
        raise ValueError("Independent trajectory includes glyph construction frames")
    config = {
        "gameplay_manifest_sha256": sha256_file(gameplay_path),
        "result_manifest_sha256": sha256_file(result_path),
        "heldout_labels_sha256": sha256_file(args.labels),
        "frames_sha256": sha256_file(args.frames),
        "max_frames": args.max_frames,
        "labels_predeclared": True,
        "all_trajectory_frames_source_disjoint": True,
        "evidence_limit": "Selected correlated frame labels and one result; no population accuracy claim",
    }
    with RunRecorder(
        args.artifacts,
        "independent-scoring-reader-evaluation",
        config,
        algorithm="frozen-glyph-readers",
        environment="saved-hcr-pixels",
        evidence_domain="real_game_readout_evaluation",
        telemetry_interval_seconds=0,
    ) as run:
        for path in [gameplay_path, result_path, args.labels, args.frames, Path(__file__)]:
            run.register_artifact(path, "readout_evaluation_source")
        evaluated, readings = [], []
        for label in labels:
            source = Path(label["source"])
            if sha256_file(source) != label["source_sha256"]:
                raise ValueError("Heldout labeled source changed")
            rgb = np.asarray(Image.open(source).convert("RGB"))
            if label["field"] == "gameplay_progress":
                result = gameplay.read(rgb)
                value = result.get("hud_displayed_progress_meters")
            elif label["field"] == "result_distance":
                result = terminal.read(rgb, result_state_confirmed=True)
                value = result.get("distance_meters")
            else:
                raise ValueError("Unknown field label")
            evaluated.append(
                {
                    **label,
                    "prediction": result,
                    "exact_match": bool(result["valid"] and value == int(label["text"])),
                }
            )
            run.register_artifact(source, "independent_labeled_readout_frame")
        for row in rows:
            source = (args.frames.parent / row["path"]).resolve()
            if (
                not source.is_relative_to(args.frames.parent.resolve())
                or sha256_file(source) != row["sha256"]
            ):
                raise ValueError("Trajectory source path or hash mismatch")
            rgb = np.asarray(Image.open(source).convert("RGB"))
            started = time.perf_counter()
            result = gameplay.read(rgb)
            readings.append(
                {
                    "frame_index": row["frame_index"],
                    "source_sha256": row["sha256"],
                    "timestamp_ns": row["timestamp_ns"],
                    "reading": result,
                    "processing_ms": (time.perf_counter() - started) * 1000,
                }
            )
            run.register_artifact(source, "independent_trajectory_frame")
        accepted = [
            row["reading"]["hud_displayed_progress_meters"]
            for row in readings
            if row["reading"]["valid"]
        ]
        report = {
            "labels": evaluated,
            "exact_matches": sum(row["exact_match"] for row in evaluated),
            "labeled_frames": len(evaluated),
            "trajectory_frames": len(readings),
            "accepted_readings": len(accepted),
            "unknown_readings": len(readings) - len(accepted),
            "last_reading": readings[-1]["reading"] if readings else None,
            "maximum_accepted": max(accepted) if accepted else None,
            "accepted_sequence_decreases": sum(a > b for a, b in itertools.pairwise(accepted)),
            "processing_ms_median": float(np.median([row["processing_ms"] for row in readings])),
            "performance_qualification": False,
        }
        for name, value in (("readout-evaluation.json", report), ("hud-readings.json", readings)):
            path = run.directory / name
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            run.register_artifact(path, "independent_readout_results")
        run.metric("readout_exact_matches", report["exact_matches"])
        run.metric("accepted_readings", report["accepted_readings"])
        run.finalize(
            episodes=0,
            training_steps=0,
            environment_steps=0,
            **{key: value for key, value in report.items() if key != "labels"},
        )
        result = {"run_id": run.run_id, **report}
    result["verification"] = verify_run(args.artifacts, result["run_id"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-frames", type=int, default=500)
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
