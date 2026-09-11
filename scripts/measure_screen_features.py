"""Bounded offline screen-to-observation run; never captures or sends live input."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments.recorder import RunRecorder, verify_run
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile
from gradientclimb.perception.screen_features import ScreenFeatureBridge


def measure(args):
    profile = MeasurementProfile(**json.loads(args.profile.read_text()))
    hud_reader = HUDDigitReader.from_manifest(args.hud)
    bridge = ScreenFeatureBridge(HCRPixelMeasurer(profile), hud_reader, history=args.history)
    rows = [json.loads(line) for line in args.frames.read_text().splitlines() if line.strip()]
    if not 1 <= len(rows) <= args.max_frames:
        raise ValueError("Frame count outside explicit bounds")
    labels = json.loads(args.hud_labels.read_text()) if args.hud_labels else None
    glyph_training = {glyph["source_sha256"] for glyph in hud_reader.glyphs}
    if labels and any(label["source_sha256"] in glyph_training for label in labels["labels"]):
        raise ValueError("HUD test source overlaps glyph training source")
    os_events = json.loads(args.os_events.read_text()) if args.os_events else []
    states, code = [], 0
    for event in os_events:
        if event.get("delivered") != len(event.get("events", [])):
            raise ValueError("Cannot reconstruct an OS state from partial/unknown delivery")
        for key, down in event["events"]:
            if key not in (0x25, 0x27):
                raise ValueError("Unexpected key in pedal trace")
            bit = 1 if key == 0x27 else 2
            code = code | bit if down else code & ~bit
        states.append((event["completed_ns"], code))
    config = {
        "frames_sha256": sha256_file(args.frames),
        "profile_sha256": sha256_file(args.profile),
        "hud_manifest_sha256": sha256_file(args.hud),
        "schema": bridge.schema,
        "schema_id": bridge.schema_id,
        "session_id": args.session_id,
        "heldout_geometry_accuracy": None,
        "max_frames": args.max_frames,
        "action_semantics": "preceding OS insertion state; game acknowledgment is separate",
    }
    outputs, timings = [], []
    with RunRecorder(
        args.artifacts,
        "offline-screen-feature-bridge",
        config,
        algorithm="screen-relative-features-1",
        environment="saved-hcr-pixels",
        evidence_domain="real_game_pixel_measurement",
        telemetry_interval_seconds=0,
    ) as run:
        for source, kind in (
            (args.frames, "frame_manifest"),
            (args.profile, "measurement_profile"),
            (args.hud, "hud_manifest"),
            (Path(__file__), "analysis_source"),
        ):
            run.register_artifact(source, kind)
        for source in (args.os_events, args.hud_labels):
            if source:
                run.register_artifact(source, "source_labels_or_events")
        for glyph in hud_reader.glyphs:
            run.register_artifact(args.hud.parent / glyph["file"], "hud_glyph")
        for row in rows:
            source = (args.frames.parent / row["path"]).resolve()
            if (
                not source.is_relative_to(args.frames.parent.resolve())
                or sha256_file(source) != row["sha256"]
            ):
                raise ValueError("Frame escaped manifest directory or SHA-256 changed")
            rgb = np.asarray(Image.open(source).convert("RGB"))
            preceding = [state for state in states if state[0] <= row["started_ns"]]
            actual_code = preceding[-1][1] if preceding else (0 if args.os_events else None)
            age = (row["started_ns"] - preceding[-1][0]) / 1e9 if preceding else None
            started = time.perf_counter()
            observation = bridge.observe(
                rgb,
                row["timestamp_ns"],
                previous_action_code=actual_code,
                action_age_seconds=age,
                episode_elapsed_seconds=row.get("elapsed_seconds"),
            )
            timings.append((time.perf_counter() - started) * 1000)
            outputs.append(
                {
                    "frame_index": row["frame_index"],
                    "source_sha256": row["sha256"],
                    "capture_started_ns": row["started_ns"],
                    "capture_completed_ns": row["completed_ns"],
                    "processing_ms": timings[-1],
                    **observation.as_dict(),
                }
            )
            run.register_artifact(source, "real_game_frame")
        report = {
            "source_frames": len(outputs),
            "geometry_valid_frames": sum(r["geometry_valid"] for r in outputs),
            "hud_accepted_frames": sum(r["hud"]["valid"] for r in outputs),
            "hud_unknown_frames": sum(not r["hud"]["valid"] for r in outputs),
            "feature_dimension": bridge.observation_dim,
            "processing_ms_median": float(np.median(timings)),
            "processing_ms_p95": float(np.quantile(timings, 0.95)),
            "processing_ms_max": max(timings),
            "measured_geometry_accuracy": None,
            "score_semantics": "displayed HUD values only; not a qualified evaluation episode",
        }
        accepted = [r["hud"]["hud_displayed_progress_meters"] for r in outputs if r["hud"]["valid"]]
        report.update(
            first_accepted_hud=accepted[0] if accepted else None,
            last_accepted_hud=accepted[-1] if accepted else None,
            maximum_accepted_hud=max(accepted) if accepted else None,
        )
        if labels:
            by_hash = {row["source_sha256"]: row for row in outputs}
            evaluated = []
            for label in labels["labels"]:
                result = by_hash[label["source_sha256"]]["hud"]
                evaluated.append(
                    {
                        **label,
                        "accepted": result["valid"],
                        "prediction": result.get("hud_displayed_progress_meters"),
                        "exact_match": result.get("valid", False)
                        and result.get("text") == label["text"],
                    }
                )
            report["independent_hud_labels"] = evaluated
            report["hud_label_exact_matches"] = sum(label["exact_match"] for label in evaluated)
        output_path = run.directory / "screen-observations.jsonl"
        output_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in outputs))
        report_path = run.directory / "screen-feature-report.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        run.register_artifact(output_path, "screen_observations")
        run.register_artifact(report_path, "offline_measurement_report")
        for name in (
            "source_frames",
            "geometry_valid_frames",
            "hud_accepted_frames",
            "processing_ms_median",
        ):
            run.metric(name, report[name])
        run.finalize(episodes=0, environment_steps=0, training_steps=0, **report)
        result = {"run_id": run.run_id, "observations": str(output_path), **report}
    result["verification"] = verify_run(args.artifacts, result["run_id"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/perception/hcr-discovery-wrapper.json")
    )
    parser.add_argument("--hud", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hud-labels", type=Path)
    parser.add_argument("--os-events", type=Path)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    print(json.dumps(measure(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
