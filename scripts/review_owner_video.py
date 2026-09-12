"""Extract preregistered sparse source frames and their actual presentation times."""

import argparse
import json
import math
import re
import time
from pathlib import Path

from acquire_owner_video import bounded_process

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import _inside
from gradientclimb.experiments import RunRecorder, load_run, verify_run


def review(root, protocol_name):
    root = root.absolute()
    path = _inside(root, protocol_name)
    plan = json.loads(path.read_bytes())
    if plan["schema_version"] != "owner-video-review-3.0" or plan["status"] != "registered":
        raise ValueError("Expected registered sparse-review protocol")
    targets = plan["requested_timestamps_seconds"]
    if (
        not targets
        or any(type(t) not in (int, float) or not math.isfinite(t) for t in targets)
        or targets[0] != 0
        or targets != sorted(set(targets))
        or len(targets) > 40
        or targets[-1] >= 300
    ):
        raise ValueError(
            "Expected distinct ordered source times, beginning at zero, within five minutes"
        )
    if plan["wall_budget_seconds"] != 120 or plan["output_budget_bytes"] != 128 * 1024**2:
        raise ValueError("Sparse review requires its fixed 120-second/128-MiB budget")
    store = root / "artifacts"
    digest = sha256_file(path)
    for start in store.glob("runs/*/run-start.json"):
        if json.loads(start.read_bytes()).get("configuration", {}).get("protocol_sha256") == digest:
            raise ValueError("Review protocol already consumed; register any successor")
    with RunRecorder(
        store,
        plan["experiment_id"],
        {**plan, "protocol_sha256": digest},
        algorithm="fixed-sparse-video-review",
        environment="owner-public-video",
        evidence_domain="construction_external_video",
        source_root=root,
        qualifies_real_game=False,
    ) as run:
        deadline = time.perf_counter() + max(0, 120 - run.elapsed_seconds)
        run.annotate(
            environment_steps=0,
            episodes=0,
            optimizer_updates=0,
            policy_decisions=0,
            new_real_interaction_seconds=0,
            synchronized_action_labels=0,
            training_eligible=False,
            prior_acquisition_run_id=plan["source_run_id"],
            actor_observations=0,
            note="Sparse decoded archive frames for human construction review. No episode segmentation, action inference, perception fitting or policy evaluation.",
        )
        run.register_artifact(path, "owner_video_review_protocol")
        if not verify_run(store, plan["source_run_id"])["valid"]:
            raise ValueError("Source video acquisition seal failed")
        source_run = load_run(store, plan["source_run_id"])
        source = _inside(root, plan["source_video"]["path"])
        if sha256_file(source) != plan["source_video"]["sha256"] or not source.is_relative_to(
            store / "runs" / plan["source_run_id"]
        ):
            raise ValueError("Source video identity mismatch")
        if not any(
            a["sha256"] == plan["source_video"]["sha256"] for a in source_run["artifact_manifest"]
        ):
            raise ValueError("Video must belong to sealed source acquisition")
        ffmpeg, ffprobe = (Path(plan[k]["path"]) for k in ("ffmpeg", "ffprobe"))
        if any(
            sha256_file(tool) != plan[key]["sha256"]
            for key, tool in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        ):
            raise ValueError("Media tool identity mismatch")
        output = run.directory / "review"
        output.mkdir()
        failure = None
        try:
            bounded_process(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(source),
                ],
                output,
                deadline,
                32 * 1024**2,
                "probe.json",
            )
            probe = json.loads((output / "probe.json").read_bytes())
            streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
            if len(streams) != 1:
                raise ValueError("Expected exactly one video stream")
            # Keep the first original frame crossing each timestamp, without interpolation.
            expression = "isnan(prev_selected_t)" + "".join(
                f"+gte(t,{t})*lt(prev_selected_t,{t})" for t in targets[1:]
            )
            bounded_process(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-nostdin",
                    "-i",
                    str(source),
                    "-an",
                    "-vf",
                    f"select='{expression}',showinfo",
                    "-fps_mode",
                    "vfr",
                    "-frames:v",
                    str(len(targets)),
                    "-compression_level",
                    "2",
                    str(output / "frame-%03d.png"),
                ],
                output,
                deadline,
                32 * 1024**2,
                "extract.log",
            )
            stamps = [
                float(x)
                for x in re.findall(
                    r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)",
                    (output / "extract.log").read_text(errors="replace"),
                )
            ]
            frames = sorted(output.glob("frame-*.png"))
            if len(stamps) != len(targets) or len(frames) != len(targets):
                raise ValueError(
                    "Sparse extraction incomplete; retain all partial frames and timestamps"
                )
            if any(
                actual < requested or (i + 1 < len(targets) and actual >= targets[i + 1])
                for i, (requested, actual) in enumerate(zip(targets, stamps))
            ):
                raise ValueError(
                    "Extracted source times do not satisfy declared first-crossing intervals"
                )
            rows = [
                {
                    "requested_seconds": requested,
                    "source_pts_seconds": actual,
                    "path": f.relative_to(root).as_posix(),
                    "sha256": sha256_file(f),
                }
                for requested, actual, f in zip(targets, stamps, frames)
            ]
            report = {
                "schema_version": "owner-video-review-report-3.0",
                "source_run_id": plan["source_run_id"],
                "source_video": plan["source_video"],
                "probe": probe,
                "frames": rows,
                "selected_frame_count": len(frames),
                "probe_video_frames_read": int(streams[0]["nb_read_frames"]),
                "source_clock": "Media presentation timestamps, not new real interaction, simulator time or learner wall clock",
                "note": "ffprobe counted video frames in a separate decoding pass; ffmpeg source decode count is not measured by selected output count. Neither is an actor decision or evaluation episode.",
            }
            (output / "review.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            run.annotate(
                selected_archive_frames=len(frames),
                probe_video_frames_read=report["probe_video_frames_read"],
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            errors = []
            for f in output.iterdir():
                if f.is_file():
                    try:
                        run.register_artifact(f, "owner_video_review_evidence")
                    except BaseException as error:  # noqa: BLE001 - retain remaining evidence
                        errors.append(error)
            if errors:
                if failure:
                    failure.add_note("Review retention also failed: " + str(errors))
                else:
                    raise RuntimeError("Review retention failed") from errors[0]
        run.finalize(
            selected_archive_frames=len(frames), training_steps=0, episodes=0, optimizer_updates=0
        )
    return {"run_id": run.run_id, "verification": verify_run(store, run.run_id)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.project_root, args.protocol), indent=2))
