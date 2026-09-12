"""Shared synthetic sealed demonstration fixture; never reads real recordings."""

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.experiments import RunRecorder


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def reference(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def make_demo_source(
    tmp_path,
    *,
    status="failed",
    purpose="imitation",
    starts=None,
    change_frames=None,
    change_controls=None,
    partial_frames=b"",
    partial_controls=b"",
    frame_bytes=None,
):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    config = {
        "version": "human-demonstration-3.0",
        "session_id": "synthetic-session",
        "purpose": purpose,
        "source_partition": purpose,
        "input_injection": False,
        "controls": {"gas": "VK_RIGHT", "brake": "VK_LEFT"},
        "benchmark_excluded_from_training": purpose == "human-benchmark",
        "operator_configuration_note": "Synthetic owner assertion only",
        "configuration_independently_verified": False,
        "prior_knowledge": {"human_practice_seconds": None, "system_prior": "fixture"},
    }
    frames, controls = [], []
    starts = starts or [1_000_000_000 + i * 100_000_000 for i in range(8)]
    with RunRecorder(
        root / "artifacts",
        "synthetic-demo",
        config,
        algorithm="human",
        telemetry_interval_seconds=0,
        source_root=root,
    ) as run:
        folder = run.directory / "demonstration"
        folder.mkdir()
        for index, start in enumerate(starts):
            path = folder / f"frame-{index:06d}.png"
            path.write_bytes(
                frame_bytes(index) if frame_bytes else f"synthetic image bytes {index}".encode()
            )
            frames.append(
                {
                    "frame_index": index,
                    "started_ns": start,
                    "completed_ns": start + 10_000_000,
                    "timestamp_ns": start + 5_000_000,
                    "observation_ready_ns": start + 20_000_000,
                    "path": path.name,
                    "file_sha256": sha256_file(path),
                    "state_evidence": {"state": "unknown"},  # Review, not template, governs.
                }
            )
        for index, tick in enumerate(
            range(starts[0] - 9_000_000, starts[-1] + 90_000_000, 10_000_000)
        ):
            controls.append(
                {
                    "sample_index": index,
                    "started_ns": tick,
                    "completed_ns": tick + 1_000_000,
                    "gas": bool(index % 2),
                    "brake": bool(index % 3 == 0),
                }
            )
        if change_frames:
            change_frames(frames)
        if change_controls:
            change_controls(controls)
        for name, rows, tail in (
            ("frames.jsonl", frames, partial_frames),
            ("controls.jsonl", controls, partial_controls),
        ):
            (folder / name).write_bytes(
                b"".join((canonical_json(row) + "\n").encode() for row in rows) + tail
            )
        for path in sorted(folder.iterdir()):
            run.register_artifact(path, "human_demonstration_payload")
        run.finalize(status=status, demonstration={"capture_wall_clock_seconds": 1.0}, episodes=0)
    evidence = [reference(root, folder / f"frame-{i:06d}.png") for i in range(len(frames))]
    profile = {
        "profile_id": "synthetic-profile",
        "vehicle_name": None,
        "map_name": None,
        "game_build": None,
        "upgrades": [],
        "evidence": [evidence[0]],
        "field_evidence": {},
        "limitations": "Names are not visually established",
    }
    review = {
        "review_id": "synthetic-review",
        "reviewer": "fixture",
        "reviewed_at": "2026-09-12T15:00:00Z",
        "source": {
            "run_id": run.run_id,
            "session_id": config["session_id"],
            "run_sha256": sha256_file(run.directory / "run.json"),
            "configuration_sha256": sha256_file(run.directory / "config.json"),
            "frames": reference(root, folder / "frames.jsonl"),
            "controls": reference(root, folder / "controls.jsonl"),
        },
        "accepted_capture_status": status,
        "capture_failure_review_note": "Capture stopped later; this span was reviewed",
        "profile": profile,
        "segments": [
            {
                "segment_id": "playing-1",
                "episode_id": None,
                "first_frame_index": 0,
                "end_frame_index_exclusive": len(frames),
                "started_ns": starts[0],
                "end_ns_exclusive": starts[-1] + 50_000_000,
                "start_context": "unknown",
                "end_context": "unknown",
                "frame_evidence": evidence,
                "note": "Each synthetic frame covered",
            }
        ],
    }
    ledger = {
        "ledger_id": "frozen-partitions",
        "created_at": "2026-09-12T14:00:00Z",
        "sessions": [
            {
                "run_id": run.run_id,
                "session_id": config["session_id"],
                "source_purpose": purpose,
                "split": "train" if purpose == "imitation" else purpose,
            }
        ],
        "note": "Whole-session assignment before fitting",
    }
    plan = {
        "dataset_id": "synthetic-windows",
        "split": "train" if purpose == "imitation" else "development",
        "partition_ledger": {},
        "reviews": [],
        "prior_evidence": [reference(root, run.directory / "run.json")],
        "note": "Synthetic construction-only windows",
    }

    def freeze():
        review_path, ledger_path = (
            root / "research/review.json",
            root / "research/partitions.json",
        )
        write_json(review_path, review)
        write_json(ledger_path, ledger)
        plan["reviews"] = [reference(root, review_path)]
        plan["partition_ledger"] = reference(root, ledger_path)
        write_json(root / "research/plan.json", plan)
        return plan

    freeze()
    return root, plan, review, ledger, freeze, frames, controls
