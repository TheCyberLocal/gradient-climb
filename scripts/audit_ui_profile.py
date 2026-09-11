"""Offline replay audit of a UI profile over every stored real frame; no native input.

Every PNG registered by a sealed run as a reset, terminal, gameplay sample, capture
or reference frame is classified with the candidate profile. The audit reports the
classification distribution, disagreements with the state recorded at capture time,
and every advertisement chrome hit, so a profile change is validated against
positive and negative evidence before it is used live. Results are sealed as a
zero-episode canonical run; source runs are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.control.game_adapter import GameUIProfile, GameUIRecognizer
from gradientclimb.experiments import RunRecorder, list_runs

FRAME_KINDS = {
    "reset_frame",
    "terminal_frame",
    "real_game_frame",
    "real_ui_reference",
    "ui_reference",
    "frame",
}


def stored_frames(root: Path, *, max_frames: int) -> list[dict]:
    rows = []
    for record in list_runs(root):
        if record["status"] == "running":
            continue
        directory = root / "runs" / record["run_id"]
        for artifact in record["artifact_manifest"]:
            if artifact["kind"] in FRAME_KINDS and artifact["path"].endswith(".png"):
                rows.append(
                    {
                        "run_id": record["run_id"],
                        "experiment_id": record["experiment_id"],
                        "kind": artifact["kind"],
                        "path": directory / artifact["path"],
                        "sha256": artifact["sha256"],
                        "recorded_state": artifact["metadata"].get("state"),
                        "operator_label": artifact["metadata"].get("operator_label"),
                    }
                )
    rows.sort(key=lambda row: (row["run_id"], str(row["path"])))
    if len(rows) > max_frames:
        raise ValueError(f"{len(rows)} frames exceed the declared bound {max_frames}")
    return rows


def classify_frames(recognizer: GameUIRecognizer, rows: list[dict]) -> list[dict]:
    results = []
    width, height = recognizer.profile.expected_size
    for row in rows:
        with Image.open(row["path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if rgb.shape != (height, width, 3):
            results.append({**row, "path": str(row["path"]), "skipped": "geometry"})
            continue
        started = time.perf_counter()
        frame = CapturedFrame(rgb, 1, 2, "audit", ClientRect(0, 0, width, height), "file", 0.0)
        observation = recognizer.observe(frame)
        results.append(
            {
                **row,
                "path": str(row["path"]),
                "state": observation.state,
                "variant": observation.variant,
                "confidence": observation.confidence,
                "controls": [c.name for c in observation.controls],
                "processing_ms": (time.perf_counter() - started) * 1e3,
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    classified = [r for r in results if "state" in r]
    by_state = Counter(r["state"] for r in classified)
    by_variant = Counter(r["variant"] for r in classified if r["variant"])
    recorded = [r for r in classified if r["recorded_state"]]
    disagreements = [
        r
        for r in recorded
        if r["recorded_state"] not in {"unknown", r["state"]}
        and not (r["recorded_state"] == "advertisement" and r["state"] == "advertisement")
    ]
    game_states = {"playing", "paused", "revive_offer", "bonus_offer", "result", "tune"}
    chrome_false_positives = [
        r for r in recorded if r["recorded_state"] in game_states and r["state"] == "advertisement"
    ]
    return {
        "frames": len(results),
        "classified": len(classified),
        "skipped_geometry": len(results) - len(classified),
        "by_state": dict(sorted(by_state.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "recorded_state_frames": len(recorded),
        "disagreements_with_recorded_state": len(disagreements),
        "disagreement_examples": [
            {k: r[k] for k in ("path", "recorded_state", "state", "variant", "confidence")}
            for r in disagreements[:40]
        ],
        "advertisement_hits_on_recorded_game_states": len(chrome_false_positives),
        "advertisement_hit_examples": [
            {k: r[k] for k in ("path", "recorded_state", "state", "variant", "confidence")}
            for r in chrome_false_positives[:40]
        ],
        "unknown_recorded_now_recognized": Counter(
            r["state"] for r in recorded if r["recorded_state"] == "unknown"
        ),
        "median_processing_ms": float(np.median([r["processing_ms"] for r in classified]))
        if classified
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ui-profile", type=Path, default=Path("configs/perception/hcr-reset-ui.json")
    )
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--no-record", action="store_true", help="Print only; seal no run")
    args = parser.parse_args()
    profile = GameUIProfile.model_validate_json(args.ui_profile.read_text(encoding="utf-8"))
    recognizer = GameUIRecognizer(profile, args.root)
    rows = stored_frames(args.root, max_frames=args.max_frames)
    results = classify_frames(recognizer, rows)
    summary = summarize(results)
    if args.no_record:
        print(json.dumps(summary, indent=2, default=str))
        return
    config = {
        "ui_profile": str(args.ui_profile),
        "ui_profile_sha256": sha256_file(args.ui_profile),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "frame_kinds": sorted(FRAME_KINDS),
        "frames": len(rows),
        "scope": "offline replay of stored frames; no capture, no input, no run mutated",
    }
    with RunRecorder(
        args.root,
        "ui-profile-audit",
        config,
        algorithm="template-recognizer-audit",
        environment="stored_real_frames",
        evidence_domain="real_game_offline",
        qualifies_real_game=False,
    ) as run:
        run.register_artifact(args.ui_profile, "ui_profile")
        path = run.directory / "classifications.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in results:
                stream.write(json.dumps(row, default=str) + "\n")
        run.register_artifact(path, "ui_profile_audit_rows")
        summary_path = run.directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        run.register_artifact(summary_path, "ui_profile_audit_summary")
        run.finalize(
            status="completed",
            episode_count=0,
            frames=len(rows),
            classified=summary["classified"],
            advertisement_hits_on_recorded_game_states=summary[
                "advertisement_hits_on_recorded_game_states"
            ],
            disagreements_with_recorded_state=summary["disagreements_with_recorded_state"],
            qualification_evidence=False,
        )
        print(json.dumps({"run_id": run.run_id, **summary}, indent=2, default=str))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
