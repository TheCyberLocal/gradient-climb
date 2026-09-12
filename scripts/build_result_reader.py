"""Extend a frozen result-distance reader with manually labeled result frames.

Every source frame must come from a sealed run and match its recorded hash. The
labels are declared before this script reads anything numeric; the frames become
construction evidence for the resulting bank, so they can never serve as held-out
evaluation of it. The output is a new sealed zero-episode run containing a fresh
``result-reader.json`` manifest, an extended glyph bank and the anchor copied
unchanged. No existing run or manifest is modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.perception.hud import HUDDigitReader, segment_digits
from gradientclimb.perception.scoring import ResultDistanceReader


def load_labels(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("The label file must contain a non-empty 'labels' list")
    for row in labels:
        for key in ("run_id", "frame", "text", "sha256", "annotation_id"):
            if not row.get(key):
                raise ValueError(f"Every label needs {key}")
        if any(character not in "0123456789" for character in row["text"]):
            raise ValueError("Labels contain visible digits only, without the unit suffix")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--field",
        choices=["result_distance"],
        default="result_distance",
        help="Only result-distance frames extend the terminal bank",
    )
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    base_dir = args.base_manifest.resolve().parent
    numeric_path = (base_dir / base["numeric_manifest"]["file"]).resolve()
    if sha256_file(numeric_path) != base["numeric_manifest"]["sha256"]:
        raise ValueError("Base numeric manifest hash mismatch")
    anchor_path = (base_dir / base["anchor"]["file"]).resolve()
    if sha256_file(anchor_path) != base["anchor"]["sha256"]:
        raise ValueError("Base anchor hash mismatch")
    reader = HUDDigitReader.from_manifest(numeric_path)
    labels = [row for row in load_labels(args.labels) if row.get("field", "result_distance")]
    glyphs_before = len(reader.glyphs)
    added = []
    sources = []
    for row in labels:
        run = load_run(args.root, row["run_id"])
        if (
            run["status"] not in {"completed", "failed"}
            or not verify_run(args.root, row["run_id"])["valid"]
        ):
            raise ValueError(f"Source run {row['run_id']} is not a sealed, verified record")
        frame_path = (args.root / "runs" / row["run_id"] / row["frame"]).resolve()
        digest = sha256_file(frame_path)
        if digest != row["sha256"]:
            raise ValueError(f"Frame hash mismatch for {row['frame']}")
        rgb = np.asarray(Image.open(frame_path).convert("RGB")).copy()
        segments = segment_digits(rgb, reader.roi)
        if len(segments) != len(row["text"]):
            raise ValueError(
                f"{row['frame']}: segmented {len(segments)} glyphs for label {row['text']}"
            )
        reader.add_labeled_region(
            rgb, row["text"], reader.roi, source_sha256=digest, annotation_id=row["annotation_id"]
        )
        added.append({**row, "glyphs": len(row["text"])})
        sources.append(frame_path)
    config = {
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "base_numeric_sha256": base["numeric_manifest"]["sha256"],
        "labels_file": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "field": args.field,
        "glyphs_before": glyphs_before,
        "glyphs_added": len(reader.glyphs) - glyphs_before,
        "construction_sources": [row["run_id"] for row in added],
        "scope": "construction only; held-out accuracy requires later sessions",
    }
    with RunRecorder(
        args.root,
        "result-reader-construction",
        config,
        algorithm="labeled-glyph-extension",
        environment="stored_real_frames",
        evidence_domain="real_game_offline",
        qualifies_real_game=False,
    ) as run:
        run.register_artifact(args.base_manifest, "base_result_reader")
        run.register_artifact(args.labels, "construction_labels")
        for path in sources:
            run.register_artifact(path, "construction_frame")
        bank_dir = run.directory / "result-glyphs"
        numeric_manifest = reader.save(bank_dir)
        for file in bank_dir.iterdir():
            run.register_artifact(
                file, "result_glyph" if file.suffix == ".png" else "glyph_manifest"
            )
        anchor_copy = run.directory / anchor_path.name
        shutil.copyfile(anchor_path, anchor_copy)
        run.register_artifact(anchor_copy, "result_label_anchor")
        manifest = {
            **base,
            "numeric_manifest": {
                "file": f"result-glyphs/{numeric_manifest.name}",
                "sha256": sha256_file(numeric_manifest),
            },
            "anchor": {"file": anchor_copy.name, "sha256": sha256_file(anchor_copy)},
            "extended_from": {
                "manifest": str(args.base_manifest),
                "sha256": config["base_manifest_sha256"],
                "run_id": run.run_id,
            },
        }
        manifest_path = run.directory / "result-reader.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run.register_artifact(manifest_path, "result_reader_manifest")
        # Construction self-check: every source frame must now read its own label.
        rebuilt = ResultDistanceReader.from_manifest(manifest_path)
        checks = []
        for row, path in zip(added, sources, strict=True):
            rgb = np.asarray(Image.open(path).convert("RGB")).copy()
            out = rebuilt.read(rgb, result_state_confirmed=True)
            checks.append(
                {
                    "frame": row["frame"],
                    "label": int(row["text"]),
                    "read": out.get("distance_meters"),
                    "valid": out["valid"],
                    "quality": out.get("quality"),
                }
            )
        run.evaluation(
            {
                "protocol": "result-reader-construction-selfcheck-1",
                "episodes": len(checks),
                "results": {
                    "checks": checks,
                    "all_match": all(c["valid"] and c["read"] == c["label"] for c in checks),
                    "scope": "construction self-match only; not held-out accuracy",
                },
            }
        )
        run.finalize(
            status="completed",
            episode_count=0,
            glyphs_total=len(reader.glyphs),
            construction_selfcheck_all_match=all(
                c["valid"] and c["read"] == c["label"] for c in checks
            ),
            qualification_evidence=False,
        )
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "manifest": str(manifest_path),
                    "glyphs_total": len(reader.glyphs),
                    "checks": checks,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
