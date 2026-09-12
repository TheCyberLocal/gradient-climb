"""Sealed bank construction and held-out label evaluation on synthetic frames only."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gradientclimb.experiments import RunRecorder, verify_run
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.scoring import ResultDistanceReader, _white_text

WIDTH, HEIGHT = 1034, 581
DIGIT_ROI = (738 / 1034, 151 / 581, 916 / 1034, 195 / 581)
ANCHOR_ROI = (602 / 1034, 152 / 581, 736 / 1034, 196 / 581)
BASE_FONT = cv2.FONT_HERSHEY_SIMPLEX
# A different typeface: the base bank refuses these glyphs at its threshold.
OTHER_FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX


def script(name):
    spec = importlib.util.spec_from_file_location(
        f"reader_tools_{name}", Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_frame(text, font=BASE_FONT):
    frame = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    cv2.putText(frame, "DISTANCE:", (606, 186), BASE_FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, text, (765, 188), font, 1.2, (255, 255, 255), 2)
    return frame


def seal_frames(root, frames):
    """Seal synthetic terminal frames in a run and return (run_id, {name: sha256})."""
    with RunRecorder(root, "synthetic-terminal-frames", {"synthetic": True}, seed=1) as run:
        hashes = {}
        for name, rgb in frames.items():
            path = run.directory / name
            Image.fromarray(rgb).save(path)
            run.register_artifact(path, "terminal_frame", {"state": "result"})
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        run.finalize(status="completed", episode_count=0)
    return run.run_id, hashes


def base_manifest(directory):
    """A frozen-style result reader with a complete base-font alphabet."""
    reader = HUDDigitReader(expected_size=(WIDTH, HEIGHT), roi=DIGIT_ROI, alignment_pixels=1)
    for digit in "0123456789":
        reader.add_labeled_region(
            result_frame(digit),
            digit,
            DIGIT_ROI,
            source_sha256=hashlib.sha256(f"base-{digit}".encode()).hexdigest(),
            annotation_id=f"base-{digit}",
        )
    numeric = reader.save(directory / "result-glyphs")
    anchor_mask = _white_text(result_frame("0")[152:196, 602:736]).astype(np.uint8) * 255
    anchor_path = directory / "distance-label-anchor.png"
    Image.fromarray(anchor_mask).save(anchor_path)
    manifest = {
        "version": 1,
        "numeric_manifest": {
            "file": "result-glyphs/hud-glyphs.json",
            "sha256": hashlib.sha256(numeric.read_bytes()).hexdigest(),
        },
        "anchor": {
            "file": anchor_path.name,
            "sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        },
        "anchor_roi": list(ANCHOR_ROI),
        "anchor_threshold": 0.9,
        "field": "right_side_result_distance",
    }
    path = directory / "result-reader.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def test_bank_builder_extends_from_sealed_labeled_frames_and_seals_a_run(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    other = result_frame("47", OTHER_FONT)
    run_id, hashes = seal_frames(root, {"terminal-000.png": other})
    base = base_manifest(tmp_path / "base")
    assert not ResultDistanceReader.from_manifest(base).read(other, result_state_confirmed=True)[
        "valid"
    ]
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "run_id": run_id,
                        "frame": "terminal-000.png",
                        "sha256": hashes["terminal-000.png"],
                        "field": "result_distance",
                        "text": "47",
                        "annotation_id": "synthetic-47",
                    }
                ]
            }
        )
    )
    builder = script("build_result_reader")
    monkeypatch.setattr(
        sys,
        "argv",
        ["build", "--base-manifest", str(base), "--labels", str(labels), "--root", str(root)],
    )
    builder.main()
    construction = [
        run for run in (root / "runs").iterdir() if (run / "result-reader.json").exists()
    ]
    assert len(construction) == 1
    manifest = construction[0] / "result-reader.json"
    assert verify_run(root, construction[0].name)["valid"]
    extended = ResultDistanceReader.from_manifest(manifest)
    out = extended.read(other, result_state_confirmed=True)
    assert out["valid"] and out["distance_meters"] == 47
    # The base alphabet still reads its own font exactly after the extension.
    assert extended.read(result_frame("380"), result_state_confirmed=True)["distance_meters"] == 380
    record = json.loads((construction[0] / "run.json").read_text())
    assert record["configuration"]["glyphs_added"] == 2
    assert record["summary"]["construction_selfcheck_all_match"] is True
    # A tampered label hash is refused before any bank is written.
    bad = json.loads(labels.read_text())
    bad["labels"][0]["sha256"] = "0" * 64
    labels.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.main()


def test_label_evaluation_refuses_construction_overlap_and_reports_fields(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    run_id, hashes = seal_frames(
        root,
        {
            "terminal-000.png": result_frame("47", OTHER_FONT),
            "terminal-001.png": result_frame("380"),
        },
    )
    base = base_manifest(tmp_path / "base")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"protocol_version": "reader-validation-2.0", "splits": {"construction": []}})
    )
    hud = tmp_path / "base" / "result-glyphs" / "hud-glyphs.json"
    rows = [
        {
            "run_id": run_id,
            "frame": "terminal-000.png",
            "sha256": hashes["terminal-000.png"],
            "field": "result_distance",
            "text": "47",
        },
        {
            "run_id": run_id,
            "frame": "terminal-001.png",
            "sha256": hashes["terminal-001.png"],
            "field": "result_distance",
            "text": "380",
        },
        {
            "run_id": run_id,
            "frame": "terminal-001.png",
            "sha256": hashes["terminal-001.png"],
            "field": "terminal_cause",
            "text": "driver_down",
            "observed_variant": "driver_down_native",
        },
    ]
    labels = tmp_path / "heldout.json"
    labels.write_text(json.dumps({"split": "heldout", "labels": rows}))
    evaluator = script("evaluate_reader_labels")
    argv = [
        "evaluate",
        "--labels",
        str(labels),
        "--protocol",
        str(protocol),
        "--result-reader",
        str(base),
        "--hud",
        str(hud),
        "--root",
        str(root),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evaluator.main()
    validation = [
        run for run in (root / "runs").iterdir() if (run / "reader-validation-report.json").exists()
    ]
    assert len(validation) == 1
    report = json.loads((validation[0] / "reader-validation-report.json").read_text())
    result = report["fields"]["result_distance"]
    # The other-font "47" is refused (unknown, not wrong); "380" is read exactly.
    assert result["labeled"] == 2 and result["available"] == 1 and result["exact"] == 1
    assert result["wrong_accepts"] == 0
    assert report["fields"]["terminal_cause"]["exact"] == 1
    assert verify_run(root, validation[0].name)["valid"]
    # Construction frames or sessions cannot be evaluated as held-out.
    labels.write_text(json.dumps({"split": "construction", "labels": rows}))
    with pytest.raises(ValueError, match="held-out"):
        evaluator.main()
    protocol.write_text(
        json.dumps(
            {"protocol_version": "reader-validation-2.0", "splits": {"construction": [run_id]}}
        )
    )
    labels.write_text(json.dumps({"split": "heldout", "labels": rows}))
    with pytest.raises(ValueError, match="construction session"):
        evaluator.main()
