"""Evaluate readers against Cycle 2 held-out labels under reader-validation-2.0.

Labels reference sealed frames by run id, file name and hash and were written from
pixels before any reader output was viewed. The evaluation refuses labels whose
frames or sessions are construction sources of the readers under test, and seals
its report as a zero-episode run. Metrics follow the registered protocol:
availability, exact accuracy, wrong-accept rate, fuel-bin agreement and terminal
cause agreement, with per-digit confusion for rejected result readings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.perception.fields import CoinCounterReader, FuelGaugeReader
from gradientclimb.perception.hud import HUDDigitReader, segment_digits
from gradientclimb.perception.scoring import ResultDistanceReader

FUEL_BINS = {"empty": (0.0, 0.02), "low": (0.0, 0.25), "mid": (0.25, 0.75), "high": (0.75, 1.01)}


def load_labels(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("The label file must contain a non-empty 'labels' list")
    if data.get("split") != "heldout":
        raise ValueError("Only held-out label files may be evaluated with this script")
    return labels


def construction_hashes(result_reader, gameplay_reader) -> set[str]:
    return {g["source_sha256"] for g in result_reader.reader.glyphs + gameplay_reader.glyphs}


def evaluate(args):
    result_reader = ResultDistanceReader.from_manifest(args.result_reader)
    gameplay = HUDDigitReader.from_manifest(args.hud)
    coins = CoinCounterReader(gameplay)
    fuel = FuelGaugeReader()
    labels = load_labels(args.labels)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    construction_runs = set(protocol.get("splits", {}).get("construction", []))
    train_hashes = construction_hashes(result_reader, gameplay)
    if train_hashes & {label["sha256"] for label in labels}:
        raise ValueError("Held-out labels overlap glyph construction frames")
    if construction_runs & {label["run_id"] for label in labels}:
        raise ValueError("Held-out labels come from a declared construction session")
    rows, sources = [], []
    for label in labels:
        run = load_run(args.root, label["run_id"])
        if not verify_run(args.root, label["run_id"])["valid"]:
            raise ValueError(f"Label source run {label['run_id']} does not verify")
        path = (args.root / "runs" / label["run_id"] / label["frame"]).resolve()
        if sha256_file(path) != label["sha256"]:
            raise ValueError(f"Frame hash mismatch: {label['frame']}")
        rgb = np.asarray(Image.open(path).convert("RGB")).copy()
        field = label["field"]
        row = {**label, "session_start": run["start_time"]}
        if field == "result_distance":
            out = result_reader.read(rgb, result_state_confirmed=True)
            value = out.get("distance_meters")
            row.update(
                valid=out["valid"], value=value, exact=out["valid"] and value == int(label["text"])
            )
            if not out["valid"]:
                segments = segment_digits(rgb, result_reader.reader.roi)
                row["rejected_digit_scores"] = [
                    sorted(
                        (
                            (str(d), round(s, 3))
                            for d, s in result_reader.reader.glyph_scores(seg).items()
                        ),
                        key=lambda item: -item[1],
                    )[:3]
                    for seg in segments
                ]
        elif field == "gameplay_progress":
            out = gameplay.read(rgb)
            value = out.get("hud_displayed_progress_meters")
            row.update(
                valid=out["valid"], value=value, exact=out["valid"] and value == int(label["text"])
            )
        elif field == "coins":
            out = coins.read(rgb, playing_state_confirmed=True)
            value = out.get("coins")
            row.update(
                valid=out["valid"], value=value, exact=out["valid"] and value == int(label["text"])
            )
        elif field == "fuel":
            out = fuel.read(rgb, playing_state_confirmed=True)
            value = out.get("fill_fraction")
            low, high = FUEL_BINS[label["text"]]
            row.update(valid=out["valid"], value=value, exact=out["valid"] and low <= value < high)
        elif field == "terminal_cause":
            variant = label.get("observed_variant")
            recognized = (
                "driver_down"
                if variant and "driver_down" in variant
                else "out_of_fuel"
                if variant and "out_of_fuel" in variant
                else None
            )
            row.update(
                valid=recognized is not None, value=recognized, exact=recognized == label["text"]
            )
        else:
            raise ValueError(f"Unknown label field {field}")
        rows.append(row)
        sources.append(path)
    report = {"protocol": protocol.get("protocol_version"), "labels": len(rows), "fields": {}}
    for field in sorted({row["field"] for row in rows}):
        subset = [row for row in rows if row["field"] == field]
        valid = [row for row in subset if row["valid"]]
        report["fields"][field] = {
            "labeled": len(subset),
            "sessions": len({row["run_id"] for row in subset}),
            "available": len(valid),
            "availability": len(valid) / len(subset),
            "exact": sum(row["exact"] for row in valid),
            "exact_accuracy": (sum(row["exact"] for row in valid) / len(valid)) if valid else None,
            "wrong_accepts": sum(not row["exact"] for row in valid),
            "wrong_accept_rate": (sum(not row["exact"] for row in valid) / len(valid))
            if valid
            else None,
        }
        if field == "result_distance":
            confusion = Counter()
            for row in subset:
                for scores in row.get("rejected_digit_scores", []):
                    if len(scores) >= 2:
                        confusion[f"{scores[0][0]}/{scores[1][0]}"] += 1
            report["fields"][field]["rejected_top_two_digit_pairs"] = dict(confusion)
    config = {
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "labels_file": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "result_reader": str(args.result_reader),
        "result_reader_sha256": sha256_file(args.result_reader),
        "hud_manifest_sha256": sha256_file(args.hud),
        "coin_reader_version": coins.version,
        "fuel_reader_version": fuel.version,
        "scope": "held-out labels from sessions disjoint from reader construction; no policy evaluation",
    }
    with RunRecorder(
        args.root,
        "reader-validation",
        config,
        algorithm="frozen-readers",
        environment="stored_real_frames",
        evidence_domain="real_game_offline",
        qualifies_real_game=False,
    ) as run:
        for path in (args.protocol, args.labels, args.result_reader, args.hud):
            run.register_artifact(path, "reader_validation_source")
        for path in sources:
            run.register_artifact(path, "heldout_labeled_frame")
        rows_path = run.directory / "label-evaluations.json"
        rows_path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
        run.register_artifact(rows_path, "reader_validation_rows")
        report_path = run.directory / "reader-validation-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        run.register_artifact(report_path, "reader_validation_report")
        run.evaluation(
            {"protocol": "reader-validation-2.0", "episodes": len(rows), "results": report}
        )
        run.finalize(
            status="completed", episode_count=0, report=report, qualification_evidence=False
        )
        return {"run_id": run.run_id, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/definitions/cycle-2-reader-validation.json"),
    )
    parser.add_argument("--result-reader", type=Path, required=True)
    parser.add_argument("--hud", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
