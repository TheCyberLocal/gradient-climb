"""Build local prototype glyphs from explicitly inspected, already saved HUD regions.

This fixed bootstrap set lacks digit nine. The reader therefore refuses numeric
readings until a later independently labeled region supplies that missing digit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments.recorder import RunRecorder, verify_run
from gradientclimb.perception.hud import HUDDigitReader

SPECS = [
    ("playing.png", "23", (343, 66, 416, 109), "displayed progress"),
    ("playing.png", "2400", (790, 48, 910, 99), "upper checkpoint"),
    ("playing.png", "2154", (790, 99, 910, 149), "lower checkpoint"),
    ("playing.png", "7750", (121, 84, 220, 125), "coins"),
    ("playing.png", "6", (121, 131, 153, 174), "gems"),
    ("paused.png", "84", (172, 43, 226, 75), "fuel-can HUD quantity; not progress"),
    ("playing-native-normalized.png", "16245", (122, 86, 221, 125), "coins"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/game-discovery"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    hashes = {file: sha256_file(args.source / file) for file, *_ in SPECS}
    config = {
        "protocol": "inspected real HUD glyph prototype construction",
        "source_hashes": hashes,
        "source_regions": [
            {"file": file, "text": text, "bbox": box, "quantity": quantity}
            for file, text, box, quantity in SPECS
        ],
        "heldout_images": 0,
        "missing_digit": "9",
    }
    reader = HUDDigitReader(
        expected_size=(1034, 581), roi=(343 / 1034, 66 / 581, 465 / 1034, 110 / 581)
    )
    with RunRecorder(
        args.artifacts,
        "hud-glyph-prototypes",
        config,
        algorithm="local-labeled-glyphs-1",
        environment="saved-hcr-pixels",
        evidence_domain="real_game_glyph_construction",
        telemetry_interval_seconds=0,
    ) as run:
        for file in hashes:
            run.register_artifact(
                args.source / file,
                "hud_glyph_source",
                {"redistribution": "local only; third-party content"},
            )
        for index, (file, text, box, _) in enumerate(SPECS):
            with Image.open(args.source / file) as image:
                if image.size != (1034, 581):
                    raise ValueError("Inspected source geometry changed")
                rgb = np.asarray(image.convert("RGB")).copy()
            roi = (box[0] / 1034, box[1] / 581, box[2] / 1034, box[3] / 581)
            reader.add_labeled_region(
                rgb,
                text,
                roi,
                source_sha256=hashes[file],
                annotation_id=f"bootstrap-region-{index}",
            )
        manifest = reader.save(run.directory / "hud-glyphs")
        for path in manifest.parent.iterdir():
            run.register_artifact(
                path, "hud_glyph_manifest" if path == manifest else "hud_glyph_crop"
            )
        missing = sorted(set("0123456789") - {glyph["digit"] for glyph in reader.glyphs})
        run.metric("prototype_glyph_count", len(reader.glyphs))
        run.metric("available_digit_classes", 10 - len(missing))
        run.finalize(
            episodes=0,
            environment_steps=0,
            training_steps=0,
            missing_digits=missing,
            accepted_numeric_readings=0,
            heldout_images=0,
            measured_generalization_accuracy=None,
        )
        result = {"run_id": run.run_id, "manifest": str(manifest), "missing_digits": missing}
    result["verification"] = verify_run(args.artifacts, result["run_id"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
