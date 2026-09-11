"""Capture one game-only UI reference after supervising visual inspection.

The label is supplied by the operator, never inferred or used to grant input.
This utility performs no input, focusing, or navigation. Images remain local.
"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from gradientclimb.capture.screen import WindowCapture
from gradientclimb.capture.windows import WindowGuard, discover_windows
from gradientclimb.experiments import RunRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        required=True,
        choices=[
            "playing",
            "paused",
            "revive_offer",
            "bonus_offer",
            "driver_down",
            "out_of_fuel",
            "tune",
            "ad_close",
        ],
    )
    args = parser.parse_args()
    targets = discover_windows()
    if len(targets) != 1:
        raise RuntimeError("Expected exactly one visible game window")
    target = targets[0]
    config = {
        "operator_label": args.label,
        "label_semantics": "supervisor-supplied; inspect saved pixels before using as a reference",
        "target": target.as_dict(),
        "frame_size": [1034, 581],
        "input": "none",
    }
    with RunRecorder(
        "artifacts",
        "real-ui-reference",
        config,
        algorithm="supervised-reference-capture",
        environment="actual_hill_climb_racing",
    ) as run:
        with WindowCapture(
            WindowGuard(target), backend="dxcam", output_size=(1034, 581)
        ) as capture:
            frame = capture.grab()
        path = run.directory / f"{args.label}.png"
        Image.fromarray(frame.rgb).save(path)
        run.register_artifact(
            path, "real_ui_reference", {"operator_label": args.label, **frame.metadata()}
        )
        metadata = run.directory / "capture.json"
        metadata.write_text(json.dumps(frame.metadata(), indent=2) + "\n", encoding="utf-8")
        run.register_artifact(metadata, "capture_metadata")
        run.finalize(status="completed", episode_count=0, qualification_evidence=False)
        print(
            json.dumps({"run_id": run.run_id, "path": str(path), "label": args.label}), flush=True
        )


if __name__ == "__main__":
    main()
