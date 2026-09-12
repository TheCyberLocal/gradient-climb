"""Build enlarged contact sheets of sealed frames' numeric fields for manual labeling.

The sheet shows only pixels; it never runs a reader, so labels written from it are
independent of any reader output. Frames are addressed by run id and file name and
their hashes are written next to the sheet so a label file can reference them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIELDS = {
    "result_distance": (600, 140, 920, 200),
    "gameplay_progress": (290, 60, 480, 116),
    "coins": (70, 78, 260, 132),
    "fuel": (68, 38, 280, 78),
    "result_header": (600, 70, 860, 130),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--run", action="append", required=True, help="Sealed run id (repeatable)")
    parser.add_argument("--pattern", default="terminal-*.png")
    parser.add_argument("--field", choices=sorted(FIELDS), default="result_distance")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()
    left, top, right, bottom = FIELDS[args.field]
    tiles, index = [], []
    for run_id in args.run:
        directory = args.root / "runs" / run_id
        for path in sorted(directory.glob(args.pattern)):
            if len(tiles) >= args.max_frames:
                break
            rgb = np.asarray(Image.open(path).convert("RGB"))
            crop = Image.fromarray(rgb[top:bottom, left:right]).resize(
                ((right - left) * args.scale, (bottom - top) * args.scale), Image.Resampling.NEAREST
            )
            canvas = Image.new("RGB", (crop.width, crop.height + 18), (0, 0, 0))
            canvas.paste(crop, (0, 18))
            tag = f"{len(tiles):03d} {run_id[:8]}/{path.name}"
            ImageDraw.Draw(canvas).text((4, 2), tag, fill=(255, 255, 0))
            tiles.append(canvas)
            index.append(
                {
                    "tile": len(tiles) - 1,
                    "run_id": run_id,
                    "frame": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "field": args.field,
                }
            )
    if not tiles:
        raise SystemExit("No frames matched")
    width = max(tile.width for tile in tiles)
    sheet = Image.new("RGB", (width, sum(tile.height for tile in tiles)), (0, 0, 0))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sheet": str(args.output), "frames": len(index)}))


if __name__ == "__main__":
    main()
