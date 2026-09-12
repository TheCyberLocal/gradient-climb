"""Seal an explicit reviewed annotation plan; no capture, decoding, fitting or truth inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.experiments.reader_annotations3 import publish_reader_annotations3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.plan.stat().st_size > 32 * 1024**2:
        parser.error("Annotation plan exceeds 32 MiB")
    result = publish_reader_annotations3(args.project_root, json.loads(args.plan.read_bytes()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
