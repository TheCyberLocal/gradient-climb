"""Build a new result glyph bank from a registered Cycle 3 construction plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.experiments.reader_validation3 import build_reader3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = build_reader3(args.project_root, json.loads(args.plan.read_bytes()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
