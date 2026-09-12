"""Evaluate a frozen result reader after all blinded Cycle 3 labels are sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.experiments.reader_validation3 import evaluate_reader3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_reader3(args.project_root, json.loads(args.plan.read_bytes()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
