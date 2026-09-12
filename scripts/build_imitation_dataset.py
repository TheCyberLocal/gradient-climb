"""Build reviewed window references; no capture, feature fitting or model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.datasets.demonstrations import publish_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan", required=True, help="Project-relative frozen plan JSON")
    args = parser.parse_args()
    print(json.dumps(publish_dataset(args.project_root, args.plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
