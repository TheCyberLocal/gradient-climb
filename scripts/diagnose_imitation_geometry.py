"""Measure fixed detector coverage on reviewed imitation windows; no fitting or live input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.datasets.geometry_diagnostic import publish_geometry_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan", required=True, help="Project-relative frozen window plan JSON")
    parser.add_argument(
        "--profile", required=True, help="Project-relative fixed perception profile"
    )
    parser.add_argument("--maximum-frames", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            publish_geometry_diagnostic(
                args.project_root, args.plan, args.profile, maximum_frames=args.maximum_frames
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
