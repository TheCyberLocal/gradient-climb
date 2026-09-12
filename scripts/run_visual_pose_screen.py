"""Publish the registered four-pose construction comparison; no fitting or live input."""

import argparse
import json
from pathlib import Path

from gradientclimb.experiments.visual_screen import publish_visual_pose_screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", required=True, help="Project-relative registered protocol")
    parser.add_argument("--replay", required=True, help="Project-relative frozen geometry labels")
    args = parser.parse_args()
    print(
        json.dumps(
            publish_visual_pose_screen(args.project_root, args.protocol, args.replay), indent=2
        )
    )


if __name__ == "__main__":
    main()
