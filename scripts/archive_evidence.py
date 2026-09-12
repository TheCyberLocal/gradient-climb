"""Inventory, archive, verify and restore private research evidence without overwriting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.artifacts.archive import (
    build_inventory,
    create_archive,
    restore_archive,
    verify_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--project", type=Path, default=Path.cwd())
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional project-relative evidence path; repeatable",
    )
    archive = commands.add_parser("archive")
    archive.add_argument("--project", type=Path, default=Path.cwd())
    archive.add_argument("--inventory", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--inventory", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = build_inventory(args.project, args.output, include=args.include)
        result = {key: value for key, value in result.items() if key != "files"}
    elif args.command == "archive":
        result = create_archive(args.project, args.inventory, args.output)
    elif args.command == "verify":
        result = verify_archive(args.archive, args.inventory)
    else:
        result = restore_archive(args.archive, args.inventory, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
