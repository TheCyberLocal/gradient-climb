"""Read-only artifact integrity inventory for a deliberate research-cycle pause."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import list_runs, verify_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Destination inventory path. Required and never defaulted: each cycle's inventory "
            "is cited by that cycle's report and notebooks."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing an inventory that already exists at --output.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output.as_posix()} already exists. A published inventory records one "
            "cycle's verified run set; write the current scan to a new path, or pass "
            "--overwrite deliberately."
        )
    rows = []
    for run in list_runs(args.root):
        directory = args.root / "runs" / run["run_id"]
        result = verify_run(args.root, run["run_id"])
        rows.append(
            {
                **result,
                "status": run["status"],
                "experiment_id": run["experiment_id"],
                "algorithm": run["algorithm"],
                "source_sha": run["git_sha"],
                "dirty_worktree": run["dirty_worktree"],
                "start_time": run["start_time"],
                "end_time": run["end_time"],
                "seal_sha256": sha256_file(directory / "seal.json")
                if (directory / "seal.json").exists()
                else None,
                "file_count": sum(p.is_file() for p in directory.rglob("*")),
                "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            }
        )
    recognized = {row["run_id"] for row in rows}
    uncatalogued = sorted(
        p.name for p in (args.root / "runs").iterdir() if p.is_dir() and p.name not in recognized
    )
    payload = {
        "schema_version": "cycle-integrity-1",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "artifact_root": args.root.as_posix(),
        "scope": "All local canonical runs; hashes do not establish scientific validity or backup.",
        "run_count": len(rows),
        "valid_count": sum(row["valid"] for row in rows),
        "unfinished_run_ids": [row["run_id"] for row in rows if row["status"] == "running"],
        "uncatalogued_directories": uncatalogued,
        "total_bytes": sum(row["bytes"] for row in rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}))
    if payload["valid_count"] != len(rows) or payload["unfinished_run_ids"] or uncatalogued:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
