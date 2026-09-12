"""Prospective training entry point; first sample precedes project/framework imports."""

import time

COMMAND_STARTED = time.monotonic()
COMMAND_STARTED_EPOCH = time.time()

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

if __name__ == "__main__":
    entry_parser = argparse.ArgumentParser(add_help=False)
    entry_parser.add_argument("--entry-receipt", type=Path, required=True)
    entry, arguments = entry_parser.parse_known_args()
    entry.entry_receipt.parent.mkdir(parents=True, exist_ok=True)
    # A durable outer receipt survives subsequent project/framework import failure.
    with entry.entry_receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": "command-entry-3.0",
                "started_monotonic": COMMAND_STARTED,
                "started_at": datetime.fromtimestamp(COMMAND_STARTED_EPOCH, UTC).isoformat(),
                "process_id": os.getpid(),
                "argv": list(sys.argv),
                "clock": vars(time.get_clock_info("monotonic")),
                "boundary": "python_entry_before_project_imports",
                "stage": "entry_recorded_before_project_imports",
                "run_id": None,
                "exclusions": "shell_and_interpreter_startup_before_first_clock_sample",
            },
            stream,
            indent=2,
        )
        stream.flush()
        os.fsync(stream.fileno())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gradientclimb.cli import main
    from gradientclimb.experiments.command_clock import clock_receipt

    receipt = clock_receipt(COMMAND_STARTED, COMMAND_STARTED_EPOCH)
    receipt["entry_receipt_path"] = str(entry.entry_receipt.resolve())
    main(arguments, command_clock=receipt)
