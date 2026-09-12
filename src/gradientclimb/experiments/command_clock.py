"""Opt-in prospective command-entry clock; old learner clocks remain unchanged."""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import UTC, datetime


def validate_clock(receipt: dict) -> dict:
    if receipt.get("schema_version") != "command-clock-3.0":
        raise ValueError("Unsupported command clock receipt")
    start = receipt.get("started_monotonic")
    if (
        not isinstance(start, (int, float))
        or not math.isfinite(start)
        or not 0 <= start <= time.monotonic()
    ):
        raise ValueError("Command clock origin must be a prior finite monotonic sample")
    stamp = datetime.fromisoformat(receipt["started_at"])
    if stamp.utcoffset() is None:
        raise ValueError("Command clock requires timezone-aware wall time")
    if receipt.get("process_id") != os.getpid():
        raise ValueError("A command clock cannot be replayed from another process")
    if stamp.timestamp() > time.time() + 1:
        raise ValueError("Command wall time is in the future")
    if receipt.get("boundary") != "python_entry_before_project_imports":
        raise ValueError("Clock boundary must be declared honestly")
    return dict(receipt)


def clock_receipt(started_monotonic: float, started_epoch: float) -> dict:
    return validate_clock(
        {
            "schema_version": "command-clock-3.0",
            "started_monotonic": started_monotonic,
            "started_at": datetime.fromtimestamp(started_epoch, UTC).isoformat(),
            "boundary": "python_entry_before_project_imports",
            "process_id": os.getpid(),
            "argv": list(sys.argv),
            "clock": vars(time.get_clock_info("monotonic")),
            "included": [
                "project_and_framework_imports",
                "argument_parsing",
                "setup",
                "provenance",
                "training",
                "in_process_evaluation_and_waits",
                "checkpoint_publication",
            ],
            "excluded": [
                "shell_dispatch_and_python_interpreter_startup_before_first_clock_sample",
                "independent_post_checkpoint_evaluation",
            ],
            "prior_costs": "separate_lineage_required",
        }
    )
