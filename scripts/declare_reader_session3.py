"""Seal a reader session purpose and sampling plan before held-out acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradientclimb.experiments.reader_receipts3 import (
    SessionReceiptPlan3,
    publish_session_declaration3,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = SessionReceiptPlan3.model_validate_json(args.plan.read_bytes())
    result = publish_session_declaration3(
        args.project_root,
        args.artifact_root,
        plan.protocol.model_dump(),
        plan.session_id,
        plan.purpose,
        plan.source_run_config_binding,
        candidate_freeze_run_id=plan.candidate_freeze_run_id,
        selection_ref=plan.selection.model_dump(),
        note=plan.note,
        source_run_ids=plan.source_run_ids,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
