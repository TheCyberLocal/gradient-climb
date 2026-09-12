"""Validate and seal a prospective study envelope linked to existing evidence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gradientclimb.experiments.efficiency import LearningEfficiencyStudy, publish_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = LearningEfficiencyStudy.model_validate_json(args.study.read_text(encoding="utf-8"))
    result = publish_study(args.root, study)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
