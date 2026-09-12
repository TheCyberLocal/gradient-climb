"""Export canonical Cycle 3 real-competence efficiency without changing source records."""

from __future__ import annotations

import argparse
from pathlib import Path

from gradientclimb.dashboard.efficiency_report import write_report
from gradientclimb.experiments.efficiency import load_studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args()
    reports = load_studies(args.root)
    for path in write_report(reports, args.output):
        print(path.resolve())


if __name__ == "__main__":
    main()
