"""Build a new read-only Cycle 3 notebook; historical notebooks remain separate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

PROJECT = Path(__file__).resolve().parents[1]


def make_notebook():
    md = nbformat.v4.new_markdown_cell

    def code(source):
        return nbformat.v4.new_code_cell(dedent(source).strip())

    cells = [
        md(
            "# Real competence efficiency — Cycle 3\n\n"
            "## Summary\n\nThis notebook presents prospective real-game competence evidence. "
            "Episodes measure experience, compute measures cost, and wall-clock time measures "
            "rapidity. No simulation result is promoted to real competence. The results below "
            "are produced by the same canonical analysis used by the dashboard and report CLI."
        ),
        md(
            "## Context and methods\n\n"
            "Inputs are sealed `artifacts/runs/*/learning-efficiency.json` envelopes and their "
            "hash-verified references. Invalid sources remain visible errors. "
            "Threshold times are first observed passing checkpoints. Censor support ends at "
            "the last eligible independent evaluation, not the unobserved budget endpoint. "
            "Prior elapsed clocks are separate; data-generation compute is a subset of total "
            "compute. Evaluation costs are separate from checkpoint costs.\n\n"
            "### Key assumptions\n\nByte identity does not prove truthful independence, "
            "representative episode sampling, or measurement validity. Protocol audit remains "
            "necessary. Pareto cohorts and eligibility come only from the canonical comparator; "
            "the notebook does not estimate speedups, confidence intervals or acquisition times."
        ),
        md(
            "## Data\n\n### 1. Set local input and display scope\n\n"
            "Use the project virtual environment. Set `ARTIFACT_ROOT` to the canonical store. "
            "An empty `STUDY_IDS` list displays all studies. This selection changes the display "
            "and leaves registered comparison cohorts intact."
        ),
        code("""
            import json
            from pathlib import Path

            from IPython.display import Markdown, display

            from gradientclimb.dashboard.efficiency_report import compare_reports, render_markdown
            from gradientclimb.experiments.efficiency import load_studies

            PROJECT = Path.cwd()
            if not (PROJECT / "src/gradientclimb").is_dir():
                PROJECT = PROJECT.parent
            ARTIFACT_ROOT = PROJECT / "artifacts"
            STUDY_IDS = []
            print("Canonical artifact root:", ARTIFACT_ROOT.resolve())
            print("Display selection:", STUDY_IDS or "all studies")
        """),
        md(
            "### 2. Load and validate one snapshot\n\nThe loader never mutates canonical runs "
            "or fills historical missing measurements. Each source is read only from the "
            "explicit canonical run directory."
        ),
        code("""
            reports = load_studies(ARTIFACT_ROOT)
            comparisons = compare_reports(reports)
            selected = [report for report in reports if not STUDY_IDS or report["study_id"] in STUDY_IDS]
            print("Study envelopes:", len(reports))
            print("Validation errors:", sum(bool(report.get("error")) for report in reports))
            print(
                json.dumps(
                    [
                        {
                            key: report.get(key)
                            for key in (
                                "study_id",
                                "training_run_id",
                                "protocol_id",
                                "profile_id",
                                "prior_class",
                                "provenance_status",
                                "error",
                            )
                        }
                        for report in selected
                    ],
                    indent=2,
                )
            )
        """),
        md(
            "## Results\n\n### 3. Read the canonical outcome and cost tables\n\n"
            "These are the shared report renderer's tables. Null values remain **Not measured**; "
            "recorded zeros remain zero. Full comparison cohorts stay visible to explain "
            "dominance or exclusion even when a study selection is active."
        ),
        code("""
            display(Markdown(render_markdown(selected, comparisons)))
        """),
        md(
            "### 4. Inspect frozen checkpoint observations\n\nDots show canonical median "
            "real-game distance. No line interpolation or monotonic learning assumption is "
            "introduced. Up to twelve selected studies are plotted; the tables retain every "
            "selected result. Different protocols or profiles are not a controlled comparison."
        ),
        code("""
            available = [
                report for report in selected if not report.get("error") and report.get("checkpoint_curve")
            ]
            if not available:
                print("Checkpoint competence curve not measured: no eligible real evaluation.")
            else:
                import matplotlib.pyplot as plt

                with plt.rc_context(
                    {
                        "figure.figsize": (10, 5),
                        "font.size": 11,
                        "axes.spines.top": False,
                        "axes.spines.right": False,
                    }
                ):
                    figure, axis = plt.subplots()
                    for report in available[:12]:
                        points = report["checkpoint_curve"]
                        axis.scatter(
                            [point["time_seconds"] for point in points],
                            [point["median_distance_m"] for point in points],
                            label=f"{report['study_id']} / {report['profile_id']}",
                        )
                    axis.set(
                        title="Independent real-game checkpoint evaluations",
                        xlabel="Own command-start elapsed seconds",
                        ylabel="Median real-game distance (m)",
                    )
                    axis.grid(alpha=0.2)
                    axis.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
                    figure.tight_layout()
                    display(figure)
                    plt.close(figure)
        """),
        md("## Takeaways"),
        code("""
            if not reports:
                display(
                    Markdown(
                        "**Real competence efficiency is not yet measured.** "
                        "No validated Cycle 3 study envelope is present in this store. "
                        "Historical simulator runs do not fill this evidence gap."
                    )
                )
            elif all(report.get("error") for report in reports):
                display(
                    Markdown(
                        "**Evidence validation failed for every available envelope.** "
                        "No competence or efficiency claim can be made from this snapshot."
                    )
                )
            else:
                display(
                    Markdown(
                        "Review the canonical threshold states, cost coverage, warnings "
                        "and comparison exclusions above. A descriptive Pareto frontier "
                        "does not establish a repeated-seed advantage or exact acquisition "
                        "time. Source protocol audit remains necessary."
                    )
                )
        """),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3 (.venv)",
                "language": "python",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"real-competence-efficiency-{index:02d}"
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "notebooks/real_competence_efficiency.ipynb"
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=PROJECT / "artifacts/notebooks/real_competence_efficiency.html",
    )
    args = parser.parse_args()
    if args.output.exists() or (args.execute and args.preview.exists()):
        parser.error(
            "Use new output and preview paths; this builder never replaces saved artifacts."
        )
    notebook = make_notebook()
    if args.execute:
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter

        runtime = PROJECT / "artifacts/notebooks/cycle3-runtime"
        for variable, leaf in [
            ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
            ("IPYTHONDIR", "ipython"),
            ("MPLCONFIGDIR", "matplotlib"),
            ("JUPYTER_CONFIG_DIR", "jupyter-config"),
        ]:
            directory = runtime / leaf
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)
        os.environ["JUPYTER_PATH"] = str(runtime)
        kernel = runtime / "kernels/gradientclimb-efficiency"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "GradientClimb efficiency (.venv)",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="gradientclimb-efficiency",
            resources={"metadata": {"path": str(PROJECT)}},
        ).execute()
        rendered, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        with args.preview.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    nbformat.validate(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    print(
        json.dumps(
            {
                "notebook": str(args.output),
                "executed": args.execute,
                "preview": str(args.preview) if args.execute else None,
            }
        )
    )


if __name__ == "__main__":
    main()
