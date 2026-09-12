"""Presentation of canonical Cycle 3 analysis; no metric recomputation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

COST_LABELS = {
    "cpu_core_seconds": "CPU core-seconds",
    "gpu_utilization_equivalent_seconds": "GPU utilization-equivalent seconds",
    "simulator_transitions": "Simulator transitions",
    "simulator_episodes": "Simulator episodes",
    "simulator_seconds": "Simulated seconds",
    "physics_steps": "Physics steps",
    "rendered_frames": "Rendered frames",
    "policy_decisions": "Policy decisions",
    "optimizer_updates": "Optimizer updates",
    "real_game_interaction_seconds": "Real-game interaction seconds",
    "real_game_episodes": "Real-game episodes",
    "data_generation_cpu_core_seconds": "Data-generation CPU core-seconds",
    "data_generation_gpu_utilization_equivalent_seconds": (
        "Data-generation GPU utilization-equivalent seconds"
    ),
}
STATUS_LABELS = {
    "reached": "Reached",
    "right_censored": "Right censored",
    "not_evaluable": "Not evaluable",
}
EXPLANATION = (
    "Real competence determines value. Episodes measure experience, compute measures cost, "
    "and wall-clock time measures rapidity. A reached threshold is supported by independent "
    "real-game evaluation of the named checkpoint. Right censoring ends at the last eligible "
    "evaluation; it does not extend to an unobserved budget endpoint. Missing values are not zero. "
    "Prior elapsed times are reported separately and are never added to the own elapsed clock."
    " Data-generation compute is a subset of the total compute fields, not an additional charge."
    " Evaluation cost is shown separately from checkpoint training and inherited cost."
)


def value_text(value: Any) -> str:
    """Format only; preserve explicit zeros, missingness, and recorded precision."""
    if value is None:
        return "Not measured"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return str(value)


def threshold_rows(report: dict) -> list[list[Any]]:
    return [
        [
            item["threshold_m"],
            STATUS_LABELS.get(item["status"], item["status"]),
            item.get("time_seconds"),
            item.get("interval_lower_seconds"),
            item.get("censor_seconds"),
            item.get("budget_seconds"),
            item.get("checkpoint_sha256"),
        ]
        for item in report.get("thresholds", [])
    ]


THRESHOLD_HEADERS = [
    "Threshold (m)",
    "Status",
    "Reached at (own elapsed s)",
    "Prior eligible evaluation (s)",
    "Censor support (s)",
    "Declared budget (s)",
    "Checkpoint SHA-256",
]


def cost_rows(threshold: dict) -> list[list[Any]]:
    return [
        [label, *(threshold.get(scope, {}).get(key) for scope in COST_SCOPES)]
        for key, label in COST_LABELS.items()
    ]


COST_SCOPES = ("own_cost", "prior_cost", "combined_cost")
COST_HEADERS = ["Resource or experience", "Own", "Prior", "Combined"]


def compare_reports(reports: list[dict]) -> list[dict]:
    """Delegate every comparison to the canonical analysis implementation."""
    from gradientclimb.experiments.efficiency import compare_studies

    thresholds = sorted(
        {row["threshold_m"] for report in reports for row in report.get("thresholds", [])}
    )
    return [compare_studies(reports, threshold) for threshold in thresholds]


def comparison_rows(comparison: dict) -> list[list[Any]]:
    return [
        [
            item["study_id"],
            "Pareto frontier" if item["pareto_frontier"] else "Dominated",
            item["cohort"],
            item["values"],
            item["dominated_by"],
        ]
        for item in comparison.get("comparisons", [])
    ] + [
        [item["study_id"], "Excluded", item["reason"], None, None]
        for item in comparison.get("excluded", [])
    ]


COMPARISON_HEADERS = [
    "Study",
    "Comparison status",
    "Cohort or exclusion reason",
    "Measured dimensions",
    "Dominated by",
]


def _html_table(headers: list, rows: list[list]) -> str:
    def escape(value: Any) -> str:
        return html.escape(value_text(value))

    if not rows:
        return '<p class="missing">Not measured</p>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def render_html(reports: list[dict], comparisons: list[dict] | None = None) -> str:
    """A self-contained, network-free report using the same analysis as the API."""
    sections = []
    for report in reports:
        if report.get("error"):
            sections.append(
                "<section><h2>Evidence validation failed</h2><p>"
                + html.escape(str(report.get("study_id")))
                + '</p><p class="warnings">'
                + html.escape(str(report["error"]))
                + "</p><p>No competence or efficiency claim is available for this record.</p>"
                + "</section>"
            )
            continue
        details = [
            ["Study", report.get("study_id")],
            ["Prior class", report.get("prior_class")],
            ["Profile", report.get("profile_id")],
            ["Protocol", report.get("protocol_id")],
            ["Real scenario SHA-256", report.get("scenario_sha256")],
            ["Comparison contract SHA-256", report.get("comparison_contract_sha256")],
            ["Wall-clock boundary", report.get("wall_clock_boundary")],
            ["Evaluation split", report.get("evaluation_split")],
            ["Compute comparison scope", report.get("compute_comparison_scope")],
            ["Provenance status", report.get("provenance_status")],
            ["Policy prior roots", report.get("policy_prior_roots")],
            ["Lineage complete", report.get("lineage_complete")],
        ]
        warning_html = "".join(
            f"<li>{html.escape(str(warning))}</li>" for warning in report.get("warnings", [])
        )
        costs = "".join(
            "<details><summary>Costs at threshold "
            + html.escape(value_text(threshold["threshold_m"]))
            + " m — "
            + html.escape(STATUS_LABELS.get(threshold["status"], threshold["status"]))
            + "</summary>"
            + _html_table(COST_HEADERS, cost_rows(threshold))
            + "</details>"
            for threshold in report.get("thresholds", [])
        )
        sections.append(
            "<section><h2>"
            + html.escape(str(report.get("study_id", "Unnamed study")))
            + "</h2>"
            + _html_table(["Study definition", "Recorded value"], details)
            + (f'<ul class="warnings">{warning_html}</ul>' if warning_html else "")
            + _html_table(THRESHOLD_HEADERS, threshold_rows(report))
            + "<details><summary>Checkpoint evaluation and verification clocks</summary>"
            + _html_table(
                [
                    "Checkpoint SHA-256",
                    "Own elapsed (s)",
                    "Median real distance (m)",
                    "Verification elapsed (s)",
                    "Evaluation elapsed (s)",
                    "Evaluation run",
                ],
                [
                    [
                        point.get(key)
                        for key in (
                            "checkpoint_sha256",
                            "time_seconds",
                            "median_distance_m",
                            "verification_elapsed_seconds",
                            "evaluation_elapsed_seconds",
                            "evaluation_run_id",
                        )
                    ]
                    for point in report.get("checkpoint_curve", [])
                ],
            )
            + "</details>"
            + costs
            + "<details><summary>Evaluation costs (separate)</summary>"
            + _html_table(
                ["Resource or experience", "Evaluation"],
                [
                    [label, report.get("evaluation_cost", {}).get(key)]
                    for key, label in COST_LABELS.items()
                ],
            )
            + "</details><details><summary>Prior elapsed clocks and ancestry</summary>"
            + _html_table(
                ["Cost ID", "Kind", "Parent IDs", "Elapsed (s)", "Human time (s)"],
                [
                    [
                        node.get(key)
                        for key in (
                            "cost_id",
                            "kind",
                            "parents",
                            "elapsed_seconds",
                            "human_seconds",
                        )
                    ]
                    for node in report.get("prior_ledger", [])
                ],
            )
            + "</details>"
            + "<details><summary>Complete canonical analysis and provenance</summary><pre>"
            + html.escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            + "</pre></details></section>"
        )
    for comparison in comparisons or []:
        sections.append(
            "<section><h2>Pareto comparison at "
            + html.escape(str(comparison["threshold_m"]))
            + " m</h2><p>"
            + html.escape(str(comparison["note"]))
            + "</p>"
            + _html_table(COMPARISON_HEADERS, comparison_rows(comparison))
            + "</section>"
        )
    body = "".join(sections) or (
        "<section><h2>Real competence efficiency is not yet measured.</h2>"
        "<p>No validated Cycle 3 learning-efficiency study is present in the selected store. "
        "Historical simulation curves remain available in the existing research views.</p></section>"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>GradientClimb · Real competence efficiency</title><style>"
        "body{font:16px/1.55 system-ui,sans-serif;color:#16312c;background:#f5f7f4;"
        "max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:36px;line-height:1.15}"
        "h2{font-size:23px}section{background:white;border:1px solid #cfdad3;border-radius:12px;"
        "padding:24px;margin:24px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;"
        "width:100%;margin:16px 0}th,td{padding:10px 12px;border-bottom:1px solid #dee5df;"
        "text-align:left;vertical-align:top}th{font-size:13px;color:#456056}"
        "td{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}"
        "summary{cursor:pointer;font-weight:600;padding:14px 0}pre{overflow:auto;font-size:12px}"
        ".warnings{background:#fff4dc;padding:16px 32px}.missing{color:#69786e}"
        "@media print{body{margin:0;max-width:none}section{break-inside:avoid}}"
        "</style><main><p>GRADIENTCLIMB / CYCLE 3</p><h1>Real competence efficiency</h1><p>"
        + html.escape(EXPLANATION)
        + "</p><p>Source: validated learning-efficiency.json envelopes in the canonical run store. "
        "All values below are canonical analysis outputs; this view does not recompute outcomes. "
        "Full checkpoint curves, cost lineage and warnings are retained in the JSON export.</p>"
        + body
        + "</main></html>"
    )


def _markdown_table(headers: list, rows: list[list]) -> str:
    def escape(value: Any) -> str:
        return (
            html.escape(value_text(value))
            .replace("|", "&#124;")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    return "\n".join(
        ["| " + " | ".join(escape(item) for item in headers) + " |"]
        + ["| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(escape(item) for item in row) + " |" for row in rows]
    )


def render_markdown(reports: list[dict], comparisons: list[dict] | None = None) -> str:
    parts = [
        "# Real competence efficiency — Cycle 3",
        EXPLANATION,
        (
            "Source: validated canonical `learning-efficiency.json` envelopes. "
            "These tables present canonical analysis outputs without recomputing outcomes. "
            "The companion JSON preserves complete curves, costs, warnings and provenance."
        ),
    ]
    if not reports:
        parts.append(
            "Real competence efficiency is not yet measured: no validated study is present."
        )
    for report in reports:
        if report.get("error"):
            parts.append(
                "## Evidence validation failed — "
                + html.escape(str(report.get("study_id")))
                + "\n\n"
                + html.escape(str(report["error"]))
                + "\n\nNo competence or efficiency claim is available for this record."
            )
            continue
        parts.extend(
            [
                "## " + html.escape(str(report.get("study_id", "Unnamed study"))),
                _markdown_table(
                    ["Study definition", "Recorded value"],
                    [
                        ["Prior class", report.get("prior_class")],
                        ["Profile", report.get("profile_id")],
                        ["Protocol", report.get("protocol_id")],
                        ["Real scenario SHA-256", report.get("scenario_sha256")],
                        ["Comparison contract SHA-256", report.get("comparison_contract_sha256")],
                        ["Wall-clock boundary", report.get("wall_clock_boundary")],
                        ["Evaluation split", report.get("evaluation_split")],
                        ["Compute comparison scope", report.get("compute_comparison_scope")],
                        ["Provenance status", report.get("provenance_status")],
                        ["Policy prior roots", report.get("policy_prior_roots")],
                        ["Lineage complete", report.get("lineage_complete")],
                    ],
                ),
                _markdown_table(THRESHOLD_HEADERS, threshold_rows(report)),
                "### Checkpoint evaluation and verification clocks",
                _markdown_table(
                    [
                        "Checkpoint SHA-256",
                        "Own elapsed (s)",
                        "Median real distance (m)",
                        "Verification elapsed (s)",
                        "Evaluation elapsed (s)",
                        "Evaluation run",
                    ],
                    [
                        [
                            point.get(key)
                            for key in (
                                "checkpoint_sha256",
                                "time_seconds",
                                "median_distance_m",
                                "verification_elapsed_seconds",
                                "evaluation_elapsed_seconds",
                                "evaluation_run_id",
                            )
                        ]
                        for point in report.get("checkpoint_curve", [])
                    ],
                ),
            ]
        )
        if report.get("warnings"):
            parts.append(
                "Warnings: " + "; ".join(html.escape(str(item)) for item in report["warnings"])
            )
        for threshold in report.get("thresholds", []):
            parts.extend(
                [
                    (
                        f"### Costs at {threshold['threshold_m']} m "
                        f"({STATUS_LABELS.get(threshold['status'], threshold['status'])})"
                    ),
                    _markdown_table(COST_HEADERS, cost_rows(threshold)),
                ]
            )
        parts.extend(
            [
                "### Evaluation costs (separate)",
                _markdown_table(
                    ["Resource or experience", "Evaluation"],
                    [
                        [label, report.get("evaluation_cost", {}).get(key)]
                        for key, label in COST_LABELS.items()
                    ],
                ),
                "### Prior elapsed clocks and ancestry",
                _markdown_table(
                    ["Cost ID", "Kind", "Parent IDs", "Elapsed (s)", "Human time (s)"],
                    [
                        [
                            node.get(key)
                            for key in (
                                "cost_id",
                                "kind",
                                "parents",
                                "elapsed_seconds",
                                "human_seconds",
                            )
                        ]
                        for node in report.get("prior_ledger", [])
                    ],
                ),
            ]
        )
    for comparison in comparisons or []:
        parts.extend(
            [
                f"## Pareto comparison at {comparison['threshold_m']} m",
                html.escape(str(comparison["note"])),
                _markdown_table(COMPARISON_HEADERS, comparison_rows(comparison)),
            ]
        )
    return "\n\n".join(parts) + "\n"


def write_report(reports: list[dict], output: str | Path) -> list[Path]:
    """Write a new report directory, refusing to replace any existing output."""
    serialized = json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    comparisons = compare_reports(reports)
    rendered = {"report.json": serialized, "report.md": render_markdown(reports, comparisons)}
    rendered["report.html"] = render_html(reports, comparisons)
    rendered["comparisons.json"] = json.dumps(comparisons, indent=2, allow_nan=False) + "\n"
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    for name, content in rendered.items():
        path = destination / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        paths.append(path)
    return paths
