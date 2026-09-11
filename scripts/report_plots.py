"""Standard scientific plotting for the portable research report."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path

COLORS = ("#087f8c", "#bc5a36", "#775a9e", "#4f8061", "#4267a0", "#ad7a23")


def scientific_plot(
    series, x_label, y_label, title, log_x=False, points_only=False, output_format="svg"
):
    # Keep the plotting cache inside the workspace, avoiding desktop/profile writes.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts/matplotlib-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    available = [(label, [p for p in pts if not log_x or p[0] > 0]) for label, pts in series]
    available = [(label, pts) for label, pts in available if pts]
    with matplotlib.rc_context(
        {
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": "#192b35",
            "axes.labelcolor": "#192b35",
        }
    ):
        figure = Figure(
            figsize=(9, 4.6 + max(0, math.ceil(len(available) / 3) - 3) * 0.2), layout="constrained"
        )
        figure.set_facecolor("#fffdf8")
        axes = figure.subplots()
        axes.set_facecolor("#fffdf8")
        if not available:
            axes.set_axis_off()
            axes.text(
                0.5, 0.5, "Not yet measured", ha="center", va="center", transform=axes.transAxes
            )
        else:
            for index, (label, points) in enumerate(available):
                # Thin display samples only; retain all observations in the JSON evidence.
                shown = points[:: max(1, len(points) // 250)]
                if shown[-1] != points[-1]:
                    shown.append(points[-1])
                axes.plot(
                    [p[0] for p in shown],
                    [p[1] for p in shown],
                    color=COLORS[index % len(COLORS)],
                    marker="o",
                    markersize=5 if points_only else 2,
                    linewidth=1.8,
                    linestyle="none" if points_only else "-",
                    label=label[:42],
                )
            if log_x:
                axes.set_xscale("log")
                values = [p[0] for _, points in available for p in points]
                lo, hi = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
                axes.set_xlim(10**lo, 10 ** max(hi, lo + 1))
            else:
                axes.set_xlim(left=0)
            axes.set_ylim(bottom=0)
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.grid(axis="y", color="#dce1df", linewidth=0.8)
            axes.set_axisbelow(True)
            axes.spines[["top", "right"]].set_visible(False)
            axes.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncols=min(3, len(available)),
                frameon=False,
                fontsize=8,
            )
        axes.set_title(title, loc="left", fontsize=13, pad=15)
        buffer = io.BytesIO()
        metadata = {"Creator": "GradientClimb canonical research report"}
        if output_format == "svg":
            metadata["Date"] = None
        figure.savefig(buffer, format=output_format, dpi=140, metadata=metadata)
        value = buffer.getvalue()
    if output_format == "svg":
        # Inline SVG must not embed an XML declaration/DOCTYPE inside the HTML body.
        text = value.decode("utf-8")
        return "\n".join(line.rstrip() for line in text[text.index("<svg") :].splitlines()) + "\n"
    return value
