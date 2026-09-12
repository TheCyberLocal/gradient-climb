# Reproducible analysis notebooks

These notebooks contain executed cell outputs from canonical local records. They do not train models, run model inference, or alter source measurements.

Research Cycle 1 is paused. The refreshed evidence cutoff is `2026-09-11T20:01:46.950554Z`, covering all 100 finalized canonical records in the matching integrity audit. Both one-hour cold starts, all twelve short ablations, the source extension and heavy/rough fine-tuning completed; the student pilot never started. The audit and plan disposition are in `research/experiments/cycle-1-integrity.json` and `cycle-1-plan-status.json`.

- `checkpoint_learning.ipynb`: parameterized DuckDB query for the primary training curve, exact requested/actual checkpoint times, independent recomputation of episode means, paired final-versus-earlier comparisons and the completed seed43 reproduction.
- `generalization_adaptation.ipynb`: four-condition primary generalization, survival/termination outcomes, and conditional parent/child comparisons when matching completed records exist.

The fixed primary run is `c0a9e142-1ad6-4d88-810d-bda4ca297f40`; checkpoint evaluation is `b8df5a43-deb0-4b3a-917b-2398322c327c`; generalization evaluation is `3a9a0c20-e544-4efd-9c68-e2830d8e8224`. Their source paths, record hashes, checkpoint lineage, and seal checks appear in the notebooks. The local `artifacts/runs/` store is required to rerun them; notebooks do not fetch missing data or substitute synthetic values.

Run from the repository root using the project virtual environment:

```powershell
.venv\Scripts\python.exe scripts/build_research_notebooks.py --execute
```

The builder uses `nbformat` and `nbclient`, validates each notebook, executes every cell in order in a fresh project-venv kernel, and saves outputs. HTML previews and extracted rendered figures are written beneath `artifacts/notebooks/`. When opening an `.ipynb` interactively, select the project's `.venv\Scripts\python.exe` kernel. The builder creates its own temporary kernel specification under `artifacts/` and does not require a globally registered kernel.

Reusable statistical and storage methods remain in `src/gradientclimb/evaluation/`, `src/gradientclimb/experiments/`, and the report analysis helper. Notebook cells contain queries, validation, and presentation rather than a separate training implementation.

The initial executed figures were visually inspected and their output tables checked against the sealed records. Intervals describe episode variation conditional on fixed policies, not independent training-seed uncertainty. All distances are nominal units in an uncalibrated simulator. Real-game competence, calibration, perception accuracy, and transfer are not established by these notebooks.

## Cycle 3 real competence efficiency

`real_competence_efficiency.ipynb` is a separate prospective companion. It loads sealed Cycle 3 efficiency study envelopes, uses the same analysis/comparison outputs as the new dashboard tab, and preserves unknown, censored and invalid evidence. Its initial executed snapshot contains zero efficiency study envelopes and therefore makes no real competence or efficiency claim. Historical notebooks above retain their earlier scope.

The notebook's input root and optional display selection are explicit parameters. New outputs can be built without replacing this snapshot using `scripts/build_efficiency_notebook.py --execute --output <new.ipynb> --preview <new.html>`. See [reporting instructions](../docs/operations/learning-efficiency-reporting.md) for the full command and cost/clock definitions.
