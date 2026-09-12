# Real competence efficiency reporting — Cycle 3

The dashboard, report CLI and notebook read the new canonical `learning-efficiency.json` study envelopes. Their scientific results come from `gradientclimb.experiments.efficiency.analyze_study`, reached through the sealed-store `load_studies` loader. Comparisons come from `compare_studies`. The presentation layers format these outputs; they do not recompute qualification, cost sums, acquisition intervals, censoring or dominance.

The existing **Learning & efficiency** dashboard view and Cycle 1 notebooks retain their historical meaning. **Real competence efficiency** is a separate tab. It shows prospective real-game evaluation, elapsed clocks, independent experience dimensions and measured compute dimensions. Simulator performance cannot fill a missing real-game result.

## Read the outcomes

- **Reached:** the first observed independently evaluated frozen checkpoint meeting the registered median real-distance threshold. This is an upper bound on observed acquisition time, not proof of continuous or monotonic competence between checkpoints.
- **Right censored:** no eligible evaluated checkpoint reached the threshold. Censor support stops at the last eligible checkpoint evaluation, even if the training budget extends later.
- **Not evaluable:** no eligible independent evaluation supports the threshold analysis. Unknown values display as **Not measured**. Explicit measured zeros remain zero.
- **Evidence validation failed:** the envelope or a referenced artifact failed validation. The error stays visible and produces no competence claim. A hash-valid artifact still needs a protocol audit for truthful independence, population definition and measurement validity.

Own command-start elapsed time, verification elapsed time, evaluation elapsed time, prior elapsed clocks and human time have separate labels. Prior wall clocks can overlap and are never added into a fictitious serial elapsed clock. Own, prior and combined additive costs remain separate columns. Data-generation CPU/GPU values are subsets of the total compute fields, not extra charges. Evaluation cost appears in its own table. Real episodes, simulator episodes, transitions, decisions, rendered frames, physics steps and optimizer updates remain distinct quantities.

The study selector controls the display. General run filters are hidden in this tab because they do not define a registered efficiency cohort. Pareto classifications are computed over the original compatible cohort and are not recomputed when the display is filtered. Excluded, censored or incompletely measured records remain visible. A per-run descriptive frontier does not establish a repeated-seed advantage.

## Reproduce the report

From the repository root and project virtual environment:

```powershell
.venv\Scripts\python.exe scripts/report_learning_efficiency.py --root artifacts --output artifacts/reports/cycle3-efficiency-review-01
```

The output directory must be new. `report.json` contains the exact list of canonical analysis reports. `comparisons.json` contains the exact canonical comparison outputs for each registered threshold. `report.md` and the standalone, network-free `report.html` display those same values. Invalid study errors remain visible. No source run is modified and no absent study is replaced with synthetic results.

The dashboard API provides `/api/learning-efficiency` as the exact analysis list. `/api/learning-efficiency/snapshot` loads once and returns `{studies, comparisons}` from that snapshot. This prevents the chart and comparison tables from reading different populations if another sealed study is appended during refresh. **Export view** preserves the selected reports and full original comparison cohorts, including errors and unknown values.

## Reproduce the notebook

`notebooks/real_competence_efficiency.ipynb` is a new runnable companion, with saved executed outputs from the available canonical store. Select the project virtual environment kernel. The notebook parameters identify the artifact root and optional study IDs. It loads one snapshot, uses the shared report renderer, and plots exact checkpoint observations only when present. An empty store has a visible missing-evidence result and no invented chart.

To build and execute another snapshot without overwriting the saved notebook:

```powershell
.venv\Scripts\python.exe scripts/build_efficiency_notebook.py --execute --output artifacts/notebooks/real_competence_efficiency_review_02.ipynb --preview artifacts/notebooks/real_competence_efficiency_review_02.html
```

The builder requires the project's `analysis` extras and writes its local kernel runtime beneath `artifacts/notebooks/cycle3-runtime`. It never captures the game, trains a policy, runs policy inference or changes canonical measurements.
