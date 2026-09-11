# Research dashboard

Start the installed package from the repository:

```powershell
.venv\Scripts\python.exe -m gradientclimb --root artifacts dashboard --port 8765
```

Open <http://127.0.0.1:8765/>. The server binds to loopback and serves its own
HTML, CSS and JavaScript. It uses no remote scripts, CDN, or separate dashboard
dataset. **Refresh** reads new canonical records. Global algorithm, environment,
status, vehicle and map filters apply to every view and exported evidence.

The run catalog also filters simulator version, model/configuration text,
start date and minimum duration. **Reset filters** clears all these restrictions.
Selecting a run opens its configuration, source SHA, worktree state, hardware,
framework versions, evaluations and artifact manifest. Its seal is checked on
demand. **Export view** downloads the selected run population and underlying
records as JSON; **Export run JSON** downloads one complete run record.

## Views and interpretation

| View | Evidence shown |
| --- | --- |
| Overview | Run counts, recorded experience/time, per-run curves, held-out evaluations |
| Runs & lineage | Searchable provenance and explicit parent run/checkpoint relationships |
| Learning & efficiency | Curves versus elapsed time, recorded steps or episodes; final metric distributions across completed runs; quality/time scatter; CPU/GPU samples; cumulative component costs |
| One-hour benchmark | Explicit `one-hour*` evaluation protocols at 5/10/20/30/45/60 minutes, plus saved checkpoint artifact times |
| Transfer & adaptation | Held-out vehicle/map conditions, adaptation measurements and matched checkpoint sim/real comparisons |
| Controls & calibration | Independent gas/brake trajectories, four-state action counts, and recorded calibration errors |

Missing required evidence is labeled **Not measured**. A simulation score never
establishes real-game qualification. Running runs have no finalized duration or
step count. A run interrupted before finalization remains unsealed; the dashboard
does not claim it completed.

The algorithm table is descriptive: it groups the final recorded metric of each
completed run by algorithm and environment. Configuration differences remain
visible in the run detail and may confound a comparison. Report independent
training seeds, matched conditions and appropriate uncertainty for scientific
claims. Its time-to-target column reports the median among runs that reached
the user-entered target, along with the attainment count; nonattainment is not
silently assigned a duration. Per-call latency is unavailable when the trainer
only reports cumulative inference/optimizer/environment time.

## Canonical analytical API

`GET /api/runs`, `/api/runs/{uuid}`, `/api/runs/{uuid}/integrity`, `/api/metrics`,
`/api/summary`, and `/api/analytics/{resources|evaluations|controls}` are read-only.
Interactive OpenAPI documentation is at `/docs`. Analytical endpoints support
`run_id` and `limit` (1–100,000; default 10,000), and return `total_rows` and
`truncated`. The browser discloses sample truncation. Narrow `run_id` queries
to inspect long individual trajectories. The frontend draws at most 12 series
per chart and decimates long paths for rendering while preserving original
observations in exports; it does not interpolate between random seeds.

Each API query opens an independent DuckDB snapshot. Finalized metrics,
telemetry, evaluations and trajectories come from Parquet. Active runs come from
complete lines of append-only JSONL journals. There is no shared DuckDB writer
file to lock a training process, and no dashboard-only source of truth.

## Recording optional evidence

- Set `metadata.evidence_domain` on a run to `real_game` or `simulation` when
  justified by the actual observation source. The versioned surrogate is
  explicitly classified as uncalibrated simulation.
- Set evaluation `protocol` to `one-hour-v1` and
  `metadata.training_minutes` to the governed checkpoint time.
- Generalization evaluations use `metadata.condition` values
  `in_distribution`, `new_map`, `new_vehicle`, or `new_vehicle_and_map` and
  preserve the actual evaluated `profile`/`terrain` in results.
- Adaptation evaluations record `metadata.adaptation_minutes` and their protocol.
- Sim/real ratios require the same `checkpoint_hash`, measurement definition,
  and declared `metadata.comparison_condition`. Ratios with a zero simulator
  denominator remain undefined.

The API does not offer file deletion, arbitrary SQL execution, artifact file
downloads, game control or training mutations.
