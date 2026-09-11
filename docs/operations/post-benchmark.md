# Completed post-hour battery — Cycle 1 paused

All 23 actions finalized successfully by 2026-09-11 20:01:47.900541 UTC.
The 15 learning runs consumed 5,523.3294751 actual training seconds against 5,520
requested. No work is queued. Read the [terminal plan-status record](../../research/experiments/cycle-1-plan-status.json)
and [interim report](../../research/reports/gradientclimb-research-report.md).
Commands below preserve reproducibility of the historical protocol; they are not
an instruction to restart it when opening the repository.

The execution plan is `experiments/definitions/post-hour-battery.json`, version `post-hour-battery-1.1`. It was registered before dispatch. Its canonical JSON SHA-256 is `5fa2b03e363f746978730b9d5535a53b69226a98e972d43befd35466e44cde23` (sorted JSON keys using the dispatcher's serialization). Version 1.0 was revised before any job executed. The plan's original registration-status field is immutable; current execution state comes from the canonical runs and `artifacts/post-benchmark/<plan_sha256>/events.jsonl`.

The governed source is run `c0a9e142-1ad6-4d88-810d-bda4ca297f40`. No job may start until that run has completed and passed artifact verification, no PPO/CEM run remains active, and the root task dispatches this sequence.

| Phase | Learning runs | Requested learning seconds | Offline evaluation |
| --- | ---: | ---: | --- |
| Source | 0 | 0 | Six scheduled checkpoints; four-condition final generalization |
| Fresh reproduction, seed 43 | 1 | 3600 | Built-in validation; six scheduled checkpoints; four-condition final generalization |
| Paired component screen | 12 | 720 | Each run's built-in 20-episode validation |
| Additional source training | 1 | 600 | Built-in validation; 300/600-second checkpoint scores; four-condition final generalization |
| Heavy/rough adaptation | 1 | 600 | Built-in validation; 300/600-second checkpoint scores; paired four-condition final generalization |
| **Total** | **15** | **5520** | **23 expanded sequential jobs including evaluation jobs** |

The plan contains the exact configurations and realized seed-paired ablation order. The initial one-hour configuration is preserved for the reproduction. Fine-tuning and adaptation each load the original source checkpoint; neither loads the other child. Both use the `fine_tuning` category because the parent is not a demonstrated broad generalist.

The 600-second children save checkpoints at requested 300/600 seconds. Explicit offline checkpoint evaluation measures both saved models on each child's training condition. The plan separately evaluates their final policies across the four generalization conditions. Requested 20/30/45/60-minute child measurements remain unmeasured. No value may be extrapolated or copied backward from a final result.

Generalization seeds 20000–20019 are paired across policies. Reuse supports before/after comparisons; it does not make subsequent evaluations independent untouched test sets. The 60-second ablation screen is separate from the earlier proposed 300-second screen, which remains unexecuted.

## Dry run and dispatch

From the repository root, this command validates the plan and prints the expanded requested budget without executing jobs:

```powershell
.venv\Scripts\python.exe scripts/run_post_benchmark.py --plan experiments/definitions/post-hour-battery.json
```

After root dispatch, execute the registered plan:

```powershell
.venv\Scripts\python.exe scripts/run_post_benchmark.py --plan experiments/definitions/post-hour-battery.json --execute
```

Resume that same plan after an interruption:

```powershell
.venv\Scripts\python.exe scripts/run_post_benchmark.py --plan experiments/definitions/post-hour-battery.json --execute --resume
```

The reproduction has an additional clean-worktree guard and now follows the source evaluations, before all other learning jobs. All source changes intended for that run must be committed by the root task before it starts. Generated reports are tracked; keep any report refreshes under `artifacts/` until the reproduction subprocess has started. A changed order or configuration must receive a new plan version/hash before execution.

## Durable records and recovery

State is written beneath `artifacts/post-benchmark/<plan_sha256>/`:

- `plan.json`: exact registered execution plan.
- `events.jsonl`: fsynced timestamped job start, completion, failure and verified-resume events.
- `*.job.json`, `*.result.json`: atomic job specifications and canonical completed-run identifiers.
- `*.console.log`: append-only subprocess output.
- `outcomes.json`: atomic action-to-run mapping, refreshed after every completed action.

Each job runs in a fresh Python process. Canonical run records retain actual monotonic training duration, environment transitions, parent hashes, source version, and offline evaluation duration. The dispatcher separately journals total subprocess duration, including process startup and its work. Requested 5520 seconds excludes offline evaluation, imports, process startup and artifact verification; actual training can slightly overshoot a deadline.

Resume skips a job only after its completed run seal verifies and its configuration, seed, budget and checkpoint lineage match the registered job. A still-running canonical learning record blocks concurrent work. If interruption occurs after sealing a run but before its result file is saved, inspect that job's console and canonical records before recovery; do not blindly start another training run.

After measurements finish, regenerate and verify the report:

```powershell
.venv\Scripts\python.exe scripts/analyze_research.py --root artifacts --benchmark-run c0a9e142-1ad6-4d88-810d-bda4ca297f40 --verify
```

All measurements concern an uncalibrated simulator on the same physical workstation. Normal desktop activity and any workload overlap are disclosed. They do not establish real-game competence or transfer.
