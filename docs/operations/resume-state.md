# Active research state — 2026-09-11

This is a continuation checkpoint, not a final completion report. The user's full
research brief remains active. Do not restart already completed work or silently
substitute simulator evidence for actual Hill Climb Racing qualification.

## Repository

- Starting SHA: b7ca30e1d9f43e701db797f73fc27d7c96f59f98.
- Validated implementation/training source:73c5ea2ec6864a90ec15c8b0b209287507da5bd7.
- Branch dev pushed/tracking origin/dev; main unchanged.
-58 tests passed locally before this continuation's additional independent work.
- GitHub Actions run34627485686 passed for73c5ea2.

## Running benchmark — do not launch a duplicate

Run:c0a9e142-1ad6-4d88-810d-bda4ca297f40. Started17:22:54.783UTC (12:22:54Chicago).
PPO CPU256, seed42, cold start,3600actual seconds. Config:
experiments/definitions/one-hour-surrogate.json. Source recorded clean73c5ea2.
No pilot weights loaded. All results are **uncalibrated surrogate**, not real-game
qualification. Compute session93709; output artifacts/one-hour-console.log.
Canonical files:artifacts/runs/c0a9e142-1ad6-4d88-810d-bda4ca297f40/.
The300s checkpoint exists. Expected completion approximately18:23UTC.

After completion, verify the sealed record and evaluate each checkpoint:

```powershell
.venv/Scripts/python -m gradientclimb experiment verify c0a9e142-1ad6-4d88-810d-bda4ca297f40
.venv/Scripts/python -m gradientclimb evaluate-checkpoints c0a9e142-1ad6-4d88-810d-bda4ca297f40
```

Pilot screen:8runs ×60s requested,481.36s actual, separate from the hour.
PPOCPU256 across3training seeds:491.38mean validation distance,35.08between-seedSD.
CEM:378.49mean,106.31SD. Source-linked records in research/experiments/runtime-screen.json.
Validation seeds10000–10019; final test20000–20019 untouched by learned-policy selection.
Random baseline run:f015c7ca-d1bd-4c8a-bfa9-1a4c2c8877c5 (median20.8918nominalm).
Scripted gas baseline:5235fad7-9d8f-4713-bb8b-1a58029020dd (median35.4455nominalm).
These brief evaluations and normal desktop/dashboard activity occurred during
the governed run; this was not an otherwise idle-machine experiment.

## Actual-game continuation

User explicitly authorized resumed game control using existing arrow keys:
Right=gas, Left=brake. The native Computer Use helper remains Escape-stop-latched
for this tool turn despite that resumption; root stopped native calls and did not
bypass the stop via a different input backend. A fresh conversation turn is needed.

Use the Computer Use skill to reacquire the returned actual game window; do not
trust cached coordinates/handles. The title starts Hill Climb Racing and the
Google Play Games process is crosvm.exe. Do not click the advertisement banner.
A temporary Tap/Q mapping draft was opened while discovering controls; inspect
and clear it before continuing. The native arrows already exist and need no remap.

Observed vehicle Hill Climber, CountrySide, max levels13/13engine,14/14suspension,
16/16tires,10/10drivetrain. No-input discovery result26m out of fuel is not a random
policy evaluation. State flow and local screenshots are documented separately.
Next: validate sustained arrow holds and both pedals; benchmark safe target-only
capture; collect timestamped independent controlled action trajectories; label
held-out frames; fit and validate effective dynamics before real-transfer claims.

Offline labels: corrected run b978c7e7-a4ff-40ae-a1bd-2577ea935010,5actual single-session
images, self-match5/5, heldout0, no measured generalization. Original b5a1748c run
retained but its mistaken default episodic-evaluation count is documented/excluded.
Template prototypes are state-only and not validated to authorize unattended input.

## Dashboard and follow-through

Dashboard localhost:8765, serverPID26828, execsession58258; six analytical views.
All views/filter/reset/details and desktop/mobile rendering inspected with no
browser errors. Query canonical records; never add a second dashboard source.

Remaining original work includes controlled real data, live unattended-loop
validation, actual calibrated simulator/alternative strategy evidence, real
one-hour qualification, transfer, held-out real vehicles/maps, adaptation and
forgetting, selected ablations, extended training, reproducibility rerun and final
scientific reporting. Preregistered pending definitions do not establish results.
The research report must remain explicit about incomplete real-game evidence.

## Paused independent implementation

The harness subagent drafted a bounded screen-session orchestrator. It is preserved
at ignored `artifacts/drafts/session.py`, outside the published package pending
focused control-flow tests. It compiles and passes lint, but was not functionally
validated. Restore it to `src/gradientclimb/control/session.py` only while completing
those tests and review. The simulation subagent's source-linked report generator
was still being prepared; no incomplete generator file was published. Both agents
were interrupted to preserve this checkpoint and let the native tool turn reset.
They can resume their existing bounded subtasks in the next turn.
