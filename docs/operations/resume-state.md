# Research Cycle 1 resumption checkpoint

**Superseded as the current entry point.** Cycle 2 ran after this checkpoint and is
itself paused at a stabilized boundary; read
[cycle-2-state.md](cycle-2-state.md) and
[cycle-2-future-work.md](cycle-2-future-work.md) first. Everything below remains the
accurate frozen record of Cycle 1 and is bound to the tag `cycle-1-paused`; Cycle 2
changed none of it. Its "exact first task" was completed by the Cycle 2
preregistration and the two reliability studies that followed.

Cycle 1 is intentionally paused. Do not resume
an old queued battery or launch a one-hour run automatically. The owner's
[pause directive](../methodology/cycle-1-pause-directive.md) supersedes the earlier
open-ended completion mandate for this release.

**Exact first task:** verify the frozen release and local artifact seals, then
preregister Cycle 2 objective/behavioral metrics and a bounded native reset/scoring
reliability test before new training. The first operational gate is repeatable,
scored native episodes with reliable natural termination/reset. Inspect the failed
native pilot below and reacquire current window/control/profile evidence. See the
canonical [resumption plan](resumption-plan.md) for prerequisites and completion
criteria. The new reward campaign is future work, not an automatically queued job.

## Exact repository checkpoint

- Project version: **0.1.0a1**; no ceremonial version bump.
- Original research starting SHA: `b7ca30e1d9f43e701db797f73fc27d7c96f59f98`.
- Stabilization starting dev SHA: `f8183772d78641dc692a1008152af9ba085207bf`.
- Frozen release ref: **`cycle-1-paused`**, an annotated Git tag published only
  after final dev CI, main synchronization and main CI succeed.
- Final main SHA and final dev SHA are the tag's target commit (fast-forward
  integration). The tag annotation records their full literal SHAs, any merge
  SHA and successful workflow IDs. This avoids a self-referential commit hash
  inside its own tracked file.

```powershell
git fetch origin --tags
git rev-parse 'cycle-1-paused^{commit}'
git for-each-ref refs/tags/cycle-1-paused --format='%(contents)'
git rev-parse origin/main origin/dev
```

The tag remains the immutable Cycle 1 checkpoint if branches later advance. Its
containing revision is also the completion audit/report revision. `dev` is
preserved. Historical `REGISTERED_PENDING_DISPATCH` text is a preregistration
snapshot, not a live queue; consult the terminal cycle plan-status record.

## Architecture and versions

| Component | Frozen implementation / boundary |
| --- | --- |
| Records | `src/gradientclimb/experiments/`; schema 1.0.0, append-only journals, sealed JSON/Parquet, artifact hashes and lineage. |
| Surrogate | `surrogate-0.1.0`, `simulation/hill.py`; original batched dynamics, uncalibrated. Oracle observations are not deployable pixels. |
| PPO | `ActorCritic`, tensor-only `.pt` format_version 1, two Bernoulli heads; primary MLP 64×64, stack 4, CPU 256 environments. |
| Simulator CEM | `LinearPolicy`, tensor-only `.pt` format_version 1, categorical four-state policy, final-distance fitness. |
| Screen CEM | JSON policy type `screen-linear-independent-pedals-1`; 34 parameters, independent heads, resumable optimizer state. An initial candidate was exercised; no trained native policy. |
| Student | `.pt` format `screen-body-student-1`; 8 body features and masks × history 4. Implementation tested; training never started. Projection uncalibrated. |
| Native bridge | `hcr-screen-relative-1`; 49 features + 49 masks × history 4 = 392 values. Image-relative estimates, not world position/contact truth. |
| Adapter | Ordinary capture/SendInput; identity/focus/geometry/freshness guards; bounded leases; explicit backend. |
| Dashboard | FastAPI/static assets; read-only DuckDB snapshots of canonical files; no separate database or scheduler. |

Screen schema SHA-256:
`f144123b15edd1dad4d05879aa06e18508cd438cbbdb59837762a3dae2bf509c`.
UI profile: `configs/perception/hcr-reset-ui.json`,
`hcr-wrapper-reset-construction-v1` (version 1, 15 construction variants).
Pixel profile: `configs/perception/hcr-discovery-wrapper.json`,
`hcr-country-hillclimber-discovery-wrapper-prototype-1`.
Both assume normalized 1034×581 images and local references. Run configuration
content hashes distinguish revisions within these prototype profile identifiers.

Historical reward semantics remain source-bound: surrogate PPO uses
`0.1 * new_best_distance - 0.0004 - 1.0 * terminated`; simulator CEM selects mean
final distance; native CEM uses eligible verified episode distance. Legacy records
are not rewritten to invent an explicit reward ID they never carried. No score,
coin or trick composite is implemented. Future version IDs, exact parameters,
observability masks and independent behavioral metrics are required research work.

## Important runs and models

All IDs resolve under `artifacts/runs/<run-id>/`. Read `run.json`, its registered
files and `seal.json`; never modify finalized directories.

| Evidence | Run ID / result |
| --- | --- |
| Primary cold-start hour | `c0a9e142-1ad6-4d88-810d-bda4ca297f40`; clean `73c5ea2`, 3600.3219 training seconds, 41,648,128 transitions; final validation mean 681.5519, median 716.1228 nominal surrogate m. |
| Primary checkpoints | `b8df5a43-deb0-4b3a-917b-2398322c327c`; actual 5/10/20/30/45/60-minute snapshots. |
| Primary held-out conditions | `3a9a0c20-e544-4efd-9c68-e2830d8e8224`; source mean 700.747, rough 122.304, heavy 681.618, combined 146.879. |
| Hour reproduction | `db77b7cd-a11a-473a-8406-06d99b5de5ad`; seed 43, 3600.3484 seconds, 31,382,272 transitions; final validation mean 696.1396, median 705.0922. |
| Reproduction evaluations | `153b8e02-dd2f-442f-9a48-6230d3f5c1cb` checkpoints; `c0a9db18-1344-4f74-b5ba-3e59d81b14ab` held-out conditions. |
| Source extension +10 min | `6f193264-faa5-47ae-88b3-9e0f7ea413a5`; fixed primary parent, 600.3199 seconds; +5 min validation mean 704.3531, +10 min 702.2869. No broad held-out improvement. |
| Combined-shift adaptation +10 min | `24eacdf3-cff5-430f-8978-b2e2bd34f396`; fixed primary parent; report includes checkpoints/retention. No +30/+60-minute result. |
| Simulator random / gas | `f015c7ca-d1bd-4c8a-bfa9-1a4c2c8877c5` / `5235fad7-9d8f-4713-bb8b-1a58029020dd`. |
| Corrected four-state input | `06a5ab40-7201-43e5-a5a8-137325c4c38d`; 35 frames / 7.5-second scripted native probe. |
| Native gas pair | `dad65c73-370f-4df9-9ff1-071ab9999680`; two 60-second episodes, paused-boundary OCR 458/411 m, release-to-pause delay included. |
| Terminal score / reset failure | `0d84a31d-4b78-4506-a869-6bdf7cbb20a3`; 203 m reading, animated result/reset failed. |
| Direct-screen pilot | `51d2527e-9274-40ec-a04d-309117de107d`; clean `f8183772`, requested 600 s, stop at 102.0990 governed s; run duration 106.6049 s, pre-finalize elapsed 107.7381 s. Stable terminal OCR 289 m; unknown ad prevented parking, 0 eligible episodes / 0 optimizer updates. |

The **fixed primary model of record** is:
`artifacts/runs/c0a9e142-1ad6-4d88-810d-bda4ca297f40/files/87c469539110d3134c04ebcf11e95f9c84e595c90ba7e6b2b106917115e04ec0_policy-final.pt`.
The reproduction final hash is
`f9ba71dc136529518c9ade34a51f1d58a3650d5b7a3c918f6994c049148cb41d`.
It has the largest final one-hour validation mean, but paired uncertainty and poor
rough-map outcomes prevent calling it universally better. The extended +5-minute
checkpoint has a higher validation mean at additional cost. Use condition-specific
comparisons rather than a single 'best' label.

## Native facts and required local artifacts

Observed game: Google Play Games, Hill Climber / Countryside, engine 13/13,
suspension 14/14, tires 16/16, drivetrain 10/10. Right Arrow is gas, Left Arrow is
brake. All four combinations are independent. No purchase/unlock was performed.
Reacquire the actual window; cached handles and coordinates are not durable target
identity. Native sessions are stopped; no game input should restart automatically.

- Gameplay bank:
  `artifacts/runs/78a29b33-8cf4-4bb4-be7a-f408acde1fcc/gameplay-glyphs/hud-glyphs.json`.
- Result reader:
  `artifacts/runs/66ce7f08-ef45-466c-ac77-33bacd0f0a41/result-reader.json`.
- UI references include `artifacts/game-discovery/` and paths in the reset profile.
  Restore every referenced file and verify hashes before native use.
- Preserve **all** `artifacts/runs/`, including failures: 100 sealed runs total
  717,929,878 bytes at the pause cutoff. Also preserve auxiliary `game-discovery`,
  `post-benchmark`, `diagnostics`, `demos`, `probe-analysis`, labeling folders and
  the [auxiliary evidence inventory](../../research/experiments/cycle-1-local-evidence.json) and local draft inventory linked by the resumption plan.
- Git contains compact reports/protocols/source, **not** trained models/game pixels.
  A fresh clone runs ordinary tests/dashboard but cannot reconstruct local evidence
  or native templates without this separate store. No external backup is claimed.

DXcam worked in the corrected native pilot after eager Torch imports were removed.
Paired diagnostics associate Torch import with DXGI unsupported failure; the exact
DLL/driver mechanism is unknown. MSS/Pillow are explicit alternatives, not proven
equivalent capture distributions. Do not silently switch within a governed
comparison. See [native operations](native-game-adapter.md).

## Environment, dashboard and reproduction

Workstation: Windows 11, Python 3.13.5, i9-14900HX (24 physical / 32 logical
cores), 63.71 GiB RAM, RTX 4090 Laptop 16 GB, driver 616.56, Torch 2.11.0+cu128.
Small-policy benchmarks selected CPU/256 environments. Concurrent desktop/research
work means these are not isolated-machine benchmarks.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train,capture,analysis]"
.venv/Scripts/python -m gradientclimb doctor
.venv/Scripts/python -m gradientclimb --root artifacts dashboard --port 8765
```

Open <http://127.0.0.1:8765/> and Refresh. Process IDs are not resumption dependencies.
Stop the server with Ctrl+C in its owning terminal. Locked CUDA wheels require the
official PyTorch CUDA 12.8 index; portable CPU installs may differ. See
`requirements-lock.txt` and [validation](stabilization-validation.md).

Evidence/report reconstruction, without new learning:

```powershell
.venv/Scripts/python scripts/audit_cycle_state.py --output artifacts/integrity-now.json
.venv/Scripts/python scripts/analyze_research.py --verify --benchmark-run c0a9e142-1ad6-4d88-810d-bda4ca297f40
.venv/Scripts/python scripts/build_research_notebooks.py --execute
```

`--output` is required and must not be the Cycle 1 inventory. A current scan covers
every canonical run present now, Cycle 2 included, so writing it over
`research/experiments/cycle-1-integrity.json` would change the 100-run number this
document, the completion audit, the report and the notebooks all cite. The Cycle 2
boundary scan is [cycle-2-integrity.json](../../research/experiments/cycle-2-integrity.json).

These commands write derived inventories/reports, never mutate sealed source runs.
Inspect changes before committing. Exact training configurations remain in
`experiments/definitions/` and canonical records; **do not execute them merely to
open this checkpoint**. Historical plans are not an automatic queue.

## Unproven boundaries

Native unattended reliability, trained native policy quality, real qualification,
calibrated world dynamics, broad independent perception accuracy, matched
sim-to-real transfer, real held-out vehicles/maps, real adaptation/retention and
human-equivalent performance remain unestablished. Surrogate adaptation/ablations
are narrower evidence. Student and ignored dynamics drafts have no trained real
performance. Future reward design must retain distance independently and test
provisional trick credit plus reward-hacking cases.

Read the [completion audit](../methodology/completion-audit.md) for original gate
coverage, the [interim report](../../research/reports/gradientclimb-research-report.md)
for results, and the [resumption plan](resumption-plan.md) for the ordered research
map. No reconstruction from chat history is required.
