# GradientClimb

GradientClimb is a research platform for rapid-learning adaptive control, using
Hill Climb Racing as its first testbed. It measures policy quality against real
wall-clock cost, experience, prior training, generalization and adaptation.

**Research Cycle 1 is intentionally paused. Version 0.1.0a1 is a research
prerelease.** Start with the [resumption checkpoint](docs/operations/resume-state.md)
and [remaining-work plan](docs/operations/resumption-plan.md). The original
scientific program is incomplete; no further experiments are scheduled.

## What the evidence establishes

- An original batched surrogate supports high-throughput PPO/CEM experiments,
  independent gas/brake controls, deterministic tests and governed checkpoints.
- Two independent cold-start PPO runs completed one-hour budgets. The primary
  final policy averaged **681.55 nominal surrogate metres** over 20 validation
  seeds; the reproduction averaged **696.14**, with median **705.09**. This does
  not establish a broad advantage for the reproduction or monotonic improvement.
- Actual-game capture and all four pedal states were exercised through ordinary
  Windows input. Two automated 60-second gas episodes produced paused-boundary
  distance readings of **458 m and 411 m**, including release-to-pause delay.
- The direct-screen CEM pilot drove and obtained a stable **289 m terminal OCR
  reading**, then safely stopped at an unrecognized advertisement. Its strict
  eligibility rules accepted **zero training episodes**; it learned no policy.

**The surrogate is uncalibrated. Its distance is not actual Hill Climb Racing
distance. Real-game learned competence, sim-to-real transfer and a real one-hour
qualification have not been demonstrated.** The actual-game observations above
are bounded integration evidence, not performance qualification.

The [interim research report](research/reports/gradientclimb-research-report.md),
[findings](research/findings/), and [completion audit](docs/methodology/completion-audit.md)
separate measured outcomes, implemented capabilities and deferred work.

## Architecture

Versioned configuration, source, hardware and seeds feed an original vectorized
surrogate or a guarded screen/input adapter. PPO uses idealized simulator state;
the native feature bridge provides masked image-relative observations. These
representations are explicitly separate. A direct-screen linear CEM learner and
an observation-compatible student implementation exist; the student was not trained.

Append-only run journals finalize into sealed JSON/Parquet. DuckDB queries the
canonical records for the six-view local dashboard. Checkpoint hashes, parent
lineage, evaluation seeds and actual checkpoint times remain inspectable.

Native input checks window identity, foreground, geometry and observation age.
Gas and brake have independent short leases. Reset actions require recognized
states and verified controls; unfamiliar advertisements halt the loop. OCR and
pose/terrain extraction have limited validation and incomplete coverage.
See [native operations](docs/operations/native-game-adapter.md).

## Install and validate

Python 3.11+ is declared; Cycle 1 local validation and CI use Python 3.13
(workstation: 3.13.5). Use an isolated environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train,capture,analysis]"
.venv/Scripts/python -m gradientclimb doctor
.venv/Scripts/python -m pip check
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check src tests scripts
.venv/Scripts/python -m ruff format --check src tests scripts
```

The `analysis` extra supports report figures and executed notebooks; `train`
adds Torch and `capture` adds optional capture/perception dependencies. Torch is
loaded lazily so ordinary screen CEM does not initialize it. On this workstation,
Torch import was associated with DXGI capture failure. Capture backends are
explicit (`dxcam`, `mss`, `pillow`), with no silent fallback.

[requirements-lock.txt](requirements-lock.txt) records exact workstation packages.
The CUDA 12.8 Torch wheel requires its official wheel index; portable CPU training
is supported. No broad dependency upgrade is required. See
[environment and validation](docs/operations/stabilization-validation.md).

## Explore existing evidence

```powershell
.venv/Scripts/python -m gradientclimb --root artifacts dashboard --port 8765
.venv/Scripts/python -m gradientclimb experiment list
.venv/Scripts/python -m gradientclimb experiment verify c0a9e142-1ad6-4d88-810d-bda4ca297f40
.venv/Scripts/python scripts/audit_cycle_state.py
```

Open [localhost:8765](http://127.0.0.1:8765/). The dashboard launches with an empty
artifact store and labels missing empirical categories. Existing local records
are required to reproduce its Cycle 1 data. It is a read-only research interface,
not a training scheduler. See [dashboard operations](docs/operations/dashboard.md)
and the resumption checkpoint for report/notebook reconstruction commands.

Generated models, frames, videos and telemetry stay under ignored `artifacts/`.
**Git alone does not contain the trained models or native reference images.**
Preserve complete run directories and the documented local dependencies; the
[integrity inventory](research/experiments/cycle-1-integrity.json) covers all 100
finalized Cycle 1 runs. Hashes detect changes but do not replace a backup.

## Future research direction

On explicit resumption, verify the release and local evidence, then preregister
Cycle 2 objective/behavioral metrics and a bounded native reset/scoring reliability
test before new training. The operational frontier is repeatable scored episodes.
Subsequent qualification must investigate versioned multi-objective fitness:
distance, useful score, pace, survival and recovery-conditioned trick credit.
Raw score can reward spectacular but fatal behavior. Existing objectives and
results remain unchanged; the reward campaign is deferred. See the
[resumption plan](docs/operations/resumption-plan.md).

## License and game boundary

Original source is MIT licensed; dependencies retain their own licenses. See the
[licensing decision](docs/decisions/ADR-001-licensing.md) and
[contributing guide](CONTRIBUTING.md). No proprietary game code or assets are
redistributed. GradientClimb is not affiliated with Fingersoft.

Only ordinary rendered pixels and input are permitted. No private game memory,
injection, protocol interception, purchases, paid-content or advertisement bypass.
Legitimate visible close/advance controls may be used; unknown UI releases input
and halts. Live game validation is separate from CI.
