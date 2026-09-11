# GradientClimb

GradientClimb is a data-science research platform for rapid-learning adaptive
control, focused on training efficiency, generalization, and sim-to-real transfer.
**Hill Climb Racing is its first experimental testbed.** The research question is
how policy quality changes with wall-clock time, computation, experience, and prior
knowledge—not just whether a controller can drive a car.

Development version: **0.1.0a1**. The scientific harness, original hill surrogate,
PPO/CEM learners, evaluation tools and local dashboard are implemented. Actual-game
qualification is **not established**. Real-game discovery has been performed;
controlled trajectory collection and validated perception/calibration are pending.
Simulation results must not be interpreted as Hill Climb Racing scores.

## Current evidence

Two synthetic convergence experiments exercise configuration/source capture,
metric persistence, hashed artifacts, finalization and DuckDB queries. Local
simulator pilots compare equal wall-clock budgets, not just sample counts.
The [experiment records](research/experiments/) and [research report](research/reports/gradientclimb-research-report.md)
distinguish measured results, exploratory comparisons and outstanding requirements.

The [qualification protocol](docs/methodology/qualification.md) fixes the
5/10/20/30/45/60-minute schedule, prior-training classes, held-out splits, and
real-game evidence requirement. High-level human-equivalent performance currently
has no measured reference distribution and is not claimed.

## Architecture

```text
Versioned configuration + source + machine + seed
                 ↓
  original vectorized surrogate / screen-only game observations
                 ↓
  stacked policy inputs → learned policy → independent gas and brake
                 ↓
  append-only run journals → sealed Parquet records → DuckDB → dashboard
                 ↓
  checkpoint hashes → held-out evaluations → source-linked findings
```

The surrogate uses two wheel-contact points, suspension, terrain, traction,
engine/brake/reverse forces, airborne torque, fuel and failure states. Its nominal
physics are **uncalibrated**. The actor currently receives idealized state estimates;
it is an experimental oracle baseline until a validated visual observation bridge
exists. Four stacked observations provide history. Gas/brake remain independent
through all four states: neither, gas, brake, both. The 10,563-parameter PPO model
uses two Bernoulli outputs; the 100-parameter CEM comparator selects among four
joint states. Scripted and random policies are labeled as baselines.

## Install

Python 3.11+; the research workstation uses Python 3.13. Create an isolated environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train,capture]"
.venv/Scripts/gradientclimb doctor
```

For the tested CUDA runtime, install PyTorch from the official CUDA 12.8 wheel
index before installing this project. CPU training is supported and should be
benchmarked for small policies. [requirements-lock.txt](requirements-lock.txt)
records the exact workstation dependencies; its CUDA wheel needs that official
index. No system driver or CUDA installation is modified.

## Run experiments

```powershell
gradientclimb experiment run --gain 0.05
gradientclimb experiment run --gain 0.4
gradientclimb experiment list
gradientclimb experiment show RUN_ID
gradientclimb experiment verify RUN_ID
gradientclimb experiment compare RUN_ID_1 RUN_ID_2
gradientclimb benchmark throughput --seconds 3
gradientclimb train --algorithm ppo --seconds 60 --envs 256
gradientclimb train --algorithm cem --seconds 60
gradientclimb evaluate --checkpoint PATH_TO_POLICY --episodes 20
gradientclimb evaluate --baseline random --generalization
gradientclimb evaluate-checkpoints RUN_ID
gradientclimb dashboard
```

Use `--root PATH` **before** the command to select another artifact store. Training
uses real monotonic deadlines and logs actual checkpoint overshoot. Post-training
evaluation time is separately reported. `--config FILE.json` passes explicit
algorithm options. Non-cold starts require a parent checkpoint and declared class.

`gradientclimb watch --checkpoint PATH_TO_POLICY` opens one selected surrogate
environment. Add `--video artifacts/rollout.mp4` to record it with installed FFmpeg.
Rendering one policy rollout is separate from headless training collection.

The dashboard runs at [localhost:8765](http://127.0.0.1:8765/) and queries canonical
run data through DuckDB. It does not maintain a second experiment database.

## Actual game integration

Only ordinary screen capture and input are permitted. The installed test instance
runs through Google Play Games. The user's existing controls are Right Arrow for
gas and Left Arrow for brake. Window identity/focus/geometry checks, MSS/Pillow
capture, a short-lease input watchdog, confidence-aware template recognition and
effective dynamics fitting are implemented as building blocks. These have unit
validation, **not a demonstrated unattended game loop**.

See [game discovery](docs/operations/game-discovery.md) and
[integration operations](docs/operations/game-integration.md) for current evidence,
limitations and the resumption procedure. Never deploy simulator-state inputs
directly to the real game, guess menu actions, or treat a timer as evidence that an
advertisement's close button is available.

## Scientific records and reproducibility

Every serious run captures source SHA/dirty state, configuration hash, machine,
framework versions, seed, environment and parent model identity. Finalized records
and Parquet tables have SHA-256 seals; artifacts are registered as private hashed
copies. Seals detect changes but do not replace backup or filesystem access control.
See [schemas](schemas/), [literature](research/literature/),
[decisions](docs/decisions/) and [contributing](CONTRIBUTING.md).

Large frames, videos, checkpoints, telemetry and datasets remain under ignored
`artifacts/`. Git contains code, protocols and compact source-linked results.
Copy complete run directories to preserve local evidence; Git alone does not
contain the generated checkpoints.

## Validation

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

CI checks package installation, lint/format, critical controls and schema behavior,
simulator determinism/reset semantics, PPO bootstraps/checkpoints, queries, synthetic
experiments and dashboard API/assets. Long training and live game automation are
excluded from CI.

## License and game boundary

Original GradientClimb source is MIT licensed following the
[dependency/reuse audit](docs/decisions/ADR-001-licensing.md). Dependencies retain
their own licenses. No proprietary game code or assets are redistributed. This
project is not affiliated with Fingersoft.

The agent must not modify or inject into the game, read private game memory,
intercept protocols, bypass paid content/advertisements, or automate purchases.
Unexpected UI states release controls and halt. Game images stay local.
