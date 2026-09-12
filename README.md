# GradientClimb

GradientClimb is a research platform for rapid-learning adaptive control, using
Hill Climb Racing as its first testbed. Rapid learning means elapsed time to
independently evaluated real-game competence. Compute, experience, prior training,
evaluation latency, generalization and adaptation are measured separately.

> **Episodes measure experience. Compute measures cost. Wall-clock time measures rapidity. Real-game capability determines whether the learning mattered.**

**Cycle 3 research is paused at the owner's 2026-09-12 stabilization checkpoint.
Version 0.1.0a2 remains a research prerelease; scientific qualification is incomplete.**
Start with the [stabilization report](research/reports/cycle-3-stabilization-report.md),
[current handoff](docs/operations/cycle-3-stabilized-state.md) and
[future goals](docs/operations/cycle-3-future-work.md). The owner authorized integration
through `dev` into `main` as a software checkpoint, with no new research dispatch.
The research scope remains recorded in the
[current mandate](docs/methodology/cycle-3-mandate.md),
[requirements/evidence register](docs/methodology/cycle-3-evidence-register.md) and
[bounded campaign ledger](experiments/definitions/cycle-3-campaign.json).
The [learning-efficiency methodology](docs/methodology/learning-efficiency.md)
governs new comparisons. Registration alone does not dispatch an experiment.

Cycles 1 and 2 remain frozen historical boundaries at `cycle-1-paused` and
`cycle-2-paused`. Their [Cycle 2 record](docs/operations/cycle-2-state.md),
[future-work queue](docs/operations/cycle-2-future-work.md),
[Cycle 1 checkpoint](docs/operations/resume-state.md) and
[plan](docs/operations/resumption-plan.md) are preserved. The current register
explicitly dispositions all five inherited, unexecuted Cycle 2 registrations.

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
separate measured outcomes, implemented capabilities and deferred work. The report and
its notebooks cover Cycle 1 only; Cycle 2 has findings but no report yet.

## What Cycle 2 added

Cycle 2 preregistered its measurements before collecting them, then spent itself on the
operational frontier: making a real episode cycle repeatable. It did not get there.

- **Two registered reliability studies ran and both failed their criteria.** Study 2.0
  met three distinct unseen UI phases and reached a longest scored-success run of **1**
  against a criterion of 10 ([F-004](research/findings/F-004-native-reliability-study-1.md)).
  Study 2.1 failed after one session when an allowlisted advertisement control opened the
  advertised app's Play Store page in the host browser
  ([F-005](research/findings/F-005-allowlisted-ad-control-opened-store-page.md)). Both
  failures became mechanism rather than relaxed criteria: the UI profile grew to 22 audited
  hash-pinned variants, advertisement controls are enabled only where a sealed run shows a
  recognized game state afterwards, unintended actions are now detected by effect as well as
  by allowlist, and the only escape from a stuck screen is an application restart that posts
  WM_CLOSE and relaunches the shortcut instead of clicking anything. `native-reliability-2.2`
  is registered with unchanged criteria and **has not been run**.
- **Live headed and headless training works and its cost is measured.** One isolated
  environment follows detached policy snapshots at about **12.7 % fewer environment steps**
  at 64 environments over 60 s, across three paired seeds
  ([F-003](research/findings/F-003-live-headed-training-overhead.md)).
- **The restricted screen-body student was trained, and the budget hypothesis failed.**
  Tripling the budget to 1,800 s left the mean at **483.6** simulator units against the
  teacher's 673.4, with 11 of 20 episodes still crashing
  ([F-006](research/findings/F-006-body-student-distillation-plateau.md)). This is an
  uncalibrated analytic projection, and it is a negative result.
- **Behavioral metrics and objectives exist ahead of their data.** `behavioral-metrics-2.0`,
  objective families A–E, a recovery definition and an eight-indicator reward-hacking battery
  are implemented and tested under the
  [preregistration](docs/methodology/cycle-2-preregistration.md). No episode is yet eligible
  for the score- or recovery-dependent arms, because their readers are unvalidated.

Cycle 2 therefore added measurement, mechanism and four findings — three of them negative —
without adding real-game competence. Reliable scored native episodes remain the gate.

## Current Cycle 3 work

New [measurement and provenance contracts](docs/methodology/cycle-3-measurement-contract.md)
separate gameplay, readiness, safety, missing measurements and censoring.
[Run recovery](docs/operations/cycle-3-run-recovery.md) documents truthful partial
work and warm-start semantics. [Environment/profile contracts](docs/operations/environment-contracts.md)
preserve seeded legacy behavior and reject unimplemented reference environments.
These are software foundations; they do not qualify the readers, simulator or
learned driver in the actual game.

The [learning-efficiency framework](experiments/definitions/cycle-3-learning-efficiency.json)
registers median 500/1,000/2,000 real-metre thresholds, 20 independent executions
per unique frozen checkpoint, three training seeds and a 900-second gameplay
horizon. Its concrete method, source, priors, scenario, native reliability/readers
and entry-clock gates remain required before collection. Checkpoint production
time and later independent verification time are reported separately. Fidelity
comparisons include measured compute and inherited costs; simulator frames per
second are diagnostic, not evidence of rapid real learning.

## Architecture

Versioned configuration, source, hardware and seeds feed an original vectorized
surrogate or a guarded screen/input adapter. PPO uses idealized simulator state;
the native feature bridge provides masked image-relative observations. These
representations are explicitly separate. A direct-screen linear CEM learner exists and
has never completed an eligible update on the real game. The observation-compatible
restricted student has now been distilled twice in the simulator and stays well behind
its teacher; no student or CEM policy has been deployed on the real game.

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

## Watch training live

One representative simulator environment can follow the evolving PPO policy
while the N training environments stay headless. `--headed` opens a Tk window;
`--headless-record` counts frames without a window and can encode an MP4.

```powershell
.venv/Scripts/python -m gradientclimb --root artifacts train --seconds 600 --envs 64 --seed 7 --headed
.venv/Scripts/python -m gradientclimb --root artifacts train --seconds 600 --envs 64 --seed 7 --headless-record --observer-video artifacts/videos/seed7.mp4
```

The observer uses its own seed, environment, model copy and RNG, contributes no
learning data, and changes no training hyperparameter: the recorded configuration
differs only by a separate `observer` block and the training metric series are
the same set either way. Observer problems (no window, no FFmpeg, a closed
window, an encoder failure) are recorded in the run's observer summary and never
fail a finished training run. `--observer-policy best` follows a lagging
training-signal selector, not held-out evaluation. Its learner-side cost is
measured per run and by the paired protocol in
`scripts/measure_headed_overhead.py`, which seals a `headed-overhead-protocol`
measurement record (never a PPO run). See
[headed training](docs/operations/headed-training.md).

## Explore existing evidence

```powershell
.venv/Scripts/python -m gradientclimb --root artifacts dashboard --port 8765
.venv/Scripts/python -m gradientclimb experiment list
.venv/Scripts/python -m gradientclimb experiment verify c0a9e142-1ad6-4d88-810d-bda4ca297f40
.venv/Scripts/python scripts/audit_cycle_state.py --output artifacts/integrity-now.json
```

Open [localhost:8765](http://127.0.0.1:8765/). The dashboard launches with an empty
artifact store and labels missing empirical categories. Existing local records
are required to reproduce its Cycle 1 data. It is a read-only research interface,
not a training scheduler. See [dashboard operations](docs/operations/dashboard.md)
and the resumption checkpoint for report/notebook reconstruction commands.

Generated models, frames, videos and telemetry stay under ignored `artifacts/`.
**Git alone does not contain the trained models or native reference images.**
Preserve complete run directories and the documented local dependencies. The
[Cycle 1 inventory](research/experiments/cycle-1-integrity.json) covers all 100 runs
finalized at that pause and is frozen; the
[Cycle 2 inventory](research/experiments/cycle-2-integrity.json) covers all **134**
verified at this boundary. `audit_cycle_state.py` requires an explicit `--output` so a
new scan cannot overwrite a published inventory. Hashes detect changes but do not
replace a backup.

## Current research gates

Repeatable scored native episodes, a representative soak and independently
validated readers precede governed real learning qualification. The
[native 2.3 amendment](experiments/definitions/cycle-3-native-reliability-2.3.json)
preserves the reliability criteria while correcting implementation semantics;
the inherited 2.2 registration remains unexecuted. Every new study requires its
specific committed protocol, source and entry gates before collection.

On explicit resumption, the campaign's future queue proceeds through bounded
demonstration, engine/fidelity and learner screens before selected replications.
Qualification still covers distance, useful
score, pace, survival and recovery-conditioned skill credit with validated readers
and controlled comparisons. Additional real profiles and retention follow
reference competence. The evidence register distinguishes implemented contracts,
validated measurements, unmet targets and external access blockers; historical
objectives, findings and sealed records remain unchanged.

## License and game boundary

Original source is MIT licensed; dependencies retain their own licenses. See the
[licensing decision](docs/decisions/ADR-001-licensing.md) and
[contributing guide](CONTRIBUTING.md). No proprietary game code or assets are
redistributed. GradientClimb is not affiliated with Fingersoft.

Only ordinary rendered pixels and input are permitted. No private game memory,
injection, protocol interception, purchases, paid-content or advertisement bypass.
Legitimate visible close/advance controls may be used; unknown UI releases input
and halts. Live game validation is separate from CI.
