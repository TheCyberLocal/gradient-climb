# Cycle 1 stabilization validation

This document records release checks, separate from scientific qualification.
The exact final dev/main SHA and successful CI workflow IDs are in the immutable
annotated tag `cycle-1-paused`; see [resume-state.md](resume-state.md). That tag is
published only after both branch checks pass. The prerelease remains 0.1.0a1.

## Local checks

- Windows 11 / Python 3.13.5; existing locked environment, no dependency upgrades.
- Editable installation rebuilt successfully with
  `python -m pip install --no-deps --no-build-isolation -e .`; package version unchanged.
- `python -m pip check`: no broken requirements.
- Full suite after final stabilization: **297 passed**, two dependency deprecation warnings, 57.88 s.
- `python -m ruff check src tests scripts`: passed.
- `python -m ruff format --check src tests scripts`: passed, 87 files already formatted.
- `python -m compileall -q src`: successful.
- Two notebooks refreshed and executed in fresh local kernels (15/11 cells),
  querying existing evidence; no model retraining. Report self-test and focused
  report/plan tests pass.

Two dependency deprecation warnings originate in Starlette's current TestClient
integration with httpx/AnyIO. They do not fail validation; no broad upgrade or
extra dependency substitution was made. Initial sandboxed pip/test attempts hit
Windows temporary-directory ACL restrictions; the authorized local validation
rerun succeeded. These were execution-environment failures, not test assertions.

Python >=3.11 is the declared package range. This cycle directly tested 3.13
locally and in CI; it did not execute a full Python-version matrix. `train`
installs Torch; `capture` installs MSS/OpenCV and Windows-only DXcam; `analysis`
installs Matplotlib/notebook tooling; `dev` installs tests/lint/API client support.
The workstation lock includes Torch 2.11.0+cu128 and needs the official CUDA 12.8
wheel index. CPU installs remain supported. Torch-free native imports are tested
in fresh subprocesses; optional backend choice is explicit and recorded.

## Bounded fixes

- Ensure menu interruption, including KeyboardInterrupt, attempts mouse-up,
  records cleanup results, latches stop and releases pedals.
- Separate hardware fingerprint identity from whether Torch has been imported.
  New observations mark `hardware-observation-v2`; historical hashes are unchanged.
- Repair native evaluation and timed-checkpoint/lineage dashboard classification
  using actual canonical contracts. Missing data stays missing, and continued
  training is distinguished from cold-start one-hour results.

No reward definition, simulator dynamics, native recognition thresholds or sealed
experiment record was changed. No live game input was needed for stabilization.

## Artifact and plan integrity

[Full inventory](../../research/experiments/cycle-1-integrity.json): all 100 local
canonical runs passed SHA-256/size/configuration/artifact/seal checks; none was
running or unsealed. They total 717,929,878 bytes, including 18 failed runs.
The [auxiliary inventory](../../research/experiments/cycle-1-local-evidence.json)
identifies 121 local files in ten evidence/draft folders and verifies all 15 native
UI reference hashes. Drafts, diagnostic scripts and proposed student/dynamics
plans are inventoried, not adopted or falsely counted as experiments.

The completed [post-hour battery](../../research/experiments/cycle-1-plan-status.json)
finished all 23 actions with exit 0 at 20:01:47.900541 UTC on 2026-09-11. The native
pilot had already safely finalized failed. No active governed process remains.
The student pilot was cancelled before dispatch. Long unexecuted preregistrations
remain historical files with a separate terminal cycle disposition.

Raw snapshot hashes describe exact local bytes. Git's `.gitattributes` normalizes
tracked text to LF; some Windows support-file snapshots used CRLF. A fresh clone
can therefore have different raw support-text hashes without different parsed
configuration/code. Use the recorded revision, parsed configuration hash and
original archived snapshot for comparison. Sealed artifact bytes must always match
exactly; never normalize or rewrite the artifact store to satisfy a hash check.

## Dashboard verification

The documented loopback startup command launches successfully. Browser checks
cover all six views, finalized run totals, real baseline presence, checkpoint
results, adaptation/retention labels, missing sim-to-real evidence, and plots.
Empty-store API/rendering contracts are tested; categories with no measurements
remain explicit. The dashboard serves canonical data and launches without local
training artifacts, though a fresh clone then has an empty catalog.

The local dashboard may remain open for review; it does not run experiments.
No production hosting or public game imagery/model upload was performed.

## Git and CI readiness

Stabilization began at dev `f8183772d78641dc692a1008152af9ba085207bf` with a clean
worktree and matching origin/dev. Origin/main was
`b7ca30e1d9f43e701db797f73fc27d7c96f59f98`; starting dev CI run 34639870366 was green.
Final release requires a clean worktree, matching dev/origin/dev, green `validate`
workflow at the exact final dev SHA, unchanged main ancestry, fast-forward merge,
remote main verification and green main CI. The annotated release receipt records
these final results outside self-referential tracked file hashes.

The workflow installs the package, runs Ruff lint/format, pytest, a small synthetic
recorder smoke test and compilation on Ubuntu/Python 3.13. It never interacts with
the actual game. A green release validates the implemented research platform; it
does not establish the deferred scientific outcomes in the completion audit.
