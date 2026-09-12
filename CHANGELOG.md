# Changelog

## 0.1.0a2 — research prerelease, Cycle 2 stabilized and paused

Research Cycle 2 is paused at a stabilized boundary with its registered program
incomplete; see [cycle-2-state.md](docs/operations/cycle-2-state.md) for the boundary
record and [cycle-2-future-work.md](docs/operations/cycle-2-future-work.md) for the
deferred queue. Nothing below demonstrates real-game learned competence.

### Measurement and preregistration

- Cycle 2 preregistration: `behavioral-metrics-2.0` per-episode outcome vector, objective
  families A–E (`distance-only-2.0` through `recovery-conditioned-2.0` plus optional
  `lexicographic-2.0`), the `recovery-progress-2.0` definition with its airtime and rotation
  event detectors, and an eight-indicator reward-hacking battery, in
  `experiments/objectives.py` with tests. Missing fields stay `None`; derived ratios publish
  numerator, denominator and eligibility instead of an extreme value. Backward loops are
  reported as unmeasurable rather than as zero. Registered protocols: native reliability
  2.0/2.1/2.2, real baselines, reader validation, context encoder, privileged critic.
- Held-out result-reader tooling: a sealed bank builder, contact-sheet labeling and a
  label evaluator (`scripts/build_result_reader.py`, `label_contact_sheet.py`,
  `evaluate_reader_labels.py`); the terminal bank reached 58 glyphs from construction
  sessions only. Coin-counter and fuel-gauge HUD readers are declared but unvalidated.

### Live headed and headless training

- One isolated simulator environment follows periodic detached policy snapshots
  (`--headed`, `--headless-record`, `--observer-*`) with reusable Tk, FFmpeg and null frame
  sinks, plus the paired `headed-overhead-paired-0.1` protocol in
  `scripts/measure_headed_overhead.py`. Observer evidence never fails a finished run;
  per-episode snapshot provenance matches the loaded weights in `best` mode; the observer's
  CPU telemetry no longer shares psutil state with the run recorder; observer seeds inside
  the held-out ranges 1000–1019 and 2000–2019 are rejected; the protocol record is labelled
  `headed-overhead-protocol` with a seed-level paired-bootstrap CI label, validates that the
  on arm's observer ran, and writes a fresh state directory per execution.
  `telemetry/provenance.py` gained `gpu_sample()`. Measured cost is in F-003.

### Native reliability hardening

- After studies 2.0 and 2.1 failed (F-004, F-005): hash-pinned variants for three Google
  interstitial player sizes, the current out-of-fuel result and the vehicle-selection cold
  boot, taking the UI profile to 22 audited variants; advertisement controls enabled only
  where a sealed run shows the close followed by a recognized game state, and required on
  two fresh frames at least 0.15 s apart.
- Effect-based unintended-action detection: a click followed within 5 s by loss of the
  foreground, with the foreground window recorded as evidence, derived session-wide from the
  adapter traces and latched for guard-class faults raised inside the capture read, the click
  path and the pedal watchdog.
- Bounded no-input waits for an advertisement without a verified control, for an unknown
  screen after an advertisement and for a contextless unknown screen; application restart
  recovery (`NativeGameAdapter.restart_app`, `--restart-shortcut`, `--probe-restart`) that
  posts WM_CLOSE and relaunches the game's shortcut instead of clicking anything, only after
  one of three bounded stuck errors, never after a latched fault and only while the game
  holds the foreground. Restart reasons and boot frames are recorded separately and validated
  against the registered protocol bounds.
- Airtime detection became radius-relative; the vehicle-selection start variant supports cold
  boots.

### Findings

- F-003: live headed training costs about 12.7 % of environment steps at 64 environments.
- F-004: native reliability study 2.0 failed on three unseen UI phases; longest scored
  success run 1 against a criterion of 10.
- F-005: an allowlisted advertisement control opened the advertised app's Play Store page in
  the host browser; seven advertisement controls were disabled in response.
- F-006: tripling the restricted student's distillation budget did not close its gap to the
  teacher (negative result, simulator only).

### Stabilization

- `scripts/audit_cycle_state.py` requires an explicit `--output` and refuses to replace an
  existing inventory without `--overwrite`. Its previous default overwrote the frozen 100-run
  Cycle 1 inventory that the Cycle 1 report and notebooks cite.
- `research/experiments/cycle-2-integrity.json` records 134 of 134 canonical local runs
  verified valid at this boundary, zero unfinished and zero uncatalogued.

## 0.1.0a1 — research prerelease, Cycle 1 stabilized and paused

- Canonical scientific run records, Parquet telemetry, DuckDB queries and artifact seals.
- Original vectorized hill vehicle surrogate and wall-clock governed PPO/CEM baselines.
- Independent gas/brake actions with ordered transitions and expiry-based input release.
- Pixel-only game discovery and explicit separation of surrogate and real evidence.
- Primary-source literature review, license audit, benchmark protocol and local dashboard.

Neither prerelease implies completion of real-game qualification.
