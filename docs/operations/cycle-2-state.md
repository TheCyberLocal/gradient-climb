# Research Cycle 2: operational state

**Cycle 2 is paused at the stabilization boundary recorded at the end of this
document.** This document is the operational state of the resumed program, updated
at every stabilization boundary on `dev`; it never rewrites Cycle 1 records, which
remain bound to the tag `cycle-1-paused`. Read
[cycle-2-future-work.md](cycle-2-future-work.md) for the deferred forward queue,
[resume-state.md](resume-state.md) for the frozen Cycle 1 checkpoint and
[cycle-2-preregistration.md](../methodology/cycle-2-preregistration.md) for the
registered metric, objective and reliability protocols.

## Checkout and storage

- Cycle 2 development started from `ebab4ef` on `dev` and was integrated into `main`
  by fast-forward at this boundary; the exact SHAs are in the `cycle-2-paused`
  annotation.
- The checkout moved to `D:\Projects\gradient-climb` on 2026-09-11 (a 1 TB drive
  provided for large files). `artifacts/` lives directly under the project again
  after a temporary junction detour. The Google Play Games emulator lives on C:,
  terminates app sessions when C: free space falls below about 4 GB and refuses to
  boot when its level is Critical; a concurrent unrelated build once drained C: at
  0.6 GB per minute and crashed the game mid-advertisement.
- Large files (models, frames, videos, Parquet) stay in ignored `artifacts/`;
  obsolete space-consuming data is deleted rather than kept.

## Native operations (real game)

- Game: Hill Climb Racing under Google Play Games (`crosvm.exe`), window title
  `Hill Climb Racing - KiraT2S`, client 2581×1449 normalized to 1034×581. Relaunch
  with the Start Menu shortcut, which resolves to
  `googleplaygames://launch/?id=com.fingersoft.hillclimb`; a cold boot lands on the
  vehicle-selection screen (Hill Climber selected, Countryside retained), which
  profile version 2 recognizes with Start as its only control.
- Configuration verified at resumption: Hill Climber, Countryside, upgrades
  13/13, 14/14, 16/16, 10/10, identical to Cycle 1.
- Focus discipline: the desktop tool session must not write or display files while
  a native session runs, because that raises the assistant window over the game and
  the adapter halts on foreground loss (exploratory session `91c3d4f4`). Native
  sessions therefore run as single foreground commands with tools idle.
- UI profile `hcr-wrapper-reset-v2` (22 variants): Meta Audience Network chrome
  (skip-forward and muted-speaker glyphs, binarized-shape matching), the
  Google-served interstitial at three player sizes (small, tiny end card, medium;
  dim skip glyph or Close pill plus a creative-independent static marker), the
  Fingersoft commercial break, the current build's out-of-fuel result and the
  vehicle-selection screen, in addition to the Cycle 1 variants. Every addition
  was audited offline over all stored frames (`scripts/audit_ui_profile.py`;
  sealed audits `7b73824a`, `745ea91e`, `b73aa968`, `5bd55d79`, `d1613fff`,
  `6b807f8e`, `8dd9d796`).
- Advertisement controls are enabled only where a sealed run shows the close followed
  by a recognized game state (`commercial_break_available_close`,
  `admob_tiny_layout_close`); the other eight advertisement variants are
  recognition-only (seven disabled after F-005; the muted-speaker variant never carried
  a control). Advertisement closes require the same control on two fresh frames
  ≥ 0.15 s apart.
- Unintended actions are measured by effect as well as by allowlist: a latched
  foreground-loss guard event within 5 s of an accepted click halts the session as
  `unintended_action`, counted session-wide from the adapter traces with the foreground
  window recorded. Stuck screens (three bounded errors only: advertisement without a
  verified control, unknown after an advertisement, unknown without context, each after
  a 45 s no-input wait) are escaped by `restart_app` (WM_CLOSE to the pinned window,
  then the Start Menu shortcut
  `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Google Play Games\Hill Climb
  Racing.lnk`). Sealed probe run `a3422d1c`: the window hid 0.5 s after WM_CLOSE,
  reappeared 9.5 s after the shortcut with the same handle, process and geometry, and
  reached the recognized vehicle-selection screen 31.7 s later. Enabled in the runner
  with `--restart-shortcut`; `--probe-restart --exploratory` seals one restart alone.

## Exploratory native sessions so far (not study attempts)

| Session | Outcome |
| --- | --- |
| `fa103b59` | Natural result at 398 m (HUD max) through revive decline, continue and bonus decline; parking stopped on a then-unknown Google small-layout interstitial after 45 s of no-input waiting. Frozen result bank refused 398 as an 8/9 ambiguity. |
| `91c3d4f4` | Click on the small-layout skip glyph after two stable frames; the session then halted on foreground loss. Attributed at the time to the assistant window; after F-005 (the pixel-identical medium-layout glyph opened a store page) the attribution is unresolved. |
| `b8e15581` | Attempt 0: natural result 246 m, full flow parked at Tune; frozen bank refused 246 as a 6/9 ambiguity (unscored). Attempt 1: random policy, 2 m, scored; the emulator then died while an advertisement loaded because the host disk was exhausted by an unrelated build. |

Every exploratory failure is sealed and preserved. Two independently labeled
result frames (246, 398) extended the terminal bank to 58 glyphs in construction
run `ca73a234`; those two sessions are construction evidence for that bank.

## Reliability studies 2.0 and 2.1 (both failed) and 2.2 (registered)

`native-reliability-2.0` ran its three sessions (`28a9d5be`, `78d51e26`,
`55b84b0f`): longest consecutive scored success run 1, zero unintended actions,
zero manual interventions, each session halted safely on a distinct unseen UI
phase (tiny Google end card, the current build's out-of-fuel result, a third
Google player size). See [F-004](../../research/findings/F-004-native-reliability-study-1.md).
Each phase became a hash-pinned, audited variant; the profile now has 22.
`native-reliability-2.1` re-registered the identical criteria against the
extended profile and failed after one session (`54d58272`): the medium-layout
skip glyph, an allowlisted control, opened the advertised app's Play Store page in
the host browser; the foreground guard halted the session on the next capture with
zero further input. See
[F-005](../../research/findings/F-005-allowlisted-ad-control-opened-store-page.md).
`native-reliability-2.2` keeps the criteria, adds the effect-based unintended-action
definition and declares application restart as the only recovery from stuck screens.

## Registered protocols awaiting execution

- `native-reliability-2.2`: 12 consecutive attempts per session, alternating gas
  and random scripted controls, ≥ 10 consecutive scored successes required, zero
  unintended actions by allowlist and by effect, restarts reported.
- `real-baselines-2.0`: random, always-gas and the stabilizing heuristic, ten scored
  attempts each, interleaved.
- `reader-validation-2.0`: held-out labels from sessions sealed after the
  registration; the reliability study sessions are the first held-out set.
- `context-encoder-2.0` and `privileged-critic-2.0`: simulator-side protocols
  registered before their implementations, which do not exist yet.

None of these started before the pause.
[cycle-2-future-work.md](cycle-2-future-work.md) records why each is blocked and
what would close it; registration is not a queue that runs itself.

## Simulator-side work

- The live headed PPO training observer (snapshot hook, observer thread, sinks,
  overhead protocol) is committed. Its measured cost is in
  [F-003](../../research/findings/F-003-live-headed-training-overhead.md): about
  12.7 % fewer environment steps at 64 environments and real-time observer frame
  rate over 60 s (three paired seeds, protocol run `238ce3a7`).
- The restricted screen-body student was trained for the first time in run
  `3ceb8b27` (600 s, seed 500, teacher `c0a9e142` final checkpoint, prior compute
  3,600.3 s): validation on seeds 30000–30019 gave joint-action agreement 0.849
  under student occupancy, mean 483.6 m and median 561.3 m against the teacher's
  673.4 m and 700.1 m, with 11 of 20 episodes crashing versus 2 for the teacher.
  This is an uncalibrated analytic-projection result, not real-game evidence.
- The 1,800 s second-seed run `807048e2` (seed 501) answered the budget question
  negatively and closed this line for now: mean 483.6 and median 585.0 against the
  same teacher, the same 11 of 20 crashes, joint agreement 0.859 and final BCE
  0.315. Budget and seed are confounded across the two runs and the episode
  bootstrap intervals overlap, so this is a negative result at this scale rather
  than a demonstrated plateau. `807048e2` is the deployable restricted candidate.
  See [F-006](../../research/findings/F-006-body-student-distillation-plateau.md).

## Stabilization boundary 2026-09-12

**Cycle 2 development is paused here by the owner, at a stabilized boundary, with
the original Cycle 2 program incomplete.** The registered protocols below have not
been executed; no experiment, native session or training run is authorized by this
document or by any registration it references.
[cycle-2-future-work.md](cycle-2-future-work.md) is the forward queue.

- Release: project version **0.1.0a2**, a research prerelease. The boundary revision
  is bound by the annotated tag `cycle-2-paused`, whose annotation records the final
  dev and main SHAs, the CI workflow results and these validation counts. Cycle 1
  stays bound to `cycle-1-paused`. The bump distinguishes the two stabilized
  prereleases and nothing else: every sealed Cycle 2 run recorded `0.1.0a1`, because
  the runs precede the bump, and a package version was never a scientific or
  objective version identifier.
- Local validation on the workstation (Windows 11, Python 3.13.5): **400 tests
  passed** in 69.75 s with the two known Starlette/httpx deprecation warnings;
  `ruff check` passed; `ruff format --check` passed over 102 files;
  `compileall -q src` succeeded; `pip check` reported no broken requirements;
  `gradientclimb doctor` reported Torch 2.11.0+cu128 with CUDA available.
- Artifact integrity: **134 of 134** canonical local runs verified valid, zero
  unfinished, zero uncatalogued directories, 991,794,172 bytes
  ([inventory](../../research/experiments/cycle-2-integrity.json)). 34 runs are new
  since the Cycle 1 pause: 26 completed and 8 failed-but-preserved, spanning
  `real-screen-episode-pilot` (10), `ui-profile-audit` (7),
  `headed-overhead-pilot` (7), `native-reliability-study` (4),
  `screen-body-distillation` (2), `real-ui-reference` and
  `result-reader-construction`, plus two runs that are validation or operator
  exercise rather than Cycle 2 evidence: one `synthetic-convergence` demonstration
  created by the CI smoke command during this validation, and failed
  `surrogate-pilot` run `e911764e` (seed 7, 64 environments, 600 s requested with a
  windowed observer, stopped at 124.9 s with zero recorded environment steps and
  zero episodes). Neither supports a finding; both are preserved because sealed runs
  are never deleted to tidy a boundary.
- `scripts/audit_cycle_state.py` now requires an explicit `--output` and refuses to
  replace an existing inventory without `--overwrite`. Its previous default wrote
  over `research/experiments/cycle-1-integrity.json`, the frozen 100-run Cycle 1
  inventory that the Cycle 1 report and notebooks cite; running the documented
  command at this boundary would have silently changed that published number.
- Native sessions are stopped. No native input should restart automatically, and
  window handles, coordinates and capture-device state from this cycle are not
  durable target identity.
- Unchanged by this boundary: the surrogate remains uncalibrated, no learned native
  policy exists, sim-to-real transfer is unmeasured, and no real one-hour
  qualification has been attempted. A green suite and a clean tag establish release
  state, never actual-game competence.
