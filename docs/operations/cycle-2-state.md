# Research Cycle 2: active state

**This document is the live operational state of the resumed program.** It is
updated at every stabilization boundary on `dev`; it never rewrites Cycle 1
records, which remain bound to the tag `cycle-1-paused`. Read
[resume-state.md](resume-state.md) for the frozen Cycle 1 checkpoint and
[cycle-2-preregistration.md](../methodology/cycle-2-preregistration.md) for the
registered metric, objective and reliability protocols.

## Checkout and storage

- Cycle 2 development started from `ebab4ef` on `dev`; `main` stays at the Cycle 1
  release until the final integration gate.
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
- UI profile `hcr-wrapper-reset-v2` (19 variants): Meta Audience Network chrome
  (skip-forward and muted-speaker glyphs, binarized-shape matching), the
  Google-served small-layout interstitial (dim skip glyph plus static marker), the
  Fingersoft commercial break, and the vehicle-selection screen, in addition to the
  Cycle 1 variants. Every addition was audited offline over all stored frames
  (`scripts/audit_ui_profile.py`; sealed audits `7b73824a`, `745ea91e`, `b73aa968`).
- Advertisement closes require the same control on two fresh frames ≥ 0.15 s apart.

## Exploratory native sessions so far (not study attempts)

| Session | Outcome |
| --- | --- |
| `fa103b59` | Natural result at 398 m (HUD max) through revive decline, continue and bonus decline; parking stopped on a then-unknown Google small-layout interstitial after 45 s of no-input waiting. Frozen result bank refused 398 as an 8/9 ambiguity. |
| `91c3d4f4` | Legitimate skip of that interstitial after two stable frames; session then halted on foreground loss caused by the assistant window. |
| `b8e15581` | Attempt 0: natural result 246 m, full flow parked at Tune; frozen bank refused 246 as a 6/9 ambiguity (unscored). Attempt 1: random policy, 2 m, scored; the emulator then died while an advertisement loaded because the host disk was exhausted by an unrelated build. |

Every exploratory failure is sealed and preserved. Two independently labeled
result frames (246, 398) extended the terminal bank to 58 glyphs in construction
run `ca73a234`; those two sessions are construction evidence for that bank.

## Registered protocols awaiting execution

- `native-reliability-2.0`: 12 consecutive attempts per session, alternating gas
  and random scripted controls, ≥ 10 consecutive scored successes required.
- `real-baselines-2.0`: random, always-gas and the stabilizing heuristic, ten scored
  attempts each, interleaved.
- `reader-validation-2.0`: held-out labels from sessions sealed after the
  registration; the reliability study sessions are the first held-out set.

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
  A 1,800 s second-seed run follows to measure whether the gap closes with budget.
