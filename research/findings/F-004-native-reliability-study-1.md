# F-004 — Unattended native episode cycling is not yet established: study 2.0 failed on unseen UI phases

Status: **NOT SUPPORTED (registered study failed its criteria)**, Cycle 2, 2026-09-12.
Protocol [`native-reliability-2.0`](../../experiments/definitions/cycle-2-native-reliability.json);
sessions `28a9d5be`, `78d51e26`, `55b84b0f` (experiment `native-reliability-study`).

## Observation

The registered study allowed three sessions of twelve consecutive attempts with alternating
always-gas and random scripted controls and required at least ten consecutive scored successes
in one session, zero unintended actions and zero manual interventions. All three sessions were
run with the version 2 UI profile as it stood at each session start; every attempt is sealed.

| Session | Attempts | Outcome |
| --- | --- | --- |
| `28a9d5be` | 1 of 12 | Natural driver-down result 243 m read exactly by the extended bank through revive decline, continue and bonus decline; parking then met a previously unseen Google end-card layout (tiny card, "Close" text pill) and halted after the bounded no-input wait: `unknown_failure`, 11 not attempted. |
| `78d51e26` | 2 of 12 | Attempt 0: legitimate close of that end card, start, driver-down result 237 m scored, parked at Tune in 25 s: `success_natural_scored`. Attempt 1: random control ran out of fuel at 19 m; the current build's OUT OF FUEL result layout differs from the Cycle 1 reference (0.85 similarity), the episode ended as an unrecognized state and parking refused: `unknown_failure`, 10 not attempted. |
| `55b84b0f` | 1 of 12 | The out-of-fuel result was dismissed, then a fifth advertisement layout (the Google interstitial at a third player size) went unrecognized for 48 s: `unknown_failure`, 11 not attempted. |

## Measurement

- Longest run of consecutive scored successes in one session: **1** (criterion ≥ 10).
- Scored attempts: 2 of 4 attempted (243 m, 237 m); score coverage of attempted cycles 2/4.
- Unintended actions: 0. Manual interventions: 0. Every click was an allowlisted named control
  (`menu-transitions` artifacts), including one legitimate advertisement close.
- Capture: DXcam initialized in every session; no foreground or geometry fault occurred while
  tools stayed idle; one stale capture was discarded and recovered.
- Result reader: the 58-glyph bank read both natural results exactly (243, 237) and the
  19 m out-of-fuel result on retained frames.

## Interpretation

The lifecycle machinery works when every screen is recognized: start, gameplay, natural result,
scoring, revive and bonus declines, one legitimate advertisement close, and parking all succeeded
inside the second session. What fails is coverage of UI phases the profile has never seen. Each
session exposed exactly one new phase, and each was a *rendering variant of a known phase*
(another emulator scale of the Google interstitial, or the current build's out-of-fuel screen)
rather than a new kind of interaction. The fail-closed design held throughout: unknown phases
released input and halted with evidence, never clicked.

## Limitation

Three sessions of one to two attempts each cannot estimate a success rate; they establish that
the unseen-phase rate per cycle was high enough to stop every session early. The study did not
reach attempts where truncation scoring at the paused boundary would have been exercised.

## Conclusion

Registered criteria not met. Unattended reliability remains unestablished. Each discovered phase
was added as a hash-pinned variant and audited offline over all stored frames after the session
that found it; the follow-up study `native-reliability-2.1` re-registers the same criteria
against the extended profile rather than relaxing them.
