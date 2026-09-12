# Cycle 2 future feature development

Status: **DEFERRED at a stabilized boundary.** This document is the forward queue for the
resumed program, not authorization and not a scheduler. No native session, training run,
labeling pass or registered protocol executes because a row here says it is next. The Cycle 1
[resumption plan](resumption-plan.md) remains the long-range scientific map; this document
records what Cycle 2 closed, what Cycle 2 opened, and what is registered but unexecuted.

The containing release is bound by the annotated tag `cycle-2-paused`, whose annotation
records the final dev/main SHAs, the CI checks and the local validation counts.
[cycle-2-state.md](cycle-2-state.md) owns the operational state at that boundary and the
[Cycle 2 integrity inventory](../../research/experiments/cycle-2-integrity.json) owns the
verified local run set. Cycle 1 records stay bound to `cycle-1-paused` and are never
rewritten, rescored or reinterpreted by anything below.

## What Cycle 2 settled, so it is not repeated

- **Live headed and headless training exists and its cost is measured.** The observer, its
  sinks and the paired `headed-overhead-paired-0.1` protocol are committed;
  [F-003](../../research/findings/F-003-live-headed-training-overhead.md) measures about
  12.7 % fewer environment steps at 64 environments over 60 s across three paired seeds.
  This closes the implementation half of original gate 7; it is a cost measurement, not a
  claim that live observation helps research throughput.
- **Distillation budget is not the restricted student's limiter.**
  [F-006](../../research/findings/F-006-body-student-distillation-plateau.md) is a negative
  result: tripling the budget to 1,800 s (`807048e2`) moved joint agreement one to two points
  and left the mean at 483.6 simulator units against the teacher's 673.4, with 11 of 20
  episodes still crashing. Do not spend further budget on the same recipe. `807048e2` is the
  deployable restricted candidate whose expected weakness is that crash rate.
- **Unattended native episode cycling is not established, and two registered attempts say
  why.** [F-004](../../research/findings/F-004-native-reliability-study-1.md): study 2.0 met
  three distinct unseen UI phases and reached a longest scored success run of 1 against a
  criterion of 10. [F-005](../../research/findings/F-005-allowlisted-ad-control-opened-store-page.md):
  study 2.1 failed after one session when an allowlisted advertisement control opened the
  advertised app's Play Store page in the host browser. Both failures are sealed and were
  converted into mechanism, not into relaxed criteria.
- **Behavioral measurement and objectives are implemented ahead of the data that would use
  them.** `behavioral-metrics-2.0`, objective families A–E, the recovery definition and the
  eight-indicator reward-hacking battery live in
  [objectives.py](../../src/gradientclimb/experiments/objectives.py) with tests, under the
  [Cycle 2 preregistration](../methodology/cycle-2-preregistration.md). No study has yet
  produced episodes eligible for arms B–E.

## Registered protocols awaiting execution

Each is already frozen in `experiments/definitions/`; none is started. Registration before
data is the point, and it is not a queue that runs itself.

| Protocol | Definition | What it would unblock | Why it has not run |
| --- | --- | --- | --- |
| `native-reliability-2.2` | [definition](../../experiments/definitions/cycle-2-native-reliability-2.2.json) | Every real-game gate downstream: baselines, readers, learning, the governed hour | Needs a supervised native session on the actual game. Criteria, the effect-based unintended-action definition and restart-only recovery are registered and implemented; the sessions were not run before the pause. |
| `real-baselines-2.0` | [definition](../../experiments/definitions/cycle-2-real-baselines.json) | Matched comparators for any learned real policy | Registered to execute only after the reliability study reports. |
| `reader-validation-2.0` | [definition](../../experiments/definitions/cycle-2-reader-validation.json) | Score, fuel and terminal-cause fields that objectives B–E depend on | Its held-out set is defined as sessions sealed after registration, and the reliability study sessions are the first such set. No held-out session exists yet. |
| `context-encoder-2.0` | [definition](../../experiments/definitions/cycle-2-context-encoder.json) | Whether learned history beats the four-frame stack under hidden dynamics variation | Registered before implementation; the context module itself is unwritten. |
| `privileged-critic-2.0` | [definition](../../experiments/definitions/cycle-2-privileged-critic.json) | Whether a privileged critic accelerates a screen-restricted actor | Registered before implementation; needs the explicitly separated critic interface so critic state cannot reach actor decisions. |

## Prioritized future feature development

All rows are **DEFERRED**. "Current state" describes what exists now; a completion criterion
describes what would justify closing the gap. Any new numerical threshold must be frozen in
the governing protocol before the data it judges are collected.

| Priority / feature | Current state | Implementation locations | Completion criterion |
| --- | --- | --- | --- |
| 1. Execute the native reliability gate | Registered `native-reliability-2.2`; mechanism implemented and restart-sealed in probe `a3422d1c` | `control/game_adapter.py`, `scripts/run_screen_episodes.py`, `configs/perception/hcr-reset-ui.json` | A session reaching at least 10 consecutive scored successes of 12 attempts, zero unintended actions by allowlist and by effect, zero manual interventions, at least 10/12 scored, with every restart reported. A failed study is a recorded finding, not a reason to relax a criterion. |
| 2. Advertisement handling that survives an unknown creative | 22 hash-pinned variants; controls enabled for exactly two variants with sealed effect evidence; eight recognition-only | `configs/perception/hcr-reset-ui.json`, `scripts/audit_ui_profile.py`, adapter allowlist | Per-variant evidence that a control returns a recognized game state without losing the foreground, on at least two independent sessions, before that control is enabled. Unknown chrome keeps releasing input and waiting. No advertisement bypass, no purchase, no paid-content path. |
| 3. Held-out reader validation | Banks built from construction sessions; 58-glyph terminal bank; coin and fuel readers declared but unvalidated | `perception/scoring.py`, `perception/hud.py`, `scripts/evaluate_reader_labels.py`, `scripts/label_contact_sheet.py` | Availability and exact accuracy per reader on frames from sessions sealed after registration and labeled before reader output is viewed, with coverage and failure counts. No objective may depend on a field before its reader reports held-out accuracy. |
| 4. Real non-learning baselines | None collected; two Cycle 1 always-gas episodes only (n=2) | `scripts/run_screen_episodes.py`, the declared stabilizing heuristic | Ten completed scored attempts per policy for random, always-gas and the stabilizing heuristic, interleaved, with every attempt classified and none excluded after the fact. |
| 5. Occupancy-corrected distillation and disagreement analysis | Teacher-occupancy imitation only; occupancy correction untested; per-state disagreement unmeasured | `algorithms/screen_distillation.py`, `scripts/distill_screen_student.py` | Registered before data: student-state data with teacher labels at equal budget against `807048e2`, plus a per-state attribution of where the body features lose the teacher's information. Report a negative result if the gap persists. |
| 6. Context encoder implementation and arm comparison | Protocol registered; module unwritten | `algorithms/policies.py`, `algorithms/ppo.py`, trainer configuration | The registered three-seed, six-condition comparison with paired differences reported per seed; non-adoption with a recorded negative result is a valid outcome. |
| 7. Privileged critic interface | Protocol registered; interface unwritten | `algorithms/ppo.py`, `algorithms/policies.py` or a new separated critic path | Actor inputs and architecture held fixed, critic privileged information varied, equal-budget learning and deployed quality reported, and a test that critic state cannot reach an actor decision. |
| 8. Objective screening across arms A–E | Implemented and tested; no eligible episodes | `experiments/objectives.py`, a screening definition still to be registered | Frozen equal-budget arms with full outcome vectors, the eight hacking indicators reported with measurability, and Pareto reasoning rather than a single scalar. Gated on priorities 1, 3 and 4. |
| 9. Governed real one-hour benchmark | Missing | Benchmark runner, declared clock, checkpoint schedule, frozen objective | Full 3,600 s measured real training with 5/10/20/30/45/60-minute checkpoint evaluations, actual overshoot, reset and advertisement costs, and independent real evaluations. The historical provisional distance gate (median at least 500 m, at least 3x the matched random median, at least 8/10 reaching 250 m) is preserved and evaluated separately from any new objective. |
| 10. Backward-motion measurement | Explicitly **not measurable** by the current bridge; the backward-loop hacking indicator reports unmeasurable | `perception/screen_features.py`, `perception/fields.py`, observation schema | A signed progress estimate with held-out error in observed units. A new actor feature requires a schema version bump and a retraining decision; the outcome vector is the default home for a new measurement. |
| 11. Dynamics strategy decision | Unchanged from Cycle 1: no real calibration or comparison | `calibration/effective.py`; the ignored screen-dynamics draft only after an explicit adoption review | Held-out prediction errors in observed units for persistence, history and action-conditioned alternatives, with rank and omission counts. Neither a fit nor a plausible randomization closes this. |
| 12. Real generalization and adaptation | Unchanged from Cycle 1: synthetic shifts only | Qualified source policy, explicitly available vehicles and maps, no purchases | Zero-shot evaluation before adaptation on a new real map, vehicle and both, with 0/5/10/30/60-minute curves, parent lineage, actual exposure and source retention. Synthetic heavy or rough conditions are never relabeled as a commercial vehicle or map. |
| 13. Cycle 2 report, notebooks and dashboard coverage | The Cycle 1 report and notebooks are current for Cycle 1 only; the dashboard reads canonical records, so Cycle 2 runs appear, but no Cycle 2 report exists | `scripts/analyze_research.py`, `scripts/build_research_notebooks.py`, dashboard views | A source-linked Cycle 2 report over the Cycle 2 record set citing its own integrity inventory, keeping unmeasured panels visibly empty. The Cycle 1 report keeps its own cutoff and its own 100-run inventory. |
| 14. Host and session operability | Known hazards handled by discipline rather than by code: emulator sessions die below about 4 GB free on C:, and any tool window raised over the game halts a native session on foreground loss | `capture/windows.py`, session runner preflight | A preflight that records host free space and refuses to start a native session below a declared margin, and a single-foreground-command protocol that does not depend on an operator remembering it. |

## Disposition of Cycle 2 registrations and plans

| Registration / work | Cycle 2 disposition | What remains |
| --- | --- | --- |
| `cycle-2-native-reliability.json` (`native-reliability-2.0`) | EXECUTED; FAILED its criteria | Nothing to rerun as registered. Its three sessions are sealed evidence and construction frames; [F-004](../../research/findings/F-004-native-reliability-study-1.md) is terminal for that version. |
| `cycle-2-native-reliability-2.1.json` | EXECUTED; FAILED after one session | Terminal. [F-005](../../research/findings/F-005-allowlisted-ad-control-opened-store-page.md) is its finding; its session is held-out reader evidence. |
| `cycle-2-native-reliability-2.2.json` | REGISTERED; NEVER STARTED | Three sessions of twelve attempts, unchanged criteria. Priority 1 above. |
| `cycle-2-real-baselines.json` | REGISTERED; NEVER STARTED | Gated behind the reliability report. Priority 4. |
| `cycle-2-reader-validation.json` | REGISTERED; labeling NEVER STARTED | Needs a held-out session. Priority 3. |
| `cycle-2-context-encoder.json` | REGISTERED; NEVER IMPLEMENTED | Priority 6. |
| `cycle-2-privileged-critic.json` | REGISTERED; NEVER IMPLEMENTED | Priority 7. |
| Restricted student protocol | EXECUTED twice (`3ceb8b27`, `807048e2`); the budget question is answered negatively | Occupancy correction and disagreement attribution, priority 5. No further budget on the same recipe. |
| Headed observer and overhead protocol | COMPLETE for this cycle | Controlled live-training comparisons and paired real/simulator video remain part of Cycle 1 priority 9. |
| Objective families A–E and the hacking battery | IMPLEMENTED; NEVER APPLIED to real episodes | Priority 8, gated on readers and reliability. |
| Cycle 1 `ablations.json`, `adaptation-pending.json`, `real-screen-cem.json` | UNCHANGED from their Cycle 1 disposition; still DEFERRED | See the Cycle 1 [plan disposition](resumption-plan.md). Cycle 2 neither started nor retired them. |

## Constraints that survive the pause

- Resuming native work requires fresh window identity, focus, geometry and observation-age
  checks. Cached handles, coordinates and a previously initialized capture device are not
  current discovery results, and a DXcam initialization failure is a failure rather than
  permission to switch a registered backend.
- The reliability gate precedes the reward campaign. A multi-objective arm must not be
  trained on a lifecycle that cannot yet cycle ten scored episodes.
- Only ordinary rendered pixels and ordinary input are permitted. Unknown UI releases input
  and halts; the recovery from a stuck screen is an application restart, never an
  exploratory click.
- Nothing here authorizes rewriting, rescoring or relabeling a sealed run in either cycle.
  Failed runs are preserved as evidence, including the seven failed Cycle 2 runs.
