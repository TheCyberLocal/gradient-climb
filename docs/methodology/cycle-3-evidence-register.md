# Cycle 3 requirements and evidence register

Authority: the owner's [Cycle 3 mandate](cycle-3-mandate.md), received 2026-09-12,
resumes development and experiments. Cycle 1/2 pauses remain historical records.
Starting source: `610dc48dc601f878086053f900ee7a005ccea821` on `dev`, fetched
2026-09-12; `origin/main`, `origin/dev` and `cycle-2-paused` agree. Cycle 1 tag
still targets `ebab4efd1bc5c456b424f5c8e590262f7cf84aa6`. Starting main/dev/tag CI
is successful (runs 34674582441 / 34674474337 / 34674706807).

This register is prospective. Status **implemented** describes available code;
**validated** requires the stated bounded evidence; **target met** requires the
acceptance criterion; **target not demonstrated** is an open scientific outcome;
**externally blocked** identifies required owner/access input. A registration is
never a result. New findings supplement historical records without changing them.

## Recovered starting evidence

[New integrity cutoff](../../research/experiments/cycle-3-start-integrity.json):
134/134 valid canonical runs, 991,794,172 bytes, no active or uncatalogued runs.
Artifacts resolve directly under `D:\Projects\gradient-climb\artifacts`.
The 100-run Cycle 1 and 134-run Cycle 2 inventories and both annotated receipts
remain unchanged. Auxiliary evidence/profile verification and archive restore
testing are separate requirements; this scan does not establish backup.
No Python training or `crosvm` game process was present in the initial host scan.
C: had about 141 GiB free and D: about 874 GiB, an observation rather than a
permanent preflight guarantee. Native identity and controls require rediscovery.

Historical evidence keys: F1–F6 refer to `research/findings/F-001*` through
`F-006*`; C1 is [completion-audit.md](completion-audit.md); C2 is
[cycle-2-state.md](../operations/cycle-2-state.md). Implementation paths below are
relative to `src/gradientclimb` unless prefixed with `scripts/`.

## Mandate coverage and gates

Each row maps the corresponding complete mandate section, including its listed
subrequirements. Acceptance is conjunctive: one completed element cannot close a
row. Component evidence will be linked as milestones complete.

| ID / required capability | Implementation and empirical evidence at start | Remaining gap | Prerequisite | Acceptance criterion |
| --- | --- | --- | --- | --- |
| C3-01 Mission, distance, pace, score, skills, reuse | Legacy learners, F1; scripted real runs, F2 | Target not demonstrated; no qualified learned real policy | Reliable native measurement | Repeatable learned reference competence, measured cost, score, adaptation and retention; no feature-count substitution |
| C3-02 State reconstruction and inherited protocols | C1/C2, F1–F6, both inventories/receipts read; this register | Auxiliary hashes and provenance reconciliation | Local evidence access | Every inherited gate mapped below; all five registrations explicitly dispositioned; preserve cutoffs |
| C3-03 Source and durable artifacts | `experiments/recorder.py`, `artifacts/integrity.py`; 134 verified runs | Archive/restore implementation and owner-selected destination | Valid inventory; destination approval | Atomic tested dev commits, exact-SHA CI, pinned experiment source, all cited evidence retrievable after restore; no paid/public storage |
| C3-04 Campaign governance | Existing experiment JSON definitions | New dependency-driven ledger and bounded stopping rules | Register before governed work | Screens precede replications; all trials/costs/negative findings retained; no unbounded retries |
| C3-05 Four evaluation clocks/horizons | `simulation/hill.py` and screen CEM use 60 s caps | Reliability/fixed-time/long-run/learning separation | Frozen prospective protocols | Reference frozen policy median >=2000 real m over 20 executions, 3 training seeds, distribution/pace/censoring, 5/10/20/30/45/60 min checkpoints; human claim requires matched records |
| C3-06 Accounting, observer, continuation | PPO/CEM, runner, recorder, visualization; F3 overhead; failed e911764e | Incremental counters, exception/checkpoint/observer evidence, resume truthfulness | Sealed diagnostic and fault fixtures | Ctrl+C/learner/window/video/write/disk faults preserve observed work, primary cause, last valid checkpoint and cleanup; explicit warm-start/exact-resume manifests; old loaders compatible |
| C3-07 Native lifecycle and host safety | `control/*`, `scripts/run_screen_episodes.py`; F4/F5 failures | Scored-streak defect, split gameplay/readiness/safety, host margin and ownership, soak | Fresh capture/profile/host checks | >=10 consecutive scored successes in 12, zero unintended actions/interventions, then representative soak; unknown UI releases and waits; effect-evidenced closes only |
| C3-08 Synchronized demonstrations/data | Capture/input journals, feature bridge, F2 | Read-only focus-gated pedal recorder; causal alignment, dropped/stale frames, partitions; no synchronized expert data | Validated recorder; owner demonstrations | Small varied action-labeled batch and separate human benchmark; monotonic acquisition/ready/action/event times; whole-session splits and visually verified build/upgrades |
| C3-09 Baselines and failure localization | Scripted gas n=2 at 458/411 m, 60 s; F2 | Matched random/gas/heuristic plus learned traces; no learned ceiling | Lifecycle gate and valid readers | Fixed-time and long-run comparisons with all attempts, pace, coverage, first divergence, verified geometry/scenario classes |
| C3-10 Environment architecture | `VectorHillEnv` hardwired throughout; legacy surrogate has approximate physics | Factory, separate legacy/reference/randomized envs and mature-engine benchmark | Provenance contracts; engine/license screen | Legacy unchanged; separate measured reference and derived randomization with units/settings; installation/stability/offscreen throughput evidence |
| C3-11 Effective dynamics | `simulation/hill.py` two contacts, torque/grip/springs/fuel; `calibration/effective.py` synthetic fitter | Articulated wheels, suspension, driveline, collisions, fuel effective fits and uncertainty | Real diagnostic trajectories and geometry | Held-out action responses; wheelie/air-pedal mechanisms; convergence, energy, penetration, tunneling and joint tests; no guessed proprietary constants |
| C3-12 Terrain, bridges and scenarios | Smooth analytic hills only | Measured route, sharp/gapped geometry, articulated collidable bridges | Real coverage and source-labeled fixtures | Measured coverage through reference range; dynamic load response and entry/exit; synthetic regions labeled; full-route qualification |
| C3-13 Visual observation contract | Observer/replay rendering and real feature extractor; analytic student projection | Shared offscreen-to-pixels pipeline, task cues/camera/noise, licensed original assets | Geometry/mechanics and observation schemas | Actor uses deployable shared preprocessing/cadence; held-out real recognition/control and rendered throughput; no hidden state substitution |
| C3-14 Fidelity and real utility | Uncalibrated surrogate; no transfer result | Camera/scale fitting, dynamics/geometry/perception and policy-ranking comparisons | Construction/heldout splits, tolerances frozen before tests | Compare persistence/history and legacy; held-out event distributions plus early frozen real probes; extra realism justified by failures or real learning |
| C3-15 Memory, observations and pedals | 34-parameter current-frame native linear actor; stacked student; four pedal states | Temporal policy with causal history, full orientation cues, replay masks/burn-in and variable durations | Action-aligned data and validated perception | Matched stack vs recurrent comparison; episode/profile resets; no future/privileged leaks; gas-first/brake-first ordering preserved |
| C3-16 Assisted learning and occupancy correction | Teacher-occupancy students, F6; seed/budget confounded | Human BC and simulator DAgger, controlled information/occupancy/budget ablations | Competent source labels and heldout evaluation | Independent learned real BC probe; DAgger labels student simulator states only; full teacher/demo/initialization cost and no false causal plateau claim |
| C3-17 Learning algorithms and discount | PPO/CEM comparators; gamma .99 per .06 s | Temporal PPO/auxiliary, justified recurrent replay candidate, asymmetric critic, physical-time discount audit | Correct temporal data and bounded pilots | Correct learner tests; equal-cost 3-seed selected comparison; no stale replay into ordinary PPO; planning only after bottleneck evidence |
| C3-18 Objectives and hacking | `experiments/objectives.py` Cycle2 versions; no qualifying native comparison | Coin delta not validated total score; overlapping attribution; weighted scalar misnamed lexicographic; censoring | Independent score/recovery validation | New version IDs; score conservation; true ordered selection; four matched families with C/D time fixed; eight hacking cases, survival/distance constraints |
| C3-19 One-hour claim and priors | Historical surrogate hour F1 with different clock boundaries | New command-start clock, deadline/checkpoint validity and full prior ledger | Credible deployable learner and independent evaluation | No new update after 60 min deadline; actual checkpoints/unchanged weights/overhead; 5 named prior classes plus policy/system distinction; censored threshold times |
| C3-20 Throughput and display cost | F1 state throughput; F3 observer 12.7% fewer steps in bounded study | Rendered-pipeline sweep and real latency, resource-isolated qualification | Useful validated experience | Physics/render/decision/update rates and improvement/min; fidelity/substeps/cadence pinned; measured batching/placement/observer overhead |
| C3-21 Composable profiles | Synthetic heavy/rough profiles; initial real configuration observed historically | Vehicle/upgrade/map/observation/control/scenario contracts and additional content | Design early; broad training after reference competence | Distinct topology/mechanisms/controls/provenance/checkpoint compatibility; 2 additional real vehicles incl special mechanism if available and 1 map; no purchases or route scripts |
| C3-22 Generalization/adaptation/retention | F1 synthetic one-parent 10 min shift and forgetting | Real heldout map/vehicle/pair, context/critic studies and retention mitigations | Qualified source and accessible target profiles | Zero-shot then 5/10/30/60 min; source retests; matched specialists/generalists/context and costs; actor/critic isolation; mitigation tested if forgetting |
| C3-23 Curriculum and bounded harder cases | Missing | Source-linked challenge fixtures and matched curriculum study | Verified failure classes and realistic distribution | Equal budget unchanged-sampling comparator, heldout full routes; bounded uncertainty randomization and all search costs; synthetic reset labels |
| C3-24 Research interface | Canonical DuckDB/FastAPI dashboard, plots/notebooks/replay | New-cycle status/counters and real/fidelity/human/objective/profile surfaces | Actual canonical measurements | Trace hashes and domain/units/priors visibly separated; playable reference, live learner, replay comparison, reproducible report; missing panels remain missing |
| C3-25 Validation and final release | 400 historical tests; Ubuntu Python 3.13 CI; alpha2 | New contracts, portable/native matrix, clean install and restore; final audit/review | Stable tested milestones and closed active runs | Focused and full checks; exact dev/main CI; reviewed FF or normal reconciliation; truthful annotated release preserving tags/dev, no force/protection bypass |
| C3-26 Evidence-based completion | C1 interim report and partial completion audit | All ten required outcome classes need new evidence | Actual campaign results | Implemented/validated/met/not-demonstrated/externally-blocked matrix, SHAs, commands, artifacts/models, all costs/results/gaps; missing capabilities cannot be claimed complete |
| C3-27 Primary-source grounding | ADR1/2 and literature register | Current DAgger/Box2D reference and candidate license review | Before adoption | Primary papers/docs consulted and attributed; measured/inferred/community/manufacturer distinctions; no research paper presented as game result |

## Inherited original completion gates

The original brief sections 1–72 remain mapped in C1's full-brief table. This
crosswalk preserves every one of its 20 conjunctive section-73 gates, with current
implementation/evidence above and its unresolved acceptance carried into Cycle 3.

| Original gate | Current scope and evidence | Cycle 3 owners / prerequisite and remaining acceptance |
| --- | --- | --- |
| 1 Actual game control | Bounded validated F2, four states | C3-07/08 fresh identity, timing and safe lifecycle |
| 2 Real trajectories | Partial F2, sparse construction frames | C3-08/09 synchronized varied data, disjoint labels |
| 3 Useful calibrated simulator or evidenced strategy | Not demonstrated | C3-10–14 real fitted and heldout fidelity plus transfer utility |
| 4 High-throughput learning | Validated legacy only, F1 | C3-17/20 rendered useful experience with cost sweeps |
| 5 Independent relevant controls | Reference four states validated | C3-07/15/21 duration/order and extra observed mechanics |
| 6 Learned actual behavior | Legacy trained; native absent | C3-09/16/17 frozen independently evaluated real learner |
| 7 Headed/headless | Implemented, F3 bounded cost | C3-06/13/24 fault-safe live new learner, same actor pixels |
| 8 Unattended real episodes | Not demonstrated, F4/F5 failures | C3-07 consecutive reliability then soak |
| 9 Reproducible runs | 134 canonical seals valid | C3-03/06 full accounting and tested preservation |
| 10 Wall-clock efficiency | Legacy result F1 only | C3-17/19/20 command-start and complete priors |
| 11 Governed hour | Two surrogate hours, no real | C3-05/19 60-minute real target and checkpoint curves |
| 12 Real transfer | Not demonstrated | C3-13/14/16 identical deployable policies, matched real/sim |
| 13 Generalization | Synthetic partial F1 | C3-21/22 real unseen map/vehicle/pair |
| 14 Adaptation | One synthetic 10 min child, F1 | C3-22 full real curves, independent seeds and retention |
| 15 Meaningful competitors | Legacy screens, F1 | C3-09/17/18 equal-cost real and objective comparisons |
| 16 Findings | F1–F6, narrow supported claims | C3-04/26 negative and positive source-linked findings |
| 17 Dashboard | Implemented and tested | C3-24 all measured surfaces traceable, missing explicit |
| 18 Final comprehensive report | Historical interim only | C3-24/26 current complete evidence matrix and reproducible outputs |
| 19 Failures/limits | Preserved failures, F1–F6 | C3-02/06/26 correction records, no retroactive relabeling |
| 20 Clean/tested/reproducible/synced | Exact starting CI green | C3-03/25 clean install/restore and exact final release CI |

## Disposition of five unexecuted inherited registrations

| Registration | Prospective disposition | Reason / executable gate |
| --- | --- | --- |
| native-reliability-2.2 | Requires implementation-correction amendment, 2.3 | `success_unscored` erroneously satisfies `endswith('scored')`; explicitly enumerate scored classes, distinguish session deadline; preserve 12 attempts, >=10 streak, zero effects/interventions, restart rules |
| real-baselines-2.0 | Supersede with split fixed-time/long-run 3.0 before collection | Retain random/gas/heuristic and all attempts; 60 s cannot qualify 2000 m; execution after reliability |
| reader-validation-2.0 | Requires provenance amendment before selecting labels | Later sessions did exist and were used in UI/result construction; seal whole-session reader-specific construction exclusions and untouched sets |
| context-encoder-2.0 | Prospective supersession to 3.0 required before experiment | New causal profile/observation contract; matched stack/context and hidden dynamics; old registration remains never executed |
| privileged-critic-2.0 | Prospective supersession to 3.0 required before experiment | New shared visual actor path and separated critic interface; identical actor access with injection tests; old remains never executed |

## Frozen starting qualification proposal

The reference target is **median >=2000 real-game metres over 20 independent
episode executions**, reporting all distances, quartiles, worst episode, natural
failure, coverage and threshold times. Reproduce the selected method over at least
three independently trained seeds. Game terrain seeds are not claimed controllable.
Human >2000 m is owner-reported; matched benchmark recordings are still required.
The initial long-run maximum is **900 gameplay seconds**, with a **60 second
no-new-verified-progress stall window**; missing progress cannot certify a stall.
This is a conservative prospective ceiling pending human pace observations, not a
claim about required driving time. Any construction-driven horizon amendment must
precede qualification and cannot lower the target. Moving timeouts are censored.
Fixed-time driving remains 60 s and instrument reliability remains its own test.
No qualification data have been collected in this cycle.

The one-hour primary clock begins at command start and includes initialization,
real acquisition/resets/ad waits, perception, learning, logging and checkpoint
serialization. Checkpoints at 5/10/20/30/45/60 minutes carry actual completion
times; no new update starts after the deadline. Finalization overhead and evaluation
costs are separate. Policies and system priors are both declared. Historical clocks
and surrogate units remain unchanged.
