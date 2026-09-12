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

The owner's subsequent 2026-09-12 clarification is implemented prospectively in
[learning-efficiency.md](learning-efficiency.md): rapid learning is elapsed time
to independently evaluated real competence, with compute, experience, system
priors and evaluation costs reported separately. The
[campaign 3.1 amendment](../../experiments/definitions/cycle-3-campaign.json) and
[learning-efficiency framework](../../experiments/definitions/cycle-3-learning-efficiency.json)
govern future dispatch. The framework is **not dispatchable** until a selected
method and all collection gates are committed; it contains no collected results.

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

## Prospective learning-efficiency clarification

This section supplements the starting requirements above. It does not reclassify
Cycle 1/2 runs or treat their historical learner timers as command-start clocks.

| Requirement | Current contract/evidence | Remaining gate and claim limit |
| --- | --- | --- |
| C3-01/05/19 rapid real competence | Registered 500/1,000/2,000 m median thresholds, 20 unique frozen real episode executions per checkpoint and three independently trained seeds | No real learned threshold demonstrated; native lifecycle/soak, reader qualification, frozen scenario/method and untouched final sessions required |
| C3-05/19 observed learning time | Actual 5/10/20/30/45/60-minute checkpoint production times; first observed pass, sampling bracket, last-observation right censoring and explicit missing coverage | Entry reading must precede project/heavy imports, with interpreter startup explicitly excluded/unmeasured; existing historical clocks cannot be repaired by relabeling |
| C3-19 prior and evaluation cost | Five policy-start classes plus system-prior inventory; hashed unique parent-cost union; independent evaluation/verification latency separate | Unmeasured inherited costs stay unknown; cold actor initialization does not erase task-trained perception or simulator construction |
| C3-20 resource/experience separation | CPU core-seconds, measured accelerator scope and memory method separate from real/simulator episodes, transitions, simulated seconds, physics substeps, renders, decisions and updates | Raw rates are diagnostic; replay reuse does not become newly acquired experience; missing metrics cannot establish a cost advantage |
| C3-10/14/20 fidelity choice | Versioned fidelity identity, fixed solver/cadence, registered physics/render/learning budgets and real-outcome Pareto comparison | Engine throughput/stability does not validate real fidelity or select the most efficient learner; extra realism needs measured utility |
| C3-04 bounded qualification cost | Maximum 18 unique-checkpoint batches/360 real executions; 900 gameplay-second horizon and 21,600-second wall cap per batch | Up to 90 gameplay hours/108 batch-wall hours are separate from three training hours; dispatch is staged and conditional, never inferred from a registration |

The governed stall window remains a censored operational stop. The primary
learning-efficiency batch requires all 20 outcomes to have qualified natural or
full-horizon endpoints; missing, unsafe or stall-truncated attempts do not get
silently removed or replaced. Report their distances and reasons even when the
batch cannot qualify. There are no qualified results under this new framework.

## Reviewed demonstration pipeline milestone

C3-08/15/16 now have an additive `imitation-windows-3.0` implementation:
[window publication](../operations/imitation-window-datasets.md) verifies source
seals, complete visual-review coverage and observation-before-control timing.
Whole-session partitions bind across cooperating publishers in the canonical
store, including failed publications; later ledgers may extend assignments but
cannot move or omit earlier sessions. Incomplete histories and stale or
unbracketed observations retain explicit exclusion reasons.

The first [frozen construction plan](../../experiments/definitions/cycle-3-imitation-window-diagnostic-001.json)
uses two individually reviewed spans from the owner's imitation recording.
Vehicle/map/build identity remains incomplete, so its windows cannot train a
policy. The [geometry diagnostic](../operations/imitation-geometry-diagnostic.md)
measures the unchanged detector on every unique accepted image and preserves
partial operations on failure. Detector validity is not labeled accuracy.

These implementations preserve source recording costs and unknown human practice,
and record each new offline publication's resource scope separately. They acquire
zero new gameplay episodes and do not establish imitation learning, simulator
fidelity or real competence. The planned executions and their results belong in
subsequent source-linked records; they are not implied by this registration.

Integrated validation passed 723 tests with four optional-engine/platform skips
and two dependency deprecation warnings. Full Ruff checks passed; two focused
checks also passed after the final explicit-LF publication adjustment.

## Addendum: executed construction plan 002

C3-08/15/16 now include the [sealed three-attempt construction audit](../../research/diagnostics/imitation-construction-002.md).
The first plan/run `8f5a7301-3cba-49c2-97e0-d39420526d7b` failed closed before
assembly because review v1 used the logical configuration hash instead of the exact
configuration-file hash. A committed hash-only successor preserved the original
review, source bytes, partition, timing limits and fixed detector.

Corrected window run `fe9dabb6-16a6-47c9-90b4-edf3bbc919b3` and fixed geometry
run `9be884a9-efe8-44c0-ac5b-ee99187bf17e`, both at clean source
`2f654286fef58d4ce36aad7e869184799d60efd6`, yielded 54 construction windows,
60 unique images and 216 references. All 60 geometry attempts completed; detector
body/wheel/terrain validity was 60/44/50, with independently measured accuracy
still null. All three individual seals passed; historical aggregate cutoffs remain
unchanged.

The selected spans cover 62 individually reviewed images, with six incomplete
history targets and two time-boundary observations excluded; 483 other source
frames remain outside reviewed segments. All windows remain ineligible for
training because profile identity is incomplete. No complete episode, learned
policy, fidelity or real competence claim follows. C3-19/20 cost evidence retains
the failed assembly, both successful offline recorder windows, all three inherited
capture attempts and explicit unknown human/review/engineering priors, with each
unique run cost counted once and sampled resource coverage shown. Independent
accuracy, matched benchmark and qualified learning requirements remain open.

## Reader and owner-archive milestone

C3-07/08 now have [prospective reader qualification tools](../operations/reader-validation-3.md)
with immutable candidate/session declarations, whole-session acquisition closure,
first-retained `terminal_frame` callback selection, distinct canonical episode units and a durable
known-exposure ledger. Failed prediction attempts consume exposure; replacement
labels and aliases cannot restore blinding. A completed verified envelope is
required for a qualification flag. The new protocol remains draft: no real bank
has been fitted or independently qualified by these tools, and the historical
reliability result keeps its rejected reading and failed gate.

C3-08/13/14/15 now include [four source/render pose pairs](../../research/diagnostics/visual-pose-screen-002.md)
and [an owner archive excerpt](../../research/diagnostics/owner-video-acquisition-001.md).
The pose screen exposes wheel-appearance and terrain-edge differences rather than
establishing fidelity. Its exact actor pixels contain no diagnostic landmarks.
The archive adds 18 fixed timestamped review frames from a 299.999-second video;
the separate counting pass reads 18,000 encoded frames. Source vehicle selection
and visibly different vehicle geometries inform future scenario/profile work,
while map/upgrades/build equivalence and synchronized actions remain unresolved.

The owner's direction is to use the archive instead of new play recordings.
Archive-inferred actions cannot be relabeled as synchronized key events, and
construction excerpts cannot become untouched benchmarks. C3-19/20 retains the
141-second acquisition recorder window, the separate extraction window and their
CPU/resource coverage alongside unknown inherited human skill, play, editing and
review costs. Stored media duration and repeated decoding are experience/data
measurements, never learning-speed results. All real competence, matched benchmark,
fidelity, selected-algorithm and multi-profile qualification gates remain open.

The [149-run integrity cutoff](../../research/experiments/cycle-3-archive-reader-integrity-001.json)
verifies every canonical run at 2026-09-12 16:47:17 UTC, with no unfinished or
uncatalogued directory. It adds a new cutoff without replacing the earlier
136-run inventory. Integrity is neither backup nor scientific qualification.
The [reader ancestry inventory](../../research/datasets/native-reader-construction-source-inventory-001.json)
locates all effective glyph/UI pixel sources and the exact sealed historical
annotation bytes. Missing original session identities remain unknown; an audit
grouping cannot create independent sessions. Candidate fitting remains pending.

## Stabilization disposition — 2026-09-12

The owner redirected current work to stabilization, documentation and integration
through `dev` into `main`. The [closing report](../../research/reports/cycle-3-stabilization-report.md)
and [handoff](../operations/cycle-3-stabilized-state.md) govern the current stop;
the [future-work queue](../operations/cycle-3-future-work.md) retains all unmet
research goals. This later authority permits a software checkpoint while
`scientific_targets_met` remains false. Cycle 1/2 records and qualification
criteria are preserved. No new release or research dispatch follows from merging.

C3-07/08 now include registered reader protocol 3.0, reviewed-label publication and
implemented opt-in native collection hooks. The older draft protocol and the
still-draft live acquisition amendment remain distinguishable. One construction
label was actually published in a completed sealed run with known prediction
exposure; [its stop receipt](../../research/diagnostics/reader3-construction-stop-001.md)
preserves the original null native result, explicit prior costs and zero builder/
freeze/evaluation invocations. No accuracy, blinding or competence gate was met.

C3-19/20 now include the [151-run integrity cutoff](../../research/experiments/cycle-3-stabilization-integrity-001.json)
without replacing the earlier 136/149-run cutoffs. Code and derived reports are
reviewable in Git; private payloads remain local. Archive/restore functionality
has not yet produced a verified backup. Reader qualification, native reliability
and soak, complete profile/action/benchmark evidence, calibrated fidelity, real
learner comparisons, three one-hour qualification seeds and multi-profile
adaptation/retention remain open at this software stabilization boundary.
