# Learning efficiency and simulator cost

Status: prospective governing methodology for Cycle 3, following the owner's
2026-09-12 clarification. It supplements the [mandate](cycle-3-mandate.md) and
[measurement contract](cycle-3-measurement-contract.md). It neither changes
historical clocks and findings nor establishes a new empirical result. The
[learning-efficiency registration](../../experiments/definitions/cycle-3-learning-efficiency.json)
fixes the reference targets and future collection gates.

> **Episodes measure experience. Compute measures cost. Wall-clock time measures rapidity. Real-game capability determines whether the learning mattered.**

## The outcome that governs the project

Rapid learning means less elapsed time to independently evaluated competence in
the actual game. A method that generates more simulator frames, consumes fewer
episodes, or reaches higher surrogate reward has not thereby learned faster in
the real game. Report three separate axes: wall time, compute consumed, and
experience acquired and reused. Episodes measure experience, with their durations
and lengths attached; they are not a unit of compute.

The reference thresholds are a frozen policy's median distance of **500, 1,000,
and 2,000 real-game metres** across **20 independent episode executions**. The
selected method must be independently trained with seeds **3301, 3302, 3303**.
Qualification remains a result for each training seed; pooling 60 episodes does
not substitute for three successful training runs. Report all seed outcomes and
their variation, including failures. Three seeds support a bounded replication,
not a precise population-wide performance guarantee.

Each episode uses the frozen reference scenario and starts through ordinary game
interaction. The maximum is **900 gameplay seconds**; a **60-second window with
no new verified progress** may govern a stall stop. Missing readings do not prove
a stall. Record natural failure, governed truncation, and administrative or
instrumentation interruption separately. A moving timeout gives finite-horizon
distance, not maximum achievable distance. Qualification and lifecycle success
remain different decisions. No observer episode, training rollout, demonstration,
best episode, scripted policy or simulator result is independent learned real
qualification.

The primary complete-batch criterion admits natural or full-horizon endpoints.
A governed stall remains visible censored evidence and makes this stricter primary
batch ineligible; it is not silently substituted for a 900-second evaluation.

All 20 scheduled attempts remain in the batch denominator. An unknown, unsafe,
manually assisted or unevaluable attempt cannot become zero, success, or an
omitted inconvenient case. The primary batch verdict is unknown until all 20
have eligible independently measured outcomes; there are no replacement attempts
selected after seeing results. Publish known distances, coverage and exclusions
even when the primary median cannot qualify. The same frozen method/configuration
is used throughout final qualification. No selection, training, reader tuning or
retry is allowed from these final results.

## Clock boundaries and what a speed claim means

| Quantity | Boundary and interpretation |
| --- | --- |
| Command elapsed training time | Earliest instrumented Python entry before project/heavy imports through the 3,600-second training deadline; includes subsequent imports, initialization, acquisition, native resets/ad waits, perception, rendering, learning, logging and checkpoint serialization; interpreter startup before that entry is explicitly excluded/unmeasured |
| Checkpoint production time | Actual elapsed time at which a complete, atomically published checkpoint became available; store weight-update time and hash separately, including unchanged weights |
| Learner-active time | Measured diagnostic intervals inside the command clock; never replaces the full command clock |
| Finalization overhead | Time after the training deadline to release inputs and preserve/seal already produced evidence; no further learner updates and no newly improved qualifying policy |
| Independent evaluation time | Separate command and completion receipts covering acquisition, resets/ad waits, rendering/perception, inference and evidence sealing for each frozen checkpoint batch |
| Verified elapsed time to competence | Actual elapsed time from training command start until a passing independent evaluation is complete; includes intervening wait and evaluation latency, with competing activity disclosed |
| New-profile adaptation time | Command elapsed target-training time starting from the declared inherited model; report zero-shot performance before adaptation and inherited costs beside the target clock |
| Prior/system development cost | All available parent training, demonstrations, teacher generation, distillation, reader training, calibration, simulator construction and search costs; retain unknown quantities and shared-cost allocation explicitly |

The headline learning curve uses checkpoint production time from command start,
with independently measured real distance on the vertical axis. Label it
**independently evaluated real competence by training command elapsed time**.
Also publish verified elapsed time to competence, so a policy produced in one
hour and verified much later is never described as independently verified within
one hour. Post-training frozen evaluation avoids secretly extending the learning
budget. A concurrent evaluator requires registered resource isolation and records
any contention; pausing the command clock is prohibited.

An entry launcher must capture its clock at its earliest instrumented Python
entry, before project/heavy imports, then persist the receipt before learner work.
Record UTC and monotonic start readings, clock identity, full source SHA,
protocol/configuration hashes, exact command, run ID, parent lineage and process
identity. Source/protocol verification and costly setup after the start reading
belong inside the command clock. A recorder created after imports cannot
reconstruct this boundary from its own start time. Interpreter startup and any
minimal code needed to acquire the initial clock precede this measured boundary;
retain them as excluded/unmeasured rather than claim exact OS-launch elapsed time.
A future outer process-launch receipt can measure that additional interval under
an explicitly versioned boundary. Comparisons must use matching boundaries.

UTC across sessions supports calendar latency; cross-process monotonic values are
comparable only when their clock identity and host semantics are verified. Clock
adjustments and unresolved receipt gaps produce unknown timing, not an invented
duration.

Checkpoints are scheduled at 5/10/20/30/45/60 minutes. Record requested and actual
completion times; do not relabel late serialization as on-time availability. No
new learner update starts after the deadline. Updates must be bounded so a
deadline-crossing operation cannot produce newly eligible weights after it.
Report any unavoidable overrun and use the latest valid pre-deadline checkpoint.
An unchanged hash is an unchanged policy, even at a new requested checkpoint time.

## First observed attainment, uncertainty and censoring

For each training seed and threshold, sort eligible frozen checkpoints by their
actual production times. The first batch whose median reaches the threshold is
the **first observed qualifying checkpoint**. Retain its checkpoint and evaluation
hashes. The previous observed nonqualifying checkpoint and first passing
checkpoint form a sampling bracket; absent a previous eligible failure, the lower
boundary is unknown. This is not an exact threshold-crossing timestamp: learning
can regress, and an unobserved policy may have qualified earlier. Do not
interpolate a crossing through missing checkpoints or assume monotonic learning.

Report the first observed time, bracket boundaries, requested/actual checkpoint
times, number of eligible evaluations and any gaps. Without an observed pass,
attainment is right-censored at the last eligible checkpoint boundary; distinguish
that boundary from the nominal budget. Missing later evaluations retain explicit
warnings and incomplete budget coverage, so this is not a claim that the full
budget failed. If no eligible evaluation/timing pair exists, the result is
`not_evaluable`. Episode horizon censoring is a separate quantity from learning-time
censoring.

For distance, retain the 20 episode values, median, quartiles, worst episode,
failure causes, horizon/stall censoring, observation coverage, verified time to
distance regions and pace. Statistical uncertainty and variation across training
seeds accompany the result. A finite speedup ratio requires comparable uncensored
times under matching scenario, prior class, budgets and evaluation rules. If a
comparator never qualifies, state its censoring bound; do not divide by a made-up
time or advertise an exact "times faster" number.

## Experience, compute and priors

Keep separate counts for real acquired transitions, unique real transitions
consumed, replay/sample uses, real gameplay seconds and episodes; simulated
seconds, environment transitions, physics substeps, rendered observations and
policy decisions; and optimizer updates. Record reset/ad/capture waiting time
separately while retaining it in the command clock. A million repeated replay
samples are not a million newly acquired transitions. State counting units,
cadence, environment count and aggregation across parallel worlds.

Compute records include process CPU core-seconds and scope (including or excluding
child processes), accelerator device/model, measured active time/utilization or
an explicit unavailable value, peak/sampled-maximum RAM with its measurement
method, peak device memory where measured, and hardware/runtime identity. CPU
core-seconds may exceed wall seconds because several cores work concurrently.
Do not infer accelerator consumption from wall time or count utilization as
energy. Rates such as physics substeps/s, renders/s and updates/s are diagnostic;
they do not prove useful learning.

The current `resources-3.0` aggregation in
[telemetry/resources.py](../../src/gradientclimb/telemetry/resources.py) measures
all threads in the recorder process and excludes child processes. Its window
starts with the initial resource sample and ends with the final sample, excluding
earlier imports/provenance and later sealing. It therefore does not automatically
measure the entire command-entry interval. GPU samples describe the whole named
device across all processes. Missing samples and long gaps reduce gauge coverage;
memory maxima are sampled maxima. Preserve these scopes when copying values into
an efficiency study, and do not silently fill the uncovered command interval with
zero cost. Resource summaries are not automatically attached to checkpoint costs.

Every result declares both a policy-start class and a system-prior class, plus a
hashed parent/data ledger. The five policy-start classes are:

| Class | Required disclosure |
| --- | --- |
| Cold-start learning (`cold-start`) | No consumed task-trained policy/encoder/teacher or task demonstrations; engineered perception/environment knowledge and its construction costs remain system priors |
| Demonstration-assisted (`demonstration-assisted`) | Human sessions, whole-session partitions, collection time and supervised initialization, including failed collection work |
| Simulation-pretrained (`simulator-pretrained`) | Simulator experience, teacher, distillation and parent policy/representation costs and domains |
| New-profile adaptation (`generalist-adaptation`) | Source/generalist model, excluded target data, inherited experience/compute and zero-shot target evaluation |
| Fine-tuning (`fine-tuning`) | Relevant target-trained checkpoint and complete available lineage |

A randomly initialized actor using a task-trained encoder is cold-policy
initialization in a task-informed system, not entirely task-naive cold start.
Mixed assistance is recorded explicitly rather than hidden behind one convenient
label. Do not silently amortize inherited costs across unlimited future profiles.
Provide unamortized totals and any separately justified allocation rule; mark
unmeasured historical development costs unknown. The same prior cost is referenced
once in a lineage union rather than double-counted through multiple descendants.

## Fidelity is a cost decision tested against real outcomes

Each simulator arm declares a fidelity-level identity and hashes for environment,
vehicle/upgrade/map, calibration, observation pipeline and rendering assets.
Record the represented geometry/mechanics, measured route coverage, uncertainty,
solver/iterations/substeps, simulated time per decision and render/sensor cadence.
Synthetic geometry and debug images remain synthetic even when the engine has
articulated contacts. The reserved reference environments are unavailable until
their measured implementation and validation exist.

Start from an explicit bottleneck and a bounded registered comparison. Hold actor
access, control cadence, training budget, evaluation, priors and all other intended
constants fixed, or disclose the confound. Compare useful low-cost approximations,
measured intermediate fidelity and richer candidates where justified. Allocate
physics/rendering/learning resource budgets before dispatch, retain failed fits
and search costs, and stop the arm at its limit. An engineering pilot can reject
an unstable installation; its raw frame rate cannot select the best learning
environment. Increasing batch size cannot silently reduce solver quality or
change physical-time semantics.

Report a Pareto comparison over independent real distance/threshold attainment,
command elapsed time, CPU/accelerator cost, experience and prior cost. A candidate
is dominated only under comparable scope when another has no worse declared
costs/outcomes and improves at least one, with uncertainty shown. Missing or
incomparable measurements cannot establish dominance. More accurate physics or
rendering is retained only when measured failures or improved real performance
justify its extra cost. A fast lower-fidelity simulator may win; a slower one may
win by delivering relevant experience sooner. The decision depends on real
learning evidence, not realism, episode count or frames per second alone.

The [campaign ledger](../../experiments/definitions/cycle-3-campaign.json) sets
bounded screens before replications. The [engine pilot](../operations/engine-pilot.md)
is an engineering prerequisite, not a real-learning comparison. Qualification
cannot begin until the specific collecting implementation, full protocol and
source revision are committed and the declared gates are satisfied. Prospective
fields below are requirements, not claims that existing CLI commands already
capture them.

## Analysis artifact contract

[efficiency.py](../../src/gradientclimb/experiments/efficiency.py) provides
versioned `EfficiencyProtocol`, `LearningEfficiencyStudy`, `CheckpointCost`,
`FrozenRealEvaluation`, `PriorCost` and `CostVector` records. Each concrete study
retains artifact-root-relative evidence paths and SHA-256 hashes. The framework
JSON is a planning registration, not a fabricated populated study envelope.

`CheckpointCost.elapsed_seconds` is the actual published-checkpoint time and
`command_clock_evidence` supplies its boundary evidence. Independent evaluations
retain `evaluation_elapsed_seconds` and `verification_elapsed_seconds` separately.
The latter cannot precede checkpoint production plus the evaluation duration.
An explicit learned-policy declaration and matching immutable real-scenario
evidence distinguish learned reference qualification from scripted controls or a
changed upgrade configuration.
`prior_roots` identify the complete system-cost graph; `policy_prior_roots`
identify its inherited policy/representation subset. This permits engineered
system-construction costs to remain visible in a cold-policy run. Fidelity is
identified by `simulator_fidelity_id`; the actual measurement scope accompanies
`compute_comparison_scope`.

`analyze_study()` evaluates supplied declarations; it does not prove their truth.
`load_studies()` verifies sealed canonical envelopes and referenced bytes before
reporting verified artifact identity. Neither function independently establishes
that episodes were untouched, readers accurate, or clock scope complete. Those
claims require the registered protocol and acquisition evidence. Missing eligible
checkpoints remain warnings, and missing inherited costs remain unknown.
`compare_studies()` forms descriptive Pareto sets only within matching protocol,
scenario, prior, evaluation and resource-accounting scope. It does not replace
the required training-seed replication or uncertainty analysis.

[train_with_command_clock.py](../../scripts/train_with_command_clock.py) is the
opt-in early-entry wrapper for the existing training CLI. Its checkpoint receipts
record actual publication time and budget eligibility; it does not make the
legacy surrogate a qualified real learner. The required `--entry-receipt` names
a new JSON file written and flushed before project imports; the canonical run
copies that receipt after recorder creation. An import/setup failure therefore
retains the outer receipt even if no canonical run exists. Failure before the
first receipt write remains a launcher-level gap. Preserve unlinked entry receipts
as incomplete attempts rather than deleting them. A concrete dispatch protocol
must satisfy every collection gate before claiming a complete command-attempt ledger.

## Research grounding

The [primary-source register](../../research/literature/references.json) contains
Agarwal et al. (LIT-024), which motivates distributional evaluation and caution
with few training seeds, and RMA (LIT-018), which motivates explicit inherited
training cost for adaptation. PPO (LIT-005) and the domain/dynamics randomization
papers (LIT-015/016) support candidate methods; none establishes learning speed
or transfer quality for this game. This methodology's target distances, time
boundaries, episode counts and budgets are prospective project decisions.
