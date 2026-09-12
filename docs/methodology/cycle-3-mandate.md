# GradientClimb — Complete the Rapid-Learning, High-Performance, Multi-Profile Research System

## Execution mandate

**Repository:** `https://github.com/TheCyberLocal/gradient-climb`

**Starting authority:** `cycle-2-paused`, currently targeting `610dc48dc601f878086053f900ee7a005ccea821`; both remote branches were at that commit when this task was prepared. The released package is `0.1.0a2`.

**Expected workstation checkout:** `D:\Projects\gradient-climb`. Discover the actual checkout and artifact locations before using them; the older C: checkout is not the assumed working directory.

This task authorizes resumption of the research program. It supersedes the pause instruction for new work, not the preservation rules for historical evidence. Develop in coherent, tested, atomic commits on `dev`; push stable increments. Integrate to `main` only at the final stable, documented release boundary, following the integration gates below.

The deliverable is a functioning data-science research system, not another scaffold, a scripted game bot, an attractive but unvalidated simulator, or a single lucky gameplay video. Continue across implementation and experimental milestones without requiring a new task between them. Ask the owner only for genuinely unavailable access, human demonstrations, unavailable game content, or consequential changes outside this authorization.

---

## 1. Mission and governing priorities

Build a visually grounded AI that learns high-quality control rapidly, first on the installed **Hill Climber / Countryside / observed maximum upgrade configuration**, then across materially different vehicle and map profiles.

The owner reports driving beyond **2,000 real-game metres** using the same car and map. Treat that as an owner-reported capability to document with matched recordings, not an independently verified benchmark. The practical target is repeatable real performance at that scale, with forward pace, resource management, score collection, and recoverable skilled maneuvers—not continued acceptance of approximately 400 m as high competence.

The scientific objective remains capability acquired per unit of **wall-clock time, real interaction, simulated experience, computation, and prior knowledge**. Pursue a one-hour learning target aggressively, while measuring cold start, demonstration-assisted learning, simulation pretraining, and adaptation separately.

Priority order:

1. Trustworthy measurements and safe, dependable operation.
2. Repeatable real-game distance, survival, and pace.
3. Fast learning and effective reuse of experience.
4. Useful score and skill acquisition without sacrificing the run.
5. Multi-profile generalization, rapid adaptation, and retention.

Do not interpret this ordering as permission to omit score research or multi-profile capabilities. Stage them so that failure causes remain distinguishable.

## 2. Reconstruct the starting state, including its limitations

Read the following before modifying code:

- `docs/operations/cycle-2-state.md` and `docs/operations/cycle-2-future-work.md`.
- `docs/operations/resume-state.md` and `docs/operations/resumption-plan.md`.
- `docs/methodology/completion-audit.md`, `cycle-2-preregistration.md`, and `research-brief.md`.
- `docs/research/reward-research.md`, the architecture ADRs, and findings F-001 through F-006.
- Both cycle integrity inventories, the local-evidence inventories, experiment definitions, and the two annotated release receipts.
- Current training, observation, scoring, session, simulator, observer, and report code—not just the README.

Establish and commit a new cycle requirements/evidence register mapping every inherited requirement to its implementation, empirical evidence, remaining gap, prerequisite, and acceptance criterion. Preserve historical documents at their historical scope; create new cycle records rather than replacing their cutoffs.

Important baseline distinctions:

| Area | Starting reality | Consequence |
|---|---|---|
| Real performance | Earlier 458/411 m examples were scripted gas runs with 60-second collection horizons and paused-boundary readings. The native learner has no qualified learned result. | Do not call those distances a demonstrated trained-policy ceiling or compare them directly with an unrestricted human run. |
| Surrogate | `surrogate-0.1.0` uses two contact points and sums of smooth sine waves. Traction, suspension, torque and crash approximations exist, but are uncalibrated. | Improve fidelity from evidence; do not inaccurately claim the old model has no physics. |
| Episode horizon | Default surrogate settings are 1,000 decisions × 0.06 seconds, or 60 simulated seconds. The registered native CEM protocol also caps episodes at 60 seconds. | Audit truncation before explaining apparent distance plateaus or qualifying 2,000 m performance. |
| Native actor | The registered actor has 34 linear parameters over eight current body/terrain features and masks. Logged history is not equivalent to learned policy memory. | Preserve it as a baseline, not a predetermined final architecture. |
| Student | Two imitation students reached about 483.6 surrogate units, with 11/20 crashes versus 2/20 for their teacher. Budget and seed changed together. | More budget did not help in those runs; an information ceiling or causal explanation has not been proved. |
| Headed training | The live PPO observer exists. A separate failed run, `e911764e`, recorded metrics but finalized with zero steps/episodes. | Investigate failure accounting and the actual exception before attributing it to rendering. |
| Objectives | Cycle 2 metric/objective code exists, but no qualifying multi-objective native comparison exists. | Audit the semantics and validate measurements before training against them. |
| Evidence | The freeze records 134 valid canonical local runs, approximately 992 MB. Raw models, frames and run bytes are not in ordinary Git. | Remote source availability is not proof that empirical artifacts are available or backed up. |

The five inherited, registered-but-unexecuted protocols are `native-reliability-2.2`, `real-baselines-2.0`, `reader-validation-2.0`, `context-encoder-2.0`, and `privileged-critic-2.0`. For each, record whether it will execute unchanged, needs a prospective amendment, or is superseded by a new version. Never silently change a registered protocol or treat its existence as a result. The reader-study documentation contains historical eligibility statements that must be reconciled against actual session provenance before selecting held-out data.

## 3. Git, artifacts, and experiment authority

Fetch branches and tags; verify `cycle-1-paused` and `cycle-2-paused`, current ancestry, local changes, active processes, and the current CI state. If branches have advanced, inspect and preserve intervening work rather than resetting to this task’s expected SHA.

Work on `dev`. Each substantive commit must contain a coherent change, its focused tests, relevant documentation, and any source-linked experimental interpretation. Separate infrastructure, experimental definitions, and findings where that improves reviewability. Do not squash the entire program into one opaque commit, force-push, change historical tags, or override branch protection.

Before important experiments, commit the governing implementation and protocol. Reports produced afterward belong in later commits linked to that experiment source. Unrelated concurrent development must not silently change running experiments; use pinned worktrees or equivalent isolation where necessary.

Verify the local evidence store before reuse. Inventory essential checkpoints, profiles, glyph banks, videos, and raw observations. Use new explicit output paths with `audit_cycle_state.py`; preserve its overwrite protection. Never delete sealed failed runs or cited artifacts as “obsolete.” Remove only regenerable caches or explicitly approved expendable data. Implement archive/restore verification for an owner-approved local or external destination; do not claim a backup until a restore test verifies it. No paid storage or public upload is authorized.

## 4. Execute a dependency-driven program, not a serial wishlist

Use these workstreams, with bounded experimental budgets and explicit gates:

| Workstream | Main work | Gate before dependent work |
|---|---|---|
| Foundation | Artifact recovery, current-state audit, horizon correction, interrupted-run accounting, host preflight | Reliable source/data identity and working smoke tests |
| Real instrument | Capture/control timing, scoring, lifecycle reliability, held-out labels | Qualified scored real episodes |
| Reference simulation | Measured terrain, articulated physics, visual/camera interface | Held-out fidelity tests and early real transfer probes |
| Learner | Human imitation, temporal observations, student-state correction, critic/replay studies | Independently evaluated learned policies |
| Objective and qualification | Distance/score/time/recovery comparisons, long-horizon and one-hour benchmarks | Frozen winners evaluated on untouched episodes |
| Multi-profile | Different dynamics, controls, maps, adaptation and retention | Profile-specific real qualification and transfer evidence |
| Release | Reports, dashboard, independent reproduction, validation, main integration | Exact-SHA green release and honest completion audit |

The native reliability gate blocks unattended native training; it does not block offline labeling, demonstration-recorder development, simulator reconstruction, or simulator experiments. Obtain an early learned real baseline once the gate passes. Do not spend the whole cycle building a visually elaborate simulator without testing whether it helps real control.

Register a campaign ledger with trial counts, budgets, stopping rules and decision criteria. Cheap screening precedes replication. After an unproductive campaign, identify and test a specific bottleneck rather than indefinitely repeating the same recipe. A research technique may be rejected by evidence; a required product capability may not be silently declared optional.

## 5. Correct performance definitions and the 400 m versus 2,000 m comparison

Create separate protocols for **instrument reliability**, **fixed-time driving**, **long-run competence**, and **time-to-learn**.

A reliability episode may remain short. It is not the high-performance test. High-performance evaluation must allow sufficient gameplay time to reach 2,000 m at the observed human pace, with a declared maximum horizon and stall handling. Record natural crash, fuel exhaustion, time-limit truncation, and administrative interruption separately. A timeout while moving is a censored outcome, not a crash or evidence of a policy’s maximum distance.

Freeze the following starting acceptance proposal before qualification data are collected:

- On the reference real configuration, a frozen policy should achieve **median distance at least 2,000 m across 20 independent episode executions**; additionally report the full distribution, lower quartile, worst run, time to distance thresholds, and natural failures.
- Compare pace and useful score with the recorded human reference under the same upgrade state, horizon and assistance rules. Do not claim human-expert equivalence from a reported personal best alone.
- Evaluate learner reproducibility over at least three independently trained seeds for the selected method, unless an explicitly documented external constraint prevents it. Repeated game episodes do not imply independent game terrain seeds.
- The one-hour target asks whether policies produced by the **60-minute training checkpoint** meet the competence criterion. Evaluate earlier checkpoints at 5/10/20/30/45 minutes too. Longer training cannot replace a failed one-hour result.

These are prospective targets, not accomplished results or established statistical certainty. The historical 500 m provisional gate remains a historical early milestone, not the revised definition of high performance. Never lower targets after viewing qualification results. If the target is missed, publish the measured shortfall and continue evidence-driven improvement within the registered program.

## 6. Fix run accounting, continuation, and headed usability first

Inspect `e911764e` and the relevant exception, console, metric journal, observer and recorder paths. Determine whether the failure was user interruption, learner failure, window failure, resource exhaustion, or a finalization defect. Do not assume the observer caused it because it was enabled.

Maintain incremental authoritative counters during training, not only when `train()` returns successfully. On interruption or failure, preserve the last known steps, optimizer updates, episode counts, elapsed time, valid checkpoint and failure cause. Distinguish observed counters from reconstructed lower bounds. Never rewrite the sealed historical run; issue a linked diagnostic/correction record.

Fault-inject Ctrl+C, window closure, learner exceptions, video failures, partial writes, disk pressure and termination between checkpoint stages. Verify input release and observer cleanup. Optional visualization failure must not erase legitimate learner work; a completed learner must not conceal a failed observer.

Retain the existing headed interface, improve its regression coverage, and extend it to the new simulator and chosen policy. Use immutable snapshots, separate observer state/RNG, and no observer-to-learner experience feedback. Offscreen pixel rendering required by the actor remains part of headless training; “headless” removes the display window, not the actor’s images.

Distinguish warm-start fine-tuning from exact resume. Existing PPO restores weights and available optimizer state but not exact in-flight simulator state. Add explicit checkpoint manifests for optimizer, normalization, recurrent state policy, RNG and sampler state as needed; never promise bitwise continuation unless tested. Old checkpoint formats must remain readable or fail with an explicit migration instruction.

## 7. Make the native environment reliable without rewarding unsafe recovery

Audit the current `native-reliability-2.2` implementation and protocol, then execute it unchanged if applicable. It requires at least ten consecutive scored successes in a twelve-attempt session, zero unintended actions and zero manual intervention. Extend this with a sustained, representative soak test before “set-and-forget” operation is claimed; a short pass is not a universal safety guarantee.

Preserve independent gas/brake state, press/release order, hold duration, freshness limits, focus/geometry checks and emergency release. Use only ordinary screen capture and authorized game input. No process injection, private memory, game binary changes, hidden game-state access, purchases or advertisement circumvention.

Separate three records: **gameplay outcome**, **readiness for the next episode**, and **session safety**. A verified distance can remain valid diagnostic/evaluation data even if later parking fails; that does not make the session reliable or authorize further input. Decide learning eligibility prospectively, record every attempt, and audit whether exclusions preferentially remove difficult or poor episodes. Keep Cycle 1/2 eligibility semantics unchanged.

Known legitimate ad-close controls require effect evidence; a recognizable icon alone is insufficient. Unknown ad/UI states release input and wait. Preserve bounded approved stuck-application recovery, without resetting timers to accelerate advertisements or using restart as a reward/entitlement bypass. Any unintended effect or latched guard fault stops the session; recovery must not conceal it.

Add preflight and runtime checks for C: and artifact-drive free space, process health, bounded disk growth, latency, and capture readiness. The repository reports emulator failures near low C: capacity; treat the value as a workstation observation, not a universal vendor threshold. Record a conservative margin and stop safely before exhausting it. Enforce exclusive native input ownership; do not bring the agent’s own tool windows over the game during collection.

## 8. Build a synchronized real-data and demonstration system

Record monotonic frame acquisition times, observation-ready times, intended actions, delivered input events, pedal changes, model latency, game-state transitions, and score/HUD evidence. Identify duplicate/stale frames and dropped intervals. Learn from the observation available **before** the action, not a conveniently later screenshot.

Prepare a read-only human demonstration mode that injects no gameplay input and records only the relevant game controls while the game is active. Capture expert successful runs, ordinary runs, and recoveries/failures—not just highlights. Ask the owner to play only after the recorder is validated and give a concrete capture procedure; their role is demonstrator, not data engineer. Start with a small useful batch and request additional maneuvers based on coverage gaps rather than an arbitrary demand for dozens of runs.

Include diagnostic throttle/coast/brake/both sequences, steep acceleration, wheelies, airborne correction, hard landings, low traction, fuel pickup and bridge traversal where accessible. Verify the actual game build, vehicle, map and upgrade levels visually. Internet upgrade statistics are priors; they are not calibrated force, torque or friction coefficients.

Partition data by whole run/session and purpose: construction, simulator fitting, imitation training, development validation, and untouched qualification. Crops and adjacent frames from one video are not independent examples. Keep human benchmark runs separate from demonstrations used to train or tune. Video without synchronized actions may support geometry/physics analysis, but is not action-labeled imitation data.

## 9. Establish fair baselines and locate the real failure mechanisms

Run random, always-gas and a clearly labeled stabilizing heuristic under matched conditions after lifecycle qualification. Add any existing actual learned model only when its checkpoint and training provenance establish that it really learned.

For both fixed-time and long-run tests, report distance, duration, pace, failure type and measurement coverage. Compare competent human and AI traces around the first divergence—not only the death frame. Diagnose excessive crest speed, late braking, airborne pitch, traction loss, fuel mistakes, bridge response, observation dropout, actuator latency and administrative termination separately.

Do not assume all failures occur at a universal 300–500 m obstacle. Group by verified location and geometry where the track permits alignment; use scenario classes when repeatability is not established. Turn repeatable failures into simulator regression fixtures and curriculum targets, not route-specific “brake at metre X” scripts.

## 10. Introduce the new simulator without breaking the old one

Preserve `surrogate-0.1.0` and its historical learning results. Implement a separately versioned, measurement-driven **reference simulator** for the initial real configuration. Treat “digital twin” as an objective, not a validation label automatically earned by implementing physics.

Separate mechanics, terrain/objects, camera/rendering, observations, reward/evaluation, profile definitions, and training. Use an environment registry or factory instead of continuing to hardwire every trainer and observer to `VectorHillEnv`. Maintain thin adapters and one canonical experiment store; avoid a second incompatible framework.

Benchmark a mature lightweight 2D physics engine first. Box2D’s wheel joints expose suspension and motor behavior and are a plausible starting point, not a mandate or proof of HCR equivalence. Select and pin an appropriate supported implementation after measuring installation, articulated-body stability and batch/offscreen throughput. Do not choose Unreal or Blender merely because installed; they may assist original asset creation or inspection without becoming the training runtime.

Provide three explicit environments: the frozen legacy surrogate, a reference simulator tied to measured scenarios, and a randomized training distribution derived from it. Never mix their units or results in an unlabeled leaderboard.

## 11. Reconstruct the dynamics that decide success and failure

Replace purely plausible constants with fitted effective parameters and uncertainty. Prioritize observable consequences over claims to recover exact proprietary physics.

For the reference car, implement and validate chassis mass distribution and rotational inertia; independently rotating wheels and their inertia; appropriate driven wheels; motor torque/speed behavior; braking, coast and reverse transitions; suspension travel, stiffness and damping; normal-load-dependent tire grip, slip and traction limits; chassis/head collision and observed death rules; airborne pedal response; fuel consumption, pickup geometry and resource timing.

The same pedal command must sometimes be good and sometimes dangerous: aggressive acceleration should produce a wheelie/flip under the measured conditions that cause one in the game, not because of a hand-coded anti-throttle penalty. Braking must affect balance and momentum through the modeled mechanism.

Add numerical tests for timestep/substep convergence, collision tunneling, joint stability, impossible energy growth, penetration, exploding forces and artificial speed caps. Record where the effective model remains uncertain. Correct contact geometry and center of mass before compensating with implausible friction or arbitrary damping.

## 12. Build terrain and interactive structures beyond smooth hills

Reconstruct the observed route through the owner’s 2,000 m reference range, extending measured coverage as successful runs reach farther. Preserve sharp slope changes, small bumps, troughs, steep faces, crests, gaps, ledges and surface transitions. Unobserved terrain is generated research terrain, not a claimed reconstruction.

Model rope/plank bridges as articulated collidable structures with measured span, anchors, sag, segment geometry, compliance, damping and load response where observed. The car must interact with moving bridge surfaces; drawing a bridge over static ground is insufficient. Do not force gaps, overlapping structures or dynamic surfaces into a single smooth height function.

Create a scenario library for crest control, traction-limited ascent, takeoff/landing, pits, bridge entry/exit, flexible traversal, low fuel and recovery. Attach source evidence or mark a case synthetic. Permit simulator-only start states near difficult scenarios for training; retain full-route starts for qualification and clearly distinguish scenario curriculum from public game capabilities.

## 13. Make rendering part of the observation contract

The simulator must preserve task-relevant visual signals: car/head/wheel proportions, suspension movement, terrain edges, roughness, bridge planks/ropes, gaps, coins, fuel objects, bonus indicators, HUD layout and camera framing/tracking/zoom. Use original or appropriately licensed art; keep private game captures outside public source history.

Implement a reference visual mode and a controlled randomization mode. Randomize nuisance appearance, background, textures, color, resolution, anti-aliasing, capture noise and camera variation within declared ranges. Do not randomize away an object’s meaning or make two controls/tasks visually indistinguishable.

For a pixel-trained actor, use the same preprocessing and observation cadence in simulation and real deployment. For a feature actor, run simulator-rendered frames through the same perception system. Exact simulator coordinates may supervise perception or train a critic, but cannot silently replace noisy screen features for the deployed actor.

Offscreen rendering must work without a window and be included in throughput accounting. A simulator running at enormous state-only FPS does not establish the speed of the pixel pipeline. Assess object/geometry recognition and policy performance on held-out real images and episodes; visual resemblance or a pixel-similarity score alone does not establish transfer.

## 14. Calibrate and qualify fidelity iteratively

Estimate camera scale/motion and geometry before treating screen displacement as vehicle speed. Compare relative geometry, time, and calibrated progress where absolute quantities are not identifiable. Fit effective parameter sets with uncertainty; do not present non-identifiable mass/friction combinations as recovered ground truth.

Use construction data to choose tolerances before held-out tests. Compare against the old surrogate and simple persistence/history predictors. Evaluate:

| Layer | Required evidence |
|---|---|
| Dynamics | Short-horizon action-response error, pitch/rate, jump/landing timing, braking, traction transitions and crash/survival agreement |
| Geometry | Terrain profile, gap/crest placement, bridge sag and load response in identifiable coordinates |
| Perception | Vehicle/object/terrain errors, score accuracy, visibility and missingness through shared image preprocessing |
| Closed-loop utility | Multiple frozen policy rankings, failure locations and actual real performance after training in each simulator variant |

Replaying recorded actions is useful for short horizons; do not demand indefinitely identical open-loop trajectories from chaotic contact dynamics. Use repeatability distributions, event timing and closed-loop outcomes as well.

Conduct early bounded real transfer probes while fidelity improves. Adopt additional realism when it corrects measured failure classes or improves real learning enough to justify its cost. Do not optimize only for prettier images, nor abandon required terrain/bridge mechanics merely to preserve legacy FPS.

## 15. Upgrade observations, memory and action modeling

Compare the current linear/stacked baselines with a compact temporal policy. A practical first candidate is a visual/geometry encoder plus GRU or LSTM and previous-action/duration inputs. Compare against a matched feed-forward stack before escalating to a transformer or larger model.

Test look-ahead terrain, relative motion, full-direction orientation, angular rate, wheel/contact/traction proxies, resource state and upcoming pickups. Retain validity/confidence masks. The current modulo-pi body-axis representation cannot distinguish every upright/inverted pose or count complete flips; resolve that ambiguity with validated cues rather than interpreting it as full orientation.

Provide independent pedal actuation with all four states, arbitrary holds, and gas-first versus brake-first combinations. A four-joint-state distribution is acceptable; independent actuation does not require statistically factorized Bernoulli heads. Preserve timing in observation/replay and include genuine additional buttons only where the selected vehicle exposes them.

Reset recurrent state appropriately at true episode/profile boundaries. For replay-based sequence learning, implement burn-in, masks and action/observation timing correctly. Never inject future recovery labels, simulator-only parameters, terrain beyond the visible sensing contract, or qualification identities into actor decisions.

## 16. Use demonstrations and student-state correction to accelerate learning

Implement action-aligned behavior cloning from the owner’s competent demonstrations as a named assisted-training branch. Evaluate it immediately on held-out real episodes. No human explanation of every driving tactic is required; collect behavior and context.

Compare the existing teacher-occupancy student with occupancy-corrected imitation: let the student visit its own states, label those states using a competent simulator teacher, aggregate the data, and retrain. DAgger provides the relevant precedent. A simulator teacher labels simulator states; do not pretend it can supply correct labels for arbitrary real frames without a validated state mapping. Real corrective labels require recorded human intervention or a separately justified teacher.

Compare representation, occupancy correction and budget with controlled ablations. The two Cycle 2 runs do not isolate budget from seed and do not prove the exact cause of the student’s deficit. Test missing-information hypotheses against observation enrichment rather than repeating a causal assertion.

After imitation, use RL to improve beyond the demonstrations where evidence supports it. Keep teacher generation, demonstration collection, supervised training and distillation costs in the parent lineage. A ten-minute fine-tune of a heavily pretrained student is not ten-minute cold-start learning.

## 17. Select learning algorithms by real progress per cost

Keep PPO and small CEM as comparators. Do not replace a working baseline solely because another algorithm is fashionable. Validate custom learner correctness against focused tests and, where useful, a mature reference implementation.

Prioritize a small set of justified candidates: temporal PPO with optional imitation auxiliary loss; an appropriate discrete recurrent off-policy/replay learner for expensive real experience; and observation-restricted actor/privileged-critic training in the new simulator. A compact four-state action space is not a reason to introduce continuous-control machinery unnecessarily.

Reuse real demonstrations/transitions with methods designed for replay or supervised/offline learning. Do not feed arbitrary stale trajectories into ordinary on-policy PPO as though they were current rollouts. Distinguish sample efficiency from wall-clock efficiency, and include perception/rendering/optimization overhead.

Audit the discount horizon as well as the episode horizon: the current default discount is 0.99 per 0.06-second simulator decision. Verify whether the resulting temporal emphasis supports fuel planning, landing preparation and long-run competence. Test changes at fixed physical-time semantics instead of blindly increasing a coefficient.

Before world-model planning or large searches, show evidence that prediction/planning—not observation, fidelity, latency, reset cost or basic training correctness—is the dominant remaining bottleneck. Register bounded pilots and eliminate underperformers early; reproduce selected gains across seeds.

## 18. Audit and complete the multi-objective reward system

Preserve Cycle 2 objective IDs and exact historical behavior. Introduce new versions for corrected semantics; do not silently “fix” old objective meanings.

Required audit items before use:

- The existing `score_gain` is constrained to a coin-counter boundary difference. Establish whether it measures the intended episode score and which bonuses it includes. Separate collections, stunt bonuses, unrelated balances, revive/ad rewards and double counting.
- Recovery windows need unique event/score attribution. Summing gains from overlapping windows does not itself prevent duplicate attribution. Test conservation of credited score and overlapping airtime/rotation events.
- The implementation named `lexicographic-2.0` is a weighted scalar. It does not provide unconditional priority guarantees when lower-priority terms can grow. Implement actual ordered comparison/constraints, or prove appropriate bounded scalarization and name it accurately.
- Validate time-to-goal, censoring, unknown recovery and terminal-cause coherence. Do not reward an early suicidal burst, punish every necessary recovery, or allow missing measurements to appear successful.

Run the four required matched families: distance; distance plus raw score; distance/score/time; and recovery-conditioned skill credit. Hold time treatment fixed when isolating the effect of recovery; otherwise identify the comparison as confounded. Use an optional true constrained/lexicographic selection method where useful.

A provisional trick reward becomes eligible only after an observed, preregistered continuation/recovery outcome. This remains a hypothesis about useful credit assignment, not proof that every conservative maneuver is better. Evaluate fatal flips, useless airtime, slow-safe inactivity, reckless speed, coin chasing with progress loss, backward loops, stagnation and local farming. Retain independent distance, score, survival, pace and coverage metrics.

Do not let a reward study postpone the first credible long-distance learner indefinitely. Establish distance/pace competence first, then demonstrate score/skill gains under declared distance and survival non-inferiority constraints.

## 19. Make the one-hour claim precise

Maintain separate experiment classes:

| Class | Permitted initial knowledge |
|---|---|
| Cold-start learning | No task-trained policy/encoder/teacher weights or task demonstrations consumed; disclose engineered environment/perception development separately |
| Demonstration-assisted | Declared human demonstrations and supervised initialization |
| Simulation-pretrained | Declared simulation experience, teacher and distillation lineage |
| New-profile adaptation | Declared generalist source model; target profile excluded from its training as specified |
| Fine-tuning | A relevant existing target-trained checkpoint |

For the new primary one-hour claim, use command-start-to-training-deadline wall time, including imports/initialization, experience acquisition, perception, optimization, logging, checkpoint serialization and native reset/ad delays. Record a secondary learner-active clock for diagnosis, never as a substitute. Infrastructure construction and pretraining belong in separately reported prior-cost ledgers. State both policy-start and system-prior classes: a randomly initialized actor using a task-trained perception network is cold-policy training, not an entirely task-naive system. Do not re-label historical clocks, which had different boundaries.

Capture the latest fully valid policy at 5/10/20/30/45/60 minutes; admit no new learner update after the deadline and report unavoidable finalization overhead separately. Report actual times and unchanged weights when no update has occurred. Independent evaluations have their own cost record and must not secretly give the learner extra updates or a paused-clock advantage. Use resource isolation or post-training frozen-checkpoint evaluation for uncontaminated learning comparisons.

Compare distance-at-time, time-to-500/1,000/2,000 m competence, useful real transitions, experience consumed and full prior cost. If the old method never reaches a threshold, report its time as censored or beyond budget; do not fabricate a finite “times faster” ratio. Select configurations on development data and reserve untouched final evaluation.

## 20. Scale throughput only after validating useful experience

Benchmark environment count, CPU/GPU placement, offscreen rendering, batching, learner updates, memory and data transfer on the actual workstation. Start with measured sweeps rather than assuming 100, 256 or 1,024 environments is optimal.

Record raw physics steps/s, rendered observations/s, policy decisions/s, optimizer updates/s and improvement per training minute separately. Prevent silent physics simplification when increasing batch count. Track rendering fidelity, solver iterations and sensor cadence as versioned settings.

Measure action-to-observation latency in the actual game and simulate the observed delays/noise where useful. Keep discounting and reward rates coherent with physical time when action cadence changes. Avoid simultaneous unrelated high-load jobs during qualification; exploratory concurrency must be disclosed.

Investigate a subprocess observer only if measured contention or reliability justifies it. Do not require zero-cost visualization: set a measured usability/overhead target and report it honestly.

## 21. Design profiles early; qualify different vehicles after reference competence

Build profile contracts while constructing the reference simulator so additional vehicles do not require a new architecture. Do not launch broad multi-profile training before the first real car/map is competent.

Separate `VehicleProfile`, `UpgradeProfile`, `MapProfile`, `ObservationProfile`, `ControlProfile`, and a composed scenario. Include version, game build, available input channels, calibration provenance, rendering assets, body/joint topology, uncertainty, meter definitions and compatible checkpoints.

Do not constrain every vehicle to the same two-wheel chassis. Support composable validated mechanisms for different wheel layouts, tracked or unusual bodies, airborne thrust/boost, transformations and mode-dependent controls as observed. The simulator may implement verified vehicle mechanics—including ordered combinations—but the driving policy must learn when to use them. Profiles must not contain route scripts or “perform trick when X” decision rules.

Vehicle identity and learned dynamics context are distinct. A known profile may select compatible inputs/assets and condition the policy; an adaptation module should infer changing dynamics from causal observation/action history. Report known-profile specialization separately from held-out-profile adaptation.

After reference qualification, require at least two additional legitimately available vehicle profiles with materially different dynamics, including a special/airborne or sequence-sensitive mechanism where available, and at least one additional real map. If content is unavailable without a purchase, report the exact access blocker; do not buy or unlock it automatically. All vehicles need sensible profile-specific targets rather than inheriting a universal 2,000 m threshold.

## 22. Demonstrate generalization, adaptation and retention

Separate four settings: unseen scenarios on a known profile; unseen map; held-out vehicle; and held-out vehicle/map pair. Real known-map route competence is valuable, but is not evidence of unseen-map generalization. Do not call procedural surrogate vehicles commercial-game vehicles.

Before adapting, measure zero-shot performance. Then measure 5/10/30/60-minute adaptation and reevaluate the original tasks. Compare target improvement, retained competence, negative transfer, prior compute and inference cost. Compare specialists, profile-conditioned generalists and learned-context policies under matched total budgets.

Implement the registered context-encoder and privileged-critic studies, or prospectively supersede them with clearly justified new versions appropriate to the new simulator. Preserve old registrations as never executed where applicable. Hold actor access fixed in critic experiments; run explicit tests that privileged fields cannot affect actor inference.

Where forgetting is measured, compare targeted replay, mixed-profile training, distillation or other bounded mitigations. A context encoder is not automatically successful because it exists. Evidence-based non-adoption is acceptable; a working multi-profile control system and actual adaptation measurements remain required outcomes.

## 23. Use curriculum and harder scenarios without manufacturing transfer claims

Build curriculum from observed failures and successful expert maneuvers. Compare it with unchanged sampling at equal total budget. Preserve a realistic base distribution and held-out full routes; do not train exclusively on impossible or arbitrarily harder-than-real worlds.

Randomize dynamics around measured uncertainty and geometry around meaningful real scenario families. Widen distributions only when this improves robustness rather than overwhelming learning. Simulator reset-to-challenge training is allowed and recorded, but final real evaluation starts through ordinary game interaction.

Select long-horizon, hyperparameter and model-based studies only where they answer an unresolved performance question. Log all search compute and negative trials. The objective is not to implement every algorithm named in prior brainstorming.

## 24. Finish the research interface and evidence trail

Extend the existing canonical datastore/dashboard rather than replacing it. Present simulator generation, actual-game results, human references, scripted baselines, learned models and prior-training class visibly and separately.

Required analytical surfaces include learning curves against actual time; matched human/AI traces; performance by distance region and failure cause; physics/geometry/visual fidelity; outcome and coverage vectors; reward tradeoffs and hacking diagnostics; profile generalization; adaptation/retention; model lineage; rendering and runtime cost; and interrupted-run status with valid counters.

Keep every graph traceable to run/configuration/checkpoint/dataset hashes. New cycle reports use new inventory paths and frozen cutoffs. Missing measurements stay missing. Notebooks consume package logic and canonical records; they are not the sole implementation of the method.

Ship a playable/inspectable reference simulator, a live headed learner, offline video/replay comparison and reproducible report generation. Mark observer episodes as visualization rather than independent qualification. Preserve private game imagery locally unless publication rights and authorization are established.

## 25. Quality gates, atomic delivery and final integration

Test the actual contracts: pedal timing and combinations; capture faults; unwanted click consequences; termination/truncation; score missingness; recovery attribution; replay sequence masks; actor/critic separation; checkpoint compatibility; interruption counters; offscreen/windowed parity; simulator contacts/bridges; deterministic seeded cases; artifact immutability; dashboard queries and documentation commands.

Add portable and native Windows validation where feasible. Ordinary CI must not control the game or run full research training. Exercise supported Python/platform claims or narrow them honestly. Test a clean installation and artifact restoration separately from an already-configured workstation.

Recommended atomic milestones are: recovery/measurement contract; failed-run accounting; native reliability; demonstration pipeline; environment/profile interfaces; physics fixtures; terrain/bridges; visual pipeline; fidelity report; temporal imitation learner; student-state/critic experiments; real learning pilot; objective comparison; one-hour qualification; additional profiles; adaptation/retention; final research presentation. Split large milestones further into independently valid commits. Push stable `dev` increments and verify their exact-SHA CI.

At the final stopping point, stop adding features. Resolve transitional code, unsafe defaults, stale docs and artifact integrity issues. Generate the current completion audit and research report, run full tests/lint/format/compilation/package checks, verify important artifacts, and close or explicitly finalize every active run.

Fetch remote state again. Prefer a fast-forward from reviewed `dev` when ancestry permits; otherwise reconcile legitimate main changes through normal review and revalidation. Never force the merge, rewrite history, or bypass protections. Push and verify `dev`, integrate and push `main`, confirm exact-SHA main CI, and create a new annotated release tag with truthful scientific and software status. Preserve both prior cycle tags and `dev`.

## 26. Completion is an evidence matrix, not a success narrative

The final report must distinguish **implemented**, **validated**, **target met**, **target not demonstrated**, and **externally blocked**. A stable research release may record a negative scientific result, but it may not claim the original mandate is complete while the required capability remains missing.

| Required outcome | Evidence needed |
|---|---|
| Reference real driver | Frozen learned policy, matched long-horizon evaluation, repeatable progress toward/through 2,000 m and measured pace |
| Rapid learning | Actual checkpoint curves, explicit priors/costs, independent seeds and honest one-hour target status |
| Useful simulator | Measured physics, geometry, visual interface, bridge/scenario coverage and real-transfer comparison |
| Reliable native operation | Completed lifecycle protocol and representative sustained test, including safety/unknown/coverage records |
| Skill-aware objective | Validated score/recovery observations and matched objective/hacking comparisons |
| Multi-profile system | Real qualified additional vehicles/maps, compatible artifacts, held-out and adaptation results |
| Retention | Before/after evaluation on source tasks and tested mitigation if needed |
| Usable research software | Working headed/headless modes, continuation, interruption recovery, dashboard, docs, tests and installation |
| Durable science | Source-linked results, negative findings, immutable historical evidence, retrievable artifacts and tested preservation |
| Presentable release | Clean reviewed branches, exact-SHA CI, accurate README/report/audit and annotated release receipt |

Provide the starting and final SHAs, atomic commit summary, exact runnable commands, artifact locations, model/profile versions, one-hour and extended results, real/sim gap, human comparison, reward findings, profile adaptation, limitations and remaining external blockers.

Do not conclude after scaffolding, one simulator video, one successful UI cycle or one best episode. Complete the bounded implementation and research program; continue toward the goals while measurements identify tractable improvements. If a target remains unmet after the registered campaigns, deliver a stable honest research state with the specific unresolved cause and next experiment, not an invented success or an endless unbounded search.

## 27. Research grounding and source authority

Repository evidence governs what GradientClimb has implemented or demonstrated. Public papers inform candidate methods; they do not establish that those methods will improve this game.

Consult and record primary sources, including:

- Ross, Gordon and Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**, AISTATS/PMLR 2011: student-induced state distributions and DAgger.
- Pinto et al., **Asymmetric Actor Critic for Image-Based Robot Learning**, arXiv:1710.06542: image actor versus privileged-state critic.
- Tobin et al., **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**, arXiv:1703.06907: rendering variation as a transfer technique, not permission to ignore geometry/dynamics.
- Kumar et al., **RMA: Rapid Motor Adaptation for Legged Robots**, arXiv:2107.04034: learned adaptation from history; a robotics precedent, not a game performance guarantee.
- Agarwal et al., **Deep Reinforcement Learning at the Edge of the Statistical Precipice**, arXiv:2108.13264: evaluation uncertainty and limits of a few stochastic runs.
- Official Box2D documentation for wheel joints, collision, articulated constraints and simulation: engineering references, not evidence of the game’s proprietary constants.

Check current official implementation documentation and third-party licenses before adopting code. Clearly label manufacturer documentation, community observations, directly measured mechanics, inferred parameters and conjectures. Do not state guessed vehicle statistics as recovered physical truth.

**Final directive:** preserve what already works, correct what makes comparisons misleading, build a demonstrably relevant visual/physical training world, learn efficiently from competent behavior and reusable experience, prove performance in the actual game, then extend and qualify the same system across multiple profiles. Let validated real improvement—not simulator reward or feature count—govern the project.
