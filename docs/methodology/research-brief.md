# GradientClimb — Complete Start-to-Finish Research and Implementation Task

## Repository

Target repository:

`https://github.com/TheCyberLocal/gradient-climb`

Project:

**GradientClimb**

Expected starting condition:

* repository exists;
* README is blank or effectively blank;
* little or no implementation exists.

This task owns development from that state through a complete, functioning, empirically evaluated data-science research project.

Do not stop after repository scaffolding, simulator construction, first successful training, or first real-game demonstration.

Continue through the full research lifecycle defined below.

---

# 1. Mission

Build GradientClimb into a rigorous data-science and machine-learning research platform for studying:

> **How rapidly can an intelligent adaptive control policy acquire, transfer, and generalize high-quality behavior under constrained wall-clock training time, experience, and computational resources?**

Hill Climb Racing is the initial experimental testbed.

The immediate engineering challenge is to produce an AI capable of playing the locally installed Hill Climb Racing at very high quality while learning unusually rapidly.

The central qualification target is:

> **Reach highly competent/high-level Hill Climb Racing performance within a governed one-hour training session.**

However, GradientClimb is not merely a Hill Climb Racing bot.

The larger research objectives include:

* rapid reinforcement learning;
* sample efficiency;
* wall-clock learning efficiency;
* high-throughput simulation;
* system identification;
* simulator calibration;
* sim-to-real transfer;
* visual control;
* learned vehicle dynamics;
* continual adaptation;
* generalization;
* curriculum learning;
* policy distillation;
* model-based learning where useful;
* experiment reproducibility;
* quantitative comparison of algorithms;
* rigorous scientific reporting.

The project should ultimately tell us not only **whether the AI learned**, but:

* how quickly;
* why;
* through which architecture;
* using how much computation;
* using how much experience;
* with what variance;
* with what transfer gap;
* with what generalization;
* and which techniques materially improved learning efficiency.

---

# 2. Governing Principle

At every architectural decision ask:

> **Which approach is most likely to produce the strongest real Hill Climb Racing agent after a fixed amount of wall-clock training?**

Prefer measured evidence over architectural preference.

Do not optimize for:

* novelty;
* code complexity;
* number of algorithms implemented;
* simulator graphical fidelity;
* fashionable ML techniques;
* synthetic reward alone.

Optimize for:

1. real-game competence;
2. time-to-competence;
3. sample efficiency;
4. generalization;
5. adaptation;
6. reproducibility;
7. scientific understanding.

---

# 3. Execution Authority

You have broad authority to:

* create the project architecture;
* install reasonable open-source dependencies;
* create isolated Python environments;
* benchmark alternative frameworks;
* research relevant literature;
* inspect open-source implementations;
* build experiments;
* revise architecture based on results;
* abandon poorly performing approaches;
* create and tune simulation;
* train models;
* collect local gameplay data;
* automate the locally installed game;
* create dashboards;
* generate charts;
* produce reports;
* create tests;
* create CI;
* commit and push completed work.

Do not require separate approval after every phase.

Continue automatically unless blocked by something genuinely external such as:

* missing authentication;
* unavailable repository access;
* a commercial license purchase;
* a destructive system change;
* something requiring explicit user credentials.

Where possible, work around nonessential blockers and continue.

---

# 4. Repository Governance

Inspect the repository before modification.

Establish:

* `main` as stable/reviewed state;
* `dev` as the active integration branch,

unless existing repository governance already establishes something else.

Perform implementation work on `dev`.

Maintain:

* clean commits;
* meaningful commit messages;
* clean worktrees at major checkpoints;
* synchronization with `origin/dev`.

Do not casually rewrite repository history.

At final completion report:

* starting SHA;
* ending SHA;
* branch;
* remote state;
* worktree state.

---

# 5. Project Identity

GradientClimb must present itself as a **machine-learning/data-science research project**, not a game automation utility.

Use a concise repository description along the lines of:

> GradientClimb is a data-science research platform for rapid-learning adaptive control, focused on training efficiency, generalization, and sim-to-real transfer.

The README must make clear that Hill Climb Racing is the **first experimental testbed** rather than the conceptual limit of GradientClimb.

---

# 6. Intellectual-Property and Safety Boundary

Hill Climb Racing is third-party software.

GradientClimb must:

* not redistribute game assets;
* not redistribute proprietary game code;
* not modify the game executable;
* not inject into the game process;
* not read private game memory;
* not intercept network protocols;
* not bypass paid content;
* not circumvent advertisements;
* not automate purchases.

The real-game agent must operate through:

* rendered screen observations;
* ordinary input controls.

Advertisements may be recognized and legitimately dismissed only after normal availability of the close control.

Unexpected menus must not trigger arbitrary clicks.

Avoid accidental:

* store interactions;
* currency spending;
* purchases;
* external advertisement clicks.

Implement safe UI-state handling.

---

# 7. Research Existing Work

Before committing to the final architecture, conduct a focused literature and open-source review.

Investigate relevant work involving:

* Hill Climb Racing reinforcement learning;
* Hill Climb Racing simulators;
* Gymnasium environments;
* Box2D implementations;
* PPO;
* recurrent PPO;
* NEAT/evolutionary approaches;
* vision-only HCR agents;
* high-throughput RL;
* PufferLib;
* Stable-Baselines3;
* asynchronous/vectorized RL;
* model-based reinforcement learning;
* world models;
* Dreamer-style approaches;
* privileged/asymmetric actor-critic;
* teacher/student policy distillation;
* curriculum learning;
* domain randomization;
* sim-to-real transfer;
* online system identification;
* meta-RL;
* rapid motor adaptation;
* contextual policy learning.

Record literature in:

`research/literature/`

Each useful reference should include:

* title;
* authors/project;
* source;
* URL/DOI;
* license where relevant;
* key idea;
* relevance to GradientClimb;
* whether anything is reused;
* limitations.

Do not blindly copy repository implementations.

Observe licensing.

Prefer clean implementations informed by published ideas where code licenses would create undesirable constraints.

---

# 8. Licensing Review

Before selecting GradientClimb's repository license:

1. inspect all major dependencies;
2. inspect any reused implementation;
3. inspect licenses of candidate simulator projects;
4. identify compatibility constraints.

Create an ADR explaining the license decision.

Do not use third-party code with incompatible obligations without documenting and intentionally accepting those implications.

---

# 9. Workstation Assessment

Inspect the actual machine before selecting the training architecture.

Record:

* CPU;
* physical/logical cores;
* RAM;
* GPU;
* VRAM;
* NVIDIA driver;
* CUDA capability;
* storage;
* relevant Python environments;
* PyTorch/CUDA visibility;
* Node;
* FFmpeg;
* available development tools.

The workstation already contains a broad validated software arsenal.

Do not reopen unrelated installation/license issues unless GradientClimb actually needs one of those tools.

Do not destabilize existing:

* graphics drivers;
* CUDA configuration;
* Python installations;
* Blender;
* Unreal Engine;
* scientific environments.

Create isolated environments where practical.

---

# 10. Initial Repository Architecture

Build a coherent research architecture similar to:

```text
gradient-climb/
│
├── README.md
├── pyproject.toml
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .gitignore
├── .editorconfig
│
├── docs/
│   ├── architecture/
│   ├── methodology/
│   ├── decisions/
│   └── operations/
│
├── research/
│   ├── hypotheses/
│   ├── literature/
│   ├── findings/
│   ├── experiments/
│   └── reports/
│
├── src/
│   └── gradientclimb/
│       ├── agents/
│       ├── algorithms/
│       ├── environments/
│       ├── simulation/
│       ├── perception/
│       ├── capture/
│       ├── control/
│       ├── calibration/
│       ├── experiments/
│       ├── evaluation/
│       ├── benchmarks/
│       ├── telemetry/
│       ├── artifacts/
│       └── visualization/
│
├── experiments/
│   ├── definitions/
│   └── suites/
│
├── benchmarks/
│   ├── one_hour/
│   ├── throughput/
│   ├── adaptation/
│   ├── generalization/
│   └── sim_to_real/
│
├── schemas/
│
├── notebooks/
│
├── dashboard/
│
├── scripts/
│
├── tests/
│
└── artifacts/
```

Adjust where evidence warrants.

Avoid empty-directory theater.

---

# 11. Scientific Data Architecture

Treat every serious training/evaluation run as an immutable scientific record.

Prefer:

* Parquet for columnar telemetry;
* DuckDB for local analytical querying;

unless measured evidence justifies another foundation.

Define canonical schemas for:

* experiments;
* runs;
* metrics;
* system telemetry;
* evaluations;
* trajectories;
* artifacts;
* model lineage;
* benchmarks;
* simulator calibration.

Every governed run should record at minimum:

```text
run_id
experiment_id

start_time
end_time
duration

git_sha
dirty_worktree
project_version

machine_fingerprint
cpu
gpu
ram
driver
cuda

python_version
framework_versions

algorithm
algorithm_version

policy_architecture
configuration

environment
environment_version

simulator_version
calibration_version

vehicle_profile
map_profile

seed

parent_checkpoint
parent_run

training_steps
environment_steps
episodes
optimizer_updates

wall_clock_seconds
environment_steps_per_second

evaluation_results

artifact_manifest
checkpoint_hash
```

Use validated schemas.

---

# 12. Artifact Governance

Do not commit large generated data directly to Git.

Large artifacts include:

* checkpoints;
* videos;
* raw frame captures;
* telemetry archives;
* large datasets.

Maintain artifact manifests containing:

* artifact ID;
* type;
* producing run;
* path/location;
* size;
* SHA-256;
* creation time;
* semantic metadata.

The repository must allow a graph or reported result to be traced back to:

```text
source
→ configuration
→ run
→ data
→ model
→ evaluation
→ result
```

---

# 13. Hypothesis-Driven Research

Create lightweight hypothesis governance.

Example:

`research/hypotheses/HYP-001.md`

Each hypothesis should define:

* proposition;
* motivation;
* independent variables;
* dependent variables;
* controls;
* success criterion;
* failure/falsification criterion;
* planned experiment;
* eventual result.

Statuses:

```text
PROPOSED
TESTING
SUPPORTED
NOT_SUPPORTED
INCONCLUSIVE
SUPERSEDED
```

Findings must be separately documented.

Example:

`research/findings/FIND-001.md`

Each finding should distinguish:

* observation;
* statistical evidence;
* interpretation;
* limitation;
* conclusion.

Do not turn every exploratory test into a formal publication exercise, but serious architectural claims must ultimately have evidence.

---

# 14. Experiment Harness

Before complex game work, implement a complete experiment harness using a trivial synthetic environment.

Prove:

1. run creation;
2. configuration capture;
3. source SHA capture;
4. machine fingerprinting;
5. metric streaming;
6. telemetry persistence;
7. artifact registration;
8. run finalization;
9. run querying;
10. run comparison;
11. dashboard visualization.

Create at least two distinguishable synthetic experiments.

Validate full provenance.

---

# 15. CLI

Create a coherent CLI such as:

```text
gradientclimb doctor

gradientclimb experiment run ...
gradientclimb experiment list
gradientclimb experiment show <run>

gradientclimb train ...
gradientclimb evaluate ...

gradientclimb capture ...
gradientclimb calibrate ...

gradientclimb benchmark ...

gradientclimb dashboard
```

Do not create speculative commands with no implementation.

---

# 16. Real Game Discovery

Inspect the locally installed Hill Climb Racing instance.

Determine:

* process/window identity;
* screen geometry;
* rendering characteristics;
* menu states;
* gameplay state;
* pause state;
* death/end state;
* restart flow;
* vehicle selection;
* map selection;
* advertisements;
* result screens;
* visible HUD;
* fuel meter;
* distance meter;
* special meters;
* control bindings.

Document observations.

Do not assume fixed screen coordinates where robust visual recognition is possible.

---

# 17. Screen Capture

Build the lowest-latency reliable screen-capture pipeline appropriate for the machine.

Benchmark candidate methods when appropriate.

Measure:

* capture FPS;
* capture latency;
* CPU usage;
* GPU impact;
* dropped frames;
* resolution.

Support:

* full-resolution capture for debugging;
* reduced/cropped observation capture for training;
* optional frame recording.

Store timestamps accurately enough to reconstruct action-response trajectories.

---

# 18. Independent Pedal Control

The agent has two principal action channels:

```text
gas
brake/reverse
```

These MUST be independent.

The action space must include:

```text
00
10
01
11
```

Do not model them as mutually exclusive.

The system must preserve action order and duration.

The policy must be capable of discovering distinctions such as:

```text
gas
→ gas + brake
```

versus:

```text
brake
→ brake + gas
```

as well as:

* taps;
* long holds;
* alternating inputs;
* simultaneous holds;
* delayed combinations.

Do not encode explicit vehicle tricks as conditional scripts.

---

# 19. Real-Game Observation Dataset

Before building the final simulator, create controlled real-game trajectories.

Collect:

* frames;
* actions;
* timestamps;
* progress/distance;
* visible meters;
* inferred pose;
* game state;
* termination condition.

Perform controlled action experiments when practical:

* no input;
* gas;
* brake;
* both;
* timed pulses;
* transitions between controls;
* airborne control;
* hill climbing;
* braking/downhill behavior.

This becomes evidence for:

* perception;
* system identification;
* simulator calibration;
* imitation;
* debugging.

---

# 20. Perception System

The deployed agent must operate using information derivable from rendered pixels and action history.

Investigate and implement efficient estimation of useful state, potentially including:

* vehicle location;
* vehicle orientation;
* angular velocity;
* translation velocity;
* local terrain;
* upcoming terrain profile;
* camera movement;
* relative motion;
* fuel;
* distance;
* progress;
* special meter state;
* UI state.

Avoid forcing the policy to rediscover trivially extractable HUD information from raw pixels if reliable structured extraction is substantially more sample-efficient.

However, do not introduce brittle hand-coded gameplay rules.

---

# 21. Temporal State

A single screenshot is generally not Markov-complete.

Provide temporal context using one or more of:

* frame stacking;
* recurrent policies;
* GRU/LSTM;
* temporal encoders;
* compact transformer;
* learned latent state.

Benchmark complexity against training speed.

Use the lightest architecture that delivers needed competence.

---

# 22. Perception Validation

Build labeled validation data.

Measure separately:

* vehicle localization;
* pose estimation;
* terrain estimation;
* distance/progress reading;
* fuel reading;
* meter extraction;
* game-over detection;
* menu-state detection;
* advertisement-state detection.

A training failure must be diagnosable as:

```text
perception
control
policy
simulation
calibration
UI orchestration
```

rather than simply “AI failed.”

---

# 23. Real-Game Automation State Machine

Build a robust UI-state machine for unattended training.

Recognize:

```text
main menu
vehicle/map screen
starting
playing
paused
game over
result
advertisement
return/restart
unexpected
```

When an episode terminates:

1. recognize the result screen;
2. collect final metrics;
3. dismiss permitted dialogs;
4. handle advertisements legitimately;
5. return to gameplay;
6. begin the next episode.

Advertisements:

* may be waited out normally;
* may be dismissed only when the legitimate close control becomes available;
* must not be bypassed;
* should not have arbitrary content clicked.

Unknown states should enter safe recovery or halt rather than clicking blindly.

---

# 24. Simulator Strategy

Determine empirically whether a simulator materially accelerates the objective.

Unless evidence strongly contradicts it, implement a lightweight high-throughput 2D simulator.

Potential physics foundation:

* Box2D;
* another performant deterministic 2D physics engine;
* custom specialized physics only if justified.

Do NOT use Unreal simply because it is available.

Graphical fidelity is not the objective.

The simulator should represent enough of:

* terrain;
* gravity;
* chassis;
* wheels;
* suspension;
* torque;
* braking;
* reverse;
* airborne rotation;
* traction;
* contact;
* crash conditions;
* fuel/resource behavior;
* vehicle-specific dynamics;
* unusual control effects.

---

# 25. System Identification

Use real trajectories to estimate simulator parameters.

Possible parameters:

* effective gravity;
* vehicle mass;
* center of mass;
* wheel radius;
* wheelbase;
* motor torque;
* friction;
* suspension stiffness;
* damping;
* rotational response;
* braking;
* aerodynamic behavior where relevant.

Use optimization or parameter inference rather than hand-tuning everything manually.

Compare simulator and real trajectories under identical action sequences.

---

# 26. Domain Randomization

Exact hidden game physics may be impossible to recover.

Represent uncertainty explicitly.

Randomize plausible distributions of:

* friction;
* mass;
* torque;
* suspension;
* terrain;
* response delay;
* perception noise;
* control latency.

Train policies robust to the plausible real-game distribution.

Measure whether domain randomization improves transfer.

---

# 27. Simulator Fidelity Metrics

Create simulator-validation metrics such as:

* displacement error;
* speed error;
* pitch error;
* angular-velocity error;
* jump-trajectory error;
* landing-position error;
* contact-state discrepancy;
* progress discrepancy.

Maintain simulator calibration versions.

Never change simulator mechanics without recording the version.

---

# 28. Parallel Simulation

Build headless vectorized simulation.

Benchmark environment counts such as:

```text
32
64
128
256
512
1024
```

or other appropriate counts.

Measure:

* steps/sec;
* CPU utilization;
* GPU utilization;
* memory;
* policy latency;
* training throughput.

Select throughput empirically.

Do not assume “100 parallel cars” is optimal.

Use the number producing the greatest **useful learning per wall-clock minute**.

---

# 29. Headed Simulator

Provide a visualization mode.

It should support visualizing:

* selected environment;
* best current policy;
* evaluation episode;
* representative random environment.

Do not render every training environment.

Headless mode must remain substantially faster.

The headed view should show useful telemetry such as:

* distance;
* reward;
* controls;
* velocity;
* orientation;
* elapsed training;
* current model;
* simulator version.

---

# 30. Baselines

Create meaningful baselines.

At minimum compare against:

### Random policy

Establishes the floor.

### Simple heuristic baseline

May use basic stabilizing logic solely as a benchmark.

Clearly label it as scripted and not a learned policy.

### Learned baseline

Start with a mature algorithm appropriate for the action space.

PPO/recurrent PPO is a reasonable initial benchmark but is not automatically the final architecture.

---

# 31. Learning Algorithm Research

Experiment with approaches that plausibly improve rapid learning.

Candidate families include:

* PPO;
* recurrent PPO;
* APPO;
* SAC-like methods if appropriately formulated;
* evolutionary approaches;
* model-based RL;
* latent world models;
* actor-critic with privileged critic;
* teacher/student distillation;
* imitation bootstrap;
* population-based training.

Do not implement every algorithm for completeness.

Use staged screening:

1. cheap pilot;
2. eliminate obvious losers;
3. deeper comparison;
4. replicate promising results across seeds.

---

# 32. Hyperparameter Optimization

Because time-to-quality matters, optimize hyperparameters for **wall-clock learning efficiency**, not eventual asymptotic reward.

Use a practical search approach such as:

* Optuna;
* successive halving;
* ASHA;
* population-based optimization;
* another justified method.

Avoid exhaustive brute-force grids.

Record every search trial.

---

# 33. Curriculum Learning

Create a performance-driven curriculum if beneficial.

Potential progression:

```text
flat terrain
→ small hills
→ steep hills
→ jumps
→ airborne control
→ irregular terrain
→ longer survival
→ fuel constraints
→ difficult procedural terrain
```

Curriculum advancement should depend on competence rather than arbitrary elapsed epochs.

Measure whether curriculum actually shortens time-to-competence.

---

# 34. Privileged Training

Strongly investigate asymmetric training.

In simulation:

### Actor

Receives only information ultimately obtainable from real visual observations.

### Critic / Teacher

May access privileged simulation state:

* exact velocities;
* terrain geometry;
* contacts;
* physics parameters;
* perfect pose.

Test whether this improves rapid learning.

The deployed actor must not depend on privileged information.

---

# 35. Teacher / Student Distillation

If privileged-state training produces a substantially better/faster expert:

1. train privileged teacher;
2. generate trajectories;
3. distill into visually deployable student;
4. optionally fine-tune through RL;
5. evaluate real-game transfer.

Quantify whether distillation improves the one-hour objective.

---

# 36. Vehicle Profiles

Support explicit vehicle profiles, but keep them descriptive rather than strategic.

Profiles may include:

* visual identity;
* dimensions;
* observation normalization;
* expected control channels;
* simulator priors;
* learned embedding;
* checkpoint associations.

Profiles must NOT contain rules like:

```text
press both pedals when X
boost at Y
brake before Z
```

The policy must learn behavior.

---

# 37. Learned Vehicle Dynamics

Develop a mechanism allowing one general agent to adapt to differing vehicle behavior.

Strongly investigate a context/dynamics encoder:

$$
z_t=f(o_{t-k:t},a_{t-k:t-1})
$$

where \(z_t\) represents inferred vehicle/environment dynamics.

Condition the policy on this latent context.

The system should eventually infer properties such as:

* acceleration;
* braking;
* rotational authority;
* unusual simultaneous-pedal effects;
* flying behavior;
* boosters;
* transformations;
* control sequencing effects.

without being explicitly told the corresponding strategy.

---

# 38. Generalist Policy

Favor a generalist architecture unless experiments prove separate specialists materially superior.

The long-term agent should be able to:

* encounter a new vehicle;
* interact with it;
* infer its dynamics;
* improve quickly.

Likewise for unfamiliar maps/terrain.

---

# 39. Procedural Terrain

Use procedural simulation to prevent memorization.

Generate:

* slopes;
* valleys;
* crests;
* jumps;
* irregular terrain;
* parameterized difficulty.

Maintain held-out terrain seeds for evaluation.

Never evaluate generalization only on terrain seen during training.

---

# 40. Reward Research

Develop a reward reflecting good gameplay.

Possible components:

* progress;
* distance;
* rate of progress;
* survival;
* stable landing;
* crash penalties;
* fuel/resource efficiency;
* controlled orientation.

Avoid reward hacking.

A policy that achieves high simulator reward but performs poorly in the real game is a failure.

---

# 41. Replay and Trajectory System

Define a trajectory format containing:

```text
timestamp
observation
frame reference
derived state
action
reward
distance
termination
vehicle
map
policy
environment version
```

Simulator trajectories may additionally store privileged state.

Support replay for:

* debugging;
* analysis;
* imitation;
* simulator calibration;
* regression testing.

---

# 42. One-Hour Qualification Benchmark

Create a formal benchmark measuring policy quality at:

```text
5 minutes
10 minutes
20 minutes
30 minutes
45 minutes
60 minutes
```

The clock must represent actual wall-clock training time.

Track at minimum:

* distance/progress;
* median performance;
* best performance;
* survival;
* crash rate;
* progress per second;
* consistency;
* reward;
* real-game performance.

---

# 43. Separate Training Claims

Never mix fundamentally different starting conditions.

Maintain separate benchmark classes.

## Cold Start

No HCR-trained weights.

## Simulator-Pretrained

Policy has simulator experience before the measured real-game session.

## Generalist Adaptation

Policy has broad prior training but has not trained on the target vehicle/map.

## Fine-Tuning

Policy explicitly starts from a relevant HCR checkpoint.

All published results must identify their category.

Do not claim “learned in one hour” if significant undeclared pretraining occurred.

---

# 44. Real-Game Evaluation

Simulation results do not qualify the system.

Evaluate promising checkpoints against the actual game.

Use enough episodes to distinguish:

* lucky runs;
* stable competence.

Record:

* median;
* mean;
* best;
* spread;
* failure modes.

---

# 45. Sim-to-Real Gap

For identical policies, compare:

```text
simulation
vs
actual game
```

Create metrics for:

* performance ratio;
* behavioral divergence;
* trajectory differences;
* crash-pattern differences.

If simulator scores improve while real scores do not, prioritize calibration rather than blindly training longer.

---

# 46. Generalization Benchmarks

After the initial target becomes competent, evaluate:

### New map

Use a map excluded from training.

### New vehicle

Use a vehicle excluded from training.

### New vehicle + map

Evaluate combined distribution shift.

Measure:

* zero-shot competence;
* adaptation speed;
* performance after 5/10/30/60 minutes;
* forgetting of prior competence.

---

# 47. Continual Learning

Investigate whether additional training causes catastrophic forgetting.

After adapting to a new vehicle/map, re-evaluate prior conditions.

If necessary, explore:

* replay;
* mixed-task training;
* regularization;
* multi-task batches;
* retained specialists;
* knowledge distillation.

Again, implement only what evidence warrants.

---

# 48. Statistical Methodology

Serious algorithm comparisons should use multiple random seeds.

Report:

* mean;
* median;
* standard deviation;
* percentiles;
* confidence intervals where meaningful;
* effect size when appropriate.

Distinguish:

* exploratory runs;
* governed benchmark runs.

Do not pretend a single stochastic result proves superiority.

---

# 49. Primary Scientific Quantity

Treat policy quality as approximately:

$$
Q = f(t,C,D,P)
$$

where:

* \(t\) = wall-clock training time;
* \(C\) = compute;
* \(D\) = experience/data;
* \(P\) = prior learned knowledge.

The research system must enable comparisons across all four.

Do not reduce everything prematurely to one scalar leaderboard.

---

# 50. Learning-Efficiency Frontier

Generate Pareto-style analyses showing tradeoffs among:

* quality;
* training duration;
* samples;
* compute.

Examples of scientific questions:

* Does doubling simulator throughput halve time-to-competence?
* Is a model-based agent more sample-efficient but slower in wall-clock time?
* Does a larger recurrent model improve generalization enough to justify lower FPS?
* Does privileged teaching reduce real-game training requirements?
* Does additional simulator fidelity outperform additional simulation throughput?
* How much adaptation is required for a new vehicle?

Document answers with evidence.

---

# 51. System Telemetry

Record training-system resource metrics.

Where available capture:

* CPU utilization;
* per-core utilization;
* GPU utilization;
* VRAM;
* RAM;
* simulator FPS;
* environment steps/sec;
* inference latency;
* optimizer latency;
* data-transfer latency.

This data is part of the research.

---

# 52. Dashboard

Build a polished interactive GradientClimb research dashboard.

It must consume the canonical experiment database.

Do not maintain a second dashboard-only source of truth.

Provide interactive views for at least:

### Overview

* best current models;
* recent experiments;
* benchmark status;
* one-hour results.

### Runs

Filter/sort by:

* algorithm;
* model;
* environment;
* vehicle;
* map;
* simulator;
* date;
* duration;
* status.

### Learning Curves

Plot against:

* wall-clock time;
* environment steps;
* episodes.

### Algorithm Comparison

Compare:

* median learning curve;
* seed distribution;
* final quality;
* time-to-threshold.

### Runtime Efficiency

Display:

* environment steps/sec;
* GPU/CPU usage;
* inference latency;
* training latency.

### One-Hour Benchmark

Visualize:

```text
5
10
20
30
45
60 minutes
```

### Generalization

Matrix of:

```text
trained vehicle/map
×
evaluation vehicle/map
```

### Adaptation

Quality versus exposure time for unfamiliar dynamics.

### Sim-to-Real

Compare real and simulated performance.

### Simulator Fidelity

Visualize trajectory/calibration errors.

### Controls

Show temporal pedal traces:

* gas;
* brake;
* simultaneous states.

### Model Lineage

Show parent/child relationships among:

* runs;
* checkpoints;
* distilled models.

### Experiment Detail

Expose:

* full configuration;
* source SHA;
* dependencies;
* machine;
* seed;
* artifacts;
* resulting metrics.

---

# 53. Scientific Visualization

Support visualizations such as:

* learning curves;
* confidence bands;
* violin/box plots;
* scatter plots;
* parameter sensitivity;
* Pareto frontiers;
* correlation matrices;
* adaptation curves;
* generalization heatmaps;
* simulator-real trajectory overlays;
* training throughput plots;
* resource-utilization plots.

Do not create graphs solely for decoration.

Each graph should answer a research question.

---

# 54. Headed Training Visualization

Provide a visualization mode allowing the user to watch training.

Possible information:

```text
current agent
distance
velocity
orientation
gas
brake
reward
episode
training elapsed
model version
```

For parallel simulation, render only selected environments.

Allow training to remain high throughput.

---

# 55. Videos and Demonstrations

Record representative videos for major benchmark checkpoints where practical:

* early/random behavior;
* 5-minute policy;
* 30-minute policy;
* 60-minute policy;
* best extended policy;
* unseen vehicle adaptation;
* simulator versus real game.

Do not store huge video files directly in Git.

Reference them through artifact manifests.

---

# 56. Notebooks

Use notebooks for exploratory science.

Examples:

* calibration analysis;
* algorithm comparison;
* hyperparameter analysis;
* simulator fidelity;
* generalization;
* adaptation.

Move reusable logic into package modules.

A notebook must not become the only implementation of important methodology.

---

# 57. CI and Quality

Configure lightweight CI.

Validate:

* package installation;
* formatting;
* linting;
* type checks where reasonable;
* unit tests;
* schema tests;
* deterministic simulator tests where applicable;
* synthetic experiment;
* dashboard build.

Do not run hours of GPU training in ordinary CI.

---

# 58. Tests

Implement tests for critical behaviors including:

* independent controls;
* simultaneous pedals;
* action durations;
* run schema validation;
* artifact hashing;
* experiment provenance;
* simulator determinism where expected;
* environment reset;
* termination;
* metric persistence;
* UI-state classifier;
* dashboard queries;
* benchmark calculations.

Add regression tests when substantive bugs are discovered.

---

# 59. Research Decision Records

Use ADRs for major architectural decisions.

Likely areas:

* project identity;
* data architecture;
* experiment provenance;
* simulator physics;
* RL framework;
* perception representation;
* vehicle-context architecture;
* dashboard stack;
* artifact strategy;
* licensing;
* game/IP boundary.

An ADR should explain alternatives and evidence, not merely announce the winner.

---

# 60. Research Findings

Throughout development create finding documents when evidence resolves meaningful questions.

Examples:

```text
FIND-001 Parallelism scaling
FIND-002 Best initial RL baseline
FIND-003 Curriculum effect
FIND-004 Privileged critic effect
FIND-005 Simulator fidelity threshold
FIND-006 Domain randomization effect
FIND-007 Vehicle context adaptation
FIND-008 One-hour qualification
```

Number/names should reflect actual findings.

---

# 61. Optimize Iteratively

Do not stop after the first working policy.

Use profiling and experiment evidence to locate the dominant bottleneck.

Possible bottlenecks:

* simulator;
* environment serialization;
* policy inference;
* GPU utilization;
* perception;
* observation size;
* reward;
* optimization instability;
* simulator mismatch;
* exploration;
* curriculum;
* model capacity.

Address the largest evidence-backed bottleneck first.

---

# 62. Failure Is Data

Record unsuccessful experiments when they teach something important.

Do not erase failed research.

For meaningful negative results capture:

* hypothesis;
* experiment;
* failure;
* evidence;
* likely explanation;
* resulting decision.

GradientClimb should accumulate knowledge, not merely final successes.

---

# 63. One-Hour Objective

Aggressively optimize toward high-quality performance within one hour.

However:

**Do not falsify this result.**

If the project fails to reach the desired performance after 60 minutes:

1. report the actual result;
2. identify dominant limitation;
3. quantify the gap;
4. continue research;
5. determine which changes improve the frontier.

The one-hour benchmark remains useful even if initially failed.

---

# 64. Extended Training

After completing the governed one-hour benchmark, continue selected strong policies for longer horizons.

Evaluate whether:

* competence continues rising;
* convergence occurs;
* overfitting develops;
* generalization changes.

Do not replace the one-hour result with extended-training performance.

Report both.

---

# 65. Final Model Selection

Define a transparent model-selection rule.

Do not select simply by maximum simulator reward.

Use real-game evaluation and research objectives.

Likely criteria include:

* real-game performance;
* stability;
* learning efficiency;
* generalization;
* adaptation;
* computational efficiency.

Record why the selected model is considered best.

---

# 66. Final Reproducibility Run

Before finalizing the repository, rerun the most important benchmark configuration from a clean/reproducible environment where practical.

Validate that the reported result can be reproduced within expected stochastic variance.

Capture:

* exact SHA;
* dependencies;
* model configuration;
* seeds;
* simulator version;
* benchmark version.

---

# 67. Final Research Report

Create a comprehensive durable final report, for example:

`research/reports/gradientclimb-final-report.md`

Also generate a polished web/HTML version if practical.

The report should read like a serious applied ML research report.

Include:

## Abstract

What GradientClimb investigated and the major result.

## Research Question

Rapid acquisition/adaptation of control competence.

## Experimental Environment

Hill Climb Racing and simulator.

## Hardware

Actual machine used.

## Data

How trajectories/telemetry were gathered.

## Simulator

Architecture, calibration, fidelity, limitations.

## Perception

Inputs, extraction, accuracy.

## Agent Architecture

Final policy architecture.

## Algorithms Tested

Approaches compared.

## Training Infrastructure

Parallelism and throughput.

## Experimental Methodology

Seeds, benchmarks, controls.

## Results

All major quantitative outcomes.

## One-Hour Benchmark

Explicit 5/10/20/30/45/60-minute results.

## Learning Efficiency

Quality over wall-clock time.

## Sim-to-Real

Transfer results and gap.

## Vehicle Generalization

Results.

## Map Generalization

Results.

## Adaptation

Results.

## Runtime Efficiency

CPU/GPU/throughput findings.

## Ablations

Which techniques actually contributed.

## Negative Results

Important approaches that failed or underperformed.

## Statistical Analysis

Variability and confidence.

## Limitations

Known weaknesses.

## Conclusions

What the evidence supports.

## Future Research

What would most improve GradientClimb next.

All claims should link back to experiment/run identifiers.

---

# 68. README Finalization

Replace the blank README with a polished project introduction.

Include:

* project purpose;
* research identity;
* major findings;
* current performance;
* architecture overview;
* screenshots/plots where appropriate;
* installation;
* experiment workflow;
* simulator usage;
* real-game usage;
* training;
* evaluation;
* dashboard;
* research report;
* data/artifact architecture;
* licensing/IP statement.

Do not advertise unsupported functionality.

---

# 69. Documentation

Create sufficient documentation for another technically capable researcher to:

1. install GradientClimb;
2. validate the environment;
3. run the simulator;
4. run a training experiment;
5. evaluate a model;
6. inspect run history;
7. launch the dashboard;
8. understand artifacts;
9. reproduce key benchmarks.

---

# 70. Final Repository State

The repository should end as a complete research artifact containing:

### Implementation

* game integration;
* simulator;
* perception;
* agent;
* training system;
* evaluation;
* experiments;
* benchmarks.

### Science

* literature;
* hypotheses;
* findings;
* experiment records;
* methodology;
* final report.

### Data Infrastructure

* canonical schemas;
* DuckDB/Parquet workflow;
* provenance;
* artifact manifests.

### Visualization

* interactive dashboard;
* generated figures;
* model/run comparisons.

### Governance

* ADRs;
* CI;
* testing;
* reproducibility;
* licensing.

---

# 71. Release

When the work reaches a coherent completed research state:

* ensure `dev` is clean;
* run full repository validation;
* confirm final report generation;
* confirm dashboard operation;
* confirm benchmark records;
* confirm reproducibility metadata;
* push `dev`.

Do not merge to `main` unless repository governance/user authorization already permits it.

Prepare the repository for a sensible research release.

A version such as:

```text
0.1.0
```

may be appropriate for the first complete experimental system.

Do not claim `1.0.0` maturity merely because this task is complete.

---

# 72. Final Completion Report

At the end of the task, produce a comprehensive implementation report.

Include:

## Repository State

* starting SHA;
* ending SHA;
* branch;
* remote synchronization;
* worktree state.

## Architecture

* final system architecture;
* key decisions;
* important deviations from the original plan.

## Research Conducted

* literature reviewed;
* hypotheses tested;
* major experiments.

## Training Infrastructure

* simulator throughput;
* parallel environments;
* hardware utilization.

## Model

* final architecture;
* algorithm;
* parameter count;
* observation space;
* action space.

## Real-Game Integration

* capture;
* perception;
* control;
* unattended loop.

## One-Hour Benchmark

Provide the actual:

```text
5-minute
10-minute
20-minute
30-minute
45-minute
60-minute
```

results.

## Extended Results

Longer training results.

## Generalization

* new vehicle;
* new map;
* combined change.

## Adaptation

How rapidly competence is regained.

## Sim-to-Real

Quantified transfer gap.

## Runtime Efficiency

* steps/sec;
* inference latency;
* training throughput;
* GPU/CPU utilization.

## Algorithm Research

* algorithms tested;
* winners;
* losers;
* why.

## Ablations

Which features materially contributed.

## Negative Findings

What did not work.

## Dashboard

Available analytical views.

## Scientific Conclusions

What GradientClimb has actually established.

## Remaining Limitations

Be explicit.

## Next Research Frontier

Identify the highest-value next scientific question.

---

# 73. Completion Criteria

This task is not complete when:

* the repository is scaffolded;
* the simulator runs;
* PPO learns something;
* one successful gameplay video exists;
* a dashboard exists.

It is complete when GradientClimb constitutes a coherent functioning data-science research project that:

1. controls the actual game;
2. collects real trajectories;
3. has a calibrated training simulator or evidence explaining why another strategy prevailed;
4. trains agents at high throughput;
5. independently controls all relevant pedal states;
6. learns rather than scripting vehicle behavior;
7. supports headed and headless training;
8. supports unattended real-game episodes;
9. records every serious experiment reproducibly;
10. measures wall-clock learning efficiency;
11. executes the governed one-hour benchmark;
12. evaluates real-game transfer;
13. evaluates generalization;
14. evaluates adaptation;
15. compares meaningful competing methods;
16. records scientific findings;
17. exposes results through an interactive dashboard;
18. contains a comprehensive final research report;
19. documents failures and limitations honestly;
20. leaves the repository clean, tested, reproducible, and synchronized.

---

# Final Research Standard

Do not approach GradientClimb as:

> “Build an AI that can play Hill Climb Racing.”

Approach it as:

> **Use Hill Climb Racing as an experimentally rich control problem to determine how quickly an intelligent system can learn useful dynamics, acquire skilled behavior, transfer that behavior from simulation to reality, adapt to unfamiliar systems, and improve the efficiency frontier of machine learning itself.**

The game is the laboratory.

The data is the evidence.

The learning curve is the primary object of study.

The final agent is both a result and an experimental instrument.

Continue from the blank repository through implementation, experimentation, analysis, visualization, documentation, and final scientific reporting without requiring another development task between phases.
