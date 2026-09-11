# Qualification protocol v0.1

Registered before serious training. Hill Climb Racing is the first testbed;
synthetic experiments and uncalibrated surrogate results do not qualify a game agent.

## Clock and starting conditions

Record actual monotonic elapsed training time and checkpoints at 300, 600, 1200,
1800, 2700 and 3600 seconds. Record actual checkpoint timestamps and overshoot;
never relabel a later checkpoint as an earlier one. Initialization, collection,
optimization, and synchronous evaluation consume the governed session clock.
Offline checkpoint evaluation has a separately reported duration. Also report
total development/search compute outside the qualification run.

Keep cold-start, simulator-pretrained, generalist-adaptation, and fine-tuning
classes separate. A simulator-only cold start must explicitly name its environment.
Register source SHA, clean/dirty state, seed, dependency lock, data/calibration
versions, parent weights and parent data. No parent checkpoint means no trained
weights; hand-designed simulator and architecture remain declared research priors.

## Evaluation and selection

Use fixed held-out seeds 10000–10019 for surrogate checkpoint comparisons and
disjoint seeds 20000–20019 for final evaluation. Compare random, clearly scripted
stabilization, PPO, and one cheap evolutionary baseline using equal actual time
budgets. Replicate promising comparisons across training seeds 0, 1, 2. Select
hyperparameters on validation results only, preserving final test seeds.

For the real game record vehicle, upgrade levels, map, game version, session date,
and at least 10 completed episodes per selected checkpoint when feasible. Report
distance median/mean/best/std/percentiles, survival, crash and fuel-end rates,
progress per second, and episode exclusions with reasons. Report uncertainty
over training seeds separately from uncertainty over evaluation episodes.

The provisional initial-vehicle competence gate is median real distance ≥500 m,
at least 3× random-policy median, and at least 8/10 episodes reaching 250 m on the
same observed vehicle/map/upgrade state. This operational threshold is a research
gate, not proof of human expert or leaderboard-level play. High-level performance
requires an independently established human/reference distribution; presently
that reference is unavailable. Do not label the project high-level solely from
passing the provisional gate.

## Transfer, generalization and adaptation

Evaluate identical policies and comparable conditions in simulation and the actual
game. Publish real/sim median-distance ratio only when both measurements exist;
otherwise report not measured, never zero. Evaluate new map, new vehicle, and both.
Adapt on a separate training split for 5/10/30/60 minutes, reevaluate old conditions
for forgetting, and retain prior compute/data lineage. A held-out surrogate profile
is a synthetic dynamics shift, not a verified new real vehicle.

## Calibration and perception

Collect pixels and timestamped ordinary actions before fitting hidden dynamics.
Use separate fitting and held-out action sequences. Report position/speed/pitch/
angular velocity errors and observation uncertainty. No real calibration claim
may be inferred from fitting synthetic trajectories. Validate each perception
component against labeled screenshots and keep unknown UI states fail-closed.

## Staged research

First validate infrastructure on two scalar synthetic conditions. Screen methods
with short pilots, record negative results, then replicate only promising choices.
Optimize measured useful learning per second. Run a full hour only with a fixed
versioned configuration; extended training is a separate run with explicit lineage.
Unattended game control requires demonstrated reliable state recognition and input
release. If those prerequisites fail, continue independent work and record the
external/engineering gate rather than substituting simulator scores.
