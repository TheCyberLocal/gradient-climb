# Screen-body student distillation protocol

Status: implemented and semantically tested; **training and real-game evaluation
were never dispatched and are deferred at the Cycle 1 pause**. The queued
60-second seed500 pilot was cancelled before it started when the user paused
research. There is no learned student artifact or measured teacher agreement,
surrogate student score, or native student evaluation. The historical proposal
below is not authorization to resume. This is a transfer comparator with a trained teacher
prior. It is neither a cold start nor evidence of successful sim-to-real transfer.

The frozen teacher is primary PPO run `c0a9e142-1ad6-4d88-810d-bda4ca297f40`,
final checkpoint SHA-256
`87c469539110d3134c04ebcf11e95f9c84e595c90ba7e6b2b106917115e04ec0`.
Its recorded 3600.3219-second surrogate training and preceding development pilots
remain prior compute, in addition to any subsequently measured distillation time.
The execution script verifies the teacher's completed canonical run, full seal,
and registered checkpoint identity before creating a derived run.

## Restricted actor and training targets

The student receives exactly the eight features and scales in
`experiments/definitions/real-screen-cem.json`: sine/cosine of twice the body
axis angle, observed body angular rate, body-to-terrain clearance in body spans,
image terrain slope, and terrain offsets one/two/three body spans ahead. Each
frame contains normalized values followed by validity bits. Four oldest-first
frames give 64 inputs; missing history has zero values and false masks. Two
64-unit tanh layers and independent gas/brake heads can output all four states.
The current body orientation must be valid or the actor releases both pedals.

Neither simulator velocity, simulator angular velocity, fuel, contact truth,
reward, previous pedal bits, absolute world position, nor privileged teacher
observations enter the student network. The projection uses world geometry only
to construct idealized analogues of observable body features. Teacher state
observations only produce detached pedal-probability targets. The loss is mean
binary cross entropy against both teacher probabilities. Nominal default/train
trajectories follow the frozen deterministic teacher; the teacher is never
updated. There is no DAgger, real-pixel training, domain randomization or hidden
use of rewards in this initial experiment.

## Projection assumptions and unresolved gap

`surrogate-continuous-chassis-body-1` uses the original polygon drawn by the
surrogate renderer. Uniform filled-polygon moments give its centroid and
principal body axis. Rotating its vertices gives the axis-aligned bounding-box
span `max(width, height)`, matching the screen bridge's normalization. The
renderer is an orthographic camera with equal scale in both axes, x increasing
right and image y increasing down. Camera translation and scale cancel in the
selected relative features. Ground height is sampled at the projected body
centroid and zero/one/two/three rotated body spans. Terrain outside the renderer's
960×540 viewport is masked. Image terrain slope is the negative of world uphill
slope. Angular rate is a finite difference of consecutive modulo-pi body axes,
using the screen bridge's alias threshold and observation interval rules;
simulator `omega` is not used. Histories are cleared at episode resets or gaps.

These are **analytic continuous silhouette features, not rendered measurements**.
The projection does not reproduce the real vehicle's red-body silhouette,
upgrades, moving camera, perspective, pixel segmentation noise, terrain sampling
resolution, wheel/body occlusion, feature dropout, source-frame cadence, or
action/capture latency. It does not reproduce rasterization of the surrogate
polygon either. Most training geometry has ideal visual support. The real
screen schema's matching name/hash only establishes interface compatibility,
not matched observation distributions or physical calibration. Unknown
front/back orientation remains ambiguous under the modulo-pi representation.

## Preparation, execution and validation

`scripts/distill_screen_student.py` defaults to a dry run. Supply the exact
destination `ScreenFeatureBridge.schema_id`, its history and maximum interval,
teacher path and parent run. `--plan-output` writes the resolved proposal;
`--execute` is an explicit execution switch, to be used only after the active
sequential surrogate battery finishes or the coordinator dispatches otherwise.
The proposed default is 60 seconds, seed500, 64 nominal environments; the caller
must predeclare any changes. Validation seeds30000–30019 are separate from the
registered surrogate validation/test seeds. No validation result affects weights.

Training reports actual monotonic initialization/collection/optimization time;
imports, parent verification/loading, final checkpoint serialization and held-out
evaluation are separately excluded in the recorded configuration. The canonical
run duration includes these in-run costs. The checkpoint contains the restricted
student weights/configuration and teacher lineage, but no optimizer/RNG state;
exact distillation resume is not claimed.

After the training deadline, validation runs the same held-out initial episodes
twice: once under the teacher and once under the student. Both record joint and
per-pedal action agreement, observed action counts, distance distribution,
termination reasons and measured evaluation time. The student occupancy exposes
compounding imitation errors. Episode uncertainty does not substitute for
independent distillation seeds. Real validation remains a separate root dispatch
through the native adapter, with UI state, freshness and bounded input leases.

Semantic tests check geometry against independently constructed screen evidence,
terrain sign/normalization, viewport missingness, finite-difference angle aliasing,
episode-history clearing, exclusion of all other screen features, schema mismatch,
validity masks and checkpoint round trips. These tests make no accuracy claim
about either game or real vision.
