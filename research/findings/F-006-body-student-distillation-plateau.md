# F-006 — Tripling the distillation budget does not close the restricted student's gap to its teacher

Status: **NOT SUPPORTED (negative result, simulator only)**, Cycle 2, 2026-09-11.
Runs `3ceb8b27` (600 s, seed 500) and `807048e2` (1,800 s, seed 501), algorithm
`screen_body_distillation`, both distilled from the final checkpoint of PPO teacher `c0a9e142`
(256 environments, 3,600 s). Validation protocol `analytic-body-distillation-validation-1` on
held-out seeds 30000–30019, 60 s horizon, projection `surrogate-continuous-chassis-body-1`.

## Observation

The restricted student sees only the eight body features a screen can supply (body angle
pair, image angular rate, body-to-terrain span, terrain slope and three relative terrain
heights), their validity bits and a short history; the teacher sees the privileged simulator
observation. Training is teacher-occupancy soft-label imitation with no reinforcement-learning
update. Cycle 2 asked whether the first student's gap to the teacher (`3ceb8b27`) was a budget
limit, and answered it with a second run at three times the budget on a different seed.

## Measurement

Teacher on the same 20 validation episodes: mean 673.4 sim units, median 700.1, 2 of 20 crashes
(the teacher's own held-out protocol reports 681.6 / 716.1). Student under its own occupancy:

| Run | Budget | Mean (CI95, episode bootstrap) | Median | Crashes | Joint agreement (student / teacher occupancy) | Pedal-bit agreement | Final BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `3ceb8b27` | 600 s | 483.6 (384.7–581.7) | 561.3 | 11 / 20 | 0.849 / 0.865 | 0.905 / 0.920 | 0.335 |
| `807048e2` | 1,800 s | 483.6 (362.1–597.5) | 585.0 | 11 / 20 | 0.859 / 0.882 | 0.912 / 0.932 | 0.315 |

Distances are simulator units of the uncalibrated surrogate, never game metres.

## Uncertainty

One run per budget, on different seeds, so budget and seed are confounded and no seed-level
interval exists. The episode bootstrap intervals of the two students overlap almost entirely
and both exclude the teacher's mean. The identical crash count and the near-identical means
are one observation each, not a demonstrated plateau at that exact value.

## Interpretation

The extra budget improved what imitation can improve: the loss fell, and joint-action agreement
rose by one to two percentage points under both occupancies. It did not change the outcome that
matters: the student still crashes in 11 of 20 episodes and reaches 72 % of the teacher's mean
distance. The remaining disagreement therefore sits where the teacher's action depends on
information the body features do not carry (wheel state, velocities, terrain beyond the sampled
offsets), and where a small disagreement compounds under the student's own occupancy into a
crash the teacher avoids. That is an observation and occupancy limit, not a training-budget limit.

## Limitation

Everything here is an analytic projection inside the uncalibrated simulator; it says nothing
about the real game. Agreement under student occupancy is measured, but the student was
trained only on teacher-occupancy states, so the occupancy correction (student-state data with
teacher labels) is untested. Which feature carries the missing information was not measured.

## Conclusion

The hypothesis that budget closes the gap is not supported at this scale. The 1,800 s student
(`807048e2`) is the deployable restricted candidate for the sim-to-real pilot, carrying the
crash rate above as its expected weakness. The next simulator experiments on this line are an
occupancy-corrected distillation and a per-state disagreement analysis, each registered before
data, rather than further budget.
