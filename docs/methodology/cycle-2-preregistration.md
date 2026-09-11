# Cycle 2 preregistration: versioned behavioral metrics, objectives and the reliability gate

Status: **REGISTERED 2026-09-11, before any Cycle 2 data collection.** Cycle 1 is frozen
at the annotated tag `cycle-1-paused`; nothing below rescored, relabels or reinterprets a
Cycle 1 run. The implementation is [objectives.py](../../src/gradientclimb/experiments/objectives.py)
with [tests](../../tests/test_objectives.py). The first empirical gate is the
[native reliability study](../../experiments/definitions/cycle-2-native-reliability.json).

## Version registry

Every serious Cycle 2 run stamps `cycle2_versions` into its configuration through
`stamp_versions()`. A run without the stamp is exploratory.

| Dimension | Identifier | Meaning |
| --- | --- | --- |
| Metric schema | `behavioral-metrics-2.0` | The per-episode outcome vector below. |
| Observation schema | `hcr-screen-relative-1` (schema hash recorded per run) | Unchanged 49-feature masked bridge. New measurements go into the outcome vector, never into actor features, unless a retraining decision explicitly bumps the schema. |
| Reward/fitness | `distance-only-2.0`, `distance-score-2.0`, `distance-score-time-2.0`, `recovery-conditioned-2.0`, optional `lexicographic-2.0` | Objective families A–E. Historical objectives are referenced as `historical:<name>` and are never renamed. |
| Benchmark | `native-reliability-2.0`, later `real-baselines-2.0`, `objective-screening-2.0`, `real-one-hour-2.0` | Each protocol is a frozen JSON definition under `experiments/definitions/`. |
| Simulator | `surrogate-0.1.0` or a later versioned dynamics | Simulator changes require a new version before any comparison. |
| Calibration | `uncalibrated` or a calibration record version | Calibrated claims require held-out real trajectory evidence. |
| UI profile | `hcr-wrapper-reset-v2` (content hash recorded) | Chrome-aware advertisement recognition; see the adapter documentation once implemented. |
| Score reader | manifest run IDs + hashes | Result and gameplay glyph banks; any new coin/fuel/cause reader gets its own manifest identity. |
| Recovery definition | `recovery-progress-2.0` | Defined below; `null` until a study depends on it. |

## Outcome vector (`behavioral-metrics-2.0`)

`EpisodeOutcome` retains, per episode and independently of any training scalar:

- Endpoint: `distance_m` with an explicit `distance_source` (stable terminal pair, verified
  paused frame or unknown), `terminal_cause` (driver down, out of fuel, truncated horizon,
  aborted, natural-but-unlabeled, unknown), `survived_horizon`.
- Time: gameplay seconds, session seconds including reset/advertisement overhead, reset
  seconds, advertisement seconds and encounters.
- Score: coin counter at the start and end boundaries and their difference `score_gain`;
  the source is declared. Lifetime currency is never a score; only the boundary difference
  of one episode is.
- Progress proxies: HUD maximum progress, HUD read fraction, time to reach frozen goal
  distances, longest stagnation gap, and an explicit flag that signed backward motion is
  **not** measured by the current bridge.
- Stunt/recovery: airtime events and seconds, rotation events, recovery events with their
  outcome (recovered, failed, fatal, censored), counts and a fatal-stunt flag.
- Measurement quality: frames, dispatched actions, stale captures, missing required
  feature frames, unknown-state frames, manual interventions and unintended actions.

Missingness rule: every field is `None` when not measured. Derived ratios
(`derived_metrics`) publish numerator, denominator and eligibility; a zero or unknown
denominator yields an unknown ratio, never an extreme value. Score gain requires both coin
boundary readings; a known distance requires a declared source. These rules are enforced by
validators and cannot be bypassed by a training script.

## Objective families

All arms share the actor, action space, observation masks, horizon, reset policy and score
eligibility rules of the run that uses them. Parameters are **not** defaulted in code: the
experiment definition must state every parameter, and those values are frozen from
construction data (baseline episodes) before any held-out evaluation. A required unknown
measurement makes the episode ineligible for that arm.

| Arm | Identifier | Scalar fitness |
| --- | --- | --- |
| A | `distance-only-2.0` | `distance_m` |
| B | `distance-score-2.0` | `distance_m + w_s · score_gain` |
| C | `distance-score-time-2.0` | Arm B `+ w_t · max(0, horizon − t_reach(goal))`, credited only when the HUD trace observably reached the frozen goal; unknown attainment is ineligible, an unreached goal earns zero time credit |
| D | `recovery-conditioned-2.0` | `distance_m + w_s · (base_gain + Σ_events credit)` where coin gain inside each maneuver window is credited in full only if the window's outcome is `recovered`, by `unrecovered_credit_fraction` if `failed`, and never if `fatal` or `censored` |
| E (optional) | `lexicographic-2.0` | `1000 · min(distance, minimum_distance) + distance + survival_bonus·[survived] + w_s · score_gain` |

Arm C deliberately avoids instantaneous speed: the same useful outcome reached earlier is
rewarded, while a reckless burst that never reaches the goal gets no time credit and a
burst that reaches it then dies still loses the distance term. Whether that is sufficient
is an empirical question answered by the reward-hacking battery, not assumed here.

## Recovery definition (`recovery-progress-2.0`) and event detector

An airtime event (`airtime-wheel-clearance-2.0`) is a window of at least two consecutive
frames in which both wheel clearance features are valid and exceed a frozen image-relative
threshold. Missing measurements end a window; they never extend it. A rotation event is a
sustained body angular rate above a frozen threshold; because the axis is modulo π it counts
attempts, never complete flips.

After an event ends at time `t_e`, the recovery window is `[t_e, t_e + W]`. The outcome is:

- `fatal` if a natural terminal failure occurs inside the window;
- `censored` if the window was not fully observed (episode truncated or aborted before
  `t_e + W`, or HUD progress unavailable at the boundaries);
- `recovered` if accepted HUD progress increased by at least the frozen increment inside
  the window;
- `failed` otherwise.

The policy never receives future observations; recovery outcomes are used only for
episode-end fitness assignment. Overlapping windows do not duplicate credit because each
event's coin gain is measured on its own window and subtracted from the base gain. These
thresholds (`W`, increment, clearance, minimum frames) are frozen from construction data
before Arm D is trained, and the detector must first be validated against independent
labels; until then Arm D is not eligible for qualification.

## Reward-hacking battery

`hacking_diagnostics()` reports the eight required indicators for one policy's evaluation
episodes: flips then death, airtime without progress, slow-but-safe, reckless speed, coin
chasing, backward loops, stagnation and local score farming. Each indicator states whether
it was measurable. Backward loops are explicitly **not measurable** with the current
image-relative bridge and are reported as such rather than as zero. Recovery-arm exploits
(repeated easy recovery states, overlapping credit, censoring failed windows, exploiting
missing observations, treating termination as recovery) are addressed by the definition
above and by reporting censored/fatal counts alongside recovered counts.

## Native reliability gate

The [registered protocol](../../experiments/definitions/cycle-2-native-reliability.json)
freezes: alternating always-gas/random scripted controls, 60 s horizon, 12 attempts per
session, at most three sessions, the terminal classification vocabulary, the measured
quantities, the primary endpoint (longest consecutive success run in one session) and the
success criteria (≥ 10 consecutive successes, zero unintended actions, zero manual
interventions, ≥ 10/12 scored). No attempt is excluded after the fact. Exploratory
advertisement-discovery sessions that precede the study are labeled exploratory in their
run configuration and are not study attempts.

## Sequencing

1. Exploratory offline and supervised native work to recognize advertisement chrome and
   grow the UI profile (version 2) with positive and negative frame audits.
2. The reliability study above.
3. Independent observation labels and validated coin/fuel/terminal-cause readers, each
   with a frozen manifest identity and held-out accuracy before any objective depends on it.
4. Real baselines (`real-baselines-2.0`): random, always-gas and a declared stabilizing
   heuristic, registered in their own definition before collection.
5. Learned real policy, objective screening at a preregistered bounded budget, selection
   by Pareto reasoning, then the governed real one-hour benchmark with the frozen objective.

Each later protocol is registered in `experiments/definitions/` before its data exist.
