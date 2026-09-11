# Cycle 2 reward and recovery research proposal

Status: **DEFERRED; design only.** The [owner's pause directive](../methodology/cycle-1-pause-directive.md) asks the next cycle to investigate distance, score, time and recovery. No objective below is implemented, tuned or validated. No old reward, checkpoint, fitness, label or run record is changed by this document. A reward campaign must wait for reliable, scored native episodes.

## Preserve the actual historical objectives

Cycle 1 was not uniformly “distance-only.” Its code-bound semantics are:

| Historical method | Optimized quantity | Interpretation |
| --- | --- | --- |
| Surrogate PPO, `original-0.1.0` on `surrogate-0.1.0` | [hill.py](../../src/gradientclimb/simulation/hill.py): `progress * 0.1 - 0.0004 - terminated.astype(float) * 1.0`; `progress = best_x - old_best` | Dense increase in maximum nominal simulator position, small per-decision cost and terminal penalty. The policy optimizes discounted return; distance remains a separate evaluation measure. |
| Surrogate CEM, original implementation | [cem.py](../../src/gradientclimb/algorithms/cem.py): mean completed episode distance per candidate | Episodic nominal simulator distance, independent of PPO's shaped return. |
| Direct screen CEM, `real-screen-cem-1` | Qualified actual terminal distance, or displayed progress at a verified paused endpoint after the full episode horizon | Unknown/unqualified endpoints are ineligible, not zero. The failed pilot completed no learner update. Its last diagnostic HUD maximum is not fitness. |

Use the original run's source SHA, configuration, simulator/profile/calibration version, checkpoint hash and recorded score semantics to identify historical results. These runs did not consistently record a standalone reward-version identifier; do not invent one retrospectively. Package version `0.1.0a1` is not a unique objective version.

The [screen bridge](../operations/screen-feature-bridge.md) measures image-relative pose and terrain with validity masks. Displayed progress behaves as a temporal maximum, not signed current position. Current readers do not validate total game score, stunt events, fuel or recovery outcomes. Currency visible in a menu or a user demonstration is not an episodic score label. Coins, airtime, flips and other skill bonuses must be distinguished as potential contributors with their actual game semantics independently observed; no additive score formula is assumed from their names.

## Questions to register before implementation

1. Does adding game score improve useful distance and survival, or primarily increase stunts followed by terminal failure?
2. Does time efficiency improve forward progress without selecting short, fatal episodes or exploiting reset-time accounting?
3. Does rewarding recovery after a maneuver preserve strategic progress better than immediate raw stunt/airtime reward?
4. Do conclusions persist across training seeds and genuinely different vehicle/map configurations, at equal actual compute and interaction budgets?

“Recovery” needs an independently checkable definition. A proposed event starts when a declared maneuver/risk condition occurs and closes after a declared observation window. A recovered event could require valid upright/ground-relative evidence sustained for a minimum duration, continued progress beyond a specified increment, and no terminal failure during the window. Those thresholds, eligible maneuvers, observation cadence, maximum window and treatment of gaps must be preregistered from construction data. Current modulo-π body orientation cannot by itself distinguish upright from inverted, and apparent wheel contact is not ground truth. Recovery labels therefore require additional validated visual evidence or manual labels before automated use.

The proposal does not claim that every jump or flip is strategically harmful. A successful maneuver can trade immediate risk for useful progression. The experiment must measure the tradeoff instead of declaring a preferred behavior from appearance alone.

## Proposed outcome vector and decision rules

Retain a vector per episode rather than hiding behavior inside a single scalar:

| Outcome | Required source and unit | Missingness / failure rule |
| --- | --- | --- |
| Distance | Qualified terminal or paused HUD metres with explicit boundary | Invalid endpoint stays unknown; no carry-forward diagnostic maximum substitution. |
| Game score | A validated episode-specific score field and documented game semantics | Do not use lifetime currency, best score or unverified OCR. Missing score does not become zero. |
| Time | Monotonic observed episode seconds; acquisition gaps, reset/ad and total training time separately | Natural termination, truncation and administrative abort remain different. |
| Survival / terminal cause | Independently recognized or labeled crash, fuel exhaustion, timeout or unknown | Unknown does not silently count as successful survival. |
| Recovery | Validated event/outcome labels and observation windows | Occluded or interrupted windows are censored/unknown; do not award recovery credit. |
| Efficiency | Qualified distance per observed gameplay second and per total session second, reported separately | Use a fixed minimum episode horizon or stratify termination; a short fatal burst must not win merely through a favorable denominator. |
| Control/measurement quality | Stale/invalid frames, neutral releases, endpoint coverage, resets and exclusions | Always report alongside performance so missing difficult episodes cannot inflate success. |

The full independent metric record must include **final distance; accumulated episode score; score gain from the episode start; score per metre; score per second; forward metres per second; episode duration; death and cause; stagnation time; backward distance; trick attempts; trick/airtime/flip/skill bonuses; successful recoveries; failed and fatal trick outcomes; and collections, including coins and fuel where observable**. These metrics remain separate from whatever scalar the learner optimizes.

For score/m and score/s, publish numerator, denominator and eligibility, not just a ratio. A zero or invalid denominator yields unknown, not an extreme reward. Score gain excludes starting balances and unrelated menu/revive/advertisement payouts; collections need episode-specific events or independently validated differences. Signed forward speed and backward distance require a validated motion/scale estimator. Until then, report HUD maximum-progress gain/time as that named proxy and backward screen displacement in pixels/body spans, with invalid world-metre fields. Neither is a measured signed world velocity. Stagnation thresholds use a declared temporal window; a HUD plateau alone cannot distinguish standing still from returning over previously covered ground. Trick counts need independent event labels; the current modulo-π silhouette axis cannot reliably count complete flips. Every metric has a confidence/validity field and raw supporting observations.

## Four required future objective comparisons

The resumed study must preserve at least these four comparisons. They are prospective objective versions, not names retroactively attached to old runs. The existing PPO progress/time/termination shaping is an additional historical reference where relevant; it must not be mislabeled as the new pure distance-only arm.

| Proposed arm | Objective to specify before execution | Key question |
| --- | --- | --- |
| A. Distance-only | Qualified final distance, with the same endpoint, horizon and eligibility rules as the other arms | What strategically useful behavior does distance alone produce under the chosen actual-game learner? |
| B. Distance + raw score | Distance plus a frozen normalized episode-score component, including observed coin/airtime/flip/skill rewards according to verified semantics | Does adding immediate game score induce fatal stunts or route loss despite high score? |
| C. Distance + score + time efficiency | The same distance/score terms plus an explicitly defined time term, with preregistered horizon and termination handling | Can time improve useful progression without rewarding reckless short bursts or punishing necessary recovery? |
| D. Recovery-conditioned | Distance and score with maneuver/bonus credit withheld or downweighted unless the registered recovery criterion succeeds; time treatment fixed in advance | Does conditioning bonus value on subsequent recovery preserve progression and reduce strategically fatal reward seeking? |

An optional fifth **lexicographic or constrained** arm may prioritize survival/recovery and minimum distance before score/time, or choose from a Pareto set. It supplements the four comparisons if the evidence warrants it; it does not replace one merely because scalar tuning is inconvenient. Normalization, weights, constraint levels, terminal costs, event windows and any time discount use construction/training data only. Freeze them before held-out evaluation; tuning and trials count as prior search compute. The same actor, data access, action space and total budget must be used to attribute differences to the objective.

For recovery-conditioned credit, an immediate event reward can be withheld until its outcome window closes, then credited only under the registered recovery criterion. This is a proposed reward-assignment mechanism, not a learned causal recovery model. A policy decision at time t must never receive future observations or the eventual label as an input. Online history/context uses only observations available before the decision. Episode-end fitness may use the subsequently observed outcome, with explicit training semantics and no future leakage into deployed features.

Report a Pareto comparison of distance, score, survival/recovery and total time before selecting a scalar winner. A candidate should not displace the baseline if its claimed benefit is entirely due to worse score coverage, more excluded failures, reset overhead omitted from the denominator or a change in endpoint semantics. Preregister uncertainty handling and a minimum practical improvement; do not choose these after viewing held-out results.

## Eight required reward-hacking diagnostics

These are future failure cases to annotate and test across all four objectives, with matched observation quality. They are hypotheses about possible exploits, not claims that Cycle 1 learned them.

| Diagnostic | Evidence to retain | Rejection signal for a purported improvement |
| --- | --- | --- |
| Repeated flips followed by death | Attempt/bonus counts, recovery windows, terminal cause, final distance | More flip score accompanied by unrecovered fatal outcomes and worse strategic distance/survival. |
| Airtime without progress | Airtime labels, bonus amounts, progress and landing outcomes | High airtime reward with no sustained forward progress or useful recovery. |
| Slow but safe behavior | Duration, stagnation, distance, survival and efficiency | Survival or time treatment rewards near-inactivity; separately check whether an aggressive time penalty unfairly rejects useful safe recovery. |
| Reckless speed | Forward/proxy speed, distance, crash timing, score and recoveries | A high early rate dominates the scalar despite earlier death and lower endpoint quality. |
| Coin chasing with route loss | Collections, route/progress history, backward motion, fuel and terminal outcome | Local coin gain outweighs lost progression, fuel viability or successful continuation. |
| Backward loops | Signed motion where validated, revisited terrain, repeated collections/bonuses, max-progress plateau | Cycling over old ground farms credit without useful net progress. |
| Stagnation | Declared stagnation windows, actions, bonuses, cumulative clock | Continuing to earn reward while stalled or leaving idle time out of the budget. |
| Local score farming | Repeated local event patterns, score rate, distance, duration and reset boundaries | A small recurring maneuver/collection loop improves raw score while strategic progression fails. |

A recovery-conditioned arm also needs checks against repeatedly entering and leaving an easy “recovery” state, censoring failed windows, exploiting missing visual features or treating a terminal event as a successful reset. Event segmentation must prevent overlapping duplicate credit. Compare raw episode vectors and recordings; a scalar increase alone cannot pass these diagnostics.

## Minimal future experiment and acceptance criteria

First register an observation/annotation study and bounded native reset/scoring test, as specified in the [resumption plan](../operations/resumption-plan.md). Seal construction and independent held-out episode/session hashes before tuning readers. Validate exact score reads, terminal labels and recovery-event agreement with errors and coverage, including examples of spectacular failure and recovered progress. Do not count crops or augmentation as independent real observations.

Only after that gate, preregister a bounded four-arm screening budget and the criteria for extending promising arms, while holding actor, observation masks, action space, initial conditions, clock, maximum horizon and reset policy fixed. All four required comparisons must be retained, including unfavorable outcomes. Use repeated training seeds where the environment permits; distinguish learner RNG from unavailable public game seed control. Evaluate frozen checkpoints on independent real episodes with the entire outcome vector and all eight diagnostics. Keep simulator experiments in a separate domain; the surrogate currently lacks validated commercial-game score and stunt mechanics.

A future objective is supported only if it improves a preregistered real outcome or tradeoff at equal declared budget, with no unacceptable regression in survival/recovery/measurement coverage and with uncertainty reported. Failure to detect benefit is publishable. No raw-score leaderboard or successful single stunt establishes strategic competence.

Implementation belongs in existing reward/fitness code, [run configuration and metadata](../../src/gradientclimb/experiments/schemas.py), evaluation result dictionaries and reporting, with new explicit objective/metric versions and parent lineage. The current schema already supports these records; propose a schema migration only if a concrete requirement cannot be represented. Preserve original distance series and tables alongside any later objective, never rewrite them.
