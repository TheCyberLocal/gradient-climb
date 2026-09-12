# Prospective native descriptive summaries

`native-descriptive-summary-3.0` records every attempted native episode by its
actual control identity, with observed and unknown result counts and lifecycle
classifications. Requested but unattempted episodes remain explicit. This is an
additive report contract in new `run_screen_episodes.py` records; it does not
alter historical records, clocks, scoring decisions or reliability criteria.
An attempt whose outcome was recorded before episode-artifact publication failed
is counted as attempted with a missing summary, rather than as never started.

An accepted result reading can remain diagnostic evidence when later parking
fails. The new inventory preserves both facts. Existing `scored_episodes` and
`distance_statistics` retain their existing selection of episodes with a distance
and no gameplay error. No unknown score is filled from a gameplay HUD maximum.

These summaries report distributions, without confidence intervals or assumed
training-seed uncertainty. In particular, pooling alternating scripted gas and
random controls cannot describe one trained policy. Per-control groups are
visible; the pooled values are only an inventory of available observations, with
potentially selective score missingness. Independent game terrain seeds,
qualification and learning speed are never inferred from these descriptive
records. Registered frozen-policy evaluation and its uncertainty analysis remain
separate.

The need was observed in the preserved
[native session 2 report](../../research/diagnostics/native-reliability-002.md).
Its original generic bootstrap label remains in the sealed source. This new
contract prevents that unsupported label in subsequent native summaries.
