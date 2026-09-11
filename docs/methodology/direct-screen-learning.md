# Direct screen-feature learning protocol

The corrected extended arrow encoding and recorded pedal animations establish
ordinary input delivery to the actual game. The existing surrogate actor still
expects privileged world velocity, contact and fuel values. The screen bridge
cannot provide those quantities reliably, and the surrogate is uncalibrated.
Direct episode-based parameter search is therefore an explicit additional
strategy to test, with its own observation schema and training clock.

`experiments/definitions/real-screen-cem.json` registers the versioned design and
requires shortened integration before a governed hour. It specifies two
linear pedal heads over eight image-derived measurements, eight validity bits,
and a bias: 34 trainable parameters. Each candidate is scored by observed episode
distance. Gas and brake remain independent; no route sequence or scripted vehicle
behavior is substituted for a learned policy. The fixed gas-only initialization
is a declared engineering prior and is measured separately as a baseline.

The optimizer reevaluates both the incumbent and current mean in each population
of eight. It updates its search distribution only after eight eligible measured
scores. Score failures, capture interruptions, ads, persistence and incomplete
generations retain their real time and experience costs. Invalid scores do not
become zeros. Numeric reading seeks two agreeing fresh frames at least 0.15 seconds
apart. After twelve attempts or three seconds without agreement, independently
verified result UI may be dismissed, retaining an unknown score. The pending
candidate is retried; three consecutive ineligible attempts stop the session.
A snapshot includes the exact pending candidates, score list,
distribution and random-generator state, so resumption can preserve the search.

The registered one-hour clock includes collection, perception, all input/reset
handling, ad waits, optimization and serialization. It is distinct from the
completed surrogate hour and its independent-seed reproduction. No surrogate
checkpoint is loaded into the screen actor. A future distillation/transfer study
must separately establish aligned observable features and evaluate the gap.
The implemented restricted student exposes the same eight body features plus
masks and history, with an explicitly uncalibrated analytic projection in its
training environment. It has not yet established transfer performance.

Before dispatch, run fixed-duration baseline episodes through the same capture,
perception and input loop, validate natural and truncated restart paths, and
verify the result-field OCR on independently labeled images. Record any manual
intervention. Unknown game screens and missing required body evidence release
both pedals. A held observation is never made fresh by assigning a new timestamp.

Random, gas-only and learned policies must use identical episode/time/measurement
rules. The installed game's level resets do not expose a public seed API; record
the observed same initial condition and do not invent real-game seed control.
The RNG seed controls the learner and random baseline only. Final evaluation
episodes follow training and are reported separately from candidate-selection
scores. Passing a provisional distance gate does not establish expert-level play.
