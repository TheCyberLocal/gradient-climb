# Native reliability 2.3 — session 2

**The process completed, but the registered reliability gate failed.** All twelve
attempts reached natural results and parked at Tune. Eleven received accepted
scores; the ninth attempt (index 8) remained unscored, breaking the longest scored
streak at **8**, below the required 10. No manual intervention or unintended action
was recorded. This is scripted lifecycle evidence, not learned driving.

Canonical run `1c263031-ac29-4dbc-8fa5-864a2473116a` used clean, isolated source
`1bbb542c37e5c15f61834e83c00045a102bdd9a5` under
[native-reliability-2.3](../../experiments/definitions/cycle-3-native-reliability-2.3.json).
Its final seal verifies all **1,953 files**. The
[numeric interpretation](native-reliability-002.json) retains hashes, every
attempt, resource coverage, input/recovery counts and explicit interpretation limits.
The registered protocol hash matches the pinned and committed LF bytes. The main
checkout initially held a CRLF copy; it was normalized to exact committed bytes
after verifying identical JSON and LF-normalized content. No protocol semantics
changed. The protocol was hashed in configuration rather than copied into this
run's manifest.

This is **session 2 of the maximum 3**. Session 1,
`915ac38b-ac38-45f1-bc16-7540f5aa2353`, remains a failed foreground-guard session
with all twelve scheduled attempts not attempted. Across both sessions, 24 attempts
were scheduled, twelve entered gameplay and eleven produced scored cycles. Neither
session is excluded or replaced. One registered session remains available; no
additional run is implied by this report.

## Every attempt

Indices are zero-based. Scores are the frozen result reader's accepted values.
Gameplay seconds are the runner's observed episode-loop intervals. All attempts
parked at Tune, including those using the registered restart recovery.

| Index | Script | Terminal cause | Accepted score (m) | Classification | Gameplay s | Frames / dispatches | Park/reset s | Restarts |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | always gas | driver down | 237 | natural scored | 16.620 | 169 | 76.911 | 1 |
| 1 | random | out of fuel | 5 | natural scored | 45.038 | 433 | 5.053 | 0 |
| 2 | always gas | driver down | 224 | natural scored | 16.734 | 162 | 5.486 | 0 |
| 3 | random | out of fuel | 1 | natural scored | 46.385 | 459 | 73.249 | 1 |
| 4 | always gas | driver down | 208 | natural scored | 16.790 | 173 | 6.504 | 0 |
| 5 | random | out of fuel | 1 | natural scored | 45.048 | 448 | 4.200 | 0 |
| 6 | always gas | driver down | 332 | natural scored | 27.703 | 275 | 76.979 | 1 |
| 7 | random | out of fuel | 15 | natural scored | 45.049 | 444 | 5.028 | 0 |
| 8 | always gas | driver down | unknown | unscored | 17.577 | 177 | 5.514 | 0 |
| 9 | random | driver down | 4 | natural scored | 15.511 | 155 | 74.648 | 1 |
| 10 | always gas | driver down | 292 | natural scored | 23.262 | 245 | 6.334 | 0 |
| 11 | random | driver down | 4 | natural scored | 15.148 | 152 | 4.201 | 0 |

Score coverage was **11/12 (91.67%)**, satisfying its separate minimum of 10/12.
The consecutive-scored-cycle requirement failed. The zero-unintended-action and
zero-manual-intervention conditions passed under their recorded definitions.
There were eight driver-down and four out-of-fuel endings. **No attempt reached
the full 60-second horizon**, so full-horizon pause scoring and long-run soak remain
untested here. Four advertisement encounters recovered through application restart;
zero in-game advertisement closes were exercised.

## Why index 8 remains unscored

All twelve bounded result-field reads returned `Unrecognized or ambiguous HUD
glyph`; no agreeing pair was accepted. The gameplay HUD maximum of 269 is a
different observation and does not replace the missing result score.

Separate post-output visual inspection of retained `terminal-024.png` and
`terminal-035.png` reads DRIVER DOWN and DISTANCE: 269 m. Exact replay of the frozen
58-glyph bank explains the rejection: the middle visible 6 ranks as 9 at 0.933116,
0 at 0.920583, 8 at 0.918138 and 6 at 0.916803. The best-class margin is 0.012532,
below the required 0.04; the result anchor passes. Lowering the margin would accept
the erroneous value **299**. This supports a prospective reader-construction
correction with new disjoint validation, not a relaxed threshold.

The selected result image for each of the other eleven attempts visually agrees
with its recorded score. These are checks after seeing outputs. They do not supply
blinded reader accuracy, change the null score, or retroactively extend the streak.

## Lifecycle and input evidence

The session dispatched **3,292 scripted gameplay actions** and recorded 3,292
gameplay observations, all with accepted HUD hypotheses. It requested 3,292
bounded 0.4-second leases; overlapping renewals are not additive gameplay duration.
The OS pedal trace reports 2,168 requested and 2,168 inserted key events in 1,616
batches, with no short insertion. Reconstructing the transition sequence leaves
both pedals released. These are OS insertion records, not direct evidence of the
game's interpretation of every event.

There were 41 accepted named menu clicks: twelve starts, eleven revive declines,
twelve result continuations and six bonus declines. The OS menu trace contains
the matching 41 click batches plus 41 pointer-parking moves, totaling 164 inserted
events. No guard fault was latched, and no accepted click was followed within five
seconds by a latched foreground-loss event.

Four park-phase application restarts succeeded, at indices 0, 3, 6 and 9. Their
combined wall interval was **98.371 seconds**, already included in park/reset and
session costs. They followed one recognized advertisement without an allowed
close and three unrecognized advertisement phases, using the registered bounded
wait and restart route. Reset observations included 499 advertisement and 1,735
unknown frames; restart boot frames have a separate state count.

The capture trace contains **one discarded initial frame**, with a 449.101 ms
capture interval and 470.919 ms total grab latency, followed by a fresh acquisition.
Each attempt's `stale_captures: 1` is a cumulative session counter, not twelve
separate stale captures. The trace does not separate backend initialization from
acquisition or establish actual compositor/display age.

## Time and resource scope

| Quantity | Observed value | Scope |
| --- | ---: | --- |
| Recorder wall | 746.285 s | Recorder start through final resource sampling; not full command entry |
| Session timer | 745.367 s | Governed native session |
| Gameplay-loop intervals | 330.865 s | Sum across twelve naturally ended attempts |
| Start/reset intervals | 16.143 s | Sum across attempts; median 0.964 s |
| Park/reset intervals | 344.106 s | Includes restart recovery; median 5.924 s |
| Attempt intervals | 691.117 s | Gameplay plus attempt lifecycle work |
| Session time outside attempt timers | 54.251 s | Remainder; not assigned to a specific stage without timing evidence |
| Process CPU | 981.766 core-s | All recorder-process threads; excludes game/emulator and children |
| Process RSS | 645.016 MiB peak; 500.147 MiB mean | Sampled peak and time-weighted sampled mean |
| Device utilization equivalent | 181.603 s | Whole GPU, all processes; not policy compute or energy |
| GPU utilization | 24.345% mean; 51% sampled peak | Whole device, including the game and other applications |

CPU coverage was 99.9928% of the 745.990-second resource window; process memory
coverage was also 99.9928%. There were 735 resource samples and 75 GPU queries,
with zero recorded query failures. GPU coverage was 99.9935%. Unsampled peaks may
be higher. Component intervals are nested and must not be added again to total
wall cost. Full command elapsed time and preceding imports/setup remain unknown
because this pinned runner did not produce an entry receipt.
GPU UUID and recorder process identifiers remain in the private canonical source;
the public interpretation keeps device names and aggregate coverage measurements.

Expected content was Hill Climber/Countryside. The coordinated preflight observed
Tune upgrade levels 13/13, 14/14, 16/16 and 10/10; vehicle/map names were not visible
there, so their identity is not independently established by this session record.
Engineered controls, UI references and frozen readers remain inherited construction
priors with incompletely measured development costs.

No policy was trained or consumed, and optimizer updates were zero. The canonical
`environment_steps` field remains zero; this report separately counts the observed
scripted action dispatches. The canonical pooled distance interval has a generic
"conditional on one trained policy" label that does not apply to these alternating
scripts. It is preserved in the sealed record and is not used as a learning result.
There is no real competence, independent reader qualification or rapid-learning
claim from this session.
