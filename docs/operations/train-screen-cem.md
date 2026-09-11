# Governed actual-screen CEM training

`scripts/train_screen_cem.py` implements the protocol in
`experiments/definitions/real-screen-cem.json`. It trains `ScreenLinearPolicy` with
`EpisodeCEM` from normalized native captures and explicit pixel measurements. It
does not load a surrogate policy or privileged state. The initial measurement and
mask weights are zero, with gas bias +1 and brake bias −1. The selected features,
required body features, scales, schema, random seed, population, and UI dependencies
are recorded before the first episode.

The implementation has been tested with synthetic frames, clocks, and input mocks.
It has not by itself established a successful real one-hour CEM run. A shortened
supervised integration must verify the complete start/score/park/update cycle on
the current registered UI before the governed run. Unknown advertisements or
uncovered controls can still stop the adapter safely.

The CLI requires `--hud PATH` and `--result-reader PATH` pointing to the frozen
gameplay-glyph manifest and result-reader manifest. Its default behavior validates
the definition and prints a plan without native actions. `--execute` enables the
authorized run. The defaults are 3,600 governed seconds and the definition's
60-second episode horizon; `--seconds 120 --episode-seconds 8` selects an explicitly
unqualified shortened integration. An episode-horizon override is rejected for a
3,600-second run. UI, measurement, and experiment-definition paths can be supplied
explicitly; all resolved dependencies are frozen and copied into the run record.

Actual execution requires a clean committed Git checkout, exactly one discovered
game window, and an initially recognized PAUSED or Tune screen. The operator must
preserve the previously inspected vehicle, map, upgrades, resolution, focus, and
layout. Neither this trainer nor the current Tune profile independently verifies
all vehicle/map identity fields. The initial source revision is checked again after
dependency registration. Every local reference, glyph, manifest, imported project
module, helper script, dependency lock, and project definition used at startup is
registered with a content hash in the canonical record.

The governed monotonic clock starts at entry into the script, before heavy package
imports. It includes package/model initialization, source verification, record
setup, menu/ad waiting, capture, perception, inference, policy updates, checkpoint
serialization, and evidence persistence. Interpreter launch before file entry is
explicitly excluded. There is one absolute deadline. A wrapper on the ordinary
Windows input sender checks it immediately before non-release packets, and the
pedal watchdog stops admitting holds at that deadline. Owned key/mouse release
cleanup remains permitted. The optimizer checks the deadline again after phase
and checkpoint callbacks, immediately before accepting a score.

An episode starts only if its complete declared horizon plus a ten-second parking
reserve remain. Menu costs are checked again after starting; insufficient remaining
time causes a verified pause and no candidate evaluation. Duration truncation uses
the full declared horizon, followed by a verified paused boundary. A naturally
ended episode reaches a verified terminal result, captures stable numeric evidence,
then advances through supported dialogs to Tune. Persistence and optimizer updates
occur while parked. The remaining budget is spent stationary when there is not
enough time for another full episode. Reaching the explicit episode-attempt cap
early is reported as an incomplete governed hour.

Only these scores are eligible:

- A natural result with exactly one current-episode accepted reading: two fresh,
  agreeing nonnegative integer distances at least 0.15 seconds apart, followed by
  successful parking at Tune.
- A completed duration truncation with a valid scoped paused-distance readout from
  the verified PAUSED boundary. This includes release-to-pause delay and is labeled
  separately from terminal distance.

An accepted 0 m is valid. Missing numeric values, raw/maximum gameplay HUD values,
stale evidence, bools, nonfinite durations, shortened horizons, capture/control
errors, and failed parking are ineligible. Unknown scores never become zero. A
bounded terminal-reader exhaustion can dismiss a separately verified result while
leaving its score unknown; the trainer retains all raw attempts and exhaustion
records. An ineligible attempt keeps the same CEM candidate pending. Three
consecutive ineligible attempts stop the run. Only a complete eligible population
updates the distribution; unfinished generations retain their exact candidates,
scores, RNG state, and costs.

Required body features are an additional gate in the policy callback. Missing
required support returns neutral even if some optional feature is present. This is
independent of the UI/focus/freshness gate. Two pedal logits retain all four joint
states. The state-history and validity-mask schema is saved with every policy.

Checkpoints target 300, 600, 1,200, 1,800, 2,700, and 3,600 seconds. A callback at
episode ticks and phase boundaries snapshots the first observed crossing. Each
JSON checkpoint contains a top-level `policy`, `optimizer_state`, pending candidate,
parent run, schema, source lineage, target time, actual snapshot time, and phase.
Snapshotting detaches mutable optimizer state; disk writes wait for the next
stationary phase. Actual persistence times are separate. A delayed snapshot is
never described as occurring exactly at the target time. The canonical checkpoint
artifact hash is the identity used by later frozen-policy evaluation.

At the deadline, no further policy selection or optimizer update is admitted.
Final release, trace storage, record sealing, and final checkpoint serialization
may finish afterward; observed overrun/persistence time is reported rather than
hidden. A final state file is also written on shortened or failed runs. An artifact
whose target has not been reached is not fabricated.

The canonical run stores per-episode observations, sampled native frames, verified
boundary frames, candidate/score decisions, requested leases, raw OS pedal events,
raw mouse delivery/cleanup events, menu actions, discarded captures, guard reasons,
raw/accepted/exhausted terminal readings, phase times, optimizer generations, and
machine/process telemetry. Episode-relative times stay in episode observations;
canonical trajectories receive their recorder-relative elapsed origin. Partial
episodes and pending candidate state remain on interruption. Training-selection
scores are not independent evaluation results, and the trainer never marks them
as qualification evidence.

Focused trainer validation currently includes 22 tests covering a full synthetic
CEM generation, partial collection failure, invalid-score candidate retries,
required-feature neutral actions, fixed-horizon admission, source and manifest
integrity, measured-zero eligibility, deadline-crossing callbacks, OS packet
admission, and detached checkpoints with actual crossing times. Native reset tests
and frozen-reader tests are maintained separately.
