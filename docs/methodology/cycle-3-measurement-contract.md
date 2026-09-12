# Cycle 3 measurement and reader-provenance contract

Status: implemented and tested on synthetic cases. No native measurement accuracy,
reader qualification, learned-policy improvement or competence target is established
by this contract. Register the collecting protocol and its source revision before
using these records in an experiment.

The new implementations are
[cycle3_measurements.py](../../src/gradientclimb/experiments/cycle3_measurements.py)
and [reader_provenance.py](../../src/gradientclimb/experiments/reader_provenance.py).
Cycle 1/2 objectives, registrations, evaluation cutoffs and sealed run bytes remain
unchanged. The old readers and native runner do not automatically acquire these
semantics: callers must explicitly construct and retain the new versioned records.

| Contract | New identifier | Historical behavior retained |
| --- | --- | --- |
| Outcome and independent eligibility | `behavioral-metrics-3.0` | `behavioral-metrics-2.0` and the strict historical CEM parking gate |
| Recovery score attribution | `recovery-score-ledger-3.0` | `recovery-progress-2.0` and `recovery-conditioned-2.0` |
| Ordered policy selection | `lexicographic-3.0` | Weighted scalar `lexicographic-2.0` |
| Reader dataset provenance | `reader-session-provenance-3.0` | Registered, unexecuted `reader-validation-2.0` |

## Endpoints and censoring

`EpisodeEndpoint` separates natural `driver_down`, `out_of_fuel` and
`natural_unlabeled` termination from governed `horizon`, `stall_limit` and
`frame_limit` truncation and `administrative_interrupt` or
`instrumentation_failure` interruption. Governed and interrupted stops are
right-censored with respect to natural failure time and achievable distance.
`unknown` and `not_started` report unknown termination/censoring flags. Unknown
measurements never become zeros or successful survival.

A horizon stop requires measured elapsed time at least as long as the registered
horizon. A session deadline at 7 seconds in a 60-second episode is administrative
interruption. A crash observed at the 60-second boundary remains a natural crash;
the clock cannot replace observed causal evidence. A timeout while moving does
not identify the policy's maximum attainable distance. A stall deadline describes
the protocol's stopping rule, not a physical crash.

The caller must preserve the reason from the acquisition/control timeline and
retain independent evidence of natural termination. `outcome_flags()` serializes
the independent flags alongside the endpoint. These are finite-horizon evaluation
records, not an estimator for an uncensored distance distribution.

## Gameplay, readiness and safety

`GameplayOutcome` retains the episode identity, endpoint, monotonic start and
verified boundary times, distance/source/evidence identifier and observation
coverage. A verified terminal source requires natural termination. A diagnostic
frame may provide a useful distance for an interrupted episode without converting
that result into completed long-run competence.

`Readiness` separately describes a checked Tune/Paused boundary or an unknown or
failed parking result. `SessionSafety` describes the continuously monitored
interval, assessment time, input-release evidence and timestamped unintended,
manual, guard or release events. The caller must derive continuous monitoring and
events from retained traces; the validator cannot prove a declaration by itself.

`assess_eligibility()` emits five independent decisions:

- Measurement validity retains a known, sourced distance even when later parking
  fails. Validity does not itself authorize policy learning or evaluation inclusion.
- Evaluation eligibility additionally requires a classified endpoint, continuous
  gameplay safety coverage and no safety/manual event at or before the boundary.
  Censored evaluation records remain explicitly censored; this is data inclusion,
  not a qualification verdict.
- Learning eligibility applies the prospectively supplied `EligibilityPolicy`:
  censored data, unlabeled natural causes, minimum observation coverage and whether
  learning requires readiness. All four choices are mandatory, with a protocol ID.
- Session safety requires verified input release, current complete monitoring and
  no recorded safety/manual event. A later event leaves completed gameplay evidence
  intact while the session remains unsafe.
- Next-episode eligibility additionally requires readiness checked after the
  outcome boundary and no later than the safety assessment. This is an accounting
  decision, not permission to bypass live freshness, focus or host checks.

For example, the historically recorded 243 m result in session `28a9d5be` remains
diagnostic evidence despite its later failed parking, as described in
[F-004](../../research/findings/F-004-native-reliability-study-1.md). This example
explains the new separation; it does not rescore that historical run or change its
failed lifecycle classification. Retain every attempt and every exclusion reason
so exclusion rates can be compared across policies and difficult episodes.

## Recovery attribution and score conservation

`ScoreLedger` partitions a known counter-boundary difference into uniquely
identified, ordered, contiguous, nonoverlapping `ScoreIncrement` intervals. Every
increment is measured over `(start, end]`, has retained evidence, and has a
nonnegative amount. Their sum must equal `coins_end - coins_start`. A reset,
counter decrease, partial partition or inconsistent difference is rejected.

`RecoveryWindow` covers the declared maneuver and continuation interval and
retains an independently supported outcome. Recovered/failed outcomes require a
fully observed window. Natural failure evidence cannot be labeled recovered;
fatal requires that evidence. A recovered/failed window extending past the episode
score boundary is rejected. The event detector and progress thresholds must be
registered and validated separately; these records do not establish detector
accuracy or infer physical recovery from modulo-pi orientation.

`recovery_score_credit()` assigns each score increment exactly once. Outside all
windows it receives full credit. Inside overlapping windows it receives the
minimum applicable factor: recovered = 1, failed = the registered fraction in
`[0, 1]`, fatal/censored = 0. An increment whose exact collection time is unknown
and whose measurement interval crosses a window boundary is conservatively
subject to every overlapping window. No fabricated within-interval timestamp is
used. Store the allocation rows, raw gain, credited gain and withheld gain.

For example, overlapping recovered and fatal windows covering the same 20 coins
withhold those 20 coins once. They neither duplicate reward nor subtract the same
coins repeatedly from a residual base. Raw gain equals credited plus withheld
gain; full raw score remains a separate outcome. Unknown score boundaries or
unknown recovery coverage return an ineligible result with null quantities.
An empty event tuple means measured absence of events; `None` means unmeasured.

This conservative overlap rule is a prospective hypothesis. Compare recovery
conditioning against the raw-score arm with identical time treatment, and measure
effects of coarse counter intervals. Do not train or qualify a skill objective
until the contributing readers and event labels satisfy their registered gates.

## Actual lexicographic ordering

`lexicographic_key()` returns the tuple
`(min(distance, minimum_distance), survived_horizon, distance, useful_score)`.
Maximization compares coordinates in that exact order. Before the minimum,
progress dominates; once the minimum is reached, survival dominates further
distance, which dominates useful score. This ordering is explicit and intentionally
does not claim that every survival preference is optimal.

All measurements must be known, and the minimum must be a positive integer fixed
before evaluation. Missing values return no eligible key. No finite multiplier or
scalar conversion is provided: an arbitrarily large lower-tier score cannot beat
a better higher-tier coordinate. This is an optional selection method, not a
replacement for the four required matched objective families.

## Whole-session blinded reader provenance

`ReaderSplitManifest` records a hashed protocol, timezone-aware registration and
cutoff, explicit session/run membership, reader/anchor/UI source roots and their
complete parent graph, hashed labels and raw-frame/crop ancestry. Every parent
must resolve and cycles, orphan sources and duplicate identifiers are rejected.
Construction exclusions use the union of all ancestor hashes and session IDs,
including earlier glyph banks, anchors and UI references. A new crop from an old
construction session stays construction evidence.

Each held-out session must start after registration, be sealed and verified before
labeling, and have its purpose declared as heldout. Every label must precede that
session's earliest viewed reader prediction and the frozen cutoff. An audited
absence of prediction exposure is represented by a null exposure time plus a
mandatory hashed exposure record whose reviewed interval covers the cutoff.
Unknown or unaudited exposure cannot qualify. The supplied audit is evidence to
verify, not an automated guarantee of a human's blindness.

One run cannot belong to multiple sessions. Duplicate frame/field labels cannot
inflate counts, and an identical frame cannot count as two independent sessions.
Different labels on the same original frame count as one unique frame. Reports
record labels, unique frames and sessions per field with `episode_count = 0`.
Provenance validation alone always reports `reader_accuracy_qualified = false`;
frozen classifier execution, accuracy, wrong accepts and availability remain
separate measured gates. The old evaluator is not modified by this module.

The inherited held-out statements require reconciliation before selection:
[Cycle 2 future work](../operations/cycle-2-future-work.md) calls study 2.0 sessions
construction, also says no held-out session exists, and calls the study 2.1 session
held-out reader evidence. [F-004](../../research/findings/F-004-native-reliability-study-1.md)
already discusses exposed 243/237/19 m reader results; the current UI profile uses
study frames for construction. Those statements cannot certify blinded future
labels. New qualification needs new registered sessions and explicit exposure
records; already inspected or unverified historical data remain development data.

## Focused validation

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_cycle3_measurements.py tests/test_reader_provenance.py tests/test_objectives.py --basetemp artifacts/cycle3-tests-measurement -q
python -m ruff check src/gradientclimb/experiments/cycle3_measurements.py src/gradientclimb/experiments/reader_provenance.py tests/test_cycle3_measurements.py tests/test_reader_provenance.py
```

Tests exercise simultaneous clock/natural termination, early interruption, missing
distance/coverage, faults on both sides of a verified boundary, parking failure,
stale readiness, score overlap/censoring, partition corruption, arbitrarily large
lower-tier scores, transitive reader/UI/anchor exclusions, blinded chronology,
duplicate frame/session evidence and preservation of historical objective versions.
