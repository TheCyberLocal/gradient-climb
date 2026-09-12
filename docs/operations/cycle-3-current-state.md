# Cycle 3 implementation state — 2026-09-12

This is a current work record, not an amendment to sealed Cycle 1/2 findings.
The governing [learning-efficiency methodology](../methodology/learning-efficiency.md)
defines the prospective objective and the [campaign](../../experiments/definitions/cycle-3-campaign.json)
keeps scientific collection conditional on measured gates.

## Implemented foundations

- Causal outcome/readiness/safety contracts, conserved score credit and explicit
  blinded-reader provenance are available as versioned libraries. They do not
  establish that current readers are independently qualified.
- Training interruption journals retain completed work and atomic checkpoints;
  warm start is distinguished from exact resume. The historical interrupted run
  remains sealed and has a separate diagnostic.
- Scenario/profile contracts preserve seeded legacy simulator behavior. Reserved
  reference environments fail explicitly until their implementation is available.
- Native reliability 2.3 has strict dispatch checks, an exclusive input lease,
  host resource guards and distinct episode/session endpoints.
- Resource accounting records measured process CPU core-seconds, sampled device
  utilization and memory with scope, coverage and missing values. Command-entry
  clocks and checkpoint cost receipts are opt-in through
  `scripts/train_with_command_clock.py --entry-receipt <new-local-json> ...`.
  The first sample precedes project imports; interpreter startup is excluded.
- Learning-efficiency envelopes link immutable scenario/checkpoint/evaluation
  evidence, adaptation costs and inherited system/policy lineage. Threshold and
  Pareto analysis feed the dashboard, report CLI and executed notebook directly.
- Read-only demonstration capture, private archive/restore utilities and a
  hash-pinned articulated-engine engineering pilot are implemented and tested.

## Observations and validation

The fresh [canonical integrity audit](../../research/experiments/cycle-3-efficiency-integrity.json)
verified **136/136** runs, with no unfinished or uncatalogued canonical directories.
Hashes establish byte integrity, not backup durability or scientific validity.
Cycle 1/2 tags and historical records retain their original clocks and semantics.

Native inspection `82bc980b-9c29-4e04-a560-dc9e13b6e9b5` completed without gameplay.
Reliability attempt `915ac38b-ac38-45f1-bc16-7540f5aa2353` stopped before its first
episode because the target was not foreground. All 12 attempts remain not started;
it contributes no scored episode or learning result. Source was pinned at
`eae208ee62280726626d7f4b5f9f9a3623934879` for these attempts.

The integrated regression suite passed **650 tests**, with four optional-engine/
platform skips and two dependency deprecation warnings. Focused review corrections
were checked separately. The notebook and dashboard report **zero** qualifying
efficiency study envelopes; no historical surrogate result was relabeled to fill
this gap. Synthetic test fixtures are not research training or qualification data.

The governing efficiency revision `b61b7e19343b37451d8af83b32fba87ce55abe08`
passed [exact-revision CI](https://github.com/TheCyberLocal/gradient-climb/actions/runs/34699410134).
The subsequent first engine pilot was retained as failed because its flat fixture
ran beyond its finite support domain; see [F-007](../../research/findings/F-007-engine-fixture-support-domain.md).
Its individual seal passed verification. The 136-run audit above remains at its
original pre-pilot cutoff; the new failed run is additional evidence.

## Remaining collection gates

No learned real-game threshold is demonstrated by this milestone. Native scored
reliability/soak, independently validated readers, measured simulator fidelity,
human demonstrations, selected learner comparisons, three one-hour qualified
training seeds, and additional-profile adaptation/retention remain unfinished.
The learning-efficiency framework is preregistered but not dispatchable until its
specific source, method, scenario, priors and evaluation manifest are resolved.

Human demonstration capture requires the owner's actual play; synthetic controls
cannot supply it. A private backup destination must be selected before archiving
real evidence, and only a successful restore test will justify a backup claim.
The engine pilot can establish engineering feasibility only; its throughput
cannot establish fidelity, transfer or faster learning. No stable release or
merge to `main` is justified by software foundations alone.
