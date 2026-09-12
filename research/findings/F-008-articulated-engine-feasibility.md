# F-008 — The extended synthetic fixtures pass the bounded engine screen

Status: **engineering gate passed, bounded**, Cycle 3, 2026-09-12. This permits
measured fixture construction with the candidate engine. It establishes no
real-game fidelity, learned driving, transfer, rapid-learning result or optimal
environment count.

Run `a6df4449-f62f-42f8-995d-3d592819b049` completed
[articulated-engine-pilot-3.1](../../experiments/definitions/cycle-3-engine-pilot-3.1.json)
from clean pinned source `1bbb542c37e5c15f61834e83c00045a102bdd9a5`.
Its canonical seal verifies, as does the preserved predecessor. The
[numeric interpretation](../experiments/cycle-3-engine-pilot-002.json) retains
source/run/result hashes, all 21 observation-artifact references, all diagnostic
repeats, every throughput arm, aggregated support measurements and resource scope.
The original [F-007 failure](F-007-engine-fixture-support-domain.md) remains failed;
this is a separately registered successor, not a reinterpretation of its verdict.

## What changed and what was measured

Fixture 3.1 extends static ground to x=−1020 through 1020, derived from the declared
100-unit/s speed bound, ten-second horizon and 20-unit margin. It preserves the
open bridge gap x=6 through 14 and the articulated planks. Support-domain exits and
separate global/static-floor/bridge-gap wheel-bottom minima are sampled every
physics substep with event timestamps. The −0.05 flat penetration threshold,
solver, speeds, control script, seeds, repeats and budgets stayed fixed.

All three diagnostic gates passed, with exact repeated state digests at seed
31000. Each repetition executed 600 scripted decisions, 1,200 physics substeps
and ten simulated seconds. No wheel support left the declared outer domain.
The unchanged limits were wheel lateral error ≤0.1, bridge anchor error ≤0.15,
body speed ≤100 and angular speed ≤300, in synthetic fixture units.

| Diagnostic | Minimum over static floor | Minimum in bridge gap | Peak wheel error | Peak bridge error | Maximum body speed | Maximum angular speed | Wheel/plank contact decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | −0.001095 | — | 0.030862 | 0 | 17.1353 | 57.5050 | 0/600 |
| Bridge load | — | −0.696911 | 0.000567 | 0.021221 | 2.6479 | 4.4397 | 584/600 |
| Bridge traverse | −0.004475 | −0.747434 | 0.046963 | 0.026873 | 16.6653 | 58.9003 | 41/600 |

The flat minimum occurred at 8.275 simulated seconds, wheel x=98.3501. Its global
minimum equals its static-floor minimum. Bridge-load wheels remained over the
open static-ground gap for all 1,200 substeps, so a static-floor minimum is absent.
Their gap minimum occurred at 0.491667 s. The traverse fixture first entered the
gap at 1.5 s and reached its gap minimum at 2.125 s. These negative gap heights
remain recorded; they are not falsely scored as penetration into an imaginary
floor. Minimum plank-centre heights were −0.763156 for load and −0.804809 for
traverse; these are positions, not identified real bridge deflections.

Across all throughput fixtures, the lowest observed static-floor wheel bottom
was −0.0061872, maximum wheel lateral error 0.0493010 and maximum bridge anchor
error 0.0269369, within the same descriptive numerical bounds. There were no
outer-domain exits. Every traverse instance recorded 41 wheel/plank contact
decisions and ended with chassis x between 95.7343 and 96.0247, beyond the bridge.
This is bounded contact/traversal evidence. Driven chassis ended rotated about
−9.5 radians; the fixture has no driver-death or driving-success criterion.

## All throughput arms

Box2D 2.3.10 used two physics substeps per 1/60-second scripted decision, eight
velocity iterations and three position iterations. Rendered arms produced an
original 320×180 debug image every fourth decision: 15 images per simulated
second per world. All 12 arms completed 600 decisions per world within their
30-second caps.

**Worlds are stepped sequentially in a Python loop.** These counts do not
demonstrate parallel CPU scaling. Physics timing includes diagnostic work;
rendering does not run the eventual real-game perception pipeline. Physics/s
below means completed solver time substeps per wall second, not solver iterations.

| Fixture | Worlds | Render | Arm wall s | Process CPU core-s | Sampled RSS MiB | Physics/s | Images/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 1 | No | 0.031843 | 0.031250 | 63.234 | 37,684.8 | 0 |
| Flat | 1 | Yes | 0.134043 | 0.140625 | 65.676 | 8,952.4 | 1,119.0 |
| Flat | 8 | No | 0.192611 | 0.187500 | 66.418 | 49,841.4 | 0 |
| Flat | 8 | Yes | 0.712015 | 0.687500 | 66.461 | 13,482.9 | 1,685.4 |
| Flat | 32 | No | 0.805138 | 0.781250 | 68.867 | 47,693.7 | 0 |
| Flat | 32 | Yes | 2.760531 | 2.750000 | 69.605 | 13,910.4 | 1,738.8 |
| Bridge traverse | 1 | No | 0.074958 | 0.062500 | 66.230 | 16,009.0 | 0 |
| Bridge traverse | 1 | Yes | 0.155648 | 0.140625 | 66.457 | 7,709.7 | 963.7 |
| Bridge traverse | 8 | No | 0.547255 | 0.531250 | 67.008 | 17,542.1 | 0 |
| Bridge traverse | 8 | Yes | 1.197207 | 1.203125 | 67.109 | 8,018.7 | 1,002.3 |
| Bridge traverse | 32 | No | 2.878494 | 2.796875 | 69.922 | 13,340.3 | 0 |
| Bridge traverse | 32 | Yes | 4.359937 | 4.375000 | 71.043 | 8,807.5 | 1,100.9 |

Scripted decision rates are exactly half the physics-substep rates; their exact
values and initialization/physics/render timing components are in the JSON.
With/without-render state digests matched for every corresponding fixture.
Each timing cell ran once in fixed order, without counterbalancing. Seed prefixes
31000 through 31031 recur across world-count and render arms. Timer quantization,
warm-up, order and workstation variability limit comparisons, especially the
shortest cells. These observations select neither a fastest configuration for
learning nor a preferred final batch size.

## Costs, coverage and inherited work

The pilot recorded **102,000 scripted decisions, 204,000 physics substeps,
1,700 summed simulated seconds and 12,300 debug renders**. Its 170 fixture
instances contain 65 distinct fixture-kind/seed pairs with reused seeds; they
are not 170 independent scenarios. There were **zero episode lifecycles,
learned-policy decisions, optimizer updates or real-game interactions**.

| Quantity | Recorded value and boundary |
| --- | --- |
| Pilot elapsed time | 21.1702 s from after source/protocol verification through measured arms; excludes caller imports and later result serialization/sealing |
| Recorder wall interval | 21.2794 s; its last measurement precedes remaining Parquet/seal writes |
| Installation | 4.2782 s, already inside pilot/run time; pip child-process CPU is unmeasured |
| Sum of arm wall intervals | 13.8497 s; a component of the pilot, not an extra charge |
| Recorder-process CPU | 14.828125 core-seconds, all process threads, excluding children |
| CPU sample coverage | 20.9073 s, 99.7450% of the 20.9607-second resource window; earlier/later uncovered costs remain unknown |
| Recorder memory samples | Peak 74,268,672 bytes; weighted sampled mean 68,895,592 bytes; 99.7450% within-window coverage |
| Per-arm memory samples | Largest sampled RSS 74,493,952 bytes; the different sampling schedule explains why it exceeds the recorder's sampled maximum |
| Device-wide GPU activity | 5.12767 utilization-equivalent seconds; sampled mean 24.5248%, 99.7490% within-window coverage; not engine-attributed GPU consumption or energy |

Arm CPU intervals include recorder threads and end after final fixture-summary
materialization, slightly after the arm wall endpoint. Their summed 13.6875
core-seconds are diagnostic components, not costs added again to the recorder
total. Sampled memory maxima are not guaranteed process peaks and are never
summed across arms.

The operator reports no local research training, tests or native capture during
the pilot; report editing occurred outside the pinned worktree. Other game and
application processes remained present. Canonical provenance verifies clean
source at capture, but neither this operational account nor device-wide sampling
establishes exclusive host/GPU attribution. The GPU readings cannot be assigned
to this CPU engine/debug renderer.

Both engine attempts remain system-construction priors. Counting each once gives
32.6289 summed recorder wall seconds, 16.046875 measured recorder-process
core-seconds, 103,200 scripted decisions and 1,720 summed simulated seconds.
The 11.0000 installation seconds are already included in those run clocks.
Development, correctness tests, interpreter setup and installer child-process
CPU are incompletely measured; a complete system-prior cost or end-to-end calendar
learning time cannot be inferred from these sums.

## Decision and limit

The registered installation/numerical/usability gate permits a measured geometry
and dynamics prototype with this engine. Reserved reference environments remain
unavailable. No proprietary coefficients, real route coverage, camera match,
deployable observation quality, fuel planning or learned control were qualified.
Next work should connect a bounded measured fixture and shared observation path
to a prospectively frozen real-outcome comparison, including inherited costs.
Independent real competence per elapsed time and compute must justify additional
fidelity; these raw engineering rates cannot do so.
