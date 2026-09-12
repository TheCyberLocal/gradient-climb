# Reviewed imitation construction: corrected plan 002

The corrected construction plan produced **54 four-image windows**, referencing
**60 unique images** through **216 overlapping frame uses** from one previously
recorded imitation session. The fixed geometry diagnostic completed all 60 image
attempts. Its body/wheel/terrain validity flags were 60/44/50; these are detector
hypotheses, not independently labeled accuracy. All windows remain construction-only
and training-ineligible because named vehicle, map and game-build identity is
incomplete. No policy was fitted or evaluated, and no new gameplay was acquired.

The [machine-readable audit](imitation-construction-002.json) preserves exact
source and artifact hashes, every measured resource field and coverage interval
apart from private host identities, reviewed bounds, exclusions and inherited
capture costs. It is a read-only analysis of sealed outputs; it did not rerun the
assembler or geometry detector. These results are additional to the historical
136-run integrity cutoff, which remains unchanged.

## Attempts and source identity

| Attempt | Canonical run | Result | Source revision |
| --- | --- | --- | --- |
| First plan 001 | `8f5a7301-3cba-49c2-97e0-d39420526d7b` | Failed before window assembly; zero output windows | `e6489b520f7a1fb2261ab408ad52127923d515f0` |
| Corrected plan 002 | `fe9dabb6-16a6-47c9-90b4-edf3bbc919b3` | 54 windows, 60 unique images | `2f654286fef58d4ce36aad7e869184799d60efd6` |
| Fixed geometry | `9be884a9-efe8-44c0-ac5b-ee99187bf17e` | 60 attempted/completed, zero failed images | `2f654286fef58d4ce36aad7e869184799d60efd6` |

All three records report clean source, and all three seals passed independent
verification for this audit. The three inherited acquisition seals also passed.
The source recording remains `fbafca5b-fd59-499a-a56d-1180528cdc21`, recorded at
`b61b7e19343b37451d8af83b32fba87ce55abe08`; its failed capture status is preserved.

Plan 001 correctly rejected a hash mismatch: review v1 supplied logical
configuration hash `0aa0af4ad757a4bddcd80847f7104d7c4060b9525cb1d59f2ae91eb5d484fcf9`
where the contract requires the exact configuration file hash,
`fa9d8f9972eb187b2af70fbe0e7bc6ec9e5de08dba35c10235079baa35740df9`.
Parsed configuration and the sealed record agree. All 65 unique review evidence
references matched source bytes and the seal. The
[failure diagnostic](imitation-window-001-hash-failure.json), original plan and
review remain unchanged. The [registered successor](../../experiments/definitions/cycle-3-imitation-window-diagnostic-002.json)
corrected only that file-hash identity before execution; it retained review bounds,
timing limits, partition, detector and eligibility. The failed publication's ledger
still binds future consumers.

Dataset content digest is
`df65ddc22e94de752125399570a4fc03eb6be0ec9f06139d3d3f3a86f44eeaae`;
the published dataset file hash is
`8c89688e29d883ec947791f39166e677623ac57b994a866bf81c63c1fc106695`.
The geometry report references the same dataset digest and has file hash
`587fab2b7667d0f1baf67dd7a5c940887b07c081894e679f7d24a91bcbab6dbf`.

## Reviewed bounds and exclusions

| Span | Reviewed image indices, inclusive | Capture-relative reviewed interval, seconds, end exclusive | Accepted target indices, inclusive | Unique referenced images | Windows |
| --- | --- | --- | --- | --- | --- |
| Early | 35–65 | [5.8730517, 10.947633401) | 38–64 | 35–64: 30 | 27 |
| Later | 350–380 | [58.5022404, 63.584438901) | 353–379 | 350–379: 30 | 27 |

The two visual reviews cover 62 images and approximately 10.157 seconds of bounded
capture intervals. Neither establishes a complete episode or a natural endpoint.
History resets at each segment. Every accepted history contains four causal
observations without padding, and every target poll completes inside its own
reviewed time interval. The audit independently checked these constraints against
the published windows. The frozen limits are 0.25 s between history frames,
0.45 s observation age, 0.1 s target lag and 0.05 s polling gap.

Of 545 source frames, 491 are excluded as targets: 483 fall outside reviewed playing
segments; 35–37 and 350–352 lack a complete history; and 65/380 have observation
readiness beyond their conservative exclusive segment cutoffs. Earlier history
images can still contribute to later windows. Both original journals have zero
trailing partial bytes; all 5,866 recorded control polls remain source evidence.

The 54 target labels comprise 24 neither-pedal states, 26 gas-only states and four
brake-only states, with no both-pedal state. These are focused key-state polls,
not human intention, OS delivery or action durations. Overlap does not create
independent experience. Episode count and complete real gameplay duration remain
unknown. The whole-session assignment remains imitation/train, while incomplete
profile evidence blocks training eligibility. Tune evidence establishes four
displayed upgrade slots at 13/13, 14/14, 16/16 and 10/10; the owner's Hill
Climber/Countryside note is retained separately from independent visual identity.

## Fixed geometry observations

The unchanged `configs/perception/hcr-discovery-wrapper.json` detector returned
valid body hypotheses for 60/60 images, wheel hypotheses for 44/60 and terrain
hypotheses for 50/60. Every selected image remains in the output, including invalid
components. Independently measured accuracy is null. There were 122 durable
operation-journal events covering selection, starts and completions.

Per-image measurement took 1.309204 s in total: median 21.895 ms and p95 26.393 ms.
Hash validation and decoding took 0.779971 s in total. These timings exclude
assembly, provenance, journal writes and artifact publication; they are components
inside the geometry recorder window. This single fixed workload does not establish
native control throughput, geometry accuracy, simulator fidelity or real learning.

## Construction and inherited costs

| Offline attempt | Recorder wall s | Measured CPU core-s | CPU coverage | Device-wide GPU utilization-equivalent s | GPU coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Failed plan 001 | 0.414035 | 0.015625 | 67.85% | 0.022794 | 60.42% |
| Corrected windows | 2.446304 | 1.984375 | 97.66% | 0.573090 | 97.39% |
| Fixed geometry | 4.843425 | 11.078125 | 98.90% | 1.050023 | 98.96% |

These sequential recorder windows total 7.703764 wall seconds and 13.078125
observed CPU core-seconds. They exclude imports, initial plan/profile parsing,
partition-authority preflight and work after the final resource sample. Full
command elapsed is unmeasured. CPU coverage uses the resource-sample window, not
the larger recorder duration. Device-wide GPU utilization includes other processes
and is neither energy nor policy-attributed GPU work. The JSON retains all sampled
process/host RAM and VRAM peaks, time-weighted means, coverage and gaps per run;
memory is not summed as consumed work.

| Inherited acquisition | Canonical run | Capture wall s | Recorder wall s | Measured CPU core-s | CPU coverage | GPU utilization-equivalent s | GPU coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DXCam stale-frame failure, zero accepted images | `febf6812-0f73-495f-a8a7-34a21a1dddf9` | 0.520308 | 0.959031 | 0.468750 | 93.13% | 0.165198 | 92.91% |
| Mixed development capture, 218 images | `789b59ed-8c97-4cb9-9d97-9b1a4aaba277` | 30.130190 | 36.345886 | 24.312500 | 99.86% | 10.756096 | 99.86% |
| Imitation capture ending target-unavailable, 545 images | `fbafca5b-fd59-499a-a56d-1180528cdc21` | 91.189077 | 106.135288 | 63.953125 | 99.94% | 25.366741 | 99.95% |

All six unique run-cost identities are retained once in this report. Source
capture costs appear in both downstream records as inherited provenance; those
copies do not constitute repeated acquisition. Geometry does reconstruct the
window selection, and its own recorder includes that repeated work. Per-image
timings must not be added again to recorder totals. Capture wall time also includes
menus and operational states, so it is not established gameplay experience.

Human practice, manual visual review, engineering, method search, prior system
construction and report-authoring costs remain explicitly unmeasured. These six
runs do not represent complete project costs or time-to-competence. The next step
is to investigate specific observation/profile limitations under a new construction
protocol. Independent labels are required before accuracy claims; fitting,
learning and qualification each retain their own unmet gates.
