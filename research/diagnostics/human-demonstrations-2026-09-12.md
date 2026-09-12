# Human demonstration timing review — 12 September 2026

The longer imitation recording provides **406 temporally eligible candidate image/control pairs**, with a stable measured **5.98 frames/s** at the requested 6 frames/s. The short diagnostic provides 63 candidates. Neither count establishes independently reviewed imitation eligibility, an episode outcome or human benchmark competence. **The next step is to review and segment the existing imitation recording before requesting more play.**

This is a new derived numeric report. The three source runs remain sealed and unchanged. No pixels, complete control journal, private window titles or device identifiers are copied here. Full measurements and source hashes are in [the companion JSON](human-demonstrations-2026-09-12.json).

## Source and integrity

All recordings used clean source `b61b7e19343b37451d8af83b32fba87ce55abe08`. `verify_run` passed for all three, and the read journal/summary hashes match their registered manifests. Both journals in every session have zero trailing partial bytes.

| Recording | Source run | Partition | Accepted frames | Control samples | Capture wall time |
| --- | --- | --- | ---: | ---: | ---: |
| DXCam attempt | `febf6812-0f73-495f-a8a7-34a21a1dddf9` | Development | 0 | 33 | 0.5203 s |
| MSS diagnostic | `789b59ed-8c97-4cb9-9d97-9b1a4aaba277` | Development | 218 | 1,939 | 30.1302 s |
| MSS imitation | `fbafca5b-fd59-499a-a56d-1180528cdc21` | Imitation | 545 | 5,866 | 91.1891 s |

DXCam stopped at the stale-frame guard, configured at 450 ms. The rejected image has no journal row, so its exact acquisition latency is unavailable. The retained control samples alone cannot supply image/action pairs. This is an acquisition failure, not a gameplay failure.

The imitation recorder ended with `WindowUnavailable: demonstration target unavailable`. The operator had been instructed to switch away, but the journal does not establish the precise cause of target unavailability. The canonical recorder status remains `failed`; that status does **not** classify the preceding gameplay outcome or erase its evidence.

## Acquisition and polling

The output resolution was **1034 × 581**, normalized from a 2581 × 1449 client. The same frozen UI profile was used in both MSS sessions. Percentiles below use NumPy's default linear quantile interpolation within each fixed session. They are descriptive timing measurements, not uncertainty over independent sessions.

| Measurement | Diagnostic median / p95 / maximum | Imitation median / p95 / maximum |
| --- | ---: | ---: |
| Native capture-call latency | 61.29 / 67.56 / 97.91 ms | 67.90 / 72.41 / 103.11 ms |
| Capture start to normalized observation ready | 83.97 / 91.72 / 121.06 ms | 94.43 / 99.86 / 129.21 ms |
| UI classification latency | 16.57 / 18.06 / 26.19 ms | 16.69 / 18.29 / 33.09 ms |
| Frame start-to-start interval | 138.49 / 147.68 / 174.61 ms | 167.02 / 167.48 / 186.35 ms |
| Control interval gap | 15.50 / 16.49 / 21.95 ms | 15.49 / 16.59 / 20.61 ms |
| Eligible pair's observation-ready to next sample start | 6.30 / 13.96 / 14.59 ms | 6.60 / 13.68 / 14.90 ms |

The diagnostic achieved **7.2353 frames/s against a 10 frames/s target**. Its median observation-ready-to-recorded interval was 54.96 ms, which includes downstream classification, hashing/encoding and file work; the whole median acquisition-start-to-recorded interval was 138.01 ms. These measurements identify real recorder overhead rather than policy inference latency: no policy model ran.

The historical `missed_capture_schedule_slots=0` value does not prove the 10 Hz target was met. The implementation floors each start interval by the requested period, and all observed intervals were shorter than two periods. It therefore reports zero despite measured throughput below target. This report retains that limitation without rewriting the source metric.

The imitation session achieved **5.9766 frames/s against a 6 frames/s target**. Observed control sampling was **64.34 Hz** and **64.39 Hz**, respectively, against the requested 100 Hz. Neither session had a gap exceeding the existing 50 ms censoring threshold. These are sampled key states; exact delivery, between-poll edges and human reaction latency are not measured.

The diagnostic's 87 consecutive identical-pixel frames were 85 Tune and 2 paused hypotheses; none were template-playing frames. The imitation recording had 20 such repeats, also outside template-playing frames. Identical pixels in a stationary screen do not establish a stale capture.

## Action coverage and causal pairing

The existing `causal_action_pairs` helper was called with `maximum_lag_seconds=0.1` and `maximum_poll_gap_seconds=0.05`. It requires the control sample to start after the normalized observation becomes available, a preceding bracket without overlap, and a frozen template `playing` hypothesis. A later screenshot never labels an earlier action in this pairing.

| Coverage | Diagnostic | Imitation |
| --- | ---: | ---: |
| Template-playing images | 65 | 406 |
| Temporally eligible candidate pairs | 63 | 406 |
| Candidate neutral `00` | 33 | 182 |
| Candidate gas `10` | 27 | 185 |
| Candidate brake `01` | 3 | 39 |
| Candidate both pedals `11` | 0 | 0 |
| Total images excluded by helper | 155 | 139 |
| Observed control changes | 5 | 94 |

The diagnostic excludes playing frames **192 and 214** because the preceding polling intervals end **0.4661 ms and 0.4602 ms after observation-ready**. No threshold was relaxed to admit them. The other 153 excluded diagnostic images and all 139 excluded imitation images have non-playing or unknown template states.

The diagnostic's five change brackets, relative to capture start, are **0.4307–0.4468 s** (brake), **25.4055–25.4202 s**, **27.8842–27.8996 s**, **28.3960–28.4115 s**, and **29.5773–29.5924 s** (gas). These are transition observation bounds, not exact key event times.

The imitation recording has **68 gas changes and 26 brake changes**. Transition-bracket widths have median **15.52 ms**, p95 **16.26 ms**, and maximum **19.32 ms**. No two-pedal change occurred in one polling interval, and state `11` was not observed. The absence of that state is a coverage fact, not proof that the demonstrator should have used it.

These counts establish candidate alignment only. The diagnostic remains development data. The imitation recording still requires reviewed gameplay/terminal segmentation and verification of appropriate action labels before training. A `playing` template is insufficient to promote every mechanically eligible pair to a training example.

## Gameplay boundaries remain distinct from session time

The short session contains 65 playing, 126 Tune, 9 paused and 18 unknown template observations. The first playing image begins acquisition **0.0004 s** into the recording; the last completes at **30.0577 s**. The coordinator's selected visual review identified the tail of an old run, Tune menus, and the start of another run. It is not a complete matched episode, and its 30.13 seconds cannot be counted as continuous gameplay.

The long session contains 406 playing, 34 Tune, 12 paused, 14 advertisement and 79 unknown observations. Its first playing hypothesis is **frame 35 at acquisition start 5.8731 s**. The main template sequence extends to **frame 437, completed at 73.1067 s**. After paused/unknown observations, **frames 458–461 at 76.5473–77.1154 s** are again labeled playing. Later advertisement/unknown content continues through the last image at **90.9803 s**.

The coordinator's subsequent visual review resolves selected boundaries: frame **438** is gameplay at **740 m**, with the vehicle upright on a bridge; **448 and 457** show **PAUSED at 745 m**; **458** is new gameplay at **0 m**; **465** shows **PAUSED at 0 m**, and **467** an **EXIT confirmation at 0 m**. Frames **475, 510 and 544** show commercial/advertisement content. Thus the main attempt reached a reviewed **745 m HUD lower bound before administrative interruption and restart**. The subsequent brief 0 m attempt was exited. **No natural failure or result was observed.** These selected observations and their image hashes are in the JSON; no unknown template image is promoted into training eligibility by this report.

Those first/last appearance bounds are not a verified continuous gameplay duration. This numeric report leaves **episode experience count, real interaction seconds, natural outcome and qualified endpoint distance unknown**. The selected 745 m pause-HUD observation is not a completed benchmark, an independently qualified distance-reader result or repeatable competence. Zero completed segmented outcome records describes annotation status, not zero gameplay experience.

## Resource and disk costs

| Recorded scope | DXCam attempt | MSS diagnostic | MSS imitation |
| --- | ---: | ---: | ---: |
| Whole recorder run wall time | 0.9590 s | 36.3459 s | 106.1353 s |
| Capture wall time | 0.5203 s | 30.1302 s | 91.1891 s |
| Recorder-process CPU core-seconds | 0.46875 | 24.3125 | 63.953125 |
| Device-wide GPU utilization-equivalent seconds | 0.1652 | 10.7561 | 25.3667 |
| PNG payload bytes | 0 | 61,912,669 | 153,003,607 |
| Reserved payload including canonical copies | 27,626 bytes | 125,913,198 bytes | 312,093,472 bytes |

CPU accounting covers the recorder process and excludes child processes; it includes acquisition, polling, encoding, registration and finalization. It is not learner compute. Its measured coverage is 93.13%, 99.86% and 99.94%, respectively, within the resource sampler's own boundaries. Device-wide GPU activity includes the game and other processes and cannot be attributed to this recorder or a policy. Energy was not measured. Run wall time includes work outside capture, while imports before resource sampling remain excluded; none of these clocks is silently substituted for a full command cost or human practice time.

At the first diagnostic's mean compression, **600 seconds at 6 frames/s** projected **1,988.27 MiB** including double storage and observed journal rates, close to a 2,048 MiB cap. Using its maximum observed PNG instead projected **2,350.31 MiB**. This was a planning estimate, not guaranteed storage use. The actual longer recording ended after 91.19 seconds with approximately 298 MiB reserved, well before the configured cap.

## Bounded next step

**Do not request another recording yet.** Review the existing imitation sequence, especially its terminal/menu boundary, and determine which of the 406 candidates can be admitted. The current recordings already give far better brake coverage than the first diagnostic; another arbitrary quota is unnecessary.

If review identifies a concrete need for another ordinary run, retain **MSS, target 6 frames/s, unchanged 1034 × 581 output and requested 100 Hz control polling**. Select its horizon and disk cap after reviewing the useful behavior to be captured; an arbitrary short batch horizon should not truncate a useful long demonstration. As a capacity illustration only, 120 seconds at 6 frames/s gives 720 images: the largest PNG observed across these recordings (341,878 bytes), doubled for canonical copies, uses 469.50 MiB before journals. A 640 MiB engineering batch would leave about 170.5 MiB for journals and overhead, but this is **not a complete-run budget recommendation**. A planned 600-second run needs a larger independently chosen disk allowance than a mean-compression extrapolation alone supports. No additional capture is requested now. Any time, focus or disk endpoint remains capture censoring until the gameplay outcome is independently established.

## Reproduction

The numeric inputs are each run's `demonstration-summary.json`, `demonstration/frames.jsonl`, `demonstration/controls.jsonl`, `run.json`, and registered manifests. Their hashes and the helper source hash are preserved in the JSON. All event times are relative to that run's `capture_session_started_ns`.

```python
from pathlib import Path
from gradientclimb.capture.demonstrations import read_complete_journal, causal_action_pairs
from gradientclimb.experiments import verify_run

run_id = "fbafca5b-fd59-499a-a56d-1180528cdc21"
assert verify_run("artifacts", run_id)["valid"]
directory = Path("artifacts/runs") / run_id / "demonstration"
frames = read_complete_journal(directory / "frames.jsonl")["rows"]
controls = read_complete_journal(directory / "controls.jsonl")["rows"]
pairs = causal_action_pairs(frames, controls,
                           maximum_lag_seconds=0.1,
                           maximum_poll_gap_seconds=0.05)
assert sum(pair["eligible"] for pair in pairs) == 406
```

This review creates no new image capture, injected input, learning run, benchmark score or relabeled source record.
