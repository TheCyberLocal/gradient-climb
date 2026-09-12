# Owner video acquisition and sparse construction review 001

Two sealed construction runs retained a 299.999-second excerpt of the owner's [All Cars video](https://www.youtube.com/watch?v=wx_cI59vFX0) and extracted all 18 preregistered original frames. Both seals verify, every requested timestamp matches its retained frame's source PTS exactly, and both final directories fit their declared byte budgets. These are actionless archive and visual construction records: zero new real interaction, policy decisions, synchronized action labels, optimizer updates or independently evaluated episodes. No learning or competence result is established.

The [machine-readable diagnostic](owner-video-acquisition-001.json) pins both protocols, run/config/seal/source-state records, four subprocess receipts, the media/probe/selection records, and both manual reviews. It retains all measured resource values and coverage while excluding raw media, logs, executable paths, host fingerprints, process identities and GPU identities. Original sealed records remain unchanged.

| Operation | Canonical run | Recorder wall seconds | Recorder CPU core-seconds | Final directory bytes |
| --- | --- | ---: | ---: | ---: |
| Acquisition | `13db846a-1e6b-4dfe-b51a-c1bacece6565` | 141.095 | 79.890625 | 94,528,815 |
| Sparse review | `4e4fae5f-d40d-4973-81a5-b3cd4c4c47a7` | 36.889 | 5.703125 | 14,742,447 |

Acquisition had a 300-second operation deadline and 256 MiB final-output budget; review had 120 seconds and 128 MiB. Their payload directories contain 54,088,395 and 7,320,002 bytes respectively. Final totals include payloads, registered copies, local tool files, journals and seals; duplicate files are storage costs, not extra acquisition events. Cooperative polling can overshoot an operation deadline by one indivisible operation. End-of-record to seal-created intervals were 20.116096 and 0.245061 seconds, outside the reported recorder/resource endpoint; these timestamp intervals are not full command clocks.

The measured file is 36,835,187 bytes, WebM/VP9, 1280×576 (20:9), with a reported average/nominal 60/1 frame rate, 1/1000 time base and one video stream without audio. These are encoded-stream properties; they do not establish distinct original captures, uniform original game timestamps, or absence of edits, duplicates or dropped frames. Its SHA256 is `91e6ecf2ad93a574f486ad5e07e9a5e010834b39b5aa8cf472eb90f8df42db9d`. The acquisition's `actual_media_seconds` remains null in its seal; 299.999 seconds is the later probe result, not a rewrite of that record.

The separate counted ffprobe pass decoded 18,000 video frames. The ffmpeg extraction pass's total decode count is unknown; its 18 output images cannot substitute for that count. Requested and observed PTS were 0, 10, 15, 20, 25, 30, 35, 37, 40, 60, 90, 120, 150, 180, 210, 240, 270 and 299 seconds. All 18 image hashes are distinct and verified. This checks the selected timestamps, not every intervening PTS, original gameplay rendering, physics steps or continuous episode boundaries.

| Child operation | Elapsed seconds | Sampled process-tree CPU lower bound, core-seconds | Process sample errors |
| --- | ---: | ---: | ---: |
| Local pinned-tool installation | 2.775 | 1.875000 | 4 |
| Public excerpt download | 136.832 | 3.328125 | 5 |
| Counted full-video probe | 25.667 | 24.515625 | 4 |
| Sparse frame extraction | 7.965 | 26.781250 | 4 |

All four receipts report exit code 0, no primary or cleanup error, and no observed live descendant after launcher exit. Their CPU figures preserve the maximum observed lifetime counter for each launched process identity sampled about every 0.2 seconds. Short-lived children, detached descendants and work after the final sample can be missed. The receipt errors remain visible; these numbers are lower bounds. Child elapsed times overlap their enclosing recorder interval and must not be added to recorder wall time. Recorder CPU covers its own threads and excludes children; it totals 85.593750 core-seconds across the two runs, separately from 56.500000 sampled child core-seconds. Neither is a complete workflow cost or an energy estimate. Pinned wheel retrieval ran inside the recorder before installation; its standalone cost is unseparated within that window, not zero.

Recorder CPU coverage was 99.9564% for acquisition and 99.7698% for review, over resource sample spans of 140.7890 and 36.4804 seconds. Initial resource samples completed about 0.3052 and 0.4084 seconds after recorder start. Resource clocks exclude work before the initial sample and after the final sample; imports, initial protocol parsing and prior-run lookup precede the recorder. The JSON retains exact covered/uncovered intervals, valid samples, interpolation rules and all fault counters.

Device-wide GPU utilization-equivalent time was 44.1052 and 13.6606 seconds, with 99.9664% and 99.8447% sampled coverage. This includes all host processes and has no policy or per-process attribution. No joules, power, or precise GPU active time were measured. Memory values are sampled peaks and time-weighted means over covered intervals; unsampled peaks may be higher, and memory is not summed across runs.

| Scope | Acquisition peak / mean MiB | Review peak / mean MiB |
| --- | ---: | ---: |
| Recorder process RSS | 66.422 / 66.274 | 62.801 / 60.833 |
| Host RAM used | 30,177.406 / 30,069.868 | 31,047.195 / 30,809.058 |
| Device-wide VRAM used | 818.000 / 808.023 | 806.000 / 806.000 |

Acquisition records commit `ea2a98b20a0319a2108bab396ac29fb416181482`, an empty tracked diff and 12 untracked reader files. Review records `596b0e36b93565b80130099347da24ce0fdd3fff`, the unrelated partition helper/test correction and 12 untracked reader files. Both report dirty worktrees and preserve exact source-state hashes. Governing scripts and imported recorder/helper source were committed before execution; the review script does not import the changed partition helper. No whole-checkout-clean claim is made. Source-state collection's `collection_is_atomic=false` flag describes provenance collection, not a demonstrated change to those governing modules.

The operator reported background synthetic reader tests during these intervals. Exact overlap and host load were not independently metered, so these records support construction provenance and observed cost, not controlled throughput, training speed or rapid competence claims.

The [early sparse review](owner-video-early-sparse-review-001.json) directly inspected nine retained images from 0–40 seconds; the [late sparse review](owner-video-late-sparse-review-001.json) inspected the other nine from 60–299 seconds. Their frame/hash/PTS references join exactly to the extraction record. These are manual narrative construction interpretations, not geometric ground truth or control supervision. Menu pixels establish the Hill Climber name in early selected frames; its complete map, upgrade and build identity remain unresolved. Later selected images reveal other vehicle appearances, menu/tune states, unequal wheel sizes and face-camera/HUD occlusions. Visible progress, mission and crown text are distinct UI fields; none is an independent benchmark outcome. Visible pedal/touch indicators do not establish synchronized delivered actions. Every gap between sampled frames remains unreviewed, and the original video's complete episode count is unknown.

The whole source remains construction evidence. Historical human play, skill acquisition, editing, uploading, prior browser discovery, manual review, engineering/test work and this report's creation were not fully metered; their costs remain unknown, not zero. The review inherits the acquisition by its unique run ID and does not create a second acquisition charge. Source-video PTS contributes no new real interaction or simulator seconds. Further use requires declared sequence selection, source-specific geometry/profile evidence, nuisance handling and separately validated video-only control inference before any training eligibility can change.
