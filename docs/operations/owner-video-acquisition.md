# Owner gameplay archive acquisition

The owner requested reuse of https://www.youtube.com/@thecyberlocal instead of
additional play recordings. The first bounded source is the beginning of
[Hill Climb Racing - All Cars Video](https://www.youtube.com/watch?v=wx_cI59vFX0).
Its whole video is construction material. No portion is an untouched benchmark.

Run the committed protocol once:

```powershell
.venv/Scripts/python.exe scripts/acquire_owner_video.py --project-root D:/Projects/gradient-climb --protocol experiments/definitions/cycle-3-owner-video-acquisition-001.json
```

The canonical recorder begins before package acquisition and records failures as
well as success. The [official yt-dlp project](https://github.com/yt-dlp/yt-dlp#installation)
documents installation from PyPI. The protocol pins a wheel digest and size from
the official registry and installs it with no dependency or global configuration
changes into the local run. An installed ffmpeg executable is hash-pinned too.
The command reads one public URL without accessing browser credentials or cookies.
All footage, partial bytes, facecam content and command logs stay local and ignored.

The 300-second budget includes download/install commands and is checked around
bounded operations. Indivisible work and final retention/sealing can finish after
the last budget check; report the measured boundary and actual final output size.
The payload limit reserves room for canonical copies and recorder metadata.
Subprocess CPU is a sampled lower bound, separate from recorder-process CPU.
Missing original play, editing, skill acquisition and browser-review costs remain
explicit inherited costs. Utilization is not energy consumption.

Verify the acquired media's actual duration, dimensions and presentation timestamps
before extracting training sequences. Requested clip duration is not an observed
episode count or new real interaction. Preserve cuts, dropped or repeated frames,
camera changes and administrative boundaries. Record every decode/extraction cost.

This source has no synchronized key journal. Geometry and temporal visual analysis
are supported uses; a future action-inference method needs separately validated
labels and uncertainty. Do not pass video-inferred states to the existing causal
key-poll dataset as if they were delivered controls. Any assisted policy must
retain archive acquisition, annotation, pretraining and teacher costs in its
lineage and still undergo independently qualified real evaluation.

After the acquisition seal verifies, a separate committed sparse-review protocol
can extract original frames for visual inspection:

```powershell
.venv/Scripts/python.exe scripts/review_owner_video.py --project-root D:/Projects/gradient-climb --protocol experiments/definitions/cycle-3-owner-video-review-001.json
```

This first review freezes 18 requested timestamps, preserves each selected source
presentation timestamp, and performs no interpolation or actor preprocessing.
The full ffprobe counting pass and ffmpeg extraction pass have separate subprocess
receipts. Eighteen retained frames do not mean only eighteen frames were decoded.
The actual media metadata and source-frame count are measured in the review run;
they are not backfilled into the sealed acquisition record.
