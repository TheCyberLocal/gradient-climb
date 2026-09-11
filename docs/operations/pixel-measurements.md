# Offline pixel measurements and their current evidence

`perception.measurements.HCRPixelMeasurer` turns a saved RGB image into explicit body, wheel-pair and visible terrain measurements. `TerrainMotionTracker` adds interval diagnostics when capture timestamps and ground texture permit them. These are profile-specific image measurements; they do not yet provide the trained simulator policy's observation vector, verified world coordinates, contact/fuel state, or permission to send input.

## Coordinate and validity contract

The profile in `configs/perception/hcr-discovery-wrapper.json` expects exactly 1034 by 581 analyzed pixels. X increases rightward and Y downward. Native physical frames of another size must be explicitly resized and registered as derived observations before this profile is applied; retain their original dimensions, scale and source hash. The reader does not silently resize or pretend analyzed pixels are world meters.

The body output is an isolated red component's centroid, bounding box, area and principal axis. It is not center of mass. A wheel pair requires two circular edges, neutral hub support, compatible radii/spacing and agreement with the body axis. These checks reject some head/sky-circle distractors but do not establish accuracy under all poses. Wheel names mean image-left/image-right; axle pitch remains modulo pi and does not resolve front/rear identity or rollover.

Terrain points follow the visible turf/soil material boundary. This suppresses decorative grass tips, but the offset to the game's physical collision surface is unknown. Occlusion and insufficient color support create invalid points. The current conservative scene ROI excludes the HUD; when terrain descends behind/below that ROI, coverage falls rather than inventing terrain.

Quality numbers summarize measured color, edge, circle, track or sample support. They are **not calibrated probabilities**. A valid image feature is also not proof that gameplay is active. UI state, foreground identity, observation age and input authorization belong to the separate guarded control loop.

## Camera-motion limits

The tracker uses near-ground soil features with forward/backward optical-flow checks and a robust affine fit. Insufficient tracks, narrow spatial support, inconsistent motion, zoom, rotation, long/nonpositive time intervals and inconsistent turf-edge movement are rejected. Failed intervals break continuity; there is no filled-in global distance across gaps.

Crucially, a 2D game's soil texture can be screen anchored. Measured soil-texture translation is therefore only an image statistic until its connection to world motion is verified using independent landmarks and capture sequences. The real profile leaves `static_ground_assumption_verified=false`: diagnostic `measured_texture_translation_xy` can exist while camera translation and vehicle displacement remain null. Do not change that flag just to obtain a numeric trajectory. The synthetic tests set it true only because their source-image transform is known by construction.

Even after anchoring verification, camera-compensated displacement is in relative ground-plane pixels. Changing scale, parallax and unobserved gaps invalidate conversion to world motion. The center HUD may show maximum progress rather than signed instantaneous position; its constancy while coasting backward must not be treated as zero vehicle movement.

## Measured source run and audit history

The real control-attempt source is `ac68937c-f5c2-442b-b4d5-f49c4f6e4df1`: 35 normalized PNGs plus capture timestamps and separately recorded OS pedal transitions. The current derived pixel run is `20ab0566-bd37-4064-9757-933664dd644e`, whose seal verification passed. It records 24 supported wheel pairs out of 35 frames and zero validated camera intervals. This is detector coverage, **not wheel accuracy**. No independent geometric labels or camera calibration exist for these frames.

Earlier derived runs `b2ba1612-38e9-49f8-b87c-b0dedd47bdd8` and `bdf76f15-5b08-48d2-a223-1b92d8ec5278` remain sealed for audit history. Inspection of the first revealed driver-head false wheel pairs in frames 18 and 27 and potentially misleading camera interpretation. Later checks added chassis-axis consistency, stricter hub color support and explicit ground-anchoring verification. These repeated analyses use the same 35 source frames; they are not 105 additional observations. Use the current run when reporting supported measurements.

Manual inspection found displayed progress of 0 m in source frame 7, 2 m in frame 27 and 4 m in frame 34. Those isolated readings do not demonstrate a gas/brake response or calibrated dynamics. OS transition delivery and visible game response remain distinct evidence.

## Reproduce measurement and evaluate labels

Run only against an already saved JSONL manifest with relative `path`, recorded `sha256` and integer `timestamp_ns` fields. Missing timestamps disable camera intervals; missing image paths reset continuity. The manifest is bounded to 500 frames by default.

```powershell
.venv/Scripts/python.exe scripts/measure_game_pixels.py --frames artifacts/runs/ac68937c-f5c2-442b-b4d5-f49c4f6e4df1/frames/frames.jsonl --session-id native-pedal-attempt-ac68937c
```

Each run registers source images/profile/manifests, emits `pixel-measurements.jsonl`, and seals its artifacts. It reports zero gameplay episodes and training/environment steps. No native input or capture API is imported.

`perception.annotations.FrameAnnotation` and `schemas/pixel_annotations.schema.json` describe independent point labels, source hashes, dimensions, session IDs, train/held-out split, annotator/method, uncertainty, wheel visibility and terrain/camera targets. Null targets mean unannotated. Camera labels identify both source frames. Held-out labels must be created without viewing model predictions; sessions and source hashes cannot overlap training, including camera-pair endpoints. Synthetic and real labels are evaluated separately. Missing predictions remain in coverage denominators, and terrain interpolation does not bridge invalid sample gaps.

Supply `--annotations path/to/labels.jsonl --split heldout` to the measurement script for a registered label report. Report center/pitch/terrain/camera errors together with coverage and label uncertainty. The current code is tested on eight generated-fixture cases, including head distractors, missing texture, zoom, timing gaps and unknown ground anchoring. Those tests do not measure real-game accuracy.

## HUD glyph prototype

`perception.hud.HUDDigitReader` reads a declared HUD region from local manually labeled glyph templates. It segments neutral digit interiors, normalizes each glyph, and rejects weak or ambiguous matches. It requires all ten digit classes before returning a numeric reading: otherwise an unseen nine could be misread as a known six. Output is named `hud_displayed_progress_meters` and is never interpreted as signed current position.

`scripts/prepare_hud_templates.py` builds the inspected initial glyph bank under ignored artifacts. Run `6fd5abdb-51e7-43a0-aecd-4cd7b9cc8298` contains 22 glyph crops from seven labeled HUD regions. At construction it covered digits 0 through 8 and refused numeric readings because digit 9 was missing. Glyphs from coin/checkpoint/fuel quantities can supply font examples, but those regions' numeric values are not progress measurements.

To extend the bank, visually label a new source region before examining its OCR result, add it with `add_labeled_region(rgb, text, roi, source_sha256=..., annotation_id=...)`, and save a new bank directory. Preserve the original source/hash and annotation. Use separate whole capture sessions for later accuracy evaluation. No game glyph images are committed or relicensed as original project content.

### Completed alphabet and bounded independent test

The completed immutable bank is `ae953116-cd5f-4f47-8313-d7350e36001a`, with its manifest at `artifacts/runs/ae953116-cd5f-4f47-8313-d7350e36001a/hud-glyphs/hud-glyphs.json`. It contains 23 glyphs covering all ten digits. The added nine comes from the visibly labeled gems quantity **19** in `user-demonstration-19-gems.jpg`. The JPEG is a saved Computer Use observation of user gameplay, not native timed capture or agent performance. Its capture timestamp is null. A smaller nine-only subregion avoids a JPEG outline artifact between the digits; thresholds and normalization were unchanged.

Before running complete-bank OCR, five probe frames were independently labeled from their center HUD: frames 0/7/13/27/34 show **0/0/0/2/4 m**. The pre-OCR label file SHA-256 is `f8f6456e7ea5691ffece6305b7b134dc5d1558ebd13ddd240a92ffa96e06bf36`. Training and held-out source hashes are disjoint, and no thresholds were tuned on these test images.

Evaluation run `74f6c15c-013d-48e2-af51-12bc95d084bf` records five accepted exact matches out of five labeled frames. These are correlated frames from one probe session and only three displayed values; this is not broad OCR accuracy, nor independent validation of every digit class. Across all 35 source frames, 26 readings were accepted and nine were rejected as ambiguous. Accepted readings were 0 m in frames 0–16, 2 m in frames 23–27, and 4 m in frames 31–34. Other frames remain unknown; they are not interpolated or silently assigned a value. Only five frames have independent labels.

Both new run seals verified successfully, and both contain zero gameplay episodes and training/environment steps. The bank run records the original user-observation JPEG and frozen parent bank; the evaluation records source frames, pre-OCR labels, predictions, confidence failures and the reproduction script. Displayed progress alone does not establish signed displacement, calibrated dynamics, or response to the attempted pedal states.
