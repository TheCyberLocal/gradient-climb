# Gameplay progress and terminal distance

The local reader artifacts are separate from source distribution. Game glyphs,
labels and source screenshots remain ignored artifacts. All reported matching
scores are visual similarities, not calibrated correctness probabilities.

Current gameplay manifest:
`artifacts/runs/78a29b33-8cf4-4bb4-be7a-f408acde1fcc/gameplay-glyphs/hud-glyphs.json`.
Load with `HUDDigitReader.from_manifest(path)` and call `read(rgb)`. An accepted
result has `valid=True`, `text`, and `hud_displayed_progress_meters`. Unknown is
not zero and must not be silently filled from a guessed digit or current action.

The field is x300–470/y68–110 in the declared 1034x581 image. Its wider left
margin avoids clipping three-digit distances. It excludes the upper-left
upcoming-fuel distance and right-side checkpoint records. The bank contains
32 labeled native glyphs covering 0–9, drawn from the separate eight-second
`72d935d4` recording's progress and currency/fuel-count text. Numeric components
are normalized with preserved aspect ratio. The new bank permits only a one-pixel
translation in the normalized 32x20 canvas, with unchanged Dice threshold .84 and
different-digit margin .04. Legacy manifests use zero translation. No OCR model
or third-party weights were downloaded.

Predeclared test labels were saved in
`artifacts/hud-independent-labels/scoring-heldout-before-construction.json`
before constructing/evaluating this bank. The entire 129-frame `09d92094` source
trajectory is disjoint from glyph construction. Sealed evaluation
`b2dcac01-ff92-4768-958d-aeedb791d9cc` accepted 129/129 readings, with no decreases
and final accepted display 248 m. Four selected manually checked frames read
129/169/210/248 exactly. **129/129 is availability, not measured accuracy for all
129 frames.** These four correlated labels do not establish general accuracy.
Median OCR processing was 2.83 ms, excluding capture and decoding. Raw predictions,
source hashes, labels and timing remain in that evaluation run.

Current native terminal manifest:
`artifacts/runs/66ce7f08-ef45-466c-ac77-33bacd0f0a41/result-reader.json`.

```python
from gradientclimb.perception.scoring import ResultDistanceReader

reader = ResultDistanceReader.from_manifest(manifest_path)
result = reader.read(rgb, result_state_confirmed=observation.state == "result")
if result["valid"]:
    terminal_distance = result["distance_meters"]
```

The result reader requires independent result-state confirmation, the right-side
`DISTANCE:` label at x602–736/y152–196, and recognized digits within
x738–916/y151–195. It does not read the tilted share card or its best record.
There is no template-threshold fallback or live UI control in the reader. Its
glyph bank has six additional construction glyphs from the separate user
demonstration's 243 m result and 900-coin text; those are not agent scores.

The initial JPEG-derived label anchor failed the predeclared native 313 m result
test (similarity .697 below .90), producing unknown. This failure remains in the
sealed `b2dcac01` evaluation. The subsequent `834be62a` artifact uses that native
313 m image to construct its label anchor. It self-matches 313 m with minimum
glyph similarity .895, but **313 m is now construction evidence for the combined
reader**. It has no new independent terminal-result accuracy measurement yet.
Use a different result image for that test. The number-recognition bank was not
adapted to 313.

The next independent native result was manually labeled 229 m before OCR.
Evaluation `b3b99b45-b82b-488d-93b8-b1e39c0324b7` matched the native label
anchor exactly, but rejected the final digit: nine's .9139 similarity exceeded
five's .8769 by only .0370, below the frozen .04 margin. That unknown result is
retained and the failed source episode `2cf160db` remains failed. The current
`66ce7f08` construction adds fourteen glyphs from native 313/229 distance and
3180/1035 coin fields, preserving all thresholds. Both result images are now
construction evidence and self-match exactly. A fresh result is needed to test
the resulting 52-glyph terminal bank independently; no completed game episode or
new generalization accuracy is claimed by that construction run.

## Truncated episodes at the paused boundary

`PausedDistanceReader.from_gameplay_manifest(gameplay_manifest)` reuses the frozen
gameplay glyphs and reads only after `read(rgb, paused_state_confirmed=True)`.
The narrower x300–417/y68–110 field excludes the bright pause-modal border, so
the adaptive threshold follows the dimmed HUD digits. The score is the displayed
progress at a verified paused frame, including input-release-to-pause delay. It
must not be described as the distance at the exact requested action cutoff.
No prior maximum is substituted for an unknown paused reading.

The paused 38 m frame `d9955d22/reset-004.png` informed the crop geometry.
The separate `reset-008.png` was labeled 49 m before running the paused reader.
Both are disjoint from glyph construction. Sealed run
`f06c4fdc-1941-4558-9060-e0f682cc0185` read 38 and 49 exactly; the latter is one
held-out paused readout with .9975 minimum glyph similarity, not a population
accuracy estimate. Preserve this verified paused frame and its capture interval
with every truncated score. The native adapter must supply its own paused-state,
target and freshness evidence.

During construction, before any new held-out evaluation, a zero/one ambiguity in
the training self-match exposed an annotation error: an enlarged upcoming-fuel
crop read **50**, previously transcribed as 51. The annotation was corrected and
the old mixed-font candidates were separated. Superseded construction runs
`c04223b9` and `f95d61d2` remain unchanged and are not deployment artifacts.
Training self-matches are not independent tests. The final source annotation and
construction script are recorded with `78a29b33`.

For episode scoring, retain the terminal value, maximum accepted gameplay value,
read coverage, last valid reading time and state/freshness evidence separately.
Displayed gameplay progress is not signed current position. Require a declared
eligibility rule before ranking CEM candidates; a candidate with sparse OCR must
not gain from missing observations. Terminal-reader failures remain unknown and
should be reviewed or excluded under that fixed rule. Synthetic tests cover
field isolation from a fake best record, required result confirmation, anchor
rejection, bounded glyph alignment and manifest round-tripping.
