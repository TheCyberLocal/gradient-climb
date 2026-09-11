# Actual-game screen feature bridge

`ScreenFeatureBridge` in `perception/screen_features.py` turns current RGB frames
into a fixed, masked observation. It uses the existing body/wheel/terrain
measurement profile and local HUD glyph bank. The actor must be trained for this
schema; the surrogate actor's world velocities, contact bits, fuel and signed
pitch are not available and must not be manufactured to fit its input shape.

```python
from gradientclimb.perception.screen_features import ScreenFeatureBridge, FEATURE_NAMES

bridge = ScreenFeatureBridge(measurer, hud_reader, history=4)
observation = bridge.observe(
    game_observation.frame.rgb,
    game_observation.frame.timestamp_ns,
    previous_action_code=last_os_state,  # None if unknown; request is not acknowledgment.
    action_age_seconds=age_since_os_transition,
    episode_elapsed_seconds=elapsed,
)
required = ["body_sin_2angle", "body_cos_2angle"]
if not observation.supports(required):
    controller.release()
else:
    # Only after independent PLAYING, target, freshness and lease checks.
    action = actor(observation.vector)
```

There are 49 values and 49 validity bits per frame. Four history frames produce
392 float32 values, oldest first: `[values, masks]` for each frame. Missing history
and measurements are zero with false masks. The schema hash includes feature
names, order, history length and maximum temporal interval. Check it when loading
a checkpoint. `reset()` requires an independently observed episode boundary.
`geometry_valid` requires body and supported wheels; a declared body-only actor
uses `supports()` for its own required feature subset. Neither returns input
permission. Source profile and glyph-bank hashes belong in the run configuration.

For a small direct-real actor, the useful current subset is:

- `body_sin_2angle`, `body_cos_2angle` and
  `body_image_angular_rate_radians_per_second`.
- `body_to_terrain_spans`, `body_terrain_slope_image_right` and
  `body_terrain_y_relative_spans_at_1`, `_at_2`, `_at_3`.
- The eight corresponding validity bits and actual preceding OS gas/brake bits.

The body span is the maximum dimension of its visible color-component bounding
box. It is an image normalization, not a wheelbase or world-unit calibration.
Terrain is the visible turf/soil boundary, not a verified collision surface.
Positive terrain y is down the image; "right" describes the screen direction.
Body color shape can change with vehicle rotation, articulation and occlusion.
Axle/body angles are modulo pi; doubled-angle encoding preserves this ambiguity.
Image angular rates are invalid on large aliased jumps, missing preceding visual
components or capture intervals outside 10–500 ms. They cannot identify a full
rollover. Image translation includes camera motion. No camera/world displacement
is supplied. Wheel-dependent features remain independently masked when absent.

HUD progress is kept separate from signed position. Only consecutive accepted,
nondecreasing readings produce a progress increment. Missing readings break the
reward interval; decreases are rejected until the prior displayed maximum is
recovered or an explicit episode reset occurs. A constant display does not imply
zero current vehicle speed. An accepted maximum is an observed-progress statistic,
not a guaranteed score if OCR is wrong or the terminal display is missing.

An initial direct-real comparator can use two small linear pedal heads and CEM
episode parameter search on this declared feature subset. Each candidate must use
the same action timing, episode budget, vehicle/map, guard policy and scoring
eligibility. Log HUD coverage, final observation gaps, unknown intervals and manual
score checks; do not rank candidates primarily by OCR missingness. A qualified
terminal score needs an independently checked result-field reader or a preregistered
coverage protocol. Keep this comparator distinct from the surrogate PPO/CEM
experiments. A teacher/student transfer experiment requires aligned observable
features and separately measured real-image performance before any transfer claim.

## Saved-frame evidence on 2026-09-11

Source corrected-arrow gas attempt `72d935d4-122f-48c7-a22f-3879504e23b0`
contains 38 frames over eight seconds. Manual inspection shows 0 m in frames 0/12,
17 m in frame 24 and 46 m in frame 37; the latter two were labeled before OCR.
GAS is visibly depressed in the later frames and RPM rises. This is scripted
control responsiveness evidence, not learned-policy performance.

Initial 38-feature bridge run `51bcbbd6-b2ac-4c83-a864-927e85292941` used the
frozen glyph bank from `ae953116-cd5f-4f47-8313-d7350e36001a` without adaptation:
body supported 38/38, wheels supported 15/38; HUD accepted 21/38 and rejected 17/38.
The four manually labeled held-out spots had two exact accepted readings (0/0)
and two explicit unknowns (17/46), with no wrong accepted reading among those four.
This small, selected, correlated sample is not a general accuracy estimate. The
last accepted HUD value was 42 m, which must not replace the manual 46 m observation.
Processing median 57.9 ms and p95 71.4 ms exclude file decoding and capture; those
times alone do not establish the live loop rate. Body-only features were added in
the next schema versioned run; the original sealed run remains unchanged.

The final 49-feature run `d909cc24-29aa-4d0a-8335-3379de56340b` retains the same
38 source frames and OCR outcomes. Body angle support is 38/38, consecutive body
angular-rate support 37/38, body-to-terrain and one-span slope support 27/38;
two-span and three-span terrain lookahead support 24/38 and 18/38. These are
measurement availability counts, not measured coordinate accuracy. Processing
median 66.3 ms, p95 75.7 ms and maximum 106.6 ms include the body features and OCR,
but exclude capture and image decoding. The run seal verifies successfully.

Separately, corrected-arrow four-state attempt
`06a5ab40-7201-43e5-a5a8-137325c4c38d` visibly includes all four pedal shapes:
35-frame exploratory upper-rim measurements yield six neutral, eleven gas-only,
eleven brake-only and seven both-pedal frames. Manual spot checks independently
confirm frames 0/7/18/28/33 as neutral/gas/brake/both/gas. A pressed plate narrows
from 137–138 px to 86.5 px at the inspected upper-rim rows. Images show animation
lag relative to OS insertion: the first both-pedal capture starts 460 ms after
the brake-down completion, while the preceding capture still shows gas-only.
These are acquisition-time relations, not presentation-latency measurements.
The ignored `artifacts/probe-analysis/06a5ab40-pedal-acknowledgment.json` stores
every source hash, capture interval, visual diagnostic and preceding OS state.

`scripts/measure_screen_features.py` reproduces offline observations, registers
source frame/profile/glyph/annotation hashes and seals a zero-episode analysis run.
It can join prior successful keyboard insertion events by capture-start time; it
does not mistake them for game acknowledgment. All game images remain ignored
local artifacts. Unit tests use synthetic evidence to verify masks, temporal
gaps, modulo-pi derivatives, missing terrain samples and reward continuity.
