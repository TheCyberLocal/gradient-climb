# F-002 — Actual input and scored baselines work; unattended learning is unproven

Evidence domain: **actual game observation/control**, with scripted baselines and failed learning attempts identified separately. Status: bounded observed result, Cycle 1 paused. Configuration was Hill Climber/Countryside with engine 13/13, suspension 14/14, tires 16/16 and drivetrain 10/10. User bindings were Right Arrow gas and Left Arrow brake. No learned real-game competence is demonstrated.

## Actual input and capture evidence

The corrected ordinary Windows extended-arrow scancodes visibly actuated all four joint pedal states in [06a5ab40](../../artifacts/runs/06a5ab40-7201-43e5-a5a8-137325c4c38d/run.json). Saved frames and preceding OS transaction intervals must be aligned with capture lag; a requested code in a frame row is not proof of the already-rendered state. Earlier virtual-key/scancode probes did not provide the same acknowledgement, and are preserved as failed exploratory methods. A touch probe visibly depressed gas but tripped the visual guard; it is not a completed control comparison.

The capture/input route uses target identity, foreground/client geometry, ordinary public screen/input APIs, freshness checks, short leases and neutral release on failure. Native inputs/capture are excluded from CI. The observed game window was Google Play Games `crosvm.exe`; normalized 1034×581 observations came from an explicitly recorded client rectangle. Future sessions must rediscover it. [Capture trials](../../docs/operations/live-capture.md) separate API timing from PNG persistence; they are not measurements of true display-to-decision latency or guaranteed new-frame rates. Three CEM startup attempts failed DXcam initialization. Explicit backend selection exists; no backend is proven universally reliable.

## Scored scripted episodes and failed learning pilot

[dad65c73](../../artifacts/runs/dad65c73-370f-4df9-9ff1-071ab9999680/run.json) completed two always-gas episodes with observed gameplay 60.174643/60.122240 s, 204/199 frames and verified paused endpoint distances 458/411 m. Mean and median are 434.5 m at n=2. Score semantics include input-release-to-pause delay; the first episode's prior diagnostic HUD maximum was 450, not its 458 endpoint. This demonstrates two bounded scored/reset flows, not reliability across natural results/ads, learning or the ten-episode competence gate. The generic CI-method string in this record says “one trained policy”; this policy was in fact scripted always-gas, and the interval is only descriptive at n=2.

Natural gas episodes produced saved result readings 229 and 203 m but failed later reset flows. Their original records remain failed. A result reader's success is not retroactive episode qualification.

The latest [CEM pilot 51d2527e](../../artifacts/runs/51d2527e-9274-40ec-a04d-309117de107d/run.json) attempted initial candidate 0:0, the declared gas-biased initialization. It collected 131 frames over 42.239441 observed seconds with 125 dispatches. Two result readings agreed on 289 m, 0.51811325 s apart, minimum glyph quality 0.972358. Parking then encountered an **unrecognized advertisement**, and the adapter made no click. The strict score decision is ineligible/null; `eligible_episodes=0`, `training_steps=0`, no optimizer update and no completed governed hour. Reading 289 m is preserved as diagnostic evidence, not training fitness or learned-agent performance. Recorded clock fields differ: run `wall_clock_seconds=106.604851`, `governed_elapsed_at_stop=102.099027`, `actual_elapsed_before_finalize=107.738070`; they must not be combined as interchangeable durations.

The [screen CEM](../../experiments/definitions/real-screen-cem.json) implementation has 34 parameters: two independent linear heads over eight measured values, eight masks and bias. Body-axis sine/cosine, angular rate, terrain slope/offsets and body-relative clearance are used; earlier history vectors are logged but not direct linear-head inputs. Population 8/elite 3 selection requires eight eligible candidate episodes before an update. No such completed generation exists. The gas-biased initialization is declared, not learned behavior.

## Measurement availability is not accuracy

The [feature bridge](../../docs/operations/screen-feature-bridge.md) retains 49 values and 49 masks per frame, four-frame history, explicit invalidity and timing. It uses screen-relative body spans and modulo-π orientation. Camera displacement/world calibration is invalid; front/back ambiguity, silhouette/terrain heuristics, occlusion and sampling remain limitations.

In [d909cc24](../../artifacts/runs/d909cc24-29aa-4d0a-8335-3379de56340b/run.json), body-axis availability was 38/38, rate 37/38, clearance/slope 27/38, two/three-span terrain 24/38 and 18/38, and wheels 15/38. Median processing 66.3 ms excludes acquisition and image decode. There are no independent coordinate labels establishing pose/terrain accuracy. Missing wheels do not become invented contact truth.

The frozen gameplay glyph bank produced 129/129 available readings on an independent trajectory; only four predeclared manually labeled frames were checked, yielding 4/4 exact selected reads in [b2dcac01](../../artifacts/runs/b2dcac01-ff92-4768-958d-aeedb791d9cc/run.json). This is not 129/129 measured accuracy. Five initial UI templates matched their five construction images; no held-out accuracy follows from that resubstitution. A later native paused frame initially failed the old template, correctly returning unknown.

Result/paused readers require explicit independently recognized UI state and restricted field geometry. Bank [66ce7f08](../../artifacts/runs/66ce7f08-ef45-466c-ac77-33bacd0f0a41/run.json) used native 313 and 229 as construction data after an earlier 229 margin rejection. Frozen 203 evaluation [043f5529](../../artifacts/runs/043f5529-1bdc-4e67-b984-7d8a6416a575/run.json) is source-disjoint but manually checked after OCR was known, and two identical image hashes mean one unique frame/episode. It is not a blinded independent accuracy study. The paused 49 check is one held-out selected frame. Full construction/holdout history and unchanged rejected records are in [scoring-readers.md](../../docs/operations/scoring-readers.md).

## Capture-backend and Torch interaction finding

**Measured observation:** early no-recording capture probes reported DXcam 11.76,
MSS 6.62 and Pillow 4.50 observations/s; the MSS run recording every PNG achieved
1.83 observations/s. These sequential, different-scene runs shared a workstation
with training and are not a randomized backend comparison. API capture cost and
PNG persistence differ; [capture trials](../../docs/operations/live-capture.md)
retain all run IDs and timings.

Three direct-screen trainer starts on source `45f11e8` failed before control with
DXGI error `0x887A0004` at desktop duplication. A baseline capture succeeded between
failures. Paired read-only diagnostic processes using the same capture path failed
with an explicit Torch import and succeeded without it. Their scripts are
`artifacts/diagnostics/capture_torch_isolation.py` and
`capture_import_isolation.py`, hashed in the
[local inventory](../experiments/cycle-1-local-evidence.json). This diagnostic pair
is an observed engineering check, not a replicated canonical performance experiment.
After lazy algorithm exports removed the unnecessary Torch import, the clean
`f8183772` native CEM pilot captured and controlled its initial episode.

**Interpretation:** on this workstation, unnecessary Torch initialization was
associated with the DXcam startup failure. Lazy imports are a practical bounded
fix for screen CEM, and explicit `dxcam|mss|pillow` selection makes future choices
auditable. **Unresolved:** the exact driver/DLL mechanism, cross-machine prevalence,
Torch-dependent student capture reliability and matched backend-policy performance.
No automatic fallback or OS graphics change was used to manufacture a success.

## Decision at pause

The operational frontier is reliable qualified endpoints and parking across observed states, particularly advertisements. The scientific frontier additionally needs independent visual labels, a completed real learning update and checkpoint evaluation, and then actual hour/transfer/generalization/adaptation evidence. The [resumption plan](../../docs/operations/resumption-plan.md) makes these dependencies explicit. Adding algorithms, relaxing score eligibility or turning unknown ads into speculative clicks would not resolve the present evidence gap.
