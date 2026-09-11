# Real-game integration status and operating contract

Last updated: 2026-09-11. The current implementation supplies capture, control, pixel-estimation, and restricted calibration building blocks. It does **not** establish a validated unattended game agent, a calibrated HCR simulator, or measured sim-to-real performance.

## Confirmed setup and current boundary

The installed game runs through Google Play Games. The discovered window title begins with `Hill Climb Racing`, and its process is `crosvm.exe`. See [game discovery](game-discovery.md) for observational screenshots, changing geometry, and the initial result screen.

The user confirmed the game's native keyboard bindings:

| Pedal | Native key | Windows virtual-key code |
| --- | --- | --- |
| Gas | Right arrow | `0x27` |
| Brake/reverse | Left arrow | `0x25` |

No custom Google Play Games mapping is required. Both keys may be held together. The current testbed is Hill Climber on CountrySide. The discovery session reported upgrade levels engine 13/13, suspension 14/14, tires 16/16, and drive 10/10; retain a labeled screen record when establishing the governed benchmark configuration. Keyboard response timing and all four action combinations still require a measured controlled session.

Computer Use reported a physical Escape stop during the discovery session. Live computer interaction was stopped. The user subsequently asked to resume, but the helper retained its stop latch for the current turn; the next live attempt requires a fresh turn. This is a temporary tool-turn boundary, not a permanent integration blocker. A custom Python backend must not be used to bypass the latched helper. The remaining work below describes the next resumed experiment, not actions performed after that stop.

The observed 26 m out-of-fuel result is discovery context. It is not a controlled baseline, a policy score, or a calibrated trajectory. Five already saved screenshots were subsequently labeled offline and used to construct five local state templates. Their five successful self-matches are a construction check, with zero independent held-out images and no measured generalization accuracy. See [discovery labeling](discovery-labeling.md) for the sealed run and provenance. No timestamped four-state control dataset, held-out real calibration set, or real capture benchmark was collected in this implementation subtask.

## Implemented interfaces

| Interface | Behavior | Evidence available |
| --- | --- | --- |
| `capture.windows.discover_windows()` | Enumerates matching visible windows and reads public process metadata/client geometry | Code and mocked identity tests; no guarantee the installed wrapper is capturable |
| `WindowGuard(target).validate()` | Rechecks handle, PID, process creation time, executable, exact title, and foreground state | Mocked identity-reuse and focus-loss tests |
| `WindowCapture(guard, backend="mss" or "pillow")` | Captures verified physical client pixels; supports normalized crop and output resizing | Mocked frame geometry/timestamp tests; native latency unmeasured |
| `WindowCapture.benchmark(...)` | Bounded frames, latency and CPU metrics, optional PNGs plus JSONL timestamps and hashes | Bounded recording tested with generated pixels; no real benchmark |
| `WindowsPedalBackend(target, gas_vk=0x27, brake_vk=0x25)` | Sends ordinary keyboard transitions; preserves the first pedal when adding the second | Mocked SendInput ordering, partial-delivery cleanup, and x64 structure layout tests |
| `PedalController(backend, is_playing)` | Short action leases, release watchdog, action trace | Unit tests; a live emergency-stop and stale-observation loop remains to be validated |
| `TemplateRecognizer.from_manifest(path).classify(rgb)` | Local labeled templates produce UI evidence; missing, resized, or ambiguous evidence gives `UNEXPECTED` | Synthetic tests plus five real discovery self-matches; no held-out real accuracy and no game templates bundled |
| `ColorGeometryEstimator(profile)` | Optional vehicle color centroid/axis and bottom-connected ground boundary | Synthetic tests; requires a map/vehicle/appearance-specific color profile |
| `fit_calibration(train, heldout, run_id=...)` | Fits eight effective input-response coefficients and evaluates disjoint trajectories | Synthetic recovery against an independent ODE solver; no real-game calibration |

The native backend uses public Windows APIs and normal input. It reads no private game memory, injects no process code, and sends no network traffic. It does not launch or focus the game, click menus, spend currency, or dismiss ads.

## Capture and trajectory contract

Resolve a unique target rather than silently selecting the first of several matches. Capture calls require the selected foreground window and recheck geometry afterward; a moved/resized frame is discarded. Client geometry is read in physical pixels using a temporary per-thread DPI context. A foreground check cannot detect every overlay or focus change occurring between the check and capture.

Each frame contains capture start/end monotonic timestamps and UTC start time. Its midpoint is an estimated sample timestamp, not the game's presentation timestamp. API latency, total capture-path latency, and end-to-end benchmark throughput are distinct. Disk encoding/writing is included in recorded-session throughput. Provider frame-drop counters and GPU impact are unavailable and remain null; missed pacing deadlines are not reported as source-frame drops.

For controlled trajectories, join the backend's requested/delivered transition timestamps, controller lease durations, and capture intervals. Keep all four states `00`, `10`, `01`, and `11`, plus sequential transitions and repeated holds. Keep initialization, menus, advertisements, pauses, failures, and termination reason explicit. Do not estimate actual delivered hold duration solely from the requested lease; the watchdog, OS scheduling, and failures can change it.

Record full resolution for labeling and smaller observations for inference when measuring both. Capture outputs belong in ignored artifact storage and must be registered with producing run ID and hashes. Do not add game imagery to normal Git history or represent it as original MIT content.

## Perception and calibration limits

Template manifests specify exact input width/height and normalized search ROIs. Thresholds are visual similarity scores, not calibrated probabilities. Label state templates separately from legitimate advertisement-close and restart controls. A positive state match cannot grant permission for an arbitrary click. No click controller is supplied by these modules.

The five discovery prototypes use 1034 by 581 screenshots including Google Play Games wrapper chrome and its sidebar. Their tight search regions apply only to that inspected layout; they are not aligned to the native client capture interface. The revive offer is labeled `selection` with substate `revive_offer`, and its X is not an advertisement-close control. Every prototype has state-only scope, with advertisement-close and restart authorization false.

The color vehicle estimator provides a principal axis modulo pi. It cannot distinguish front/back or reliably identify a rollover. It does not provide validated speed, fuel OCR, HUD distance, camera motion, wheel contact, or a deployable simulator observation vector. The terrain estimator expects calibrated ground colors connected to the bottom of its ROI; missing boundaries remain invalid. Validation metrics report coverage so rejecting hard frames cannot silently improve average error.

Simulator training and evaluation use simulator observations. The current real-screen geometry estimator has no held-out labeled accuracy, and its output has not been validated against the policy's observation contract. A successful simulated policy therefore does not establish a working real-game actor; pixel feature construction, coordinate alignment, timing, and their errors remain separate integration work.

Effective calibration fits acceleration and angular acceleration contributions of gas, brake, and simultaneous input, plus linear/angular damping. Actions at index `i` are held between timestamps `i` and `i+1`. Position must be world distance or camera-compensated pixels with stable scale, and pitch must have resolved orientation. Raw car screen-x and modulo-pi color orientation alone are inadequate.

Training and held-out trajectories must have distinct IDs, artifact references, and content hashes. Synthetic and real provenance cannot be mixed. Successful parameter recovery in synthetic fixtures does not calibrate the numerical hill simulator: suspension, contact, slopes, collisions, fuel, and camera dynamics are outside this effective model. A fitted record is versioned and hashed, with separate training/validation errors and these limits attached.

## Readiness checks before a future live experiment

- Explicitly resume after the user's stop and keep the current native arrow bindings.
- Verify the target vehicle/map/upgrades and output geometry against labeled current screens.
- Collect a small controlled capture comparison for MSS and Pillow, including occlusion and resize behavior.
- Label whole independent gameplay/menu/end/ad sessions; tune thresholds on training labels and report held-out detection errors and rejected-frame coverage.
- Supply `is_playing` with current foreground identity, fresh finite confidence, observation age, and a latched stop state. A constant `True` callback is only a test fixture.
- Verify that callback exceptions, release failures, expired leases, process restarts, focus loss, and emergency stop latch input off; report cleanup failures rather than silently restarting.
- Run short observed pedal probes covering all four states and both transition orders. Verify actual key-release delivery before permitting unattended operation.
- Build correctly timed real trajectories, calibrate on one partition, and report error on untouched sessions before training-transfer claims.
- Validate permitted restart and advertisement handling as a separate state machine. Unknown UI releases pedals and halts; advertisements may only close through a recognized legitimate available control.

No checklist item is marked complete by the synthetic tests. The targeted integration suite passed 15 tests in 4.51 seconds using mocked Windows APIs, generated pixels, and synthetic ODE trajectories. Initial sandbox execution hit pytest temporary-directory ACL errors; the same suite passed outside that sandbox without any native input or capture calls.
