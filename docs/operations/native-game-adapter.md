# Guarded native game adapter

`gradientclimb.control.game_adapter.NativeGameAdapter` connects the inspected local
UI profile to normalized captures and narrowly named menu clicks. It contains no
pedal policy, training loop, reward reader, or generic screen-coordinate click API.
The initial profile is construction evidence from this session, not a claim of
held-out classification accuracy or complete unattended menu coverage.

The profile is `configs/perception/hcr-reset-ui.json`; its schema is
`schemas/game_ui_profile.schema.json`. It names hash-pinned full-image variants for playing,
paused, revive offer, result, bonus offer, Tune, and specifically observed advertisements.
Reference images stay under ignored `artifacts/`; their bytes must match the listed
SHA-256 before the adapter opens. The expected frame is 1034 × 581, including this
wrapper's title and sidebar. Each state needs at least two matching local patches,
and each action needs its own exact control patch. The default similarity threshold
is 0.97 with a search margin of four pixels. Known modal evidence vetoes the playing
HUD. No sky/terrain color region is used as a gameplay-state guard.

Result variants use a narrower `outlined_white` matcher: static DRIVER DOWN or
OUT OF FUEL header plus DISTANCE label establish state; Continue is a separate
control. White glyph cores and their neighboring dark outlines are compared while
antialiased boundaries and scene background are excluded. The 0.97 threshold is
unchanged. A result remains recognized while Continue blinks, but no dismissal is
sent until its own glyph patch is visible. This matching mode is restricted to
result variants and rejects references lacking both glyph and outline evidence.

## Advertisement-network chrome (profile version 2)

The Cycle 1 pilot `51d2527e` stopped on a Meta Audience Network advertisement whose
creative had never been seen. Per-creative brand anchors cannot cover an unbounded
creative space, so profile version 2 (`hcr-wrapper-reset-v2`) adds a third matcher,
`white_glyph`, for creative-independent network chrome: the skip-forward glyph at
the top-right of the game area and the muted-speaker glyph at the top-left. Both
are solid white shapes on translucent disks, so the reference patch is binarized
(min channel ≥ 225, channel spread ≤ 25) and compared to the binarized capture by
Dice overlap at the best translation inside the box plus margin; the creative
behind the disk is never evidence. State evidence needs Dice ≥ `glyph_threshold`
(0.90); a control needs Dice ≥ `glyph_control_threshold` (0.95). Scores are ranked
as margins above each variant's own threshold so the two scales never compete.

Chrome variants may carry a single anchor because the glyph shape is the evidence;
they are restricted to the advertisement state. Neither Meta chrome variant carries a
control any longer (see the verified-controls rule below), so both only extend the
bounded no-input wait. Before any advertisement close is clicked the control
must be observed on two separate fresh frames at least 0.15 s apart at the same
location (±2 px); a control that appears once, fades or moves restarts the
confirmation. The offline replay audit (`scripts/audit_ui_profile.py`) classifies
every stored real frame: on 907 classifiable frames the version 2 profile produced
zero disagreements with recorded game states, zero advertisement hits on game
screens, and recognized 52 previously unknown advertisement frames. That is
construction/offline evidence; live coverage of other networks remains open and
any new chrome is added as another hash-pinned variant after the same audit.

## Verified controls, effect-based unintended actions and application restart

Study 2.1 (F-005) showed that recognizing chrome is not verification of what a control does:
an allowlisted skip glyph opened the advertised application's store page in the host browser.
Three rules follow.

- **Advertisement controls are enabled only with effect evidence.** A `legitimate_ad_close`
  control stays in the profile only where a sealed run shows the close followed by a
  recognized game state (`commercial_break_available_close`, run `dad65c73`;
  `admob_tiny_layout_close`, run `78d51e26`). Every other advertisement variant is
  recognition-only: it extends the bounded no-input wait and is then reported as stuck.
- **Unintended actions are measured by effect as well as by allowlist.** The window guard
  records the title and process of whichever window took the foreground
  (`foreground_note`), and every guard-class fault, including one raised inside the
  capture read or first seen by the pedal watchdog through `is_playing()`, is latched and
  recorded in `guard_trace` with its reason. The native runner derives unintended actions
  from the traces alone (`unintended_action_events`): an accepted click (`accepted_ns`,
  stamped when the click was delivered) followed within 5 s by a latched foreground-loss
  event is one, whichever thread observed the loss and whatever exception the runner
  finally saw. The attempt is classified `unintended_action`, the evidence goes into the
  attempt block and the session-wide reliability summary, and the session halts.
- **Stuck screens are escaped by restarting the application, never by clicking.**
  `restart_app(launch)` requires the game to hold the foreground, posts WM_CLOSE to the
  pinned game window (the emulator's own exit path), waits for it to hide (bounded by
  `hide_seconds`, 20 s, and the overall deadline), calls the caller's `launch` (the game's
  Start Menu shortcut), waits for the window to reappear with the same identity and
  client geometry, records who holds the foreground, activates the game, and waits without
  input until a recognized state appears (bounded by `max_seconds`, at most 300 s).
  Exactly three reset errors (`STUCK_RESET_MARKERS`) may be followed by a restart, each
  raised only after a full `ad_transition_seconds` no-input wait: "Advertisement without
  legitimate control persisted" (a recognized advertisement variant without a control),
  "Unrecognized advertisement; no click" (unknown frames after a recognized
  advertisement) and "Unknown screen persisted without recognized context" (an unknown
  screen with no pending click, result or advertisement context, which now waits the same
  bound instead of failing at once). Deadline, click and capture limits, post-click
  transition timeouts and recognized-but-unauthorized states halt the session, so a
  restart can never follow a click closely enough to hide its effect nor interrupt an
  advertisement inside its wait. A latched fault (foreground loss, geometry change,
  capture, click or operator fault) refuses the restart. Every restart is recorded in
  `restart_trace` (`app-restarts.json`) with its relaunch boot frames (`restart-*.png`),
  which never count as unknown-state incidence of the reset flow. The runner enables it
  with `--restart-shortcut`, bounds it with `--restart-seconds` and `--max-restarts`
  (validated against a registered protocol's recovery block), records the policy and the
  marker list in the run configuration, reports restarts per attempt and per session, and
  `--probe-restart --exploratory` seals one restart on its own (probe run `a3422d1c`:
  hidden after 0.5 s, visible 9.5 s after the shortcut, vehicle selection recognized
  31.7 s later).

Compose it with an already discovered `WindowTarget`, `PedalController`, and a
required release callback. No input occurs while loading the profile or calling
`observe()`. The default capture backend is DXcam, normalized to the profile size.
The explicit `capture_backend="dxcam"|"mss"|"pillow"` parameter selects the reader;
both native runners expose `--capture-backend` and record it in run configuration.
No automatic fallback changes the capture protocol after a failure.
Window identity, foreground status, unchanged physical client geometry, Escape,
monotonic timestamps, and a maximum observation age of 0.45 seconds guard every
action. The first stale capture is discarded and one new capture is attempted;
`capture_trace` records the discarded metadata. Two stale captures raise without
returning an observation. `is_playing()` returns false on aged observations so the
independent pedal watchdog releases; a subsequent fresh observation can recover.
Focus, identity, geometry, capture errors, and operator-stop faults latch the adapter
off. `guard_trace` records changes in guard reason, including recoverable staleness.

Useful methods are:

- `observe()` returns `GameObservation(frame, state, confidence, variant, controls,
  pixels_sha256)`; state is `unknown` when the evidence does not match uniquely.
- `is_playing()` is the non-throwing state callback for the pedal watchdog.
- `reset(allow_initial_start=True)` starts from the inspected Tune screen or restarts
  a paused episode, then returns the first recognized playing observation.
- `reset(truncate=True, start_next=False)` pauses a playing episode and returns at
  PAUSED. It does not restart while the caller stores data or updates its policy.
- `reset(start_next=False, on_terminal=callback)` declines an observed revive offer,
  retains a terminal result through the callback, continues through supported optional
  dialogs, and returns at Tune. The next separate reset starts the next episode.
- `click_verified(name, observation)` is used internally and accepts only the named
  controls listed in the profile and their assigned states. It reclassifies the exact
  source pixels immediately before mapping the verified control center into the
  pinned physical client rectangle.

`reset()` has a default overall deadline of 60 seconds (callers may raise it to
120 seconds to cover advertisement sequences), at most eight clicks (bounded at
ten), at most 2,000 captures, and at most ten seconds (bounded at fifteen) for an
ordinary expected transition. Transitions involving an advertisement allow 45
seconds of no-input waiting by default (bounded at 90), measured from the most
recent recognized advertisement chrome and always clamped to the remaining overall
deadline. `on_observation` and `on_terminal` must do cheap memory
copies or bounded fast measurements only: slow work can stale the frame. Returning
`False` from `on_terminal` requests another fresh result frame without dismissal;
the callback repeats within the same overall deadline. `None` or `True` confirms
that the result may be dismissed once Continue is visibly verified. This permits
a reader to require multiple agreeing fresh measurements. Persist the terminal frame and
interpret its score before training uses it; the adapter itself does not read final
distance. Supplying an empty callback is not evidence that distance was measured.
Capture and callback implementations must have their own execution bounds; Python
cannot interrupt an arbitrary blocking callback. Keep the independent pedal
watchdog active throughout.

Supported named actions are pause, paused restart/resume, revive-offer decline,
bonus-offer decline, result continue, Tune Start, and the exact available close on
each registered advertisement. A recognized ad without its own close patch waits
within the overall deadline and grants no input. After an inspected action whose
expected route includes an ad, unknown video/fade/interstitial frames may wait for
up to the advertisement transition bound. After a recognized result, temporary unknown frames may wait for
up to ten seconds to cover reward-text animation and label occlusion. An initial
unknown screen without this transition context stops the reset. Unknown frames
never authorize a click; fresh recognized state and exact named control are still
required after every wait. No elapsed-time rule grants a click, and no generic X, reward button,
creative, shop, purchase, vehicle, or stage selector is actionable.

Menu clicks use ordinary Windows SendInput with physical virtual-desktop coordinates.
The backend refuses to click while the physical left mouse button is held, checks
the inserted event count, and attempts mouse-up cleanup on a partial/error response,
including `KeyboardInterrupt` and `SystemExit`. The original interruption and any
cleanup failure are retained separately; an interrupted menu dispatch latches the
adapter off and releases owned pedals before propagating the interruption.
After a successful click on the inspected 1034 × 581 wrapper, it rechecks the
window/focus/geometry guard and sends one separate movement-only packet to the
verified blank title-bar point (520, 18). This avoids cursor hover obscuring future
control templates. The raw trace labels this `pointer_park`; it contains no mouse
button event. The governed trainer's packet deadline guard also covers this move.
`trace` records each menu action with source-pixel hash, bounds, mapped point, time,
and delivery status. The native sender additionally retains packet/cleanup status.
OS insertion is not game acknowledgment; the following classified state provides
separate evidence of a menu transition. Archive these traces, the UI profile, its
reference images, and reset/terminal frames in the canonical run record.

Focused validation covers adapter/session tests with synthetic frames
and mock I/O, including cold capture retry, stale recovery, focus/geometry/Escape
faults, exact state-role checks, stationary handoff, a 30-second known-ad wait,
two ad phases, unknown creatives, timeout, callback failure, and partial mouse input.
Local reference self-matches are construction consistency checks, not held-out
accuracy; adding a reference does not establish performance on independent frames.
The adapter does not verify
vehicle/map selection from the Tune screen alone: the operator must preserve the
inspected configuration until an independently validated identity reader exists.

The initial supervised 229 m Driver Down result exposed a real failure: the original
313 m reference's DISTANCE label matched at 0.99969, but its Continue rectangle
matched at 0.91284 because the underlying terrain differed. That failure is retained
in run `2cf160db-8ecb-49a7-b61b-77785ef433b3`. The separately captured 229 m image
in `9ba8d77d-b284-4fb0-b3c6-7b6057ce6f99` now serves as a development regression
example: the new state score is 0.99927 and Continue score 0.97606. It is no longer
untouched held-out evidence. The matcher still uses the original 313 m reference;
no completed run or source image was changed.

To add another observed UI phase, save a normalized frame and capture metadata in a
new immutable run, inspect its state and narrowly scoped control, add a hash-pinned
variant, then test positive and negative examples before live use. Existing source
references and completed run records must not be rewritten.

The last Cycle 1 native CEM pilot, `51d2527e-9274-40ec-a04d-309117de107d`, reached
one natural result with two agreeing 289 m readings, then stopped on an unrecognized
advertisement during parking. Its partial evidence is sealed, its episode remained
ineligible, and no optimizer generation completed. That record is unchanged; Cycle 2
answers it with the chrome matcher above rather than by weakening a classifier or
extending an allowlist retroactively. The successful Cycle 1 baseline
`dad65c73-370f-4df9-9ff1-071ab9999680` recorded 458 m and 411 m at verified paused
boundaries after its two 60-second always-gas episodes, including release-to-pause
delay. General unattended reset coverage is the subject of the registered
[native reliability study](../../experiments/definitions/cycle-2-native-reliability.json);
until that study reports, unattended reliability remains unestablished.

The episode runner (`scripts/run_screen_episodes.py`) now pauses and restarts a game
left mid-episode by the operator before its first attempt, records the recognized
result variant with every terminal reading as terminal-cause evidence, classifies
every attempt with the registered vocabulary (`classify_attempt`), records reset,
parking and advertisement overhead per attempt, and freezes the protocol hash plus
Cycle 2 versions into the run configuration when a registered protocol is supplied.
