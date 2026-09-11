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

Compose it with an already discovered `WindowTarget`, `PedalController`, and a
required release callback. No input occurs while loading the profile or calling
`observe()`. The normal capture backend is DXcam, normalized to the profile size.
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

`reset()` has a default overall deadline of 60 seconds, at most six clicks, at most
2,000 captures, and at most ten seconds for an ordinary expected transition.
Transitions involving an advertisement allow up to 30 seconds of no-input waiting,
always clamped to the remaining overall deadline. `on_observation` and `on_terminal` must do cheap memory
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
up to 30 seconds. After a recognized result, temporary unknown frames may wait for
up to ten seconds to cover reward-text animation and label occlusion. An initial
unknown screen without this transition context stops the reset. Unknown frames
never authorize a click; fresh recognized state and exact named control are still
required after every wait. No elapsed-time rule grants a click, and no generic X, reward button,
creative, shop, purchase, vehicle, or stage selector is actionable.

Menu clicks use ordinary Windows SendInput with physical virtual-desktop coordinates.
The backend refuses to click while the physical left mouse button is held, checks
the inserted event count, and attempts mouse-up cleanup on a partial/error response.
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

Focused validation currently covers 82 adapter/session tests with synthetic frames
and mock I/O, including cold capture retry, stale recovery, focus/geometry/Escape
faults, exact state-role checks, stationary handoff, a 30-second known-ad wait,
two ad phases, unknown creatives, timeout, callback failure, and partial mouse input.
All 12 local references recognize their intended state and controls; this is a
construction consistency check, not held-out accuracy. The adapter does not verify
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
