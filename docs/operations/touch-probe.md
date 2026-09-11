# Supervised desktop touch compatibility probe

`gradientclimb.control.touch.WindowsTouchPedalBackend` implements only two
explicitly located pedal contacts through the public Windows desktop touch API.
It has no automatic UI navigation, target selection, or menu-click path. Live
Google Play Games support is unvalidated; successful API acceptance does not
establish that the game received or acted on either contact.

The supervised probe CLI now accepts `--input-mode touch`, with `vk` remaining
the default. Before starting or renewing touch, that same fresh normalized capture
must match both BRAKE and GAS label patches within an eight-pixel displacement
and pass the PLAYING anchors. Scores and offsets are retained per frame. Threshold
0.96 accepted the native positive reference (both above 0.9998) and rejected the
native paused reference (both below 0.596), result, revive-offer, and Tune images.
This local construction check is not held-out recognition accuracy. Twenty-four
combined mock backend/probe tests passed, including sealed capture/submit failures
and Escape before any input.

Construct a `TouchPedalProfile` from the currently verified physical client
rectangle, normalized screenshot size, inspected gas/brake centers, and SHA-256
of the reference image. The observed 1034×581 layout has gas at (925,480) and
brake at (170,480). For client rectangle (724,294,2581,1449), these map to physical
desktop positions (3033,1491) and (1148,1491), respectively. Any window movement,
resize, reference change, or mismatch requires a newly verified profile.

The required evidence callback returns `TouchPositionEvidence` containing the
profile hash, capture source rectangle, capture-start monotonic timestamp, and
explicit `playing=True` and `positions_verified=True`. The caller must establish
both booleans from fresh inspected pixels; constructing the object does not
validate pixels. Evidence older than 0.45 seconds is rejected by default. The
backend checks foreground identity and unchanged geometry before and after the
callback. Existing active contacts are canceled on verification/callback failures.
The separate `PedalController` watchdog must use the same fresh gameplay/focus
guard, including physical Escape. Do not use this backend without that controller.

Gas uses contact ID 1 and brake ID 2. Every renewal emits UPDATE for held
contacts, while newly pressed contacts emit DOWN. Releasing one contact sends UP
and includes UPDATE for any contact still held, at its original pinned location.
Failure cleanup uses CANCELED|UP and never introduces a new contact. All API
calls retain their start/end times, exact contacts/flags, acceptance result, and
exception. The native sender retries only ERROR_NOT_READY, at most twice with
one-millisecond waits, and leaves Windows to assign event timestamps. Use one
native touch backend at a time in a process.

Fifteen focused mock-only tests passed, covering Windows x64 structure sizes,
coordinate mapping, four states, held-contact updates, stale/future evidence,
profile mismatch, geometry/focus changes, invalid actions, sender/callback failure,
and release on expired leases or failed gameplay guard. No native touch call was
made during those tests.

Microsoft's public documentation establishes this desktop API and its contact
lifecycle, but not emulator compatibility:

- [InitializeTouchInjection](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-initializetouchinjection)
- [InjectTouchInput lifecycle, held updates, and bounded retry condition](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-injecttouchinput)
- [POINTER_INFO layout](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-pointer_info)
- [POINTER_TOUCH_INFO layout](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-pointer_touch_info)

This is ordinary session-desktop input. It does not access game memory, install
hooks, modify the game, or use private APIs. Only the supervising operator may
run a bounded live comparison after verifying the current frame/profile.

The first touch pilot (`55c8b960-be08-403c-b5d5-5e272da32fff`) stopped safely when
Windows' injected-contact feedback overlapped the Gas label. The backend now
defaults explicitly to `TOUCH_FEEDBACK_NONE` (3) in `InitializeTouchInjection`.
This affects only this process's injected-contact feedback and changes no OS or
game setting. Its mode is recorded in run configuration. The pedal similarity
threshold remains 0.96. A mocked native-API test verifies initialization with
maximum two contacts and mode 3; no real touch call is needed for that test.

The follow-up `7b8d80e5-16b3-43e6-afe2-76521c19ae9b` again stopped under the
unchanged label guard after about 1.7 seconds. The supervising operator inspected
frame 8 and observed the Gas pedal visibly tilted/depressed. This is evidence of
a visible UI response during the injected contact, not yet a validated sustained
control/physics response. The resting-label prototype does not cover this pressed
appearance. A future recognizer needs separately inspected resting and pressed
templates with the same pinned pedal regions and fresh-state constraints; no
threshold reduction or speculative template is implemented here.
