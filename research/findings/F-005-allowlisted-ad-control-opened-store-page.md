# F-005 — An allowlisted advertisement control opened a store page: study 2.1 failed on an effect-based unintended action

Status: **NOT SUPPORTED (registered study failed its criteria)**, Cycle 2, 2026-09-11.
Protocol [`native-reliability-2.1`](../../experiments/definitions/cycle-2-native-reliability-2.1.json);
session `54d58272` (experiment `native-reliability-study`), one session of the three allowed.

## Observation

Session 1 of study 2.1 started while the game showed the Google-served interstitial at its
third player size (variant `admob_medium_layout_skip`, added after study 2.0 from sealed frame
`55b84b0f/reset-025.png` with the top-right skip glyph as its close control). The adapter
recognized the advertisement on four fresh frames, confirmed the control on two frames at the
same location, and clicked it once (`menu-transitions`: one accepted `legitimate_ad_close`,
about 0.6 s into attempt 0, which itself began about 1.5 s after the run started). The capture
after the click still showed the advertisement (guard trace: "Awaiting expected transition; no
input while unknown"); the following capture failed its post-read foreground check
(`WindowUnavailable: Target is not the foreground window`, raised inside the capture read).
Sealed evidence ends there: the run predates the guard's foreground evidence, so the following
is the operator's unsealed observation made immediately afterwards: the foreground window was
the host browser on the advertised application's Google Play store page ("Whiteout Survival -
Apps on Google Play - Google Chrome", process `chrome.exe`), and the game window behind it
still showed the advertisement's end card with its Install pill (screenshots retained outside
the repository). No purchase, install or further interaction occurred; the page was opened,
nothing on it was touched, and it was left for the operator.

## Measurement

- Attempts completed: 1 of 12 (`recoverable_failure` by the 2.1 vocabulary; attempt duration
  0.77 s including the failed capture), 11 not attempted.
- Unintended actions by the registered allowlist definition ("any click outside the allowlisted
  named controls"): **0** (the click was an allowlisted named control on a verified frame).
- Unintended actions by effect (a click that transferred the foreground to another application
  and opened a store page): **1**. The study reports the stricter figure as binding.
- Manual interventions: 0. Advertisement closes: 1. Stale captures discarded: 1.
- Session 91c3d4f4 (Cycle 2 exploratory) clicked the pixel-identical glyph of the small-layout
  variant and lost the foreground on the next capture; that loss was attributed at the time to
  the operator's window. It is equally consistent with this effect and the attribution is now
  recorded as unresolved.

## Interpretation

Visual identification of chrome is not verification of what a control does. The glyph region
in this layout is drawn by the advertisement's renderer and, on the end card at least, sits
inside the click-through area. Two variants (`commercial_break_available_close`, run
`dad65c73`; `admob_tiny_layout_close`, run `78d51e26`) have sealed evidence that their close
was followed by a recognized game state; every other advertisement control had been enabled on
appearance alone.

## Limitation

One event; whether the same region is a working skip control during the video phase of this
layout is unknown and will not be tested by clicking. The foreground guard detected the effect
within one capture, but it detects only effects that move the foreground; an in-page effect that
keeps the game in front (for example an in-game purchase dialog) is not counted by this measure.
It is caught by the post-click transition timeout ("Unexpected or timed-out reset transition"),
which halts the session with the frames retained and is deliberately not a restart trigger, so
such an effect can never be recovered silently. The browser title, process and end card above
are unsealed operator observations; the guard now records the foreground window on every loss
so the next such event is sealed.

## Conclusion

Registered criteria not met; study 2.1 is closed after one session. Consequences, all applied
before any further data collection and registered in `native-reliability-2.2`:

1. Advertisement controls are enabled only where a sealed run shows the close followed by a
   recognized game state; seven variants became recognition-only.
2. Unintended actions are defined by effect as well as by allowlist: a latched foreground-loss
   guard event within 5 s of an accepted click classifies the attempt as `unintended_action`,
   is counted session-wide in the reliability summary from the adapter traces alone (whichever
   thread observed the loss, whatever exception the runner finally saw) with the foreground
   window's title and process as evidence, and halts the session. Faults raised inside the
   capture read, the path of this very event, are now latched and recorded with their reason.
3. A stuck screen is escaped by restarting the application: a WM_CLOSE message to the pinned
   game window followed by the game's shortcut, never a click inside the game. Exactly three
   adapter errors qualify, each raised only after a full 45 s no-input wait: a recognized
   advertisement without a verified control, unknown frames following a recognized
   advertisement, and an unknown screen with no context at all. Deadline and click limits,
   post-click transition timeouts and every guard fault halt instead, and the adapter refuses a
   restart after any latched fault or when the game no longer holds the foreground. The path is
   sealed in exploratory probe run `a3422d1c`: the window hid 0.5 s after WM_CLOSE, reappeared
   9.5 s after the shortcut with the same handle, process, title and client geometry, and the
   game reached the recognized vehicle-selection screen 31.7 s later (46.6 s in total).
   Restarts are reported per attempt as a secondary endpoint.
