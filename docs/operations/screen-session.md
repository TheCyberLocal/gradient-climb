# Generic screen episode contract: live use unvalidated

This document describes the generic `ScreenEpisodeSession` abstraction. The
separate native adapter and executable episode runner have later bounded real-game
evidence; see [native operations](native-game-adapter.md). The generic abstraction
itself was tested with synthetic callbacks and is not the native pilot's loop.

`gradientclimb.control.session.ScreenEpisodeSession` supplies a bounded,
fail-closed episode loop. It is tested with synthetic frames, mock controls and
injected clocks. There is no live-play CLI. These tests do not establish
unattended Hill Climb Racing operation, perception accuracy or real-game competence.

Supply a `PedalController`, a `CapturedFrame` capture callable, RGB classifier,
pixel/action-history encoder, policy returning `PedalAction`, foreground checker,
named-control detector and guarded click callback. Native implementations must
be separately validated on labeled game observations before connecting them.
The session closes the controller on every exit and cannot be reused. `stop()`
requests cancellation at the next polling point.

```python
from gradientclimb.control.session import ScreenEpisodeSession, SessionLimits

session = ScreenEpisodeSession(
    controller=controller,
    capture=capture.grab,
    classify=recognizer.classify,
    encode_observation=validated_encoder,
    policy=policy_callback,  # Returns PedalAction(gas, brake, lease_seconds).
    verify_focus=verified_foreground_callback,
    locate_control=validated_named_control_locator,
    click_control=guarded_named_control_click,
    limits=SessionLimits(max_seconds=60, max_episodes=3),
    recorder=run,
    episode_metrics=validated_terminal_metric_reader,
    on_session_end=save_session_summary,
)
result = session.run()
```

This is an integration contract, not runnable live-game configuration. The
listed callbacks must be provided and validated; none are inferred by the session.

The encoder receives tuples containing up to `history_length` current-episode
frames and completed `ActionInterval` records. Each frame passed to the encoder
has a private read-only pixel array. Histories reset between episodes. Independent
gas/brake states preserve 00, 10, 01 and 11. Canonical trajectories record capture,
dispatch and interval-end timestamps, pixel hashes, classifier confidence,
requested lease duration and bounded command duration.

Actual operating-system key-hold duration is not measured. Capture and inference
overhead can leave neutral gaps between expiring leases. These gaps must be
measured and scheduling improved before timing-sensitive live use. The recorded
duration is explicitly labeled as a command-lease duration, rather than an
observed physical key-hold time.

Only a recognized `RESULT` state with restart evidence and a matching fresh
`VerifiedControl(name="restart")` can restart. Advertisement closure requires
visible legitimate-close evidence and a matching
`VerifiedControl(name="legitimate_ad_close")`. Waiting never authorizes a click.
Control detections must match the classified state, frame timestamp, confidence
threshold and current frame bounds. Repeated result frames do not repeatedly
click the same control. Menus, vehicle/map selection, purchases and arbitrary
button names have no click path.

Unknown, low-confidence, stale, nonmonotonic and invalid observations halt input.
Session time, frame count, completed episodes, restart count, advertisement-close
count and UI transition wait time are bounded by `SessionLimits`. Focus and
controller faults are rechecked while waiting. An elapsed wait limit halts the
session; it never bypasses an advertisement.

Callbacks are synchronous and must return promptly. Python cannot forcibly
cancel a blocked callback; the independent pedal watchdog expires the input
lease during such a stall. The click backend must itself guard current window
identity and foreground state. The orchestrator supplies no native desktop calls.

`episode_metrics` extracts measured results from a recognized terminal frame,
including when no restart is available. `on_episode_end` receives the result,
and `on_session_end` receives the final session summary after controller closure.
Failures are retained as explicit status, reason, error and release-failure
fields. A supplied run recorder receives trajectories and episode evaluations
with the `screen-session-unvalidated` protocol. The caller owns run finalization
and must retain the session outcome in the scientific record.

Validation: 41 focused tests cover all pedal states, action history and durations,
episode completion without a restart button, legitimate ad waiting/closure,
control identity/bounds, stale/repeated/future captures, callback failures,
watchdog faults, focus loss while holding pedals, cancellation, count/time bounds
and failed final release. All use mocks; no native game input is sent.
