# Observed reset flow: implementation requirements

The generic screen session currently permits only a separately verified restart
button on a recognized RESULT screen. It cannot yet complete the actual observed
Google Play Games end flow. Do not connect the live game to that generic restart
callback or label a multi-step click sequence as one verified restart.

The operator observed the following optional route: SECOND CHANCE revive offer
decline X → terminal result → TOUCH TO CONTINUE → Tune/selection → START → fresh
PLAYING. The offer is a purchase/reward-video choice, not an advertisement-close
control. Its decline X must remain separate from `legitimate_ad_close`; elapsed
time alone must never grant any click. Starting from a paused game has a separate
observed manual Restart route, which is outside the current RESULT-only contract.

Before implementing automatic reset, collect and independently verify:

| State | Permitted named control | Required evidence and next-state check |
| --- | --- | --- |
| Recognized revive offer | `decline_revive_offer` | Visible decline X, offer-specific state, fresh bounded control location; transition to terminal result |
| Recognized terminal result | `continue_result` | Read and retain final measured distance/reason first; fresh continue region; transition to Tune/selection |
| Recognized Tune/selection | `start_episode` | Intended vehicle/map identity and visible Start button; exclude adjacent upgrades/shop/currency controls; transition through starting to PLAYING |
| Recognized PLAYING | Pedal policy | Fresh gameplay evidence, verified pedal locations, new episode boundary, reset history |

Each control requires its own allowlist entry, state/role label, normalized bounds,
profile/reference hash, confidence threshold, fresh frame timestamp, pinned physical
geometry, and focus check. A visible general state template alone is insufficient.
After each click, release controls and wait for the expected next state with a
bounded deadline. Do not replay a click on repeated frames. Unexpected states,
ambiguous controls, changed geometry, unavailable terminal measurements, ad offers,
or exceeded click/episode/session limits stop the loop and preserve evidence.

Tests must cover the optional absent offer, repeated frames, missed transitions,
ambiguous X controls, a lookalike reward/ad button, purchases adjacent to Start,
outdated bounds, focus loss, Escape, capture/click exceptions, and release failure.
Those tests and held-out visual evidence are prerequisites for implementation.
No reset-flow extension or new live click permission is implemented by this note.
