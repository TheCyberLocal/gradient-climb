# Observed reset flow: implementation requirements

The generic screen session permits a separately verified restart button on a
recognized RESULT screen. The inspected game's multi-step flow is implemented
separately in [the native game adapter](native-game-adapter.md), with a distinct
state and named control for each step. Do not label a multi-step sequence as one
verified restart or connect it to the generic RESULT-only restart callback.

The operator observed the following optional route: SECOND CHANCE revive offer
decline X → terminal result → TOUCH TO CONTINUE → Tune/selection → START → fresh
PLAYING. The offer is a purchase/reward-video choice, not an advertisement-close
control. Its decline X must remain separate from `legitimate_ad_close`; elapsed
time alone must never grant any click. Starting from a paused game has a separate
observed paused Restart route, which is outside the generic RESULT-only contract.

The adapter's local profile supplies construction references for these contracts:

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
Mock safety tests and local reference consistency now pass. Held-out visual coverage
and supervised integration evidence are still needed before claiming general
unattended reset reliability. The implementation additionally supports stationary
handoff at PAUSED or Tune, so data persistence cannot silently consume the next
episode's initial seconds. See the adapter document for the specific optional bonus
and advertisement routes, time limits, and remaining coverage limits.
