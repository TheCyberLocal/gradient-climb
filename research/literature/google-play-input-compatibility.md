# Google Play Games ordinary-input compatibility review

Reviewed 2026-09-11 against official Google and Microsoft documentation. This is a focused compatibility review, not evidence that any proposed input was received by Hill Climb Racing. No native interaction, package installation, process access or input dispatch was performed for this review.

## Current evidence and unresolved cause

The project session reports that physical arrow keys work, ordinary Computer Use clicks operate Pause/Restart, and `SendInput` keyboard trials using virtual keys and scan codes returned successful insertion counts without an observed gas/RPM response. A first desktop-touch trial stopped because visual contact feedback interfered with its own observation check. These observations do not identify the mechanism behind keyboard failure, and an interrupted touch trial does not establish touch incompatibility.

The official pages examined do **not** document either a blanket rejection or a compatibility guarantee for Windows `SendInput`, `InjectTouchInput`, or `InjectSyntheticPointerInput` in the consumer Google Play Games client. Queries included the official Android/GPG documentation and Google Play Help for `SendInput`, synthetic input, keyboard, mouse and touch. The absence of a documented guarantee is not proof that a mechanism is blocked.

## What Google documents

**Mouse translation.** Google Play Games ordinarily translates a primary mouse click into one virtual tap. A game can opt into a PC-oriented mouse mode through its own Android manifest; the production client selects its mode at launch. Android mouse handlers can distinguish button press/release and motion. These facts make ordinary mouse input a relevant comparison, but they do not promise that a held Windows mouse button becomes a sustained Android touch in this specific game. The developer-emulator mode switch is documentation for game developers, not a reason to change the installed consumer game. [Google: Mouse input](https://developer.android.com/games/playgames/input-mouse).

**Keyboard mapping.** The Controls Editor can associate keyboard keys with on-screen controls for applicable games. Optimized games instead use their Game Controls settings. This establishes a supported user mapping feature; it does not describe how Windows synthetic events are handled. The user's working native Left/Right bindings are already the relevant configuration and do not need replacement merely to repeat this test. [Google Play Help: Use the Controls Editor](https://support.google.com/googleplay/answer/16263766?hl=en).

**Input SDK scope.** A game must already handle keyboard/mouse through its engine before integrating the Input SDK. The SDK supplies control descriptions and optional remapping to the GPG overlay; it is not a public external automation endpoint for an already installed game. Integrating or modifying that SDK is outside this screen/input project. [Google: Get started with the Input SDK](https://developer.android.com/games/playgames/input-sdk-start).

## What Microsoft documents

| Public interface | Supported behavior and relevant limit | Implication here |
| --- | --- | --- |
| [`SendInput`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) | Inserts keyboard/mouse events into the system input stream. The result counts inserted events. UIPI restricts delivery toward higher-integrity applications, and held keyboard state can interfere. | A full insertion count is not an Android-game acknowledgment. Do not attribute the observed failure to UIPI without evidence, or change integrity/security settings speculatively. |
| [Raw Input](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-raw-input) | Applications register devices and receive device-specific `WM_INPUT`, a different interface from ordinary window messages. | This makes differing input paths a possible hypothesis. It does not establish that this GPG build uses a particular raw-input path or filters synthetic events. |
| [`InjectTouchInput`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-injecttouchinput) | Sends contact arrays to the calling process's desktop/session after initialization. A held touch needs repeated update frames; down, update and up flags must form a valid sequence. | Maintain contact IDs/coordinates and send updates while held. Successful acceptance still does not prove the emulator propagated the contact into gameplay. |
| [`InitializeTouchInjection`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-initializetouchinjection) | Sets contact capacity and contact-visualization mode. `TOUCH_FEEDBACK_NONE` affects injected feedback, but application/control feedback may remain. | Suppressing OS contact visualization is a documented per-context option; it does not guarantee unobstructed observation anchors. Keep independent visual/focus checks. |
| [`InjectSyntheticPointerInput`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-injectsyntheticpointerinput) | Public touch/pen synthesis using a pointer-device handle; supports contact arrays and virtual-screen coordinates on Windows 10 1809+. | An ordinary OS-level alternate touch interface, with no need to inject code into the game. It is an exploratory fallback, not a documented cure for GPG routing. |

Microsoft specifically documents repeated updates for press-and-hold contact sequences, consistent positions on release, cancellation flags and special timestamp/retry rules. The project's current touch backend already sends updates when a held action is renewed. A controlled trial should record the actual update cadence and maximum gap; the API page does not give a universal safe maximum gap that can be assumed from capture FPS. [Microsoft: InjectTouchInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-injecttouchinput).

Posting `WM_KEYDOWN`/`WM_CHAR` messages is not equivalent to entering the Windows input stream. Microsoft's explanation distinguishes posted-message queues from actual input processing and recommends UI Automation or `SendInput` for ordinary automation. It also notes that synthetic input can be identified, which does not show that this game actually rejects it. Replacing the current backend with `PostMessage`, or adding a library that only wraps the same `SendInput` call, has little diagnostic value. [Microsoft / Raymond Chen: You can't simulate keyboard input with PostMessage, revisited](https://devblogs.microsoft.com/oldnewthing/20250319-00/?p=110979).

## Highest-value next comparison

First finish the already prepared corrected touch trial with feedback suppression and valid repeated contact updates. If that still produces no clear game response, test **one ordinary primary-mouse down/hold/up at the visually verified GAS pedal**. This is a distinct route with Google's documented mouse translation and existing evidence that ordinary pointer clicks reach the game's menu UI. Mouse hold semantics in gameplay remain the test question.

Use a bounded neutral–gas–neutral observation sequence, with a short approximately 300–500 ms hold and capture before, during and after release. Keep the current map/vehicle/upgrades, fresh playing state, pinned physical pedal coordinates and target identity. Log the mouse event timestamps, API results, capture intervals and cleanup result. Evaluate visible RPM/pedal response near the event in addition to body motion or displayed distance; gravity and the HUD's possibly cumulative progress make motion/distance alone weak evidence. Repeat from a comparable starting state only when the first comparison remains ambiguous. These durations and measurements are experimental choices, not vendor compatibility guarantees.

Interpret the comparison narrowly:

- A reproducible gas/RPM response supports that one pointer route and its observed hold behavior. It does not validate all keyboard controls or simultaneous gas/brake.
- Response only after release may indicate tap rather than hold handling and should remain explicit in the action model.
- OS success without a visible game response leaves application delivery unverified. Preserve the failure trace rather than counting an accepted API call as a controlled trajectory.

A single mouse pointer cannot independently hold two separated pedal contacts. If a direct pointer test works, separately verify two-contact touch or the existing keyboard route before claiming support for all four joint pedal states. The newer synthetic-pointer touch interface is a subsequent public-OS comparison if needed; do not assume its result from the older touch API.

No recommendation here requires game-memory access, hooks, code injection into the game process, additional drivers, security/integrity changes, a private game API, or ADB. No source/code from the documentation was copied into the implementation.
