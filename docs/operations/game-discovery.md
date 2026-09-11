# Observed Hill Climb Racing installation

Historical initial discovery: 2026-09-11. Later controlled input/capture, scoring
and reset evidence is consolidated in [native operations](native-game-adapter.md)
and the [paused resumption checkpoint](resume-state.md).

Date: 2026-09-11. Google Play Games, `crosvm.exe`, target title starts with
`Hill Climb Racing`. Game imagery remains in ignored local artifacts only.

Observed menu: Hill Climber vehicle selected, alongside Automobile and Motocross
Bike previews. Bottom navigation: Shop, Stage, Vehicle, Tune, Start. Currency and
boosters are visible; those controls are excluded from the permitted interaction
surface. CountrySide was confirmed on the actual result screen. The observed
Tune screen shows MAX levels: engine13/13, suspension14/14, tires16/16, drivetrain10/10.
Game version is not yet verified.

Window geometry changed from 1448×814 to 1034×581 when activation restored the
window. Inputs must revalidate geometry, not retain absolute screen coordinates.
The Computer Use capture initially showed occluding apps until activation.
Google Play Games accessibility exposes the wrapper, not the rendered game UI.

Observed gameplay HUD: fuel icon and upcoming-fuel distance, total coins, current
distance/record markers, pause button, left Brake pedal, right Gas pedal. The
upper-left box is not the final traveled-distance measurement; do not confuse the
78 m fuel-related reading with the result's 26 m traveled distance.

One observational episode with no deliberate gameplay control ended OUT OF FUEL
at **26 m**, on CountrySide. This is a discovery observation, not a governed random
baseline, capture benchmark, or learned-policy evaluation. Start/termination times
were not measured precisely. Local screenshots: game-discovery/result-no-input.png
and game-discovery/second-chance.png under the artifact root.

End flow observed: optional SECOND CHANCE revive-video offer with a visible close
X → OUT OF FUEL result → TOUCH TO CONTINUE. The offer was declined through its X;
no advertisement was bypassed and no reward was claimed. Banner advertisement
regions at the bottom were never targeted. The native control panel initially
showed no configured overlay mappings. The user subsequently confirmed the game's
native Right Arrow gas and Left Arrow brake bindings. At this early snapshot, sustained response and
simultaneous input had not yet been measured; both were exercised later. A temporary overlay Tap draft was
opened during discovery; its placement/assignment was not established. Computer
Use was stopped by physical Escape before cleanup could be verified. No persistent mapping change is established by this observation. On any future
resumption, inspect current controls rather than assuming an old draft is active;
the later successful native tests used the existing arrow bindings.
