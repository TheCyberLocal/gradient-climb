# Supervised keyboard compatibility probe

`scripts/collect_control_probe.py --protocol gas --input-mode scancode` selects the
optional scan-code encoding. The existing `vk` mode remains the default. Both use
ordinary Windows `SendInput`, the same foreground/identity checks, bounded schedule,
physical Escape guard, and short control leases. This option does not establish that
the game accepts either encoding. Only the supervising operator starts or restarts
the game; this script has no menu-click path.

Microsoft documents that `KEYEVENTF_SCANCODE` makes `wScan` identify the key and
ignores `wVk`. The backend uses `MapVirtualKeyW` with `MAPVK_VK_TO_VSC_EX` (4),
extracts the low scan-code byte, and represents an E0 prefix with
`KEYEVENTF_EXTENDEDKEY`. Thus the mapped arrow encodings are Right E0 4D and Left
E0 4B: key-down flags 0x09, key-up flags 0x0B, and `wVk=0`. Mapping failures and
unsupported E1 prefixes fail before sending any input. The virtual-key path remains
a supported encoding, not a demonstrated bug.

Sources, consulted 2026-09-11:

- [KEYBDINPUT and scan-code flags](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput)
- [MapVirtualKeyW extended mapping](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-mapvirtualkeyw)
- [SendInput insertion count and limitations](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)

The run configuration stores the mode and mapped codes. Every OS trace batch stores
the mode, exact `wVk`/`wScan`/`dwFlags`, call start/end timestamps, and inserted-event
count. A sender exception records an unknown count and the error before cleanup.
An inserted count means insertion into the Windows input stream; it is not an
acknowledgment from the game. No process-memory access, hooks, game modifications,
or alternative injection mechanism are used.

The 29 focused tests in `test_scancode_input.py`, `test_perception.py`, and
`test_control_probe.py` passed with all native input, capture, and focus APIs mocked.
They cover scan-code construction, all four states, held-key preservation, focus
loss, partial insertion, sender exceptions, release cleanup, and sealed failure
artifacts. Ruff passed. These tests do not qualify live game control.

## First four-state trial: OS timing only

Run `ac68937c-f5c2-442b-b4d5-f49c4f6e4df1` used virtual-key mode. Its nominal
schedule was 7.5 seconds and it retained 35 frames. The reconstruction below starts
at the first recorded command and ends at the final release batch completion,
covering 7.508726 seconds. It initializes neutral, integrates the preceding state
until each successful batch completion, then applies that batch's transitions in
order. Actual insertion occurred during each call; completion is the explicit
timing convention. This is not a measurement of the game's pedal state.

| OS state inferred from inserted events | Seconds |
| --- | ---: |
| Neutral | 1.5964061 |
| Gas only | 2.4699011 |
| Brake only | 1.7470920 |
| Both | 1.6953268 |

Six interior neutral spans apart from the planned neutral segment lasted 43.7,
47.6, 211.1, 193.7, 2.6, and 24.2 milliseconds, totaling 0.5228875 seconds. The
planned interior neutral segment lasted 0.4212055 seconds; initial neutral lasted
0.6523131 seconds. Differences from the nominal schedule also include capture and
command-boundary delays.

These additional gaps must not be reported as confirmed lease expirations. All
interior releases occurred 0.021–0.243 seconds after the preceding command, below
the 0.4-second lease. The conservative capture-start freshness watchdog can release
keys before a lease expires, but the original trace did not record each release's
cause. The measured gaps are certain under the timestamp convention; their exact
cause is not.

Frame 7 (gas interval) and frame 13 (both interval) show the RPM needle near the
lower end. They do not independently prove control responsiveness or its absence.
A matched neutral/gas trial and visible response are required before treating the
ordinary input backend as validated for gameplay. Render settings must remain fixed
within that comparison. These probes are supervised system identification, not
learned-policy evaluation or qualified episodes.

The complete derived intervals and SHA-256 references are stored locally at
`artifacts/probe-analysis/ac68937c-f5c2-442b-b4d5-f49c4f6e4df1-os-timing.json`.
The finalized original run has not been modified.

## Scan-code trial caveat found during record review

Trial `cf88e319-fba1-4b4b-adf2-1bc6aeb2abb8` recorded raw MapVirtualKey results
75 and 77 (0x4B and 0x4D), without the expected E0 prefix. Its initial encoder
therefore sent flags 0x08 rather than the intended extended-arrow flags 0x09.
That trial cannot establish that correctly extended arrow scan input is ignored.
The backend now preserves the known extended identity of Left/Right and the other
already allowlisted navigation keys even when the native mapping omits E0. The
raw mapping remains in configuration, while exact effective flags remain in every
OS event. Added regression cases cover both prefixed and unprefixed arrow mapping
results. Previous trial records are retained unchanged.
