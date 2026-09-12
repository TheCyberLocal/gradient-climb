# Cycle 3 native foundation

`native-reliability-2.3` prospectively corrects the 2.2 implementation; no 2.2
study was executed. Its criteria remain 10 consecutive scored successes within
12 attempts, at least 10/12 scored, zero unintended actions and zero intervention.
The old substring check counted `success_unscored`. The new exact class set does
not. Every registered command is checked against count, schedule, horizon, seed,
backend, reset/restart bounds and profile/reader hashes before window discovery.
Old registrations remain readable but require an explicit amendment to execute.

The collector now records acquisition, observation-ready, decision-ready and
input-start/completion times. A session deadline, frame bound and verified-progress
stall are administrative interruptions; they cannot earn full-horizon success.
Observed natural failure has precedence at a simultaneous boundary. The separate
`--long-run` option allows up to 900 gameplay seconds and 28,800 session seconds;
it is not accepted by the reliability protocol. Its 60 s stall check requires
continuous valid HUD progress; a missing read resets the evidence window. HUD
maximum-progress stagnation is not proof of zero signed motion.

Each native adapter checks host budgets at most one second apart, including reset
and ad waits: at least 12 GiB on the system drive, 20 GiB on the artifact drive,
and no more than 8 GiB net drive growth during a session. These conservative margins
are workstation policy, not universal Google Play Games thresholds. Process identity
is checked as well. A fault is latched; recovery cannot clear it. Existing capture
freshness, focus, geometry and watchdog checks remain in force. Other host activity
can consume this growth budget and intentionally stop a run.

Native pedal backends hold a session-wide Windows named mutex. A second collector
fails before sending any input. The OS releases handles on process exit. Ctrl+C
during a possibly delivered press attempts neutral release and retains the original
exception even if cleanup also fails. Sudden OS/process termination still cannot
guarantee delivery of a release to the game; the lease does not claim otherwise.

`tests/test_native_foundation.py` uses synthetic captures/input callbacks; its
Windows-only test exercises the mutex without game input. Real lifecycle and soak
qualification are separate experiments. Passing these tests establishes neither.

For a pinned experiment checkout, pass `--root D:\Projects\gradient-climb\artifacts`
and absolute reader paths. Source provenance follows the working directory; the
artifact store is explicit. UI references must resolve inside that artifact root.
Linking only its `runs` subdirectory into another root correctly fails the existing
containment check, so it is not a supported substitute for `--root`.
