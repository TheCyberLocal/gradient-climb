# Read-only human demonstrations

The recorder saves only the selected Hill Climb Racing client pixels and focused
Left/Right arrow states. It never presses or releases a pedal, clicks a menu,
launches/restarts the game, closes an advertisement or changes focus. Native
recording is exclusive with GradientClimb's automated collectors through
`NativeInputLease`. Capture and key sampling stop on lost focus, changed process
identity or changed client geometry. Other keys and other applications' pixels
are not collected; foreground error text is sanitized to avoid retaining another
application's title.

Implementation: [demonstrations.py](../../src/gradientclimb/capture/demonstrations.py),
[record_human_demonstration.py](../../scripts/record_human_demonstration.py), and
[fault tests](../../tests/test_demonstrations.py). Version: `human-demonstration-3.0`.
Synthetic tests establish software behavior. Native timing accuracy, gameplay
segmentation and demonstration quality still need real recordings and review.
Commit the implementation and collecting protocol before real capture.

## Small first capture batch

The owner is the demonstrator, not the data engineer. Keep the game in its current
window geometry and inspect the vehicle, map and four upgrade values before the
first recording. Begin at the Tune/vehicle screen so its appearance is retained.
The existing reference is Hill Climber/Countryside with previously observed
13/13, 14/14, 16/16 and 10/10 upgrades; that history is not a fresh verification.
Use the values currently visible in the configuration note.

After the recorder is validated and committed, capture a short diagnostic session
first, then two ordinary complete runs. The diagnostic can show neutral, gas,
brake and both, including releases and an ordered change while safely playing.
The ordinary runs should retain normal recoveries and failures as they happen.
Keep initial menus, terminal screens and ordinary advertisement handling in the
recording. The demonstrator operates every menu and legitimate close manually.
Further requests should follow measured coverage gaps, not an arbitrary large
demonstration quota.

Start this from the repository root, replacing the note with freshly inspected
values:

```powershell
.venv/Scripts/python.exe scripts/record_human_demonstration.py --root D:/Projects/gradient-climb/artifacts --purpose development --seconds 30 --max-mib 128 --configuration-note "Hill Climber; Countryside; upgrades visually inspected: 13/13, 14/14, 16/16, 10/10; diagnostic controls"
```

For each ordinary training demonstration:

```powershell
.venv/Scripts/python.exe scripts/record_human_demonstration.py --root D:/Projects/gradient-climb/artifacts --purpose imitation --seconds 300 --max-mib 512 --configuration-note "Hill Climber; Countryside; upgrades visually inspected: 13/13, 14/14, 16/16, 10/10; ordinary complete run"
```

1. Run the command. It waits up to 30 seconds for the owner to focus the already
   open game; no pixels or key states are recorded during that wait.
2. Select the game manually, leave the configuration screen visible briefly, and
   start/drive normally using the arrow controls. Do not move or resize its window.
3. After the result screen is visible, switch back to the terminal. Losing game
   focus ends capture and seals the available session as interrupted/failed capture
   evidence; it does not rewrite the gameplay outcome as a crash. The fixed session
   time or payload budget also stops recording. Ctrl+C is supported.

The command prints its run ID, independent session ID and artifact directory.
No upload or publication occurs. PNGs and journals remain in the ignored local
artifact store. The configured payload cap reserves original bytes plus their
canonical registered copies; canonical metadata and reports have additional small
overhead. The shared host guard also enforces C:/artifact-drive free-space margins,
bounded drive growth and process health. It may refuse capture when the workstation
has insufficient space.

## Benchmark separation and prior knowledge

Use `--purpose human-benchmark` for a separately recorded matched human reference.
Those whole sessions are excluded from imitation training and tuning. `imitation`
sessions are training data and cannot be relabeled untouched qualification data.
`development` sessions, including the initial diagnostic, support construction and
validation. The CLI retains the purpose, demonstrator pseudonym, session identity,
source revision/configuration, UI-profile hash and operator configuration note.
The note is explicitly not independent visual verification.

The controller is human and its prior practice time is unknown. The system has
engineered capture/guard/UI prior knowledge. These records are not cold-start
learning runs. A human best distance or an actionless video does not establish
repeatable expert equivalence or action-labeled imitation data.

## Timing, coverage and usable action labels

Every frame retains acquisition start/end, observation-ready time, recording time,
client geometry, backend and hashes. Observation-ready means the normalized RGB
array became available to the recording process; actual display presentation and
human reaction times are unknown. UI template states are diagnostic and never
authorize input. Actor model latency is null because no model drives this recorder.

A separate bounded thread polls only Right/Left high bits. It checks focus/identity
before and after the poll and discards a sample if the target changes during it.
Each sample retains its start/end, per-pedal read times, both pedal states, changes
and any sampling gap. A change is bracketed between observations; events between
polls and the order of two changes in one interval are unknown. These are observed
key states, not exact intended actions, OS delivery events or game acknowledgments.

`causal_action_pairs()` pairs a playing frame only with a control sample starting
at or after that frame's observation-ready time. Unknown/nonplaying states,
unbracketed intervals, excessive lag and control gaps remain ineligible with null
targets. Earlier key states are retained separately as history. A later image
cannot become the observation for an earlier action. This proves recorded temporal
ordering, not that the human reacted to that particular screenshot. Reader and
segmentation validation remain necessary before training.

Identical pixel hashes are flagged; a stationary scene is not automatically a stale
capture. Repeated/nonmonotonic capture timestamps are rejected, and maximum frame
age is bounded. Missed capture schedule slots and control gaps are reported;
source compositor drop counts remain null because these capture APIs do not expose
them.

The summary separates capture wall-clock duration, frame counts, control samples,
changes, polling span and recognized/unknown frame coverage. Demonstration
`episode_count` and `real_interaction_seconds` stay null until reviewed episode
segmentation establishes experience and gameplay duration. The separate
`completed_segmented_episodes` annotation-work count is zero initially. The canonical
run's zero episode-record count has an explicit scope: no completed segmented outcome
records; it does not mean zero gameplay experience. Review the frame timeline to derive actual
gameplay duration, natural cause and verified outcome. Recording length is not
automatically gameplay time, and a failed parking/capture boundary does not erase
an earlier verified result. Episodes measure experience, computation measures cost,
wall-clock measures rapidity, and independent real competence determines value.

## Files, interruptions and reproduction

Each run owns `demonstration/frames.jsonl`, `controls.jsonl` and numbered PNGs,
registered as canonical local artifacts after writers close. The main script adds
the summary, host checks and frozen UI profile. Journals flush complete rows
incrementally. Failed capture, polling, encoding or disk writes preserve previously
completed counters and any partial payload. `read_complete_journal()` returns
complete rows plus trailing partial-byte count without rewriting the source.
Malformed complete rows are errors. Abrupt process termination may leave an
unfinished run for the canonical integrity/recovery workflow; it is not silently
sealed as complete.

The regression suite injects focus/window/capture faults, Ctrl+C, control polling
failure, stale/repeated timestamps, disk budget/host failures, partial journals,
ambiguous simultaneous changes and causal-pair gaps. It also seals and verifies a
synthetic canonical recording with zero qualification episodes:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_demonstrations.py --basetemp artifacts/cycle3-tests-demonstrations-review -q
python -m ruff check src/gradientclimb/capture/demonstrations.py scripts/record_human_demonstration.py tests/test_demonstrations.py
```

These tests do not inspect native key state or capture the desktop. On the managed
Windows sandbox, pytest's private temporary-directory ACL can cause WinError 5;
the same scoped synthetic command passes outside that sandbox. This is a test
environment limitation, not a reason to weaken recorder filesystem checks.
