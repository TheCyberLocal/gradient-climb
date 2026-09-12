# Reviewed human demonstration windows

The offline assembler produces references to already recorded observations and
subsequent observed pedal polls. It performs no capture, pixel extraction, model
training, input injection or qualification. Source recordings, including failed
captures and partial journal tails, remain unchanged.

Implementation: [dataset contracts and assembler](../../src/gradientclimb/datasets/demonstrations.py),
[command](../../scripts/build_imitation_dataset.py), and
[synthetic fault tests](../../tests/test_imitation_dataset.py).

## Freeze the inputs before building

All reference paths use forward slashes and are relative to the project root.
Every reference has the exact file SHA-256. Absolute paths, traversal, alternate
data streams, Windows device names, symlinks and junctions are rejected. Only
explicitly selected canonical source runs are traversed and verified.

Three versioned, extra-field-forbidding contracts govern an assembly:

1. `demonstration-partitions-3.0` assigns each recording `run_id` and original
   `session_id` once, with its original `source_purpose` and one `split`. Use one
   frozen ledger for every consumer. Published ledgers may add sessions but must
   retain every prior assignment unchanged. The pure assembler checks the
   selected ledger and original recording declarations; the publication context
   described below enforces compatibility across this canonical artifact store.
2. `demonstration-segmentation-3.0` pins the source run/configuration/journal
   hashes, reviewer and timezone-aware review time, reviewed profile claims,
   capture-status acceptance, and nonoverlapping gameplay segments. Failed or
   cancelled captures require a nonempty acceptance note. This is separate from
   the gameplay outcome. Each frame in every proposed index span needs its own
   exact PNG evidence reference; sparse selected-image review cannot authorize
   unviewed intervening frames.
3. `imitation-window-plan-3.0` pins that ledger, review files and prior evidence,
selects `train` or `development`, and fixes history/timing/size limits.

The frame and control row bounds apply across the selected sources; each journal
also has its own declared byte bound. Oversized or malformed complete journals
fail assembly rather than being silently truncated.

The source session purpose is checked against the sealed configuration. Training
accepts only `imitation`; development accepts `development` or a whole imitation
session explicitly reserved for development. Human-benchmark, qualification and
actionless construction sources cannot produce windows through this command.
Adjacent frames or episodes within one recording must not be randomly split.

`ReviewedProfile` keeps `vehicle_name`, `map_name`, `game_build` and upgrade
observations separate. Unknown names stay null. Each known field requires
`field_evidence` referencing an entry in the profile's `evidence` list. Displayed
upgrade slots may be named `displayed_slot_1`, etc., without inventing a mechanism
name. Original operator configuration notes and their verification flag remain
separate in the output. Incomplete profile identity produces
`construction_only_profile_incomplete` windows with `training_eligible=false`.
A complete profile is still a review attestation, not proof of independent
qualification or calibrated perception.

## Review boundaries and temporal rules

A segment has `first_frame_index` and `end_frame_index_exclusive`, plus
`started_ns` and `end_ns_exclusive`. Index and time ranges are half-open. Its
`frame_evidence` list must cover every frame in the index span. A conservative
time cutoff may exclude the final reviewed image's target when normalization or
control polling completed after that cutoff. Do not stretch review time merely
to keep an example. `episode_id` may be null; unknown start/end context remains
unknown. A reviewed fragment does not establish a complete episode or duration.

The default history is four actual frames, with no padding. Histories reset at
each segment boundary, excluded/stale observation, or capture midpoint gap over
0.25 seconds. Frame age is measured from acquisition start to RGB availability
and defaults to at most 0.45 seconds. All history images must be available by the
target poll's start. The target and its preceding polling bracket must be inside
the same segment, and target completion must be strictly before the exclusive
segment end. Default maximum target lag is 0.1 seconds; maximum polling gap is
0.05 seconds. The existing conservative pairing helper also rejects a poll that
overlaps observation-ready time or lacks a preceding bracket.

Template states are retained in the immutable source but do not authorize the
dataset. Reviewed segment state controls inclusion. Missing target labels remain
explicit exclusions; the assembler does not replace them with neutral actions.
The target contains observed `gas`, `brake`, four-state `code`, poll start/end,
source index, bracket index and lag. `action_duration_seconds` and
`previous_os_action` remain null. A focused key-state poll does not establish
human intent, OS delivery, continuous holding between polls, or a reaction to
that particular screenshot.

Each `windows[]` entry contains its stable content-derived `window_id`,
run/session/segment/episode IDs, original purpose, split, review reference,
profile, eligibility and latest `history_reset` record. `frames[]` contains
chronological project-relative `path`, `file_sha256`, `frame_index`,
`timestamp_ns`, `started_ns`, `completed_ns` and `observation_ready_ns`.
Consumers must preserve order, masks and reset boundaries when later extracting
features. The existing feature bridge's `previous_os_*` inputs cannot silently
be populated from human key polls.

## Plan and command

After completing actual reviews, freeze a plan with the following shape. Replace
the descriptive paths and digests with real pinned evidence; the example is not
an executable populated dataset registration.

```json
{
  "schema_version": "imitation-window-plan-3.0",
  "dataset_id": "reviewed-imitation-v1",
  "artifact_root": "artifacts",
  "partition_ledger": {"path": "research/datasets/partitions-v1.json", "sha256": "<64 hex>"},
  "reviews": [{"path": "research/datasets/review-v1.json", "sha256": "<64 hex>"}],
  "split": "train",
  "history_frames": 4,
  "maximum_frame_gap_seconds": 0.25,
  "maximum_frame_age_seconds": 0.45,
  "maximum_target_lag_seconds": 0.1,
  "maximum_poll_gap_seconds": 0.05,
  "maximum_source_frames": 10000,
  "maximum_source_control_samples": 200000,
  "maximum_journal_bytes": 134217728,
  "prior_evidence": [{"path": "research/datasets/prior-ledger-v1.json", "sha256": "<64 hex>"}],
  "note": "Reviewed source windows; no model fitting or qualification"
}
```

From the project root, after implementation and concrete protocol are committed:

```powershell
.venv/Scripts/python.exe scripts/build_imitation_dataset.py --project-root . --plan research/datasets/plan-v1.json
```

The command creates a new canonical derived run and prints its identity. It seals
`imitation-windows.json`, the input plan, partition ledger, review files and prior
evidence. Repeated assembly of identical inputs produces the same dataset digest;
new publication runs have distinct run identities and measured resource costs.
An empty accepted set stays an explicit zero-window dataset, not fabricated data.

The summary retains all target exclusions, source row counts, complete-record
recovery and trailing partial-byte counts, action coverage and unique source
frames. Repeated overlapping windows are not independent episodes. Source
collection clocks/resources, unknown human practice, original prior declarations
and builder resource cost remain separate. The new recorder resource window
includes source verification and assembly but excludes imports, initial plan
parsing, partition-authority preflight and work after its final sample. No
whole-command or energy cost is inferred. Source prior evidence must include failed acquisition and other
construction work when applicable; window selection must not erase those costs.

Actionless YouTube/video material belongs in construction evidence with its own
provenance, permission scope and work cost. It cannot become action-labeled
imitation examples. Training and independently evaluated real competence require
their own committed protocols, complete identity/lineage and native reliability
gates.

## Store-level partition authority

[partitions.py](../../src/gradientclimb/datasets/partitions.py) exposes
`partition_publication(project_root, plan)` for dataset, geometry and future
training/fitting publishers. Acquire it before constructing a recorder, put
`binding.configuration(config)` in the recorder configuration, call
`binding.register(run)` before consuming source material, and hold the context
until the run is sealed. Both current publishers follow this sequence.

A nonblocking OS-held lock serializes compatibility checks through final sealing
for one artifact root. It uses a Windows byte-range lock or POSIX `flock`; its
scope and file are separate from native input ownership. A concurrent publisher
fails immediately. Closing the handle or exiting the process releases ownership;
there is no stale PID marker to delete. The reserved one-byte
`artifacts/.imitation-partitions.lock` is operational metadata, narrowly excluded
from default evidence inventories. Unexpected bytes at that path are an error,
not silently excluded evidence.

The authority comes from registered `whole_session_partition_ledger` artifacts
in verified canonical runs. Proposed ledgers must retain the union of previous
assignments unchanged, including reserved sessions not selected by a particular
dataset. Additional sessions are allowed. Both recording-run and session aliases
are checked. A failed or cancelled sealed publication still binds its registered
ledger. An unfinished publication blocks subsequent publishers until explicit
recovery/audit; releasing an OS lock does not erase possible earlier data exposure.
This includes the recorder's initial `run-start.json` identity when final
`run.json` has not yet been written.
Corrupt historical seals or missing registered ledgers likewise fail closed.

This enforcement covers cooperating publishers in the same canonical store.
Pure `assemble_windows()` does not mutate or lock a store and makes no global
assignment claim. External data consumption, other stores and legacy consumers
without the shared publication context require separate provenance auditing.
Never delete or rewrite historical runs to clear a partition conflict.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_imitation_dataset.py tests/test_imitation_partitions.py --basetemp artifacts/cycle3-tests-imitation-review -q
.venv/Scripts/python.exe -m ruff check src/gradientclimb/datasets scripts/build_imitation_dataset.py tests/test_imitation_dataset.py tests/test_imitation_partitions.py tests/demo_dataset_fixtures.py
```

Fixtures use synthetic bytes and recorded timestamps. They test identity/hash
failures, protected purposes, session aliases, reviewed coverage, failed-capture
acceptance, partial tails, temporal/segment cutoffs, history resets, deterministic
output and sealed publication. No real recording is consumed by these tests.
