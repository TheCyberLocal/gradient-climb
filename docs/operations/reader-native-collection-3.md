# Native reader acquisition hooks 3.0

The additive hooks are implemented for synthetic verification. The
[amendment](../../experiments/drafts/reader-native-collection-3/amendment.json)
is **DRAFT and non-dispatchable**. Existing native 2.3 commands keep their
behavior. Passing reader acquisition options with native 2.3 or a draft
amendment fails before native objects are constructed.

The two completed reliability sessions remain counted: the foreground failure
`915ac38b-ac38-45f1-bc16-7540f5aa2353` and twelve-attempt session
`1c263031-ac29-4dbc-8fa5-864a2473116a`. One of three sessions remains.
The future amendment preserves twelve attempts, 60-second horizons, a
2,400-second session limit, seed 4401, alternating scripted controls, and every
original profile, legacy reader, capture, input and recovery limit. It cannot
replace the two prior results. A new run-start consumes the remaining dose,
including a failure, regardless of its experiment name.

Before a future dispatch, register and commit the reader protocol and amendment,
pin both implementation files, freeze the candidate, and seal a heldout session
declaration with `declare_reader_session3.py`. The native entry point adds
`--project-root`, `--reader-session-receipt` and `--reader-sampling-source-id`.
The latter must equal `reader_sampling_source_id` in the declaration's original
`source_run_config_binding`. The original native configuration embeds the exact
returned `required_source_configuration`; conflicting native settings fail.
The public `prepare_native_reader_collection` helper verifies the actual receipt
and candidate seals and their chronology, using the same receipt primitives as
reader3. Later source verification is still mandatory.

The selected result population is the first retained **terminal callback** frame
per attempt after that attempt's start reset. Every callback frame is saved,
independently of numeric success, with original capture timing and a matching
`terminal_readings` row. These are capture-call midpoint timestamps, not measured
presentation times. The existing collector can first retain a result-looking
image as `real_game_frame` before the callback; that earlier population is not
reclassified or substituted. Initial cleanup and start-reset results are saved
as `reader_out_of_attempt_terminal_frame`, with their phase and null attempt
identity. They never become new independent endpoints. The new structural path
also works with `reader=None`; registered reliability still requires its original
legacy reader and numeric dismissal behavior.

Non-result selections use the existing `reader-source-selection-3.0` schema.
Declare stages such as `attempt:0:start:tune`, one per chosen attempt, before
acquisition. Each stage means the first recognized Tune observation from the
existing start-reset callback. `capture_ordinal` is the preregistered reader
sampling slot, not an assertion about all camera captures. Its source-local
ordinal and unit ID are unique. The artifact kind is `reader_non_result_frame`;
metadata retains `reader_sampling_unit_id`, `reader_sampling_stage`,
`reader_sampling_ordinal`, UI state and original capture timing. No extra native
capture or click is requested by this hook. A stage never reached remains
missing. A restart never replaces an already retained selection. Recognition is
a collection trigger; independent annotation supplies non-result truth later.

`reader-collection/sampling.jsonl` appends and fsyncs opportunity starts, capture
starts, exact PNG artifact identities, completion and missing opportunities.
PNG writes are fsynced and registered immediately. The final journal is
registered on normal completion and callback/capture failure; a hard process or
storage failure can leave an unsealed journal requiring recovery. Such a run
cannot qualify. Timing rows retained before reader inference preserve endpoint
identity if inference raises. Every attempt, unavailable endpoint and planned
but unattempted index remains in the native source inventory.

The journal opens before declaration dependencies are copied. Scoped setup
ownership closes both controller and backend if adapter construction, recorder
creation or evidence publication fails, including when one cleanup itself raises.
After setup succeeds, the existing native runtime cleanup takes ownership.
Source provenance identifies the checkout containing the executing script.

Opt-in console output contains only run/attempt identity, completed-attempt
counts and recorded/halted status. It excludes scores, numeric validity,
score-derived classifications and exception text. Full original numeric records
remain in sealed canonical artifacts. Keep those artifacts, logs and automatic
summaries unviewed by labelers until **all** annotations are sealed and record
the exposure audit. The console allowlist alone cannot establish blinding.
The candidate is never loaded for online inference by this instrument.

Receipt preflight elapsed and process CPU costs are recorded separately and
exclude imports, amendment/budget validation and recorder initialization.
PNG/journal work is inside native recorder elapsed and resource measurements;
these scopes overlap and must not be added as independent totals. Added callback
time can cause the unchanged freshness/deadline guards to halt. No hardware
speed or capture-fidelity claim follows from the instrument tests. Native
scripted attempts are experience, with no optimizer updates. One new source
session cannot meet reader3's three-session, twenty-positive-endpoint requirement.
Missing examples leave qualification incomplete and authorize no replacement
collection or new human recording.
