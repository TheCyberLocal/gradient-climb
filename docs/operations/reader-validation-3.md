# Prospective result-reader workflow

The additive tools implement `reader-validation-3.0`. The definition in
`experiments/definitions/cycle-3-reader-validation.json` remains a **draft** and
both tools reject it. No new reader has been fitted by adding this workflow.

Native session `1c263031-ac29-4dbc-8fa5-864a2473116a` supplied development evidence:
the right-side result in retained terminal frames 024 and 035 visibly reads
269 m. The frozen bank ranks the middle visible 6 as 9 (0.933115823817292), ahead
of 0 (0.9205834683954619), 8 (0.9181380417335474) and 6 (0.9168026101141925).
The top-two margin, 0.012532355421830066, fails the configured 0.04 requirement.
Reducing that margin would accept 299 m. The original null score stays null.

The proposed minimal extension labels the full 269 region in construction data
and adds its three segmented glyphs to a new bank. Only the 6 appearance needs
new support, but retaining the full region avoids selecting a digit using its
prediction. The inherited 0.84 threshold, 0.04 ambiguity margin, one-pixel
alignment, numeric ROI and DISTANCE anchor are preserved by construction.

The exported contracts and functions are in
`src/gradientclimb/experiments/reader_validation3.py`. Paths in all evidence
references are canonical paths relative to the supplied project root. Symlinks,
Windows junctions, path traversal, missing ancestors and changed hashes are
rejected. The registered 200-label and 512 MiB total-source bounds apply before
construction or inference output is published; free disk must also cover source
copies and bounded report overhead.

Before fitting, register and commit the protocol, assign the entire exposed
session to construction, and produce a `ConstructionPlan3`. Its graph must name
every source ancestor and bind every node to a real file hash: the base and
parent manifests, numeric manifests, glyphs, original glyph frames, anchor,
anchor source and UI references. Every embedded dependency must be in the
closure. Construction labels are individual `ResultAnnotation` JSON files,
registered and sealed in a separate zero-episode canonical label run. Supply the
exact registered artifact paths, their hashes, and that label run's ID.

`ResultAnnotation` accepts only `field="result_distance"`. A readable result has
`is_result=true` and digits-only `text`; an unreadable result has `is_result=true`
and `text=null`; a non-result scene has `is_result=false` and `text=null`. Each
annotation pins its original frame, source ancestry, original session/run,
sampling unit, labeling time and reviewer. Repeated frames or alternate images
of the same sampling unit cannot inflate counts. Do not label a second animation
frame as another episode endpoint.

Each source session has an immutable declaration receipt and a hashed
`ExposureAudit`. `session_declarations` maps session IDs to declaration **run
IDs**, not mutable JSON declarations. Use `SessionReceiptPlan3` and
`declare_reader_session3.py` to seal the protocol and selection bytes. For
construction/development, the receipt names the exact existing source run IDs.
For heldout, it binds the candidate freeze and returns
`required_source_configuration`, which the future acquisition must embed in its
original canonical configuration. Backdating a later document cannot supply this
binding. The actual `seal.json.created_at` is used for declaration, source,
annotation and candidate chronology. A heldout session's purpose must be declared
after registration and before acquisition. Its entire
set of annotation bytes must be sealed before the earliest viewed prediction
from that session. Keep prediction logs and automatic summaries hidden from
labelers until then. An audited absence of exposure has a null timestamp and a
reviewed-through cutoff; an absent audit is rejected.

The current source is a single continuous native acquisition, so the draft uses
its run ID as the explicit construction session identity. This does not permit
splitting future runs from one recording session across purposes. Every future
run-to-session mapping must be declared and audited as a whole-session mapping.

The new evaluator takes an `EvaluationPlan3` with the existing strict
`ReaderSplitManifest`. It verifies the same graph, session seals, annotation
bytes and exposure bytes, and requires the successor's completed construction
run to be sealed before its declared freeze time and before heldout acquisition.
`freeze_result_reader3.py` first publishes a separate immutable candidate receipt
from `ReaderFreezePlan3`, pinning the registered protocol, successor manifest,
UI profile and construction run. Supply its returned `run_id` as
`reader_freeze_run_id`, and its `frozen_at` as `reader_frozen_at`. The evaluator
requires that receipt's actual seal to precede every heldout acquisition; a
self-asserted earlier freeze timestamp is insufficient. The candidate must have
been built successfully under that same registered protocol.
It executes the frozen UI recognizer on each stored image and supplies that
recognizer's state to the result reader. Truth labels never authorize the UI
state, so non-result negative testing does not become a tautology.

The draft calls for at least 20 unique readable result endpoints and **20 accepted
positives**, at least **90% positive availability**, and accepted positives from
at least three new sessions. All ten digits must appear in both labeled and
correctly accepted positive coverage. At least ten non-result negatives and five
unreadable-result negatives are required, with negatives spanning three sessions. Positive
availability, exact accuracy when available, positive wrong accepts and negative
false accepts are separate. Missing classes or digits leave qualification
incomplete. Zero observed wrong accepts is a finite-sample gate; it does not
establish zero population error. All inspected historical material and the
successor's full construction ancestry remain excluded from heldout evidence.

Endpoint identity comes from original `terminal_frame` capture timing joined to
the sealed `episode_summaries[].attempt.index` and `terminal_readings`, not the
annotation's free-form `sampling_unit_id`. Qualification takes the first retained
`terminal_frame` callback artifact of every retained endpoint; this is not necessarily
the first visible result or earliest result pixels retained as `real_game_frame`
by the gameplay loop. Later callback animation frames and omission of
hard endpoints are rejected. Attempts with no retained terminal pixels appear
separately as unavailable source evidence. Non-result sampling uses a sealed
`reader-source-selection-3.0` document with `non_result_units` containing
`source_config_id`, `unit_id`, `stage` and `capture_ordinal`. These must match
original `reader_non_result_frame` metadata and the source configuration's
`reader_sampling_source_id`. Every preregistered negative must be present; capture
intervals cannot overlap. Historical menu crops without that original metadata
cannot become qualified negatives. The next native collector must implement
these declarations and sampling receipts before collection; this workflow does
not fabricate them retrospectively.

After review and registration, explicit commands are:

```powershell
.venv/Scripts/python.exe scripts/build_result_reader3.py --project-root D:/Projects/gradient-climb --plan PATH_TO_REGISTERED_CONSTRUCTION_PLAN.json
.venv/Scripts/python.exe scripts/freeze_result_reader3.py --project-root D:/Projects/gradient-climb --plan PATH_TO_CANDIDATE_FREEZE_PLAN.json
.venv/Scripts/python.exe scripts/declare_reader_session3.py --project-root D:/Projects/gradient-climb --plan PATH_TO_SESSION_RECEIPT_PLAN.json
.venv/Scripts/python.exe scripts/evaluate_reader_labels3.py --project-root D:/Projects/gradient-climb --plan PATH_TO_SEALED_LABEL_EVALUATION_PLAN.json
```

These are offline commands, with no capture, native input, automatic labeling or
publication to an external service. Each produces a new canonical zero-episode
run. Hash checks establish file identity; the truth of session and exposure
attestations still needs review. The old builder, evaluator, Cycle 2 protocol and
all existing sealed outcomes remain unchanged.

Construction and evaluation start a canonical attempt after minimal plan/root
parsing and before source validation, image processing, fitting or predictions.
`reader-operations.jsonl` fsyncs operation starts before consuming each frame and
completed prediction rows afterward. Exceptions and Ctrl+C preserve completed
work, conservative exposure starts, counters and measured elapsed/process CPU
cost. A publication copy failure keeps the partial report without qualification.
The standalone report artifact remains pending with qualification false. After
the operation journal is closed and registered, the final canonical envelope
receives the candidate decision. `load_reader_report3` verifies that envelope and
its seal and grants qualification only for completed runs; the evaluator returns
this verified view. Late journal fsync or registration faults downgrade both
summary reports and preserve their counts with qualification false.
GPU work remains unknown. These event costs overlap canonical resource samples;
do not add them together. The final event precedes journal registration and final
filesystem sealing, so its cost is a lower bound through that stated boundary.
Python startup/imports and minimal argument parsing are excluded, so these are
not full-command costs. An external durable entry receipt
is needed for that broader boundary. Source episodes, image labels, glyph
additions and newly executed interaction experience remain separate; these
offline tools execute zero new environment episodes.

A nonblocking store publication lease covers preflight through final seal. Before
evaluation reads annotation contents it checks earlier canonical reader journals.
Prior image processing, fitting or prediction starts consume that entire original
source session for future blinded qualification, including failed/cancelled
attempts. New names, audits or replacement labels cannot erase this known
exposure. A preflight-only failure can be retried, with the earlier attempt and
its costs retained. External or unrecorded exposure still requires honest review.

Focused checks use synthetic images and records only:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_reader_validation3.py tests/test_reader_receipts3.py tests/test_reader_exposure3.py tests/test_reader_units3.py tests/test_reader_provenance.py tests/test_reader_tools.py -q --basetemp artifacts/cycle3-tests-reader3
```
