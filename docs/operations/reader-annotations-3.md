# Offline publication of reviewed reader labels

`scripts/seal_reader_annotations3.py` publishes an explicit reviewed list of
`ResultAnnotation` records into a new zero-episode canonical run. It neither
decodes images nor infers label text, UI state or human truth. No annotation or
fitting of actual records is authorized by adding this implementation. The
governing `ReaderProtocol3` must already be registered before use.

```powershell
.venv/Scripts/python.exe scripts/seal_reader_annotations3.py --project-root D:/Projects/gradient-climb --plan PATH_TO_REVIEWED_ANNOTATION_PUBLICATION_PLAN.json
```

The public helper is `publish_reader_annotations3(project_root, plan)` in
`src/gradientclimb/experiments/reader_annotations3.py`. Its strict
`ReaderAnnotationPublicationPlan3` has these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `reader-annotation-publication-3.0` |
| `artifact_root` | Project-relative canonical store, default `artifacts` |
| `protocol` | Exact project-relative path and SHA256 of registered `ReaderProtocol3` |
| `sessions` | Whole-session `AnnotationSession3` entries |
| `annotations` | Explicit reviewed `ResultAnnotation` objects, without inferred additions |
| `note` | Required reviewer/publication context |

Each session entry supplies `session_id`, `purpose` (`construction`, `development`
or `heldout`), exact `run_ids`, `declaration_run_id`, and a hash-pinned
`exposure_audit` reference. The existing session declaration API verifies exact
source membership, original source configuration bindings, actual seal chronology
and selection bytes. Failed original captures remain failed and may supply their
retained registered artifacts; missing source episode images are not invented.
Missing original session IDs remain null in the index. A prospective reviewed
group does not establish an originally recorded independent session.

Every annotation pins an original registered frame path/hash, its source run and
session, complete declared source hashes, reviewer and aware `labeled_at` time.
Only a readable result has digits-only `text` with `is_result=true`. Unreadable
results use `is_result=true,text=null`; non-result scenes use
`is_result=false,text=null`. Labels must follow sealing of the entire supplied
source session and precede its exposure-audit cutoff. Every extra source hash
must belong to a verified source artifact in the supplied sessions. Duplicate
label IDs, original-frame hashes and resolved canonical endpoint units are
rejected. Free-form `sampling_unit_id` does not override original native attempt
metadata. Only construction permits an explicit legacy-artifact fallback where
canonical endpoint metadata is absent; that fallback cannot qualify.

New publication requires an explicit `ExposureAudit.prediction_exposure_status`.
`known` with a null `first_prediction_exposure_at` preserves known exposure whose
first-view time is unresolved. `unknown` preserves unresolved exposure. Neither
means absence, and construction labels may be post-prediction. Heldout publication
requires `none_reported`, an audit covering label times, a candidate frozen before
its immutable session declaration, and that declaration sealed before acquisition
and embedded in the original acquisition configuration. It also checks canonical
prior exposure and requires all first-retained terminal endpoints and all frozen
negative sampling units. Attempts lacking retained terminal pixels remain
separately visible. Future independent qualification still requires the complete
reader split, candidate/source ancestry, chronology and fresh exposure review;
publication alone establishes neither blinding nor correctness.

The returned `annotation_run_id` and `annotations` list are the exact registered
private paths/hashes accepted by the reader construction/evaluation plans.
Individual label JSON files use the existing `ResultAnnotation` contract. The
`reader-annotations.json` index retains unit resolutions, source statuses,
declarations, audit references and heldout selection coverage. It declares that
its owning completed seal is required. Successful helper return includes that
verified seal and `publication_state="completed"`; qualification, independent
truth and blinding flags remain false. Do not infer success from a partial index
or an individually retained label in a failed run.

A shared nonblocking canonical-store publication lease covers validation through
the final seal. The recorder begins after minimal JSON/root parsing and before
full plan, protocol, source or label validation. The plan is bounded to 32 MiB;
the registered label count and total source-byte bound apply to verified named
dependencies, original frames and serialized labels. Hashes and source seals are
checked without decoding. Source metadata, protocol, selection and audit bytes
are retained; frames stay at their immutable original paths.

`reader-operations.jsonl` flushes/fsyncs source checks, label-write starts,
completed publications and failures, with sequence numbers, counters and elapsed
and process CPU cost through each event. Earlier completed labels and incomplete
write bytes remain in the failed canonical attempt. Cleanup faults are attached
to the primary exception. Known or unresolved prior exposure is retained both as
an attestation in the owning envelope and an explicit event, so even event-write
failure or a later replacement audit cannot erase it. This records a prior
attestation, not a new prediction. Annotation publication without such exposure
does not itself constitute model prediction or fitting.

Journal cost overlaps `resources-3.0` recorder measurements; do not add them.
Startup/imports, initial JSON/root parsing and later sealing are outside the
stated resource boundary. GPU measurements remain device-wide, not publication
attribution or energy. Human labeling/review labor and compute are inherited and
unmetered, not zero. This offline operation performs zero image decodes,
predictions, policy decisions, optimizer updates and real interaction.

Synthetic fault and lineage checks:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_reader_annotations3.py tests/test_reader_exposure3.py -q --basetemp artifacts/cycle3-tests-reader-annotations
```

The broader construction/freeze/evaluation workflow remains documented in
[reader-validation-3.md](reader-validation-3.md). Existing sealed labels, readers,
rejected historical readings and prior protocol cutoffs are unchanged.
