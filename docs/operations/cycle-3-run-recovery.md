# Cycle 3 training failure accounting and continuation

The linked [diagnostic record](../../research/diagnostics/e911764e-accounting.json)
corrects interpretation of `e911764e` without changing its sealed files. Its
exception was `KeyboardInterrupt`; retained evidence does not identify who or what
raised it. The final zero counters are a runner accounting defect. Its metric
journal establishes lower bounds of 1,433,600 transitions, 1,663 completed episodes,
and 11,200 completed optimizer steps at 122.2213 learner seconds. No model or
observer report survived in that run. This is not evidence of a rendering failure.

PPO and CEM now retain a `TrainingProgress` object outside the learner stack. Counts
advance immediately after each returned environment operation and completed
optimizer operation. A failed operation may have produced unobserved partial
work: its phase remains explicit and it is excluded from completed counters.
An exception inside an optimizer step makes current weights uncertain, so recovery
retains the preceding valid checkpoint instead of serializing those weights.

The canonical run includes fsynced `progress.jsonl` snapshots at approximately
one-second intervals and at initialization, checkpoints, and orderly termination.
Snapshots include the last registered checkpoint path and hash. A live or abruptly
terminated run exposes these as `persisted_completed_operations_lower_bound` through
the existing datastore/dashboard query. The last partial journal line is ignored.
This does not automatically seal or repair an unfinished run. Abrupt termination
can lose work since the last complete snapshot; disk failure can prevent even the
last snapshot from being saved.

On handled failure, finalized counters reflect observed completed operations,
the original exception and traceback are retained, and observer shutdown and
recording run on failure as well as success. `KeyboardInterrupt` gives new runs
status `cancelled`; historical status is unchanged. Visualization problems appear
under `summary.observer.status = degraded`; learner status is recorded separately.
If storage fails during finalization, the original learner exception remains
primary, its exception note names the secondary failure, and the run stays unsealed.

Checkpoint format 2 publishes through a flushed temporary file, restricted-loader
readback, and atomic replacement. Serialization failure preserves the previous
complete destination. Initial, scheduled, final, and safe interruption checkpoints
use separate filenames within each run; publication and artifact registration have
their own failure boundary. A file published immediately before process termination
may not yet have a registered manifest entry and requires explicit verification.

PPO continuation remains a **warm start**: weights and available optimizer state
are restored, a new environment is reset, and this run's counters restart. The
manifest specifies fixed normalization, a feed-forward actor, captured Torch/CUDA
and sampler RNG states, and absent environment/rollout state. Captured RNG states
are not restored by the warm-start API. CEM checkpoints support inference only.
Version 1 policies remain readable with explicit legacy warm-start metadata;
unknown versions fail with an instruction to export using their producing release.
Neither format promises exact or bitwise resume.

Focused verification command (use a fresh temporary directory):

```powershell
.venv/Scripts/python.exe -m pytest tests/test_training_recovery.py tests/test_learning.py tests/test_records.py tests/test_headed_training.py tests/test_algorithm_imports.py -q --basetemp artifacts/cycle3-tests-accounting-review
```

Tests inject learner exceptions and Ctrl+C after environment work, interrupted
optimizer mutation, partial serialization, termination before atomic publication,
partial progress tails, disk exhaustion during finalization, display closure, and
video failure. Native input release is validated separately by control/session
tests; these simulator tests never issue gameplay input. A blocked FFmpeg pipe
remains the separately documented headed-observer limitation and is not a cause
established for `e911764e`, which requested no video.
