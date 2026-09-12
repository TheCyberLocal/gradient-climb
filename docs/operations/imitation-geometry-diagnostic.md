# Reviewed demonstration geometry diagnostic

`scripts/diagnose_imitation_geometry.py` applies the fixed existing pixel measurer
to every unique image referenced by an accepted window plan. This is a bounded
construction diagnostic before choosing a representation or fitting a human
imitation policy. It does not train, inspect the live desktop, read a HUD, infer
control delivery, or evaluate a policy.

The [window assembler](../../src/gradientclimb/datasets/demonstrations.py)
checks source seals, whole-session partitions, visual-review evidence and causal
timing first. Incomplete profile identity remains construction-only. The diagnostic
then checks image hashes again and measures each unique source frame once, while
retaining how many overlapping windows reference it. A failed detector hypothesis
stays in the report; invalid examples are not silently dropped.

The recorder appends, flushes and synchronizes each frame attempt and completed
measurement to `geometry-measurements.jsonl`. It also records attempted, completed
and failed frame counts in the canonical progress journal. If decoding or the
detector raises an exception, or the operator interrupts, the partial journal is
registered and sealed with the failed or cancelled run. Completed measurements
remain available; the remaining selected frames are unattempted. An abrupt
process termination can leave an unsealed journal, whose complete records provide
a lower bound on completed work. A failed diagnostic has no completed aggregate
report and cannot be treated as full coverage.

After committing the source and concrete window plan, run:

```powershell
.venv/Scripts/python.exe scripts/diagnose_imitation_geometry.py --project-root D:/Projects/gradient-climb --plan <project-relative-frozen-plan.json> --profile configs/perception/hcr-discovery-wrapper.json --maximum-frames 1000
```

The canonical result contains source references, all per-image measurements,
body/wheel/terrain validity counts, processing and decoding times, window reuse,
and inherited capture costs. Recorder resource measures cover their declared
sampling interval, including assembly inside the recorder; per-image timing is a
component, not an extra cost to add again. Per-image timing excludes journal
writes; the recorder's wall clock and sampled resources include those writes.
No image copies enter public source
history. Real gameplay and human-practice prior costs are inherited, not replaced
with this diagnostic's zero *new* game interactions.

Validity counts mean the detector returned a hypothesis under its existing rules.
Geometry accuracy remains null until compared against independently labeled
geometry. Pixel resemblance, valid-feature rate and processing speed do not show
transfer or real competence. Any later feature change needs a new profile/version
and a frozen comparison; the selected method must improve real learning per
elapsed time with compute and experience costs visible.

Focused tests cover unique-frame accounting, retention of invalid measurements,
conflicting identities, changed bytes, path escape and a bounded frame count.
Synthetic sealed PNG recordings exercise the complete assembler and publisher,
including source preservation, zero new episodes and evaluation records,
inherited costs, and partial evidence after decoding, detector or interruption
failures:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_imitation_geometry.py -q
```
