# Cycle 3 private evidence preservation

An inventory proves which bytes were inspected. A verified archive proves that
those bytes can be read from the archive. Only a successful restore test supports
the `archive_and_restore_verified` status. None proves independent hardware or
offsite durability. The destination must be selected by the owner before archiving
real local evidence; paid storage and public upload are outside this workflow.

The implementation copies complete sealed canonical run directories, including
failed runs, and checks their seals again after restoration. It also preserves
known auxiliary evidence directories, historical local-inventory file selections,
top-level artifact files, configs (including perception profiles), research,
documentation, and experiment definitions. A manifest records exactly which
regenerable test directories were excluded and which other artifact directories
were not selected. Review those lists before creating the archive. Add required
new evidence paths with repeated `inventory --include relative/path` arguments.
Pinned source checkouts such as `artifacts/cycle3-native-source` are not traversed.

Use a new manifest path for every cutoff. The commands refuse existing inventory
and archive files, and restoration requires a destination directory that does not
exist. Keep the external frozen manifest alongside the archive; verification
requires its exact original bytes. Paths, duplicate/case-colliding members,
symlinks, Windows junctions/reparse points, and alternate data stream names are
rejected. Files are extracted manually under the destination; archive path names
are never handed to a bulk extractor. Every byte count and SHA256 must reconcile.

After source is committed, freeze a new cutoff without changing earlier inventories:

```powershell
.venv/Scripts/python.exe scripts/archive_evidence.py inventory --output research/experiments/cycle-3-preservation-001.json
```

For the concrete archive and restore commands, replace the example destination
with the owner-selected location. The archive is uncompressed ZIP64, making its
required space approximately the manifest's `total_bytes` plus archive headers;
the independent restore needs that space again. No actual destination has been
selected or actual archive/restore claimed by this document.

```powershell
.venv/Scripts/python.exe scripts/archive_evidence.py archive --inventory research/experiments/cycle-3-preservation-001.json --output E:/GradientClimbEvidence/cycle-3-preservation-001.zip
.venv/Scripts/python.exe scripts/archive_evidence.py verify --inventory research/experiments/cycle-3-preservation-001.json --archive E:/GradientClimbEvidence/cycle-3-preservation-001.zip
.venv/Scripts/python.exe scripts/archive_evidence.py restore --inventory research/experiments/cycle-3-preservation-001.json --archive E:/GradientClimbEvidence/cycle-3-preservation-001.zip --destination E:/GradientClimbEvidence/restore-test-001
```

The restored `restore-receipt.json` contains archive/inventory hashes, run seal
verification, and the limited durability claim. A failed restore leaves
`.restore-incomplete.json` and partial output for inspection; it never overwrites
an existing evidence tree or announces success. Source changes after the frozen
cutoff fail archive creation rather than silently updating the inventory.

Portable verification uses synthetic private fixtures and no game input:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_evidence_archive.py -q --basetemp artifacts/cycle3-tests-archive-review
```
