# Source capture and reproduction limits

New runs retain a sealed `source-state.json` artifact as well as the Git SHA,
dirty-worktree flag, and source-diff hash in the canonical run record. The artifact
contains the exact tracked-source patch for `src`, `tests`, `scripts`, `configs`,
`schemas`, `pyproject.toml`, and `requirements-lock.txt`. It also contains hashes
of non-ignored untracked files only under `src`, `tests`, `scripts`, and `configs`.
Ignored files, artifact folders, personal folders, and symlink targets outside the
source root are excluded. No personal or artifact contents are copied.

The source-diff hash is SHA-256 of canonical JSON with `diff` and `untracked` keys,
where `untracked` maps relative paths to content hashes. Both inputs are retained
so this hash can be recomputed. A failed source query leaves the hash unavailable
and marks the missing component in the artifact. Git queries and file hashes are
not one atomic filesystem snapshot; avoid editing source while starting runs.

Untracked source contents are not copied. A matching clean Git commit is still
required for a self-contained reproduction when untracked files were present.
The tracked patch alone cannot reconstruct those files. A dirty-worktree flag is
reported even when dirtiness comes from files outside the scoped source paths.

Earlier runs, including the first keyboard and touch probes, recorded a source
hash but did not retain its underlying patch or untracked manifest. Their sealed
observations and artifacts remain inspectable, but the exact dirty source cannot
be reconstructed from the run alone. Those finalized records are not rewritten.
Use a clean source checkpoint for major governed training and reproduction runs.

For records created after Cycle 1 stabilization, `hardware.machine_fingerprint_basis`
is `hardware-observation-v2`. The fingerprint hashes the recorded OS, CPU, core
counts, RAM, GPU/driver observations, and that basis marker before adding optional
framework runtime measurements. Importing Torch therefore cannot make the same
observed machine configuration appear to be a different machine. The optional
`hardware.pytorch_cuda_available` and top-level `cuda` values remain diagnostic
observations only; provenance collection does not import Torch to obtain them.

Older records have no basis marker and hashed the entire hardware dictionary,
including Torch availability when Torch was already loaded. Their fingerprints
are not directly comparable with v2 or necessarily with each other across import
paths. Compare the retained hardware fields explicitly; sealed records are unchanged.
Even v2 identifies an observed configuration, not a unique physical host: OS/driver
changes or unavailable hardware queries can change its hash.

There is no separately populated historical `reward_version` field. Source SHA,
retained scoped diff, configuration hash, and saved policy/measurement schema
identify the implemented objective. Configuration already permits an explicit
reward version for future protocols. Existing PPO runs used shaped progress reward
(`0.1 * progress - 0.0004 - 1.0 * terminated`); screen CEM selects eligible distance
at a declared terminal or paused boundary. Neither history is relabeled as a new
objective. Versioned score/time/recovery reward comparisons are deferred by the
[Cycle 1 pause directive](../methodology/cycle-1-pause-directive.md).
