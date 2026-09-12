# Changelog

## 0.1.0a1 — research development

- Canonical scientific run records, Parquet telemetry, DuckDB queries and artifact seals.
- Original vectorized hill vehicle surrogate and wall-clock governed PPO/CEM baselines.
- Independent gas/brake actions with ordered transitions and expiry-based input release.
- Pixel-only game discovery and explicit separation of surrogate and real evidence.
- Primary-source literature review, license audit, benchmark protocol and local dashboard.
- Live headed/headless training observer: one isolated simulator environment follows periodic
  detached policy snapshots (`--headed`, `--headless-record`, `--observer-*`), reusable frame
  sinks (Tk, FFmpeg, null), measured snapshot cost, and the paired
  `headed-overhead-paired-0.1` protocol in `scripts/measure_headed_overhead.py`.
  Observer evidence never fails a finished run; per-episode snapshot provenance matches the
  loaded weights in `best` mode; the observer's CPU telemetry no longer shares psutil state
  with the run recorder; observer seeds inside the held-out ranges 1000–1019 and 2000–2019 are
  rejected; the protocol record is labelled `headed-overhead-protocol` with a seed-level
  paired-bootstrap CI label, validates that the on arm's observer ran, and writes a fresh
  state directory per execution. `telemetry/provenance.py` gained `gpu_sample()`.

This development version does not imply completion of real-game qualification.
