# F-003 — Live headed training costs about one eighth of learner throughput at 64 environments

Status: **SUPPORTED (bounded)**, Cycle 2, 2026-09-11. Protocol `headed-overhead-paired-0.1`,
sealed comparison run `238ce3a7` (algorithm label `headed-overhead-protocol`), arm runs
`e55b690b`/`e3b10061` (seed 0), `e375b119`/`efe64b07` (seed 1), `46857c13`/`1803c02d` (seed 2).
Implementation: [headed training](../../docs/operations/headed-training.md).

## Observation

Cycle 1 supported headless batched training and headed replay of saved checkpoints. Cycle 2
added a live observer: one private simulator environment driven by periodic detached policy
snapshots on a daemon thread, rendered at the simulator's real-time rate (16.67 frames per
second), with a telemetry overlay, while the N training environments stay headless. Enabling it
leaves the recorded training configuration identical except for a separate observer block; the
resolved training configuration artifact is byte-identical and the training metric series carry
the same keys with or without the observer (tested).

## Measurement

Three paired 60 s PPO trainings (64 environments, seeds 0/1/2, counterbalanced order, each arm a
fresh process and a canonical run), headless-record observer versus no observer, on the Cycle 1
workstation while an unrelated 1,800 s distillation run was also training.

| Seed | Off: environment steps | On: environment steps | Difference | Off steps/s | On steps/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 566,976 | 502,016 | −64,960 | 8,978 | 7,978 |
| 1 | 647,168 | 556,160 | −91,008 | 10,326 | 8,858 |
| 2 | 638,976 | 559,360 | −79,616 | 10,202 | 8,919 |

Relative change in environment steps (on minus off, per seed): mean −12.7 %, median −12.5 %,
range −14.1 % to −11.5 %, n = 3 pairs (seed-level percentile bootstrap; no episodes resampled).
Each on arm rendered 969–977 frames from 12 snapshots and completed one observer episode. The
learner-side snapshot cost was negligible: 3–6 ms of weight copying and 0.2 ms of callback time
per 60 s run. The loss therefore comes from the observer thread's inference, rendering, overlay and
sink work competing with the learner inside one interpreter, not from the snapshot hook.

## Uncertainty

Three seeds bound the estimate loosely (the paired differences differ by up to 26,000 steps) and the
workstation carried concurrent load; the protocol records the absolute figures so a repeat on an
idle machine or at other environment counts can be compared directly. The relative cost should
shrink with more environments per step (each training step does more work between observer
frames) and grow with smaller counts; that dependence was not measured here.

## Interpretation and limitation

Watching training live is affordable but not free: at 64 environments the learner loses about one
eighth of its steps at real-time observer frame rate. Lowering `--observer-fps` reduces the cost
proportionally; a subprocess observer would remove the interpreter contention and is deferred
until the measured cost justifies it. The observer is a visualization surface, never learning data.

## Conclusion

Live headed training is implemented, isolated from learning and its overhead is measured: about
12.7 % fewer environment steps at 64 environments and 16.67 observer frames per second over 60 s.
Governed benchmarks continue to run headless; headed mode is for inspection, and any run that used
it carries the observer block and the measured snapshot counters in its record.
