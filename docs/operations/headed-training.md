# Live headed simulator training

Watch one representative simulator environment driven by the evolving PPO policy
while the governed training run proceeds unchanged. The observer is an
observation side channel: it is not a training hyperparameter, it never
contributes learning data, and its cost on the learner is measured rather than
assumed.

## Architecture

```
train_ppo (learner thread, governed clock)
  N headless VectorHillEnv ──rollouts──> ActorCritic ──Adam──> updates
  boundaries(): every --observer-interval seconds (first at t≈0):
      detached CPU copy of state_dict; best-so-far tracked; snapshot(payload)   [timed]
                                   │  O(1) reference swap under a Lock
                                   ▼
TrainingObserver (daemon thread "gradientclimb-observer")
  private VectorHillEnv(1, --observer-seed)  +  private ActorCritic (cpu)
  loop: at episode boundary (or first snapshot) load newest payload -> act(deterministic)
        -> step -> env.render() -> compose_overlay(frame, telemetry) -> sinks[*].write(frame)
        -> wait for the next frame (wall clock, no busy wait, interrupted by stop())
  sinks: TkSink (window) | FfmpegSink (mp4) | NullSink (count) — any combination
  buffers per-episode records; never touches the RunRecorder while training runs
                                   │
run_training: after train() returns -> observer.stop(timeout) -> metrics/artifacts/summary
              (observer evidence can never fail the finished run: problems are recorded)
```

Only one observer environment is rendered; the N training environments stay
headless. Rendering, overlay composition and sinks live on the observer thread.

## Commands

Start fresh and watch in a Tk window:

```powershell
.venv\Scripts\python.exe -m gradientclimb --root artifacts train --seconds 600 --envs 64 --seed 7 --headed
```

Continue an existing PPO model (the first snapshot at t≈0 already shows the
continued policy) and follow the best-so-far selector:

```powershell
.venv\Scripts\python.exe -m gradientclimb --root artifacts train --seconds 600 --envs 64 --seed 7 `
  --parent-checkpoint artifacts\runs\<run-id>\policy-final.pt --benchmark-class fine_tuning `
  --headed --observer-policy best --observer-interval 10
```

Record without a window (CI, remote machines, macOS) and keep an MP4:

```powershell
.venv\Scripts\python.exe -m gradientclimb --root artifacts train --seconds 600 --envs 64 --seed 7 `
  --headless-record --observer-video artifacts\videos\seed7-live.mp4 --observer-fps 16.667
```

| Flag | Meaning |
| --- | --- |
| `--headed` | Live Tk window of the observer environment |
| `--headless-record` | Observer without a window; frames are counted, optionally encoded. Mutually exclusive with `--headed` |
| `--observer-policy {current,best}` | Follow the newest snapshot or the best-so-far snapshot (default `current`; see the lag caveat below) |
| `--observer-interval SECONDS` | Snapshot refresh on the governed training clock (default 5) |
| `--observer-seed INT` | Observer environment seed (default 41000). Rejected inside the held-out ranges 1000–1019 (`evaluate()` default), 2000–2019 (`generalization_suite()` default), 10000–10019 (run validation), 20000–20019 (`evaluate`/`watch` CLI defaults), 30000–30019 (`evaluate_checkpoints`) or when equal to `--seed` |
| `--observer-fps FLOAT` | Rendered frames per second. One simulator action per frame, so this is also the simulation speed; the default `1/0.06 = 16.667` is real time |
| `--observer-video PATH` | Optional MP4 through a local FFmpeg. Without FFmpeg the video is skipped and `video_skipped_reason` is recorded; an existing file is refused before training starts |
| `--observer-mode {thread}` | Only `thread` is implemented (see below) |

`--headed` opens and destroys a hidden probe window on the calling thread before
the run directory exists (`TkSink.probe`): without Tk, `PIL.ImageTk` or a display
the command fails immediately instead of training for its whole budget without a
window. Programmatic callers of `run_training(..., observer=...)` can also set
`telemetry_interval` (default 2 s) and `gpu_telemetry_interval` (default 10 s,
`null` disables `nvidia-smi` entirely) in the observer block.

The `watch` command is unchanged for replaying saved checkpoints.

## Determinism and isolation guarantees

- **No hyperparameter changes.** `--headed` / `--headless-record` build the
  training configuration exactly as before. The run's `configuration` differs
  from an unobserved run only by a separate `observer` block, and the
  `resolved-training-config.json` artifact (and the `config` stored inside every
  checkpoint) is byte-identical. `tests/test_headed_training.py::test_headless_observer_changes_only_the_observer_block`
  asserts this on a paired run. Snapshot options placed in a training config are
  rejected by the runner.
- **RNG isolation.** The observer's `VectorHillEnv(1, observer_seed)` owns its
  NumPy generator. Its private `ActorCritic` is built under
  `torch.random.fork_rng`, so orthogonal initialization does not consume the
  learner's Torch generator (asserted by
  `test_observer_construction_does_not_consume_torch_rng`). The learner's
  `torch.manual_seed(seed)` and permutation RNG are never touched; observer
  inference is deterministic (`logits >= 0`) and draws no random numbers.
- **No data feedback.** The observer never produces rollouts, rewards or
  gradients for the learner. The only learner→observer interface is
  `snapshot(payload)`: a lock, three attribute writes and an event. The
  observer→learner interface is empty; the learner never waits on it, and
  `stop()` runs only after `train_ppo` has returned. Observer episodes are
  written to the run afterwards as `observer-episodes.json` and
  `observer/episode_distance` metrics, clearly separate from training metrics.
- **Measured learner-side cost.** Copying the weights (detached, CPU, cloned)
  and invoking the callback happen inside the governed clock. Every metric row
  carries `snapshot_count`, `snapshot_copy_seconds`, `snapshot_callback_seconds`
  and `snapshot_errors` (zero when disabled); the set of metric-row keys is the
  same with and without the observer — best-selector values (`is_best`,
  `best_index`, `best_snapshot`, `best_mean_episode_distance`) live only in the
  snapshot payload, so no training series depends on the snapshot cadence. The
  run summary's `observer` block repeats the final counters together with
  `frames_rendered`, `snapshots_received`, `snapshots_loaded`,
  `observer_episodes`, `render_seconds_total`, `inference_seconds_total`,
  `sink_errors`, `error`, `record_error`, `stopped_cleanly`, `thread_joined`,
  `video` and `video_skipped_reason`. A failing snapshot callback is counted and
  logged; it never interrupts training.
- **Observer problems never fail the run.** A sink that cannot open, an
  observer-thread exception, an encoder that produced no file, a thread that
  outlives `stop()` or an evidence-writing error all leave the training run
  `completed`: they are recorded as `error`, `sink_errors`,
  `video_skipped_reason` (the MP4 is registered only when the encoder finished
  and the file exists) or `record_error` in the observer summary.
- **Recorder telemetry untouched.** The observer's SYSTEM line uses private
  `psutil.cpu_times()` deltas on its own thread; it never calls
  `psutil.cpu_percent(interval=None)`, whose process-global "since last call"
  state belongs to the run recorder's 1 s telemetry stream. GPU/VRAM come from
  one `nvidia-smi` subprocess every `gpu_telemetry_interval` seconds (default
  10) rather than every sample; that subprocess load is part of the observer as
  shipped and is therefore inside the paired protocol's on arm.
- **Snapshot provenance matches the loaded weights.** Each observer episode
  record (`snapshot_index`, `snapshot_elapsed`, `snapshot_optimizer_updates`,
  `snapshot_environment_steps`) and the SNAPSHOT overlay line describe the
  snapshot whose weights are actually loaded. In `best` mode that is the best
  snapshot's own provenance (`best_snapshot` in the payload), not the newer
  payload that carried it along.

## Best selector semantics

`--observer-policy best` follows `best_state_dict`: the snapshot whose recent
training mean distance (the learner's rolling window of at most 100 completed
training episodes, sampled at snapshot time) is the highest so far. It is a
training-signal selector, not held-out evaluation: it is recorded as
`best_selector: recent_training_mean_distance` in the observer block and must
not be reported as validated policy quality. The first snapshot (t≈0) has no
completed episodes and therefore no best entry; the observer shows the current
weights until one exists.

**The selector lags the weights.** The window holds episodes completed *before*
the snapshot, produced by the weights of earlier iterations, while the stored
`state_dict` is the weights at snapshot time. A snapshot taken right after a
policy-degrading update can therefore rank best because the window still
reflects the previous, better policy, and `best` may show a policy that never
produced the peak mean. Treat the selector as "the snapshot taken when recent
training looked best", not as the best policy.

## Overlay

Four lines are drawn below the render header (`n/a` when a value is unavailable):

```
TRAINING  t 123.4 s  steps 1,234,567  episodes 3,210  updates 456  9870 steps/s  ppo original-0.1.0 / surrogate-0.1.0
SNAPSHOT  #12 current  taken 120.0 s / 450 updates  age 3.1 s  best recent mean 84.2 m
OBSERVER  ep 7  distance 84.3 m  vx 6.21 m/s  theta 0.12 rad  gas 1  brake 0  reward +0.012  step 233  frames 4021
SYSTEM    cpu 37%  gpu 12%  vram 1.2 GB
```

Training time is the latest snapshot's elapsed time plus the wall time since it
arrived; throughput is steps over elapsed time of the latest snapshot. The
SNAPSHOT line describes the loaded snapshot only: `taken`/`updates` are its
provenance and `age` is training time elapsed since it was taken (in `current`
mode this equals the wall time since it arrived; in `best` mode it grows while
newer, non-best snapshots arrive). On the terminal frame of an observer episode
the simulator has already reset in the same step, so the OBSERVER line shows the
finished episode from its record — final distance, length, the pedals of the
final action and `ended: <termination>` — with `vx`/`theta` as `n/a`; the
rendered car below it is the freshly reset one. CPU comes from private
`psutil.cpu_times()` deltas (the first sample is `n/a`); GPU percent and VRAM
come from `nvidia-smi` when it answers and are `n/a` otherwise. CPU samples are
taken every `telemetry_interval` seconds (default 2) and GPU samples every
`gpu_telemetry_interval` seconds (default 10) on a separate daemon thread.

## Thread mode and the deferred process mode

The observer runs on a daemon thread (`--observer-mode thread`). Per frame it
holds the GIL for inference, render, overlay and sink writes (on the Cycle 1
workstation about 4.5 ms for `env.render` at 960×540 and 1.4 ms for the bitmap
overlay, measured in isolation) and then sleeps until the next frame, yielding
at least a few milliseconds even when it has fallen behind schedule; Torch
kernels release the GIL and `torch.set_num_threads` is left untouched. Weights
are reloaded at observer episode boundaries so one episode is driven by one
policy. Lower `--observer-fps` reduces the cost proportionally; the learner's
relative loss is largest for very small `--envs` counts, where each training step
is itself only a fraction of a millisecond.

`stop()` runs after `train_ppo` returns: it sets the stop flag (the frame wait
is interruptible even at very low `--observer-fps`), then joins the thread for
`timeout` (10 s in the runner) plus the longest sink close bound (30 s for an
FFmpeg flush). If the thread still has not finished, the report says so:
`thread_joined`/`stopped_cleanly` are false, counters are a mid-flight reading,
and `video_skipped_reason` explains why no MP4 was registered.

`process` mode is deferred: a child process would need its own Torch import
(seconds and hundreds of megabytes, with spawn re-import on Windows), pickled
snapshots over a pipe, and a second Tk/FFmpeg owner. The thread observer's
contention is bounded and is precisely what the paired protocol below measures.
`ObserverConfig.mode` and the picklable CPU payload keep the
`snapshot()/start()/stop()` interface ready for a process observer should the
measured overhead justify one.

## Overhead protocol `headed-overhead-paired-0.1`

`scripts/measure_headed_overhead.py` runs K paired short trainings: for each
seed, one arm without an observer and one with the headless-record observer
(no window unless `--window`), same seed, budget and configuration,
counterbalanced order (even seed positions off→on, odd positions on→off). Each
arm runs in a fresh Python process and is a canonical `run_training` record.
Dry run (default) prints the plan; `--execute` trains:

```powershell
.venv\Scripts\python.exe scripts/measure_headed_overhead.py --root artifacts --seeds 0 1 2 --seconds 60 --envs 64
.venv\Scripts\python.exe scripts/measure_headed_overhead.py --root artifacts --seeds 0 1 2 --seconds 60 --envs 64 --execute
```

The script refuses to start while any PPO/CEM run is `running` under the root.
`--envs` (default 64) must agree with a `num_envs` in `--config` if both are
given; a config may not set `algorithm`, `seconds`, `seed`, `callback`,
`parent_checkpoint`, `snapshot`, `snapshot_interval` or `observer`. State is
written beneath a fresh directory per execution,
`artifacts/headed-overhead/<utc-timestamp>-<plan_sha256[:12]>/` (plan, per-arm
job/result/console files, `pairs.json`, `outcome.json`), so re-running an
identical plan never overwrites an earlier execution; the directory is named in
the console output, the result (`state_directory`) and the protocol run's
metadata. Each pair is validated: same seed, budget and experiment, both
completed and verified, configurations identical except for `observer`, and the
on arm's observer must have run to completion — `error` and `record_error`
null, `stopped_cleanly` true, no `sink_errors`, `frames_rendered > 0` and
`snapshots_received > 0`. A pair whose observer crashed, never received a
snapshot or stopped early (a closed window) is rejected rather than sealed as a
near-zero cost.

The sealed protocol run (`RunRecorder`, protocol `headed-overhead-paired-0.1`,
algorithm label `headed-overhead-protocol`, artifact `pairs.json`) is a
measurement record, not a training run: it is never ingested as a PPO/CEM run
by `scripts/analyze_research.py`, and a leftover `running` protocol record does
not block `ensure_idle`. Its metadata names the arm algorithm and benchmark
class, the paired seeds and the state directory. It records per pair
`environment_steps_difference` and `environment_steps_per_second_difference`
metrics and, in its summary, summaries of `environment_steps_difference`,
`environment_steps_per_second_difference` and
`relative_environment_steps_change` computed with
`gradientclimb.evaluation.benchmark.summarize` over the per-seed differences.
Their `ci_method` is relabelled truthfully — *percentile bootstrap over paired
seeds (n = pairs), each pair an independently trained policy; no episodes
resampled; undefined for a single pair* — because `summarize`'s default label
describes an episode bootstrap conditional on one policy, which this is not.
Sign convention: **on minus off; negative means headed cost**. The rate is
`environment_steps / summary.training_clock_seconds` of each arm (the training
clock, not the recorder's whole-run duration that includes evaluation). The on
arm's clock includes its snapshot copy and callback time, and the on arm also
carries the observer's telemetry thread (CPU deltas every 2 s, `nvidia-smi`
every 10 s) as part of the observer as shipped. Read the CI of the relative
change as the cost estimate for that machine and configuration; the record
states the number of paired seeds and that other machine load was uncontrolled.
`--in-process` exists for tests only.

## Tk and FFmpeg availability

- No display or no Tk: `--headed` fails before the run directory exists (the
  probe window cannot open); use `--headless-record`. All headless paths are
  exercised in CI without a window; `TkSink` tests skip when Tk cannot open.
- macOS requires Tk on the main thread; the observer thread cannot own a window
  there. Use `--headless-record` (optionally with `--observer-video`).
- Closing the window destroys it at once and stops rendering for that sink
  only; the run continues, the event is recorded in `sink_errors`, and a video
  sink keeps encoding. (Such a run is not a valid overhead-protocol arm.)
- Without FFmpeg, `--observer-video` is skipped with
  `video_skipped_reason: "ffmpeg executable not found"`; the `watch` command still
  raises because its sole purpose is the video. An encoder that never started
  (an earlier sink failed to open), exited with an error, stalled at close past
  its 30 s bound or produced no file leaves `video` null and names the cause in
  `video_skipped_reason` (built from the per-sink reports or the observer error);
  the run itself still completes.
- Known limitation: an encoder that stops reading its input **while frames are
  being written** blocks the observer thread inside the write, so the close-time
  bound cannot reach it. The training run still completes; `stop()` then reports
  `stopped_cleanly: false` after its join budget, `video_skipped_reason` says the
  thread was still running, and the encoder child is not killed. Frame writes
  would need a separate writer thread with its own bound to remove this.
- Reserved seed ranges protect only the documented defaults. `evaluate
  --seed-start` and `evaluate-checkpoints --episodes` accept any seeds, so an
  operator who evaluates on a custom range must also keep `--observer-seed` out of
  it; the run configuration records both for inspection.
