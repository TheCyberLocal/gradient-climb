# ADR-002: Measured compact baselines before larger learning systems

Date: 2026-09-11
Status: Provisional; implementation choices require local throughput and game-transfer evidence
Decision driver: Best measured real-game policy after a declared wall-clock budget.

## Context

The initial workstation assessment reports an i9-14900HX with 32 logical processors and an RTX 4090 Laptop GPU with 16 GB VRAM. Hardware capability does not determine whether a tiny policy benefits from GPU execution or whether subprocess vectorization beats an in-process batch. A faithful one-hour result must also account for calibration, prior data, pretrained components, checkpoint evaluation, and training-session overhead under its declared budget rules.

The [literature register](../../research/literature/references.json) gives the full source trail. [PPO](https://arxiv.org/abs/1707.06347) offers a compact initial learner. [SB3 documentation](https://stable-baselines3.readthedocs.io/en/v2.8.0/modules/ppo.html) supports testing stacked observations before recurrence. [PufferLib](https://arxiv.org/abs/2406.12905) and [Sample Factory](https://arxiv.org/abs/2006.11751) motivate throughput measurement but do not settle the choice for this machine.

## Initial decision

Use Python with NumPy, PyTorch, and a Gymnasium environment contract. Start with an original, batched numerical simulator and a small PPO policy as experimental baselines. Explicitly label simulator-state actors as privileged/oracle policies until their inputs are supplied by validated screen/history estimators. An uncalibrated simulator is useful for harness tests and algorithm experiments; it is not evidence of game competence.

Represent the physical controls as two independent bits. A categorical policy over the four combinations is valid if its mapping preserves `00`, `10`, `01`, and `11`; independent Bernoulli outputs are another valid parameterization. Choosing one combination per decision does not make the two pedals mutually exclusive. Log transitions, hold durations, requested actions, and delivered actions so ordering remains observable. Never collapse both-pedal input to a no-op by computing only `gas - brake` without separate physical effects.

Use a short history of observations and prior actions as the first deployable temporal baseline. Compare recurrence only when hidden dynamics or aliasing limit performance. Keep a privileged critic or teacher behind an explicit training interface, informed by [asymmetric actor-critic](https://arxiv.org/abs/1710.06542) and [distillation](https://arxiv.org/abs/1511.06295). Add bounded randomization around measured uncertainty, rather than treating arbitrary parameter variation as calibration.

## Alternatives and evidence needed

| Candidate | Potential advantage | Adoption evidence |
| --- | --- | --- |
| SB3 PPO | Established, tested algorithm comparator | Matched observation/action/reward conditions; total time and learning quality compared with original learner. |
| Recurrent PPO / small GRU | Infer hidden velocity/dynamics from history | Lower held-out failure rate or better time-to-threshold than stacked observations at equal elapsed time; sequence/reset correctness verified. |
| PufferLib or asynchronous sampler | Reduce collection and synchronization overhead | Native install works without disrupting the workstation; same-policy quality improves per minute; policy lag and added resource use recorded. |
| Box2D simulation | More general contact, suspension, and articulation | Held-out real trajectory errors identify dynamics the lightweight model cannot represent; improved transfer outweighs throughput loss. |
| Evolutionary or cross-entropy learner | Simple optimization, little backpropagation overhead | Equal-budget multi-seed comparison. A fixed-topology search is labeled accurately and is not called NEAT. |
| World model / Dreamer | Reuse expensive experience in imagined rollouts | World-model fit and policy learning, including initialization, provide better real quality per minute; model exploitation checked. |
| UP-OSI / RMA-style context | Adapt to new vehicle dynamics | Held-out vehicles and maps improve with history-inferred context; oracle parameters excluded at deployment; prior training cost reported. |

## Measurement and revision rules

Before changing frameworks, record environment decisions/sec, physics substeps/sec, inference latency, optimizer time, device transfer time, CPU/GPU utilization, and memory. Compare batch sizes and CPU/GPU execution under the same policy and environment settings. A bigger batch or more processes can improve raw throughput while degrading learning; retain both axes.

Before declaring transfer, pair the same checkpoint with simulator and real-game evaluation under recorded vehicle/map/upgrades and comparable metrics. Calibrate on training trajectories and reserve whole trajectories for validation. A deployment uses only rendered pixels and recorded input history. Missing real measurements must remain missing, not zero or inferred from synthetic reward.

The main criterion for retaining complexity is improvement in held-out real-game quality or time-to-competence across training seeds. [Statistical evaluation work](https://arxiv.org/abs/2108.13264) supports interval reporting and discourages single-run winner claims. Revise this ADR when local experiment records select a framework, representation, or dynamics model; leave rejected candidates as documented negative findings rather than erasing them.

## Cycle 1 disposition

The CPU/256-environment PPO choice was supported as a provisional surrogate
engineering choice by three 60-second training seeds: mean validation distance
491.38 ±35.08 between-seed SD, versus implemented CEM 378.49 ±106.31. Two governed
surrogate hours and a source extension/adaptation battery subsequently completed.
This evidence does not select a real-game learner or establish simulator fidelity.
See [F-001](../../research/findings/F-001-surrogate-learning.md).

A separate compact direct-screen CEM avoids deploying oracle simulator state.
Its native pilot collected an initial candidate trajectory but failed parking on
an unknown advertisement with zero eligible episodes and no update. Student
projection code is implemented but was never trained. These remain unproven
alternatives; see [F-002](../../research/findings/F-002-native-observation-and-control.md).

Cycle 1 is deliberately paused. The existing architecture is retained, with
bounded cleanup/provenance/dashboard fixes and explicit lazy Torch/capture
boundaries. The next cycle must investigate versioned multi-objective reward and
recovery-conditioned skill credit after native episode reliability, preserving
historical objectives and independent distance evaluation. No larger algorithm,
physics redesign or reward campaign was adopted during stabilization. See the
[resumption plan](../operations/resumption-plan.md).
