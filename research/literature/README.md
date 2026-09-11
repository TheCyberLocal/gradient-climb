# Focused literature review

Reviewed 2026-09-11. The machine-readable [reference register](references.json) records titles, authors/projects, primary sources, URLs/DOIs, licensing, relevance, reuse decisions, and limitations for 24 references. This is an architecture review, not a systematic survey or a replication of the cited results. Most papers were inspected at abstract level; official project READMEs, API documentation, and license files were also inspected. Quantitative claims below are not GradientClimb measurements.

## What the evidence supports

The most relevant existing HCR environment is [alexzh3/hillclimbracing](https://github.com/alexzh3/hillclimbracing), a Box2D/Gymnasium recreation with PPO examples. Its GPL declaration and derivation from a separate browser clone need attention before reuse. Its three-way discrete or scalar motor-speed action interface does not meet GradientClimb's independent gas/brake requirement. The [Code Bullet](https://github.com/Code-Bullet/Hill-Climb-Racing-AI) and [0ql](https://github.com/0ql/AI-Hill-Climb-Racing) projects establish precedents for neuroevolution in clones; they do not establish transfer to the installed game. No code, assets, or checkpoints from these projects were imported.

The [vision-based behavioral-cloning project](https://github.com/FahzainAhmad/agent-hill-climb-supervised) provides a useful alternative direction: learn from recorded gameplay rather than approximate its physics first. Its three action labels also omit simultaneous pedals, and its public listing did not establish reuse rights. GradientClimb should collect its own timestamped four-state actions and split validation by entire episodes or sessions. That split is our methodological recommendation, not a claim that the reference used an invalid split.

[PPO](https://arxiv.org/abs/1707.06347) is a defensible baseline, not an established winner for this game. [SB3's own documentation](https://stable-baselines3.readthedocs.io/en/v2.8.0/modules/ppo.html) recommends first trying observation stacking before recurrence. The project should compare a small stacked-observation network to [recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) at equal elapsed training time. A simulator-state policy establishes an oracle baseline only until every actor input has a validated pixel/history equivalent.

[Asymmetric actor-critic](https://arxiv.org/abs/1710.06542) permits simulator state in the critic while keeping actor inputs deployable. [Policy distillation](https://arxiv.org/abs/1511.06295) suggests a second route from a privileged teacher to a smaller visual student. Both need explicit actor-input contracts and student evaluation; neither allows the teacher's simulator score to stand in for deployed quality.

[PufferLib](https://arxiv.org/abs/2406.12905) and [Sample Factory](https://arxiv.org/abs/2006.11751) motivate profiling collection, inference, learning, and synchronization separately. Their reported throughput does not predict this Windows host's speed. Benchmark an original batched simulator before introducing processes or a new training framework. Compare matched policies and sample definitions; a physics substep, a policy decision, and an observed frame are different quantities.

The main transfer candidates are [dynamics randomization](https://arxiv.org/abs/1710.06537), [online system identification](https://arxiv.org/abs/1702.02453), and [RMA](https://arxiv.org/abs/2107.04034). Our inference is to randomize around fitted uncertainty, then test whether action/observation history improves adaptation to held-out vehicles. Randomization cannot establish calibration, and fast test-time adaptation does not erase the cost of prior training. [RL²](https://arxiv.org/abs/1611.02779) makes the latter distinction especially explicit.

[World Models](https://arxiv.org/abs/1803.10122) and [DreamerV3](https://arxiv.org/abs/2301.04104) merit a later measured comparison if real experience becomes the bottleneck. An additional latent dynamics learner has a cost under a one-hour budget. The reviewed sources provide no evidence that Dreamer is the fastest route to competent HCR on this workstation.

## Recommended decision order

1. Validate the scientific harness, independent action channels, episode semantics, and traceable elapsed-time accounting.
2. Measure screen/control reliability and collect short controlled real trajectories. Keep uncalibrated simulation explicitly labeled as synthetic.
3. Fit and validate dynamics on disjoint trajectories. Compare control-relevant errors before adding physics complexity; [Box2D](https://github.com/erincatto/box2d) is a candidate engine, not a calibration result.
4. Establish random, constant-pedal, and learned baselines on identical evaluation conditions. Behavioral baselines are comparators, not hand-coded vehicle tricks inside the learned policy.
5. Benchmark stacked PPO and a meaningful competing learning method across seeds. Add recurrence, privileged critics, curriculum, or randomization one factor at a time when an observed failure motivates it.
6. Select using real-game quality, stability, latency, and transfer. Run the governed hour with declared prior training and report missing checkpoints as missing.

## Evidence and reuse boundaries

Use [statistical evaluation guidance](https://arxiv.org/abs/2108.13264) to report between-training-seed uncertainty separately from within-policy episode variability. Predeclare target vehicle, map, upgrades, held-out seeds, distance/survival criteria, and model-selection rules. Record a human baseline if available; do not silently invent one or equate a synthetic finish flag with high-level game play.

The [license review](dependency-licenses.md) and [license ADR](../../docs/decisions/ADR-001-licensing.md) distinguish package imports from vendored code and artwork. The [learning-architecture ADR](../../docs/decisions/ADR-002-learning-architecture.md) records provisional choices and the measurements that would change them. Literature results are motivation. GradientClimb run records are the evidence for GradientClimb claims.
