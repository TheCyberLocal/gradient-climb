# F-001 — Fast surrogate learning, limited shift robustness and a retention tradeoff

Evidence domain: **uncalibrated simulator**. Status: observed result, Cycle 1 paused. Source cutoff 2026-09-11T20:01:46.950554Z. The [canonical summary](../experiments/results-summary.json) and [report](../reports/gradientclimb-research-report.md) provide complete run/hash mappings, distributions and figures. No value here is an actual Hill Climb Racing distance.

## Result and architecture

The original PPO learned a compact controller in two independent cold-start hour runs. [Primary c0a9e142](../../artifacts/runs/c0a9e142-1ad6-4d88-810d-bda4ca297f40/run.json), seed 42, trained 3600.321910 s over 41,648,128 transitions; [reproduction db77b7cd](../../artifacts/runs/db77b7cd-a11a-473a-8406-06d99b5de5ad/run.json), seed 43, trained 3600.348389 s over 31,382,272 transitions. Final validation medians were 716.122801 and 705.092202, respectively, on 20 episodes each. Different transition totals under the same time budget are recorded workload outcomes, not equal-sample comparisons.

Both use `original-0.1.0` PPO, `surrogate-0.1.0`, default/train profiles, calibration `uncalibrated`, CPU 256 environments and one Torch thread. The actor/critic network has 10,563 parameters, four stacked 24-feature idealized simulator observations, two 64-unit hidden layers and independent Bernoulli gas/brake outputs. This actor is not deployable from the current real pixels by filling unknown state values. The governed [configuration](../../experiments/definitions/one-hour-surrogate.json) fixes rollout 128, minibatch 512, four optimizer epochs, learning rate 0.0003, gamma 0.99 and GAE 0.95. [Reward semantics](../../docs/research/reward-research.md) distinguish PPO shaped return from CEM distance fitness.

Final checkpoint hashes are `87c469539110d3134c04ebcf11e95f9c84e595c90ba7e6b2b106917115e04ec0` (primary) and `f9ba71dc136529518c9ade34a51f1d58a3650d5b7a3c918f6994c049148cb41d` (reproduction). Primary source was clean 73c5ea2; reproduction source was clean dfb6dc7; full SHAs and dependency/hardware identities are in their canonical records.

[HYP-001](../hypotheses/HYP-001.md) is supported at the measured nominal five-minute checkpoints: median/random-median ratios 32.0993 and 30.0792. Actual checkpoint times 300.0599465 and 301.2673551 s overshoot the literal 300 s boundary; an exact ≤300 s crossing remains unmeasured. Later checkpoint quality was not monotonic. Two training seeds do not establish population-level consistency or superiority.

## Throughput is not the selection objective

The prior [runtime screen](../experiments/runtime-screen.json) spent 481.36 actual training seconds across eight requested 60 s runs. CPU 256 PPO averaged 491.38 validation distance across seeds 0/1/2 (between-seed SD 35.08), versus 378.49 for the tested CEM configuration (SD 106.31). CEM nevertheless stepped faster: about 19,713 versus 8523 transitions/s. Different policy/observation/configuration choices confound a pure algorithm comparison. One CUDA 256 PPO trial and one CPU 64 trial are exploratory device/batch observations, not replicated hardware conclusions.

The [environment-only sweep](../../artifacts/runs/3d4dc35e-25c1-4671-8e8b-5c9bb150ac6c/run.json) reached about 60,618 transitions/s at 1024 environments. It includes action generation and excludes policy/optimizer costs. The learner therefore used the configuration favored by the measured quality/time screen, not the largest environment-only batch.

## Shift and adaptation results

The primary [final test](../../artifacts/runs/3a9a0c20-e544-4efd-9c68-e2830d8e8224/run.json) uses paired seeds 20000–20019 and 60 s horizons. Mean nominal distance was 700.747 on default/train, 122.304 on default/rough, 681.618 on heavy/train and 146.879 on heavy/rough. Rough-condition crash rates were 90% and 85%. This large terrain sensitivity limits the source policy even within its surrogate.

The [ten-minute heavy/rough child](../../artifacts/runs/24eacdf3-cff5-430f-8978-b2e2bd34f396/run.json) loaded the primary weights and optimizer, then trained 600.379853 additional seconds. Its checkpoint is `50b1010efc2acfb78d254bad87a8d80bc29d85ac29757ea6c50d9df8ce13be9b`. It is fine-tuning with a trained prior, not a cold start. The [paired final evaluation](../../artifacts/runs/59768cd6-6640-4e20-8198-0158e7eb108c/run.json) found:

| Condition | Parent mean | Child mean | Paired mean change | Conditional episode-bootstrap 95% interval |
| --- | --- | --- | --- | --- |
| Heavy/rough adaptation target |146.878933|405.005874|+258.126941|[204.865297, 307.918968]|
| Default/train source retention |700.747019|597.719298|−103.027720|[−198.577041,−36.440008]|

The target improves while source quality declines. These are paired episode-seed differences conditional on one parent/child policy pair; they are not training-seed confidence intervals or evidence about commercial-game adaptation. Saved 5/10-minute validation means 359.947717/396.526813 use seeds 10000–10019 and must not be mixed with the final-test means above. The original replicated 30/60-minute adaptation program is unexecuted.

The separate [default/train extension](../../artifacts/runs/6f193264-faa5-47ae-88b3-9e0f7ea413a5/run.json), 600.319895 additional seconds from the same parent, had final-test source mean 688.653 and rough mean 66.896 in [44ad7e2e](../../artifacts/runs/44ad7e2e-6758-4957-aa21-1d953d807bd7/run.json). Additional training did not produce broad improvement in this comparison. Neither child was selected to become the other's parent.

## Completed short component screen

The registered [post-hour battery](../../experiments/definitions/post-hour-battery.json) ran four conditions at requested 60 s on training seeds 101/102/103, with fixed randomized order. Its 12 learning runs are separate from the original unexecuted 300 s proposal. Mean validation distances were 442.211 baseline, 503.802 two optimizer epochs, 424.545 one stacked frame and 529.526 randomization. Paired mean changes across the three training seeds were+61.591,−17.666 and+87.315, with between-seed SDs of differences 35.306, 35.568 and 32.472, respectively.

These results identify candidates for a later experiment; they do not retroactively change the governed configuration, establish a long-budget winner, or demonstrate real transfer. The one-frame actor still sees idealized velocity/current state, so this is not removal of all temporal information. Randomization used plausible synthetic ranges, not uncertainty fitted from the game. The [plan disposition](../experiments/cycle-1-plan-status.json) records all 23 dispatched jobs and 15 learning runs complete, actual learning 5523.329475 s against 5520 requested, separate from the primary hour and earlier pilots.

Follow-up is **DEFERRED**: reliable real endpoints first, then narrowly justified component/transfer comparisons with preregistered budgets. The scientific conclusion is useful learning in the chosen surrogate plus measured weaknesses, not qualification of an actual game agent.
