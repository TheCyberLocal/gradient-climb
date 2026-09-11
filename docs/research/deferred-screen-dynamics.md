# Deferred observable-dynamics draft inventory

Status: **DEFERRED — preserved locally, unadopted, not part of the installed package.** The [Cycle 1 pause](../methodology/cycle-1-pause-directive.md) stopped this work before restoration, fitting or policy integration. These files establish no completed calibration, world model, adaptation mechanism or control improvement.

The draft was moved out of `src/` before a clean native pilot checkpoint. Only the specifically named draft files below were preserved; no other source was replaced. Paths are relative to the repository and ignored by Git. Another clone needs the local artifact archive; the committed inventory preserves design and identity, not the code bytes.

| Local file | SHA-256 |
| --- | --- |
| `artifacts/wip-screen-dynamics/screen_dynamics.py` | `f9b5f5b95b2235b3066f4c7e308c35e71b4e890daef4f1da916ed70ce9b68282` |
| `artifacts/wip-screen-dynamics/identify_screen_dynamics.py` | `be011118bcc77b52252591c0e4f01952d3edd89c1539b1e910c6d0fc2c1fa512` |
| `artifacts/wip-screen-dynamics/test_screen_dynamics.py` | `10ce00cde22999b5385a374b81e0cf7e97f431c56c33297017c9522244dbff78` |
| `artifacts/wip-screen-dynamics/screen-dynamics.md` | `468b52f0dc3016ad710d4581b1a0aa88be585641b5db8db2e31e478ffc687ca4` |

The unexecuted proposal is `artifacts/screen-dynamics/exploratory-manifest-v2.json`, SHA-256 `ac4fb43441be9836c5e9dd99e060e5a85be05bd962fa1a9a1ffe5352811d11da`. Its companion `plan-v2.json` is a dry-run readiness report with `executed: false`; earlier v1 files are also preserved. This is not a canonical completed research run.

## Preserved design

Predict next observed body-axis change modulo π and body-to-terrain clearance change in body spans. Compare zero-change persistence, action-free autoregression and standardized ridge with separate integrated gas, brake and simultaneous-pedal exposure. Targets are observed image quantities, not physical position, metres, mass, traction or signed speed. Error against extracted features would still not measure true pose error without independent labels.

The proposed input contract retains capture start/midpoint/completion, feature availability, explicit playing state and masks, plus original OS insertion transactions. It rejects partial insertions, uncertain boundary transactions, uncovered trace tails, invalid poses, 10–500 ms interval violations and large modulo-axis jumps. Both-pedal exposure is integrated directly; it is not the product of marginal gas and brake fractions. Dispatch timestamps are not proof of when the game acknowledged an action.

Whole episodes are split before fitting. Delay candidates 0/0.1/0.2 s would be chosen by leave-one-training-episode-out error on their common eligible intervals, with fixed ridge penalty 1. Every fold must have enough body targets; optional clearance must be evaluable across folds. Held-out data never select delay, scaling or coefficients. The selected delay would be a predictive parameter, not measured display or input latency.

The retrospective, already-viewed draft split uses training episodes from `2cf160db-8ecb-49a7-b61b-77785ef433b3` and `d9955d22-5f78-4753-a66f-8b549a076093`, and held-out episode `0d84a31d-4b78-4506-a869-6bdf7cbb20a3`. The dry plan found 47 + 19 common training intervals. These are same-session mostly-gas observations; they cannot identify all independent pedal effects or establish cross-session generalization. Source runs with reset failures remain failed, even if an earlier recorded trajectory is usable offline.

`RollingResponseContext(window=64, delay_seconds=0, ridge=1)` proposes eight local response coefficients plus eight masks. `update(frame, trace, available_ns=...)` ingests only completed measurements; `encode(decision_ns)` requires the decision to be strictly later than their availability. Future trace suffixes cannot affect earlier context. Missing observations, gaps and episode boundaries clear history; rank-deficient input designs keep coefficients invalid. These are local predictive associations, not causal physical parameters. The frozen eight-feature CEM actor was never changed.

## Readiness and adoption gate

Fourteen focused tests passed on synthetic data (last observed 0.87 s); Ruff was clean. Tests cover action integration, simultaneous exposure, partial/unknown actions, bounded trace coverage, episode leakage, persistence, missing targets, fold readiness, held-out noninterference, context causality/reset and action-design rank. An independent review found and resolved equal-time context eviction, insufficient cross-validation folds, mismatched source-record membership and unbounded action-log coverage. These tests are outside the release's adopted test suite and establish no real predictive accuracy.

Future adoption requires explicit research resumption, recovering and verifying these hashes, fresh source review/tests, trusted acquisition/OS coverage and episode manifests, and new independent labels. Register a new immutable analysis run before any fit. Report per-target/per-episode errors, omissions and input rank against both baselines. Only a separate equal-budget policy ablation could establish control benefit. The first cycle contains **no real fit, fitted coefficients, held-out prediction metrics or learned-context control experiment**.
