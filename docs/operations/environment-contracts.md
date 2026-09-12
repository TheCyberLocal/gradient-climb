# Cycle 3 environment and profile contracts

The available engine remains `surrogate-0.1.0`. The factory returns the original
`VectorHillEnv`, with identical numerical state, seeded resets, rewards, terminal
records and rendering. Its default episode limit remains 1,000 decisions at
0.06 seconds (60 simulated seconds). This preserves historical experiments;
new long-run protocols must explicitly declare a longer horizon.

```python
from gradientclimb.environments import make_environment, environment_from_config

env = make_environment(profile="heavy", terrain="rough", max_steps=5000)
print(env.scenario.model_dump_json(indent=2))
print(env.scenario.sha256)
replay_env = environment_from_config(checkpoint_config, num_envs=1, seed=41000)
```

`reference-0.1.0` and `reference-randomized-0.1.0` are reserved unavailable
generations. Construction raises `EnvironmentUnavailableError`; no fallback
substitutes surrogate dynamics. Their registry entries describe intended
interfaces, not implemented physics, calibrated metres, or validated fidelity.
The next simulator milestone is an engine installation/stability/throughput
pilot followed by measured fixtures. No physics dependency is selected here.

The immutable `VehicleProfile`, `UpgradeProfile`, `MapProfile`,
`ObservationProfile`, and `ControlProfile` compose into a `Scenario`. They
contain versions, build identity, provenance, asset references, checkpoint
hashes, topology, input channels, observation cadence, and distance definitions.
Parameter uncertainty is explicit; missing bounds remain unknown. Measured
evidence requires an artifact hash, and measured route coverage requires measured
geometry evidence. Profiles may represent different body counts, joints and
input channels; this expresses compatibility and does not implement a mechanism.
There is no route-script or policy-decision field.

Nested records and collections are immutable. A scenario's canonical JSON
SHA-256 identifies its complete contents, including the observation contract.
`Scenario.model_validate_json(...)` validates a serialized profile bundle.
Composition rejects different builds, mismatched upgrade vehicles, invalid joint
endpoints, and unavailable required control channels. Idealized simulator state
cannot be marked deployable from game pixels. Existing synthetic profiles have
no observed game upgrades, calibrated constants or measured route coverage.

Evaluator, saved-checkpoint replay, and headed observer resolve the checkpoint's
profile, terrain, history, cadence and horizon through the factory. Explicit
evaluation arguments select transfer conditions; fixed evaluation disables
parameter randomization. Replay/observer output is visualization, never
independent qualification. Historical checkpoints without an environment version
resolve to the only generation their format supported. Declared unknown or
contradictory versions fail closed.

New evaluations use `simulator-evaluation-3.0` and record units, physical cadence,
scenario/profile hashes, and the serialized scenario. Paired comparisons check
domain, generation, calibration, vehicle/map, horizon, units, cadence, solver
settings, randomization and profile hashes. Duplicate episode seeds and missing
scope are rejected. The explicitly named historical
`surrogate-evaluation-0.1.0` protocol can still be compared using its fixed
surrogate units and 0.06-second cadence; other records must state these fields.
No historical artifacts are rewritten. These comparisons concern repeated
episodes of fixed policies, not independent training-seed uncertainty.

Focused validation:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_environment_contracts.py tests/test_simulation.py tests/test_report_analysis.py --basetemp artifacts/cycle3-tests-architecture
```

Tests establish exact legacy parity and interface behavior. They establish no
measured reference physics, visual transfer, real-game competence, adaptation or
additional commercial-game vehicle support.
