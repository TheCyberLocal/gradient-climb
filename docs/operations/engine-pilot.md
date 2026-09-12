# Synthetic articulated-engine pilot

This prospective pilot tests optional engine installation, numerical stability,
moving collidable structures and offscreen cost. It does not expose a new
training environment or claim game calibration. Both reference factory IDs stay
unavailable until a later implementation and qualification milestone.

The first candidate is [Box2D 2.3.10](https://pypi.org/project/Box2D/2.3.10/),
whose publisher provides a CPython 3.13 Windows x86-64 wheel. The binding uses the
older Box2D 2.3 API and a [zlib license](https://github.com/pybox2d/pybox2d/blob/master/LICENSE).
The project will retain its notices and binary identity without vendoring code.
The [binding manual](https://github.com/pybox2d/pybox2d/wiki/manual) documents
separate solver iterations and timestep integration. Current
[Box2D wheel-joint](https://box2d.org/documentation/group__wheel__joint.html) and
[simulation](https://box2d.org/documentation/md_simulation.html) documentation
inform the engineering approach; their 3.x API and solver features are not claims
about this 2.3 binding.

[box2d-python 0.1.2](https://pypi.org/project/box2d-python/0.1.2/) is a second
candidate with a 3.x CFFI interface and a Windows CPython 3.13 wheel. Its author
labels the API a development preview and requires Python 3.12 or newer. It has
not been rejected by performance evidence. A follow-up pilot is justified if the
first candidate reveals a relevant limitation.

The [3.1 successor protocol](../../experiments/definitions/cycle-3-engine-pilot-3.1.json)
fixes one bounded screening campaign. The
[3.0 predecessor](../../experiments/definitions/cycle-3-engine-pilot.json) remains
unchanged: run `a88aeae4-ccdf-4f84-a178-bc5e58feebd8` failed its flat gate after the
vehicle left a floor ending at x=100. The observed minimum was outside that floor;
the retained aggregate does not rule out earlier penetration within its extent.
Original synthetic fixtures contain
independently rotating wheel bodies and wheel joints, plus twelve articulated
collidable planks spanning an actual gap. A stationary bridge load and scripted
traversal exercise contact. None of the dimensions, stiffness, torque, friction,
or appearance is inferred from game images.

Fixture version 3.1 extends the outer ground to `[-1020, 1020]`, derived from the
unchanged speed bound of 100 units/second times the 10-second simulated horizon,
plus 20 units covering initial positions/body extent. It preserves the open
static-ground bridge gap `(6, 14)`, articulation, solver, actions and all numerical
thresholds, including the `-0.05` flat penetration limit.

Every physics substep records support-domain coverage and first-exit evidence.
Global wheel-bottom minima, minima over actual static floor and minima inside the
bridge gap are separate, with wheel position and decision/substep/time evidence.
Flat qualification requires zero outer-domain exits in both repeats. A gap fall
remains visible in global/gap measurements; the implementation never fills the
gap or silently removes unsupported states. Script dispatch checks protocol/source
fixture identity and the speed/horizon envelope. The successor permits one screen
within the original 600-second total wall cap; another failure requires a new
diagnosis and committed prospective amendment.

Review the plan without installing or stepping an engine:

```powershell
.venv\Scripts\python.exe scripts/run_engine_pilot.py
```

After committing the governing source and protocol, execute once from a clean
checkout using its full source SHA:

```powershell
.venv\Scripts\python.exe scripts/run_engine_pilot.py --execute --expected-sha <full-commit-sha>
```

The script installs only the hash-pinned wheel in a fresh directory within its
canonical run. It leaves the project environment and dependency declarations
unchanged. Installation logs, pip receipt, installed source/binary/notices,
diagnostics and throughput arms remain linked to that run. The initial installer
is explicitly limited to CPython 3.13 on Windows x86-64.

Each completed diagnostic repeat, gate and throughput arm is immediately retained
as an `engine-observation-3.0` JSON artifact, including when a later gate or arm
fails. Synthetic fixtures do not create episode or evaluation records: all
training/evaluation/real-game episode counts remain zero. Scripted fixture
decisions are counted separately from policy decisions and optimizer updates.

Report all 1/8/32-world arms, with and without 320×180 offscreen rendering. Physics
substeps, scripted decisions and rendered observations have separate counters;
there are no policy decisions or optimizer updates. Physics-loop timing includes
state diagnostics, and rendering uses original debug geometry. This measures an
engineering workload, not the eventual shared perception pipeline. An incomplete
arm is capped, and failed diagnostic gates stop the campaign without replacements.
Passing permits measured fixture development, not a digital-twin label, selected
learning batch size, or real transfer claim.

Rapid learning means elapsed time to independently evaluated real-game competence.
This pilot measures no such outcome. Its `fidelity_level` identifies synthetic
articulation plus debug rendering, with real validation and shared perception both
false. Wall time, process CPU core seconds, sampled process memory, physics steps,
rendered observations and summed simulator seconds remain separate. The process
memory maximum is a sampled value, not a guaranteed peak. Episode counts are
experience descriptors; they are not a speed result. Installation and simulator
development remain system-prior costs for later learning claims. Later
preregistered fidelity comparisons must judge real improvement against added
wall time, experience and compute through Pareto comparisons.
