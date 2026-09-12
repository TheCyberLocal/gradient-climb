# F-007 — The first articulated-engine pilot exceeded its fixture's floor

Status: **engineering gate not passed**, Cycle 3, 2026-09-12. This is not a
learning, real-fidelity, or transfer result. It changes no Cycle 1/2 finding.

Run `a88aeae4-ccdf-4f84-a178-bc5e58feebd8` executed
`articulated-engine-pilot-3.0` from clean source
`b61b7e19343b37451d8af83b32fba87ce55abe08`. The hash-pinned Box2D 2.3.10 wheel
installed into the run-local directory. The canonical failure seal verifies.
The [machine-readable interpretation](../experiments/cycle-3-engine-pilot-001.json)
retains source/evidence hashes and measured costs.

Both 600-decision flat-fixture repetitions had the same state digest. The wheel
lateral-constraint error was 0.0007581 fixture units; maximum body speed 19.5298
and angular speed 57.505 were within their registered limits. The global minimum
wheel bottom was −10.9801, failing the unchanged −0.05 floor tolerance.

However, the floor covered only x=−20 through 100. Final wheel centres were
x=108.7822 and 109.4014, beyond its endpoint. The recorded minimum equals the
final first-wheel centre height minus its radius. This supports a fixture-extent
and measurement-domain defect, rather than demonstrating penetration through
existing ground. Aggregate diagnostics do not locate the first domain exit or
rule out separate earlier penetration; those remain unmeasured in this run.

The protocol stopped immediately. Bridge diagnostics, rendering arms and
throughput comparisons were not executed. No engine, batch size, fidelity level
or learner was selected from this attempt.

| Quantity | Observation and scope |
| --- | --- |
| Recorder wall time | 11.3495 s; distinct from a learning command clock |
| Installation wall time | 6.7218 s, included above; child-process CPU cost unmeasured |
| Process CPU | 1.21875 core-seconds across the sampled recorder-process interval; excludes pip and other child processes |
| Resource coverage | 10.9753 s sampled window, with approximately 99.43% CPU/memory coverage within that window |
| Process memory | Sampled peak 65,691,648 bytes; weighted sampled mean 61,554,586 bytes; unsampled peaks unknown |
| Device activity | 5.3694 GPU utilization-equivalent seconds over measured device-wide samples; other applications were present, so this is **not** engine-attributed GPU consumption |
| Simulated work | 1,200 scripted fixture decisions; derived 2,400 physics substeps and 20 summed simulated seconds across two fixture instances |
| Learned or real experience | No policy updates, learned-policy decisions, real interaction or real evaluation episodes; these fixtures do not implement episode outcome lifecycles |

The next bounded experiment must extend ground coverage from its declared
duration/speed envelope and report first domain exit plus in-domain penetration
separately. Solver settings, drive/control sequence and physical sanity limits
stay fixed. Register and commit that successor before collecting it, preserve this
failed run, and stop again on any unmet gate. Engineering feasibility remains a
prerequisite; independent real competence at declared costs still determines
whether the eventual simulator helps GradientClimb learn rapidly.
