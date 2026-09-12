"""Dry-run by default; execute a committed, hash-pinned synthetic engine screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
PROTOCOL = PROJECT / "experiments/definitions/cycle-3-engine-pilot.json"


def plan() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def check_source(expected_sha: str) -> str:
    if len(expected_sha) != 40 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("Execution requires the full expected committed source SHA")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT, text=True
    ).strip()
    if actual != expected_sha or dirty:
        raise RuntimeError("Engine experiment requires clean source at the expected SHA")
    return actual


def install_engine(run, protocol: dict) -> dict:
    """Install only the pinned wheel in a new local artifact directory."""
    if sys.version_info[:2] != (3, 13) or sys.platform != "win32" or platform.machine() != "AMD64":
        raise RuntimeError("This installation protocol is pinned to CPython 3.13 Windows x86-64")
    runtime = run.directory / "engine-runtime"
    runtime.mkdir(exist_ok=False)
    requirements = run.directory / "engine-requirements.txt"
    requirements.write_text(
        f"Box2D==2.3.10 --hash=sha256:{protocol['candidate']['wheel_sha256']}\n", encoding="utf-8"
    )
    receipt = run.directory / "engine-install-report.json"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--no-deps",
        "--no-cache-dir",
        "--require-hashes",
        "--target",
        str(runtime),
        "--report",
        str(receipt),
        "-r",
        str(requirements),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT,
            text=True,
            capture_output=True,
            timeout=protocol["budgets"]["installation_timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired as error:

        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        log = run.directory / "engine-install.log"
        log.write_text(decode(error.stdout) + "\n" + decode(error.stderr), encoding="utf-8")
        run.register_artifact(log, "engine_installation")
        run.register_artifact(requirements, "engine_installation")
        raise
    elapsed = time.perf_counter() - start
    log = run.directory / "engine-install.log"
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    for path in (requirements, log, receipt):
        if path.exists():
            run.register_artifact(path, "engine_installation")
    completed.check_returncode()
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()
    for path in sorted(runtime.rglob("*")):
        if path.is_file() and path.suffix != ".pyc":
            run.register_artifact(path, "engine_runtime")
    return {"elapsed_seconds": elapsed, "returncode": completed.returncode, "runtime": str(runtime)}


def diagnostic_gate(first: dict, repeated: dict, limits: dict) -> dict:
    checks = {
        "deterministic_repeat": first["state_sha256"] == repeated["state_sha256"],
        "wheel_lateral_constraint": first["peak_wheel_lateral_constraint_error"]
        <= limits["max_wheel_lateral_error"],
        "bridge_joint_constraint": first["peak_bridge_joint_anchor_error"]
        <= limits["max_bridge_joint_error"],
        "bounded_speed": first["max_body_speed"] <= limits["max_body_speed"],
        "bounded_angular_speed": first["max_abs_angular_speed"] <= limits["max_abs_angular_speed"],
    }
    if first["kind"] == "flat":
        checks["flat_penetration"] = (
            first["minimum_wheel_bottom_y"] >= limits["minimum_flat_wheel_bottom_y"]
        )
    elif first["kind"] == "bridge_load":
        checks["moving_bridge_contact"] = first["bridge_contact_steps"] > 0
    return {"passed": all(checks.values()), "checks": checks}


def record_engine_observation(run, name: str, kind: str, result: dict, protocol: str) -> dict:
    """Publish synthetic measurements immediately without inventing episode lifecycles."""
    path = run.directory / f"engine-{name}.json"
    payload = {
        "schema_version": "engine-observation-3.0",
        "measurement_kind": kind,
        "protocol": protocol,
        "episodes": 0,
        "evaluation_episodes": 0,
        "real_game_evaluation_episodes": 0,
        "results": result,
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return run.register_artifact(path, "engine_observation", {"measurement_kind": kind})


def execute(root: Path, expected_sha: str) -> dict:
    from gradientclimb.experiments import RunRecorder
    from gradientclimb.simulation.engine_pilot import (
        ArticulatedFixture,
        SolverSettings,
        benchmark_arm,
        engine_identity,
    )

    check_source(expected_sha)
    protocol = plan()
    start = time.perf_counter()
    with RunRecorder(
        root,
        "cycle-3-articulated-engine-pilot",
        protocol,
        seed=31000,
        algorithm="scripted_engine_diagnostics",
        environment="synthetic_articulated_engine_pilot",
        simulator_version=protocol["fixture_version"],
        calibration_version="synthetic-unmeasured",
        evidence_domain="synthetic_engineering",
        qualifies_real_game=False,
    ) as run:
        run.annotate(evaluation_episodes=0, real_game_evaluation_episodes=0, policy_decisions=0)
        run.register_artifact(PROTOCOL, "governing_protocol")
        installation = install_engine(run, protocol)
        identity = engine_identity()
        run.annotate(installation=installation, engine=identity, rapid_learning_result=None)
        settings = SolverSettings(**protocol["solver"])
        total_decisions = 0

        def progress(steps):
            run.record_progress(
                {
                    "environment_steps": steps,
                    "training_steps": 0,
                    "training_clock_seconds": 0,
                    "episodes": 0,
                    "optimizer_updates": 0,
                    "counter_provenance": "observed_completed_fixture_decisions",
                    "scripted_fixture_decisions": steps,
                    "fidelity_level": "synthetic_articulated_geometry_with_debug_render",
                }
            )
            if time.perf_counter() - start >= protocol["budgets"]["maximum_total_wall_seconds"]:
                raise TimeoutError("Engine screen reached its total command wall budget")

        diagnostics = []
        definition = protocol["diagnostics"]
        for kind in definition["fixtures"]:
            repeated = []
            for repeat_index in range(definition["repeats"]):
                fixture = ArticulatedFixture(kind, definition["repeat_seed"], settings)
                for step in range(definition["decisions"]):
                    action = (
                        0
                        if kind == "bridge_load"
                        else (1 if step < definition["decisions"] * 0.7 else 2)
                    )
                    fixture.step(action)
                    total_decisions += 1
                    if total_decisions % 60 == 0:
                        progress(total_decisions)
                result = fixture.diagnostics()
                repeated.append(result)
                record_engine_observation(
                    run,
                    f"diagnostic-{kind}-repeat-{repeat_index}",
                    "diagnostic_repeat",
                    result,
                    protocol["protocol_version"],
                )
                progress(total_decisions)
            gate = diagnostic_gate(*repeated, definition["engineering_sanity_limits"])
            diagnostic = {"kind": kind, "repeats": repeated, "gate": gate}
            diagnostics.append(diagnostic)
            record_engine_observation(
                run,
                f"diagnostic-{kind}",
                "diagnostic_gate",
                diagnostic,
                protocol["protocol_version"],
            )
            if not gate["passed"]:
                raise RuntimeError(f"Synthetic diagnostic gate failed: {kind}: {gate['checks']}")
        arms = []
        screen = protocol["screen"]
        for kind in screen["fixtures"]:
            for worlds in screen["world_counts"]:
                for render in (False, True):
                    if (
                        time.perf_counter() - start
                        >= protocol["budgets"]["maximum_total_wall_seconds"]
                    ):
                        raise TimeoutError("Engine screen reached its total command wall budget")
                    result = benchmark_arm(
                        kind,
                        worlds,
                        screen["decisions_per_world"],
                        screen["seeds"][0],
                        render=render,
                        render_every=screen["render_every_decisions"],
                        wall_limit_seconds=screen["maximum_wall_seconds_per_arm"],
                        settings=settings,
                        progress=lambda steps, base=total_decisions: progress(base + steps),
                    )
                    total_decisions += result["scripted_decisions"]
                    arms.append(result)
                    record_engine_observation(
                        run,
                        f"arm-{kind}-{worlds}-render-{int(render)}",
                        "throughput_arm",
                        result,
                        protocol["protocol_version"],
                    )
                    for name in (
                        "physics_substeps_per_wall_second",
                        "scripted_decisions_per_wall_second",
                        "rendered_observations_per_wall_second",
                    ):
                        run.metric(name, result[name], fixture=kind, worlds=worlds, render=render)
        summary = {
            "installation": installation,
            "engine": identity,
            "diagnostics": diagnostics,
            "arms": arms,
            "selected_for_measured_fixture_work": any(
                arm["completed"] and arm["render"] for arm in arms
            ),
            "scope": "synthetic_articulated_engine_pilot",
            "qualifies_game_fidelity": False,
            "fidelity_level": "synthetic_articulated_geometry_with_debug_render",
            "rapid_learning_result": None,
            "competence_measurement": "Not measured; independent real-game evaluation required",
            "selection_basis": "Synthetic engineering sanity and bounded usability only",
            "pilot_elapsed_seconds": time.perf_counter() - start,
            "pilot_clock_definition": (
                "After protocol/source verification through the last measured arm; "
                "caller imports and final result serialization/sealing are separate overhead"
            ),
        }
        path = run.directory / "engine-pilot-results.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        run.register_artifact(path, "engine_pilot_results")
        run.finalize(
            episodes=0,
            evaluation_episodes=0,
            real_game_evaluation_episodes=0,
            training_steps=0,
            optimizer_updates=0,
            policy_decisions=0,
            environment_steps=total_decisions,
            scripted_fixture_decisions=total_decisions,
            **summary,
        )
        return {"run_id": run.run_id, "directory": str(run.directory), **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-sha")
    args = parser.parse_args(argv)
    result = execute(args.root, args.expected_sha or "") if args.execute else {"plan": plan()}
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
