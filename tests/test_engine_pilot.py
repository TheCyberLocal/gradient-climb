"""Pilot governance tests and optional real-engine numerical fixture checks."""

import importlib.util
import json
from pathlib import Path

import pytest

from gradientclimb.simulation.engine_pilot import SolverSettings

SPEC = importlib.util.spec_from_file_location(
    "run_engine_pilot", Path(__file__).resolve().parents[1] / "scripts/run_engine_pilot.py"
)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_dry_run_has_no_engine_import_installation_or_artifacts(monkeypatch, capsys):
    monkeypatch.setattr(PILOT, "execute", lambda *_: pytest.fail("dry run attempted execution"))
    result = PILOT.main([])
    assert result["plan"]["screen"]["maximum_arms"] == 12
    assert result["plan"]["budgets"]["native_interactions"] == 0
    assert result["plan"]["candidate"]["version"] == "2.3.10"
    capsys.readouterr()


@pytest.mark.parametrize(
    "settings", [{"substeps": 0}, {"substeps": 1.5}, {"decision_seconds": float("nan")}]
)
def test_invalid_solver_settings_fail(settings):
    with pytest.raises(ValueError):
        SolverSettings(**settings)


def test_execution_rejects_missing_source_pin_before_work():
    with pytest.raises(ValueError, match="expected committed source SHA"):
        PILOT.check_source("")


def test_diagnostic_gate_rejects_disconnection_and_requires_bridge_contact():
    row = {
        "kind": "bridge_load",
        "state_sha256": "a",
        "peak_wheel_lateral_constraint_error": 0.0,
        "peak_bridge_joint_anchor_error": 0.0,
        "max_body_speed": 1.0,
        "max_abs_angular_speed": 1.0,
        "bridge_contact_steps": 2,
    }
    limits = PILOT.plan()["diagnostics"]["engineering_sanity_limits"]
    assert PILOT.diagnostic_gate(row, row, limits)["passed"]
    assert not PILOT.diagnostic_gate({**row, "peak_bridge_joint_anchor_error": 1.0}, row, limits)[
        "passed"
    ]
    assert not PILOT.diagnostic_gate({**row, "bridge_contact_steps": 0}, row, limits)["passed"]


@pytest.mark.parametrize("failure", [None, "diagnostic_gate", "second_arm"])
def test_synthetic_measurements_are_retained_without_evaluation_episodes(
    tmp_path, monkeypatch, failure
):
    """Run the real recorder with fake physics; no installation or game input occurs."""
    import pyarrow.parquet as pq

    from gradientclimb.experiments import RunRecorder, list_runs, verify_run
    from gradientclimb.simulation import engine_pilot

    protocol = PILOT.plan()
    protocol["diagnostics"].update(fixtures=["flat"], decisions=2)
    protocol["screen"].update(fixtures=["flat"], world_counts=[1], decisions_per_world=2)
    monkeypatch.setattr(PILOT, "plan", lambda: protocol)
    monkeypatch.setattr(PILOT, "check_source", lambda value: value)
    monkeypatch.setattr(PILOT, "install_engine", lambda *_: {"test_double": True})
    monkeypatch.setattr(engine_pilot, "engine_identity", lambda: {"test_double": True})
    monkeypatch.setattr(
        RunRecorder, "evaluation", lambda *_: pytest.fail("fixtures became evaluation episodes")
    )

    class Fixture:
        def __init__(self, *_):
            self.steps = 0

        def step(self, _action):
            self.steps += 1

        def diagnostics(self):
            return {
                "kind": "flat",
                "state_sha256": "identical-test-state",
                "peak_wheel_lateral_constraint_error": 1 if failure == "diagnostic_gate" else 0,
                "peak_bridge_joint_anchor_error": 0,
                "minimum_wheel_bottom_y": 0,
                "max_body_speed": 0,
                "max_abs_angular_speed": 0,
                "completed_steps": self.steps,
            }

    def arm(kind, worlds, decisions, seed, *, render, progress, **_):
        if render and failure == "second_arm":
            raise RuntimeError("test arm failure")
        progress(worlds * decisions)
        return {
            "kind": kind,
            "worlds": worlds,
            "seed": seed,
            "render": render,
            "completed": True,
            "scripted_decisions": worlds * decisions,
            "physics_substeps_per_wall_second": 1,
            "scripted_decisions_per_wall_second": 1,
            "rendered_observations_per_wall_second": int(render),
        }

    monkeypatch.setattr(engine_pilot, "ArticulatedFixture", Fixture)
    monkeypatch.setattr(engine_pilot, "benchmark_arm", arm)
    if failure:
        with pytest.raises(RuntimeError, match="diagnostic gate failed|test arm failure"):
            PILOT.execute(tmp_path, "a" * 40)
    else:
        PILOT.execute(tmp_path, "a" * 40)
    record = list_runs(tmp_path)[0]
    directory = tmp_path / "runs" / record["run_id"]
    assert record["status"] == ("failed" if failure else "completed")
    assert record["episodes"] == record["training_steps"] == record["optimizer_updates"] == 0
    assert record["summary"]["evaluation_episodes"] == 0
    assert record["summary"]["real_game_evaluation_episodes"] == 0
    assert record["evaluation_results"] == []
    assert pq.read_table(directory / "evaluations.parquet").num_rows == 0
    observations = [a for a in record["artifact_manifest"] if a["kind"] == "engine_observation"]
    assert len(observations) == {None: 5, "diagnostic_gate": 3, "second_arm": 4}[failure]
    payloads = [json.loads((directory / a["path"]).read_text()) for a in observations]
    assert all(p["episodes"] == p["evaluation_episodes"] == 0 for p in payloads)
    if not failure:
        assert record["environment_steps"] == record["summary"]["scripted_fixture_decisions"] == 8
    assert verify_run(tmp_path, record["run_id"])["valid"]


@pytest.mark.parametrize("kind", ["flat", "bridge_load", "bridge_traverse"])
def test_optional_engine_fixture_is_deterministic_and_articulated(kind):
    pytest.importorskip("Box2D", reason="Optional pinned engine pilot runtime is not installed")
    from gradientclimb.simulation.engine_pilot import ArticulatedFixture

    first, second = ArticulatedFixture(kind, 23), ArticulatedFixture(kind, 23)
    for _ in range(30):
        first.step(1)
        second.step(1)
    assert first.diagnostics() == second.diagnostics()
    assert first.world.bodyCount == (4 if kind == "flat" else 16)
    assert first.world.jointCount == (2 if kind == "flat" else 15)
    assert first.render().size == (320, 180)
