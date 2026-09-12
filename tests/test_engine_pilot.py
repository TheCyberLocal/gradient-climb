"""Pilot governance tests and optional real-engine numerical fixture checks."""

import importlib.util
import json
from pathlib import Path

import pytest

from gradientclimb.simulation.engine_pilot import (
    GroundSupportDiagnostics,
    SolverSettings,
    ground_extent,
)

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


def test_protocol_must_match_source_identity_and_declared_extent():
    protocol = PILOT.plan()
    PILOT.check_protocol(protocol)
    assert ground_extent() == (-1020, 1020)
    protocol["fixture_version"] = "articulated-engine-fixtures-3.0"
    with pytest.raises(ValueError, match="fixture versions"):
        PILOT.check_protocol(protocol)
    protocol = PILOT.plan()
    protocol["screen"]["decisions_per_world"] = 601
    with pytest.raises(ValueError, match="horizons"):
        PILOT.check_protocol(protocol)


def test_successor_preserves_solver_thresholds_schedule_and_budgets():
    prior = json.loads(
        (PILOT.PROJECT / "experiments/definitions/cycle-3-engine-pilot.json").read_text()
    )
    successor = PILOT.plan()
    for key in ("solver", "screen", "budgets", "candidate"):
        assert successor[key] == prior[key]
    for key in ("engineering_sanity_limits", "fixtures", "repeat_seed", "decisions", "repeats"):
        assert successor["diagnostics"][key] == prior["diagnostics"][key]
    assert successor["predecessor"]["run_id"] == "a88aeae4-ccdf-4f84-a178-bc5e58feebd8"
    assert not successor["data_collected"]


def test_domain_exit_does_not_become_penetration_and_gap_evidence_is_retained():
    lower, upper = ground_extent()
    flat = GroundSupportDiagnostics([((lower, 0), (upper, 0))])
    flat.observe([(0, 0.29)], decision=1, physics_substep=1, simulated_seconds=1 / 120)
    flat.observe([(upper + 1, -4)], decision=1, physics_substep=2, simulated_seconds=1 / 60)
    observed = flat.to_record()
    assert not observed["ground_domain_contained"]
    assert observed["outside_ground_domain_substeps"] == 1
    assert observed["first_ground_domain_exit"]["physics_substep"] == 2
    assert observed["minimum_wheel_bottom_y"] == -4.3
    assert observed["minimum_wheel_bottom_over_static_floor_y"] == pytest.approx(-0.01)
    assert observed["minimum_wheel_bottom_over_static_floor_evidence"]["simulated_seconds"] == (
        1 / 120
    )
    bridge = GroundSupportDiagnostics([((lower, 0), (6, 0)), ((14, 0), (upper, 0))])
    bridge.observe([(10, -3)], decision=1, physics_substep=1, simulated_seconds=1 / 120)
    observed = bridge.to_record()
    assert observed["ground_domain_contained"]
    assert observed["minimum_wheel_bottom_over_static_floor_y"] is None
    assert observed["minimum_wheel_bottom_y"] == observed["minimum_wheel_bottom_in_bridge_gap_y"]
    assert observed["minimum_wheel_bottom_in_bridge_gap_y"] == -3.3
    assert observed["bridge_gap_substeps"] == 1
    assert observed["first_bridge_gap_entry"]["wheel_center"] == [10, -3]


def test_flat_gate_retains_penetration_threshold_and_requires_domain_coverage():
    row = {
        "kind": "flat",
        "state_sha256": "a",
        "peak_wheel_lateral_constraint_error": 0,
        "peak_bridge_joint_anchor_error": 0,
        "max_body_speed": 1,
        "max_abs_angular_speed": 1,
        "minimum_wheel_bottom_over_static_floor_y": -0.049,
        "ground_domain_contained": True,
    }
    limits = PILOT.plan()["diagnostics"]["engineering_sanity_limits"]
    assert limits["minimum_flat_wheel_bottom_y"] == -0.05
    assert PILOT.diagnostic_gate(row, row, limits)["passed"]
    assert not PILOT.diagnostic_gate(
        {**row, "minimum_wheel_bottom_over_static_floor_y": -0.051}, row, limits
    )["checks"]["flat_penetration"]
    assert not PILOT.diagnostic_gate(row, {**row, "ground_domain_contained": False}, limits)[
        "checks"
    ]["ground_domain_contained"]


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
                "minimum_wheel_bottom_over_static_floor_y": 0,
                "ground_domain_contained": True,
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
    support = first.diagnostics()
    assert support["checked_physics_substeps"] == 60
    assert support["ground_domain_x"] == [-1020, 1020]
    assert support["ground_domain_contained"]
    if kind == "bridge_load":
        assert support["bridge_gap_substeps"] > 0
