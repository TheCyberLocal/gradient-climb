"""Presentation cannot invent competence, replace missingness, or reinterpret costs."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient
from test_dashboard import _run_dashboard_checks

from gradientclimb.dashboard.app import create_app
from gradientclimb.dashboard.efficiency_report import (
    COST_LABELS,
    compare_reports,
    render_html,
    render_markdown,
    threshold_rows,
    value_text,
    write_report,
)
from gradientclimb.experiments.efficiency import CostVector


def report_fixture():
    """Synthetic presentation fixture only; it never enters the canonical run store."""
    point = {
        "threshold_m": 500,
        "status": "reached",
        "time_seconds": 301.27,
        "interval_lower_seconds": None,
        "censor_seconds": None,
        "budget_seconds": 3600,
        "checkpoint_sha256": "a" * 64,
        "own_cost": {"cpu_core_seconds": 42.125, "real_game_interaction_seconds": 0},
        "prior_cost": {"cpu_core_seconds": 2.5, "real_game_interaction_seconds": 60},
        "combined_cost": {"cpu_core_seconds": 44.625, "real_game_interaction_seconds": 60},
    }
    return {
        "schema_version": "efficiency-analysis-3.0",
        "study_id": "synthetic-renderer-fixture",
        "training_run_id": "synthetic-run",
        "prior_class": "demonstration-assisted",
        "profile_id": "fixed-profile",
        "protocol_id": "prospective-protocol",
        "evaluation_split": "development",
        "compute_comparison_scope": "process-cpu/device-gpu-unknown",
        "comparison_contract_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "wall_clock_boundary": "command_start_to_frozen_checkpoint",
        "lineage_complete": True,
        "provenance_status": "supplied_attestations_not_file_verified",
        "policy_prior_roots": ["human-demo"],
        "prior_ledger": [
            {
                "cost_id": "human-demo",
                "kind": "human-demonstration",
                "parents": [],
                "elapsed_seconds": 11.5,
                "human_seconds": 60,
            }
        ],
        "evaluation_cost": {"cpu_core_seconds": 10},
        "warnings": ["Synthetic renderer test; no gameplay evidence"],
        "checkpoint_curve": [
            {
                "time_seconds": 301.27,
                "median_distance_m": 500,
                "checkpoint_sha256": "a" * 64,
                "verification_elapsed_seconds": 650,
                "evaluation_elapsed_seconds": 348.73,
                "evaluation_run_id": "synthetic-evaluation",
            }
        ],
        "thresholds": [
            point,
            {
                **point,
                "threshold_m": 1000,
                "status": "right_censored",
                "time_seconds": None,
                "censor_seconds": 301.27,
            },
            {
                **point,
                "threshold_m": 2000,
                "status": "not_evaluable",
                "time_seconds": None,
                "censor_seconds": None,
                "checkpoint_sha256": None,
                "own_cost": {},
                "combined_cost": {},
            },
        ],
    }


def test_formatting_preserves_missing_zero_and_precision():
    assert value_text(None) == "Not measured"
    assert value_text(0) == "0"
    assert value_text(301.27) == "301.27"
    point = threshold_rows(report_fixture())[0]
    assert point[2:6] == [301.27, None, None, 3600]


def test_all_canonical_cost_dimensions_have_explicit_display_labels():
    assert set(COST_LABELS) == set(CostVector.model_fields)


@pytest.mark.parametrize("renderer", [render_html, render_markdown])
def test_report_is_presentation_of_canonical_status_costs_and_clocks(renderer):
    fixture = report_fixture()
    before = copy.deepcopy(fixture)
    text = renderer([fixture])
    assert fixture == before
    for required in [
        "Reached",
        "Right censored",
        "Not evaluable",
        "Not measured",
        "301.27",
        "3600",
        "44.625",
        "42.125",
        "348.73",
        "650",
        "Evaluation costs (separate)",
        "Prior elapsed clocks and ancestry",
        "supplied_attestations_not_file_verified",
        "Synthetic renderer test",
    ]:
        assert required in text
    for label in COST_LABELS.values():
        assert label in text
    assert "312.77" not in text  # Own and prior elapsed must never be silently summed.


@pytest.mark.parametrize("renderer", [render_html, render_markdown])
def test_source_validation_errors_are_visible_and_no_empty_success_is_fabricated(renderer):
    text = renderer(
        [
            {
                "study_id": "bad-seal",
                "error": "Hash mismatch",
                "thresholds": [],
                "checkpoint_curve": [],
            }
        ]
    )
    assert "Evidence validation failed" in text
    assert "Hash mismatch" in text
    assert "No competence or efficiency claim" in text
    assert "No validated study" not in text
    assert "not yet measured" in renderer([])


@pytest.mark.parametrize("renderer", [render_html, render_markdown])
def test_untrusted_identifiers_and_warnings_cannot_inject_markup(renderer):
    fixture = report_fixture()
    fixture["study_id"] = '<img src=x onerror="alert(1)"> | unsafe'
    fixture["warnings"] = ["</script><script>bad()</script>"]
    text = renderer([fixture])
    assert "<img src=x" not in text
    assert "<script>bad()" not in text
    assert "&lt;img" in text


def test_export_preserves_exact_analysis_and_refuses_replacement(tmp_path):
    fixture = [report_fixture()]
    paths = write_report(fixture, tmp_path / "new-report")
    assert {path.name for path in paths} == {
        "report.json",
        "report.md",
        "report.html",
        "comparisons.json",
    }
    assert json.loads((tmp_path / "new-report/report.json").read_text()) == fixture
    assert json.loads((tmp_path / "new-report/comparisons.json").read_text()) == compare_reports(
        fixture
    )
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        write_report([], tmp_path / "new-report")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_nonfinite_values_cannot_produce_successful_json_export(tmp_path):
    fixture = report_fixture()
    fixture["thresholds"][0]["time_seconds"] = float("nan")
    with pytest.raises(ValueError):
        write_report([fixture], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_api_returns_exact_canonical_analysis_and_one_snapshot(monkeypatch, tmp_path):
    fixture = [report_fixture()]
    calls = []

    def loader(root):
        calls.append(root)
        return copy.deepcopy(fixture)

    monkeypatch.setattr("gradientclimb.dashboard.app.load_studies", loader)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/learning-efficiency").json() == fixture
    calls.clear()
    snapshot = client.get("/api/learning-efficiency/snapshot").json()
    assert snapshot == {"studies": fixture, "comparisons": compare_reports(fixture)}
    assert calls == [tmp_path.resolve()]


def test_api_reports_loader_failure_without_breaking_historical_routes(monkeypatch, tmp_path):
    def broken_loader(_root):
        raise ValueError("private local path should not appear in API exception")

    monkeypatch.setattr("gradientclimb.dashboard.app.load_studies", broken_loader)
    client = TestClient(create_app(tmp_path))
    result = client.get("/api/learning-efficiency")
    assert result.status_code == 409
    assert "private local path" not in result.text
    assert client.get("/api/runs").json() == []


def test_dashboard_renders_exact_states_and_separate_cost_scopes():
    fixture = json.dumps(report_fixture())
    _run_dashboard_checks(
        "const fixture="
        + fixture
        + r""";
assert.match(realEfficiency(),/Evidence loading/);
assert.doesNotMatch(realEfficiency(),/not yet measured/);
state.efficiencyLoaded=true; state.efficiency=[fixture]; state.efficiencyComparisons=[];
const source=JSON.stringify(fixture), rendered=realEfficiency();
for(const word of ['Reached','Right censored','Not evaluable','Not measured','301.27 s',
 '3,600 s','44.625','42.125','Own','Prior','Combined','Evaluation costs (separate)',
 '348.73 s','650 s','human-demo','Synthetic renderer test'])assert.ok(rendered.includes(word),word);
assert.equal(JSON.stringify(fixture),source);
assert.doesNotMatch(rendered,/312\.77/);
fixture.study_id='<script>alert(1)</script>';
assert.doesNotMatch(realEfficiency(),/<script>alert/);
state.efficiency=[{study_id:'bad-seal',error:'Invalid seal',thresholds:[],checkpoint_curve:[]}];
assert.match(realEfficiency(),/Evidence validation failed/);
assert.match(realEfficiency(),/Invalid seal/);
state.efficiency=[];
assert.match(realEfficiency(),/Real competence efficiency is not yet measured/);
state.efficiencyError='Invalid response';
assert.match(realEfficiency(),/Could not validate efficiency evidence/);
assert.match(realEfficiency(),/Invalid response/);
assert.ok(titles.learning); assert.ok(titles['real-efficiency']);
"""
    )
