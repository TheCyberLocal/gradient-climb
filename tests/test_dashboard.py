"""API reconciliation, empty-state and read-only provenance checks."""

from fastapi.testclient import TestClient

from gradientclimb.dashboard import create_app
from gradientclimb.experiments import RunRecorder, run_synthetic


def test_empty_dashboard_is_truthful_and_assets_are_local(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "GradientClimb" in response.text
    assert "cdn" not in response.text
    assert client.get("/dashboard.css").status_code == 200
    script = client.get("/dashboard.js")
    assert script.status_code == 200
    assert "Not measured" in script.text
    assert client.get("/api/runs").json() == []
    assert client.get("/api/runs?algorithm=ppo").json() == []
    assert client.get("/api/metrics").json() == []
    assert client.get("/api/summary").json()["wall_clock_seconds"] is None
    assert client.get("/api/analytics/resources").json()["rows"] == []


def test_dashboard_queries_canonical_records_and_filters(tmp_path):
    slow = run_synthetic(tmp_path, seed=1, gain=0.05, steps=4)
    fast = run_synthetic(tmp_path, seed=2, gain=0.4, steps=4)
    client = TestClient(create_app(tmp_path))
    runs = client.get("/api/runs").json()
    assert {row["run_id"] for row in runs} == {slow["run_id"], fast["run_id"]}
    assert all(isinstance(row["configuration"], dict) for row in runs)
    assert client.get("/api/runs?algorithm=missing").json() == []
    assert len(client.get("/api/runs?algorithm=scalar-gradient").json()) == 2
    metrics = client.get(f"/api/metrics?run_id={slow['run_id']}&name=quality").json()
    assert len(metrics) == 4
    assert {row["run_id"] for row in metrics} == {slow["run_id"]}
    assert all(row["name"] == "quality" for row in metrics)
    summary = client.get("/api/summary").json()
    assert summary["total_runs"] == 2
    assert summary["environment_steps"] == 8
    assert summary["metric_count"] == 16
    assert summary["evaluation_count"] == 2
    detail = client.get(f"/api/runs/{slow['run_id']}").json()
    assert detail == slow
    assert client.get(f"/api/runs/{slow['run_id']}/integrity").json()["valid"]
    evaluation = client.get(f"/api/analytics/evaluations?run_id={fast['run_id']}").json()
    assert evaluation["rows"][0]["results"]["quality"] == fast["summary"]["final_quality"]
    assert client.get("/api/analytics/resources?limit=1").json()["truncated"]


def test_live_run_is_visible_and_no_arbitrary_sql_or_files(tmp_path):
    client = TestClient(create_app(tmp_path))
    with RunRecorder(tmp_path, "live-dashboard", {}, telemetry_interval_seconds=0) as run:
        run.metric("distance", 4)
        assert client.get("/api/runs?status=running").json()[0]["run_id"] == run.run_id
        assert client.get("/api/metrics").json()[0]["value"] == 4
        assert not client.get(f"/api/runs/{run.run_id}/integrity").json()["valid"]
    assert client.get("/api/runs/invalid").status_code == 404
    assert client.get("/api/metrics?run_id=invalid").status_code == 422
    assert client.get("/api/analytics/secrets").status_code == 404
    assert client.post("/api/runs", json={"delete": True}).status_code == 405
