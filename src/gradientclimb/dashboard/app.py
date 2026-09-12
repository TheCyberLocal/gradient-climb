"""Read-only FastAPI endpoints and bundled, dependency-free browser assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from gradientclimb.dashboard.efficiency_report import compare_reports
from gradientclimb.experiments import connect_database, load_run, query_metrics, verify_run
from gradientclimb.experiments.efficiency import load_studies


def _rows(connection, sql: str, parameters: list[Any] | None = None) -> list[dict]:
    result = connection.execute(sql, parameters or [])
    names = [column[0] for column in result.description]
    rows = [dict(zip(names, values, strict=True)) for values in result.fetchall()]
    for row in rows:
        for key in list(row):
            if key.endswith("_json"):
                value = row.pop(key)
                row[key.removesuffix("_json")] = json.loads(value) if value is not None else None
    return rows


def create_app(root: str | Path = "artifacts") -> FastAPI:
    """Create a localhost dashboard; it never mutates run or artifact records."""
    artifact_root = Path(root).resolve()
    assets = Path(__file__).resolve().parent
    app = FastAPI(title="GradientClimb research API", version="0.1.0")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/dashboard.css", include_in_schema=False)
    def css():
        return FileResponse(assets / "dashboard.css", media_type="text/css")

    @app.get("/dashboard.js", include_in_schema=False)
    def javascript():
        return FileResponse(assets / "dashboard.js", media_type="text/javascript")

    @app.get("/api/runs")
    def runs(
        algorithm: str | None = None, environment: str | None = None, status: str | None = None
    ):
        with connect_database(artifact_root) as connection:
            # Empty databases expose only identity columns; no filter is needed.
            if connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0:
                return []
            filters = {"algorithm": algorithm, "environment": environment, "status": status}
            active = [(key, value) for key, value in filters.items() if value is not None]
            where = " WHERE " + " AND ".join(f"{key} = ?" for key, _ in active) if active else ""
            return _rows(
                connection,
                "SELECT * FROM runs" + where + " ORDER BY start_time DESC",
                [value for _, value in active],
            )

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        try:
            return load_run(artifact_root, run_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=404, detail="Run not found") from error

    @app.get("/api/runs/{run_id}/integrity")
    def integrity(run_id: str):
        try:
            load_run(artifact_root, run_id)
            return verify_run(artifact_root, run_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=404, detail="Run not found") from error

    @app.get("/api/metrics")
    def metrics(run_id: str | None = None, name: str | None = None):
        try:
            result = query_metrics(artifact_root, run_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Invalid run identifier") from error
        return [row for row in result if name is None or row["name"] == name]

    @app.get("/api/summary")
    def summary():
        with connect_database(artifact_root) as connection:
            total = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
            counts = _rows(connection, "SELECT status, count(*) AS count FROM runs GROUP BY status")
            aggregates = (
                _rows(
                    connection,
                    "SELECT sum(environment_steps) AS environment_steps, sum(wall_clock_seconds) AS wall_clock_seconds FROM runs",
                )[0]
                if total
                else {"environment_steps": None, "wall_clock_seconds": None}
            )
            return {
                "total_runs": total,
                "status_counts": {row["status"]: row["count"] for row in counts},
                "metric_count": connection.execute("SELECT count(*) FROM metrics").fetchone()[0],
                "evaluation_count": connection.execute(
                    "SELECT count(*) FROM evaluations"
                ).fetchone()[0],
                **aggregates,
                "source": "Canonical run records and Parquet, queried through DuckDB",
            }

    @app.get("/api/learning-efficiency")
    def learning_efficiency():
        try:
            return load_studies(artifact_root)
        except (ValueError, OSError) as error:
            raise HTTPException(
                status_code=409,
                detail="Cycle 3 efficiency evidence could not be validated; "
                "no efficiency claims are available from this request.",
            ) from error

    @app.get("/api/learning-efficiency/snapshot")
    def learning_efficiency_snapshot():
        reports = learning_efficiency()
        return {"studies": reports, "comparisons": compare_reports(reports)}

    @app.get("/api/analytics/{view}")
    def analytics(
        view: str, run_id: str | None = None, limit: int = Query(default=10000, ge=1, le=100000)
    ):
        tables = {
            "resources": "telemetry",
            "evaluations": "evaluations",
            "controls": "trajectories",
        }
        if view not in tables:
            raise HTTPException(status_code=404, detail="Analytical view not found")
        parameters: list[Any] = []
        where = ""
        if run_id:
            where = " WHERE run_id = ?"
            parameters.append(run_id)
        parameters.append(limit)
        with connect_database(artifact_root) as connection:
            table = tables[view]
            # Table names are chosen from the allowlist, never caller SQL.
            rows = _rows(
                connection,
                f"SELECT * FROM {table}{where} ORDER BY elapsed_seconds LIMIT ?",
                parameters,
            )
            count = connection.execute(
                f"SELECT count(*) FROM {table}{where}", parameters[:-1]
            ).fetchone()[0]
            return {"rows": rows, "total_rows": count, "truncated": count > len(rows)}

    return app
