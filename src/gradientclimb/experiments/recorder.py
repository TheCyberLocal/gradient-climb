"""Append-only run journals, sealed records and analytical Parquet snapshots.

The filesystem is canonical. DuckDB opens independent snapshots in memory, so a
dashboard reader never locks a trainer's writer connection. A seal detects later
tampering; filesystem permissions remain the operator's responsibility.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import traceback as traceback_module
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from gradientclimb.artifacts import canonical_json, hash_config, sha256_file
from gradientclimb.telemetry import capture_provenance, resource_sample
from gradientclimb.telemetry.resources import ResourceAccumulator

from .schemas import (
    ArtifactRecord,
    EvaluationRecord,
    MetricRecord,
    RunRecord,
    SystemTelemetryRecord,
    TrajectoryRecord,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _write_new(path: Path, value: Any) -> None:
    # Publish an entire immutable JSON object atomically. Readers must never see
    # a half-written run.json while the trainer is finalizing it.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


PARQUET_SCHEMAS = {
    "metrics": pa.schema(
        [
            ("schema_version", pa.string()),
            ("run_id", pa.string()),
            ("name", pa.string()),
            ("value", pa.float64()),
            ("step", pa.int64()),
            ("elapsed_seconds", pa.float64()),
            ("timestamp", pa.string()),
            ("dimensions_json", pa.string()),
        ]
    ),
    "telemetry": pa.schema(
        [
            ("schema_version", pa.string()),
            ("run_id", pa.string()),
            ("timestamp", pa.string()),
            ("elapsed_seconds", pa.float64()),
            ("cpu_percent", pa.float64()),
            ("per_core_cpu_percent", pa.list_(pa.float64())),
            ("ram_used_bytes", pa.int64()),
            ("process_rss_bytes", pa.int64()),
            ("gpu_percent", pa.float64()),
            ("vram_used_bytes", pa.int64()),
            ("measurements_json", pa.string()),
        ]
    ),
    "evaluations": pa.schema(
        [
            ("schema_version", pa.string()),
            ("evaluation_id", pa.string()),
            ("run_id", pa.string()),
            ("timestamp", pa.string()),
            ("elapsed_seconds", pa.float64()),
            ("environment", pa.string()),
            ("seed", pa.int64()),
            ("episodes", pa.int64()),
            ("results_json", pa.string()),
            ("protocol", pa.string()),
            ("checkpoint_hash", pa.string()),
            ("metadata_json", pa.string()),
        ]
    ),
    "trajectories": pa.schema(
        [
            ("schema_version", pa.string()),
            ("run_id", pa.string()),
            ("episode_id", pa.string()),
            ("step", pa.int64()),
            ("timestamp", pa.string()),
            ("elapsed_seconds", pa.float64()),
            ("gas", pa.bool_()),
            ("brake", pa.bool_()),
            ("action_duration_seconds", pa.float64()),
            ("observation_json", pa.string()),
            ("reward", pa.float64()),
            ("terminated", pa.bool_()),
            ("truncated", pa.bool_()),
            ("termination_reason", pa.string()),
            ("frame_artifact_id", pa.string()),
        ]
    ),
}


def _columnar(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key + "_json" if isinstance(value, dict) else key: canonical_json(value)
        if isinstance(value, dict)
        else value
        for key, value in row.items()
    }


class RunRecorder:
    """Own one experiment run; finalized records cannot be changed by this API.

    ``root`` is the artifact root (normally ``artifacts``), not a run directory.
    Configuration must contain JSON values. Caller metadata matching RunRecord
    fields (e.g. vehicle_profile) is promoted; other metadata remains searchable
    in the run's metadata field. Source and identity fields cannot be overridden.
    """

    def __init__(
        self,
        root: Path,
        experiment_id: str,
        config: dict[str, Any],
        seed: int = 0,
        algorithm: str = "synthetic",
        environment: str = "synthetic",
        **metadata: Any,
    ) -> None:
        self.root = Path(root).resolve()
        self.run_id = str(uuid.uuid4())
        self.directory = self.root / "runs" / self.run_id
        self._clock = time.perf_counter()
        self._closed = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._artifacts: list[ArtifactRecord] = []
        self._evaluations: list[EvaluationRecord] = []
        self._progress_state: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        source_root = Path(metadata.pop("source_root", Path.cwd()))
        telemetry_interval = float(metadata.pop("telemetry_interval_seconds", 1.0))
        if not 0 <= telemetry_interval < float("inf"):
            raise ValueError("telemetry_interval_seconds must be finite and non-negative")
        self._gpu_interval = float(metadata.pop("resource_gpu_interval_seconds", 10.0))
        if not 0 <= self._gpu_interval < float("inf"):
            raise ValueError("resource_gpu_interval_seconds must be finite and nonnegative")
        gap = float(metadata.pop("resource_max_gap_seconds", 30.0))
        self._resources = ResourceAccumulator(monotonic_origin=self._clock, max_gap_seconds=gap)
        self._last_gpu_sample: float | None = None
        metadata["resource_sampling"] = {
            "version": "resources-3.0",
            "gpu_interval_seconds": self._gpu_interval,
            "max_interpolation_gap_seconds": gap,
        }
        configuration = json.loads(canonical_json(config))
        source_state: dict[str, Any] = {}
        supplied = {
            "run_id": self.run_id,
            "experiment_id": experiment_id,
            "status": "running",
            "start_time": _now(),
            "configuration": configuration,
            "config_sha256": hash_config(configuration),
            "seed": seed,
            "algorithm": algorithm,
            "environment": environment,
            **capture_provenance(source_root, source_state=source_state),
        }
        allowed = {
            "algorithm_version",
            "policy_architecture",
            "environment_version",
            "simulator_version",
            "calibration_version",
            "vehicle_profile",
            "map_profile",
            "parent_checkpoint",
            "parent_run",
        }
        supplied.update({key: metadata.pop(key) for key in list(metadata) if key in allowed})
        supplied["metadata"] = metadata
        self._record = RunRecord.model_validate(supplied)
        # Validate arbitrary metadata before creating any files.
        start = self._record.model_dump(mode="json")
        canonical_json(start)
        # Detach nested caller-owned metadata just as configuration is detached.
        self._record = RunRecord.model_validate(json.loads(canonical_json(start)))
        self.directory.mkdir(parents=True, exist_ok=False)
        _write_new(self.directory / "run-start.json", start)
        _write_new(self.directory / "config.json", configuration)
        _write_new(self.directory / "source-state.json", source_state)
        self._streams = {
            name: (self.directory / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n")
            for name in PARQUET_SCHEMAS
        }
        self._streams["progress"] = (self.directory / "progress.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        )
        self.register_artifact(self.directory / "source-state.json", "source_provenance")
        self.telemetry()
        if telemetry_interval:
            self._thread = threading.Thread(
                target=self._sample_loop, args=(telemetry_interval,), daemon=True
            )
            self._thread.start()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if not self._closed:
            failure = (
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": "".join(
                        traceback_module.format_exception(exc_type, exc, traceback)
                    ),
                }
                if exc is not None
                else {}
            )
            try:
                if failure:
                    _write_new(self.directory / "failure.json", failure)
                self.finalize(
                    status="cancelled"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed"
                    if exc is not None
                    else "completed",
                    **failure,
                )
            except BaseException as finalization_error:
                cleanup_errors = self._abandon()
                if exc is None:
                    raise
                # Storage or cleanup failure must not replace the actual learner exception.
                exc.add_note(
                    f"Run finalization also failed; preserved unsealed journals at {self.directory}: "
                    f"{type(finalization_error).__name__}: {finalization_error}; "
                    f"cleanup_errors={cleanup_errors}"
                )
                logging.getLogger(__name__).exception("Run finalization failed")
        return False

    def _abandon(self) -> list[str]:
        errors = []
        self._stop.set()
        if self._thread:
            try:
                self._thread.join(timeout=5)
            except BaseException as error:  # noqa: BLE001 - release streams despite a second interrupt
                errors.append(f"thread: {type(error).__name__}: {error}")
        self._closed = True
        for stream in self._streams.values():
            try:
                stream.close()
            except BaseException as error:  # noqa: BLE001 - release every stream, retaining evidence
                errors.append(f"stream: {type(error).__name__}: {error}")
        return errors

    def record_progress(self, snapshot: dict[str, Any]) -> None:
        """Persist completed-operation counters independently of metric or model writes.

        Caller updates its own counters every operation; these bounded-cadence,
        fsynced snapshots support explicit lower bounds after abrupt termination.
        """
        with self._lock:
            self._ensure_open()
            canonical_json(snapshot)
            counters = {}
            for key in ("environment_steps", "training_steps", "episodes", "optimizer_updates"):
                value = snapshot[key]
                if type(value) is not int or value < getattr(self._record, key):
                    raise ValueError(f"Progress {key} must be a monotonic nonnegative integer")
                counters[key] = value
            self._record = self._record.model_copy(update=counters)
            self._progress_state = json.loads(canonical_json(snapshot))
            row = {"run_id": self.run_id, "elapsed_seconds": self.elapsed_seconds, **snapshot}
            self._streams["progress"].write(canonical_json(row) + "\n")
            self._streams["progress"].flush()
            os.fsync(self._streams["progress"].fileno())

    def annotate(self, **summary: Any) -> None:
        """Retain lifecycle diagnostics for either explicit or exception finalization."""
        self._ensure_open()
        self._summary.update(json.loads(canonical_json(summary)))

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._clock

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Run {self.run_id} is finalized and immutable")

    def _append(self, name: str, record: Any) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            row = record.model_dump(mode="json")
            self._streams[name].write(canonical_json(row) + "\n")
            self._streams[name].flush()
            return row

    def metric(self, name: str, value: float, step: int = 0, **dimensions: Any) -> dict[str, Any]:
        return self._append(
            "metrics",
            MetricRecord(
                run_id=self.run_id,
                name=name,
                value=value,
                step=step,
                elapsed_seconds=self.elapsed_seconds,
                timestamp=_now(),
                dimensions=dimensions,
            ),
        )

    def evaluation(self, result: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(canonical_json(result))
        fields = {
            key: payload.pop(key)
            for key in list(payload)
            if key in {"environment", "seed", "episodes", "protocol", "checkpoint_hash", "metadata"}
        }
        fields.setdefault("environment", self._record.environment)
        record = EvaluationRecord(
            evaluation_id=str(uuid.uuid4()),
            run_id=self.run_id,
            timestamp=_now(),
            elapsed_seconds=self.elapsed_seconds,
            results=payload.pop("results") if "results" in payload else payload,
            **fields,
        )
        with self._lock:
            row = self._append("evaluations", record)
            self._evaluations.append(record)
        return row

    def trajectory(self, record: dict[str, Any]) -> dict[str, Any]:
        return self._append(
            "trajectories",
            TrajectoryRecord.model_validate(
                {
                    "timestamp": _now(),
                    "elapsed_seconds": self.elapsed_seconds,
                    **record,
                    "run_id": self.run_id,
                }
            ),
        )

    def telemetry(self, include_gpu: bool = False, **measurements: Any) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            due = self._gpu_interval > 0 and (
                self._last_gpu_sample is None
                or self.elapsed_seconds - self._last_gpu_sample >= self._gpu_interval
            )
            sample = resource_sample(include_gpu or due)
            if include_gpu or due:
                self._last_gpu_sample = self.elapsed_seconds
            details = sample.pop("measurements", {})
            row = self._append(
                "telemetry",
                SystemTelemetryRecord(
                    run_id=self.run_id,
                    timestamp=_now(),
                    elapsed_seconds=self.elapsed_seconds,
                    measurements={**measurements, **details},
                    **sample,
                ),
            )
            self._resources.add(row)
            return row

    def resource_snapshot(self) -> dict[str, Any]:
        """Latest sampled resource window, without relabeling it as checkpoint cost."""
        with self._lock:
            return self._resources.snapshot()

    def _sample_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.telemetry()
            except RuntimeError:
                return
            except (OSError, ValueError):
                # A transient optional resource query must not terminate training.
                logging.getLogger(__name__).warning(
                    "Optional telemetry sample unavailable", exc_info=True
                )
                continue

    def register_artifact(
        self, path: str | Path, kind: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            source = Path(path).resolve(strict=True)
            if not source.is_file():
                raise ValueError("An artifact must be a regular file")
            digest = sha256_file(source)
            destination = self.directory / "files" / f"{digest}_{source.name}"
            # Store a private copy: changing the caller's checkpoint cannot mutate this run.
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                with source.open("rb") as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
            if sha256_file(destination) != digest:
                raise ValueError(
                    "Artifact content changed while registering or existing copy is corrupted"
                )
            record = ArtifactRecord(
                artifact_id=str(uuid.uuid4()),
                kind=kind,
                run_id=self.run_id,
                path=destination.relative_to(self.directory).as_posix(),
                size_bytes=destination.stat().st_size,
                sha256=digest,
                created_at=_now(),
                metadata=json.loads(canonical_json(metadata or {})),
            )
            canonical_json(record.model_dump(mode="json"))
            self._artifacts.append(record)
            return record.model_dump(mode="json")

    def finalize(self, status: str = "completed", **summary: Any) -> dict[str, Any]:
        self._ensure_open()
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("Final status must be completed, failed, or cancelled")
        # Validate summary before stopping the live recorder.
        summary = {**self._summary, **summary}
        if self._progress_state:
            summary = {
                "training_clock_seconds": self._progress_state["training_clock_seconds"],
                "accounting": self._progress_state,
                **summary,
            }
        canonical_json(summary)
        duration = self.elapsed_seconds
        promoted = {
            key: summary.pop(key)
            for key in list(summary)
            if key
            in {
                "training_steps",
                "environment_steps",
                "episodes",
                "optimizer_updates",
                "checkpoint_hash",
            }
        }
        steps = promoted.get("environment_steps", self._record.environment_steps)
        preliminary = self._record.model_dump()
        preliminary.update(
            status=status,
            end_time=_now(),
            duration=duration,
            wall_clock_seconds=duration,
            environment_steps_per_second=steps / duration if duration else 0,
            summary=summary,
            evaluation_results=self._evaluations,
            **promoted,
        )
        RunRecord.model_validate(preliminary)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            self.telemetry(include_gpu=self._gpu_interval > 0 or self._last_gpu_sample is not None)
            summary["resources"] = self.resource_snapshot()
            duration = self.elapsed_seconds
            preliminary.update(
                duration=duration,
                wall_clock_seconds=duration,
                end_time=_now(),
                environment_steps_per_second=steps / duration if duration else 0,
                summary=summary,
            )
            for stream in self._streams.values():
                stream.close()
            # If a disk write fails below, leave an explicitly unsealed record.
            # Prevent a context-manager retry from masking the original error.
            self._closed = True
            for name, schema in PARQUET_SCHEMAS.items():
                journal = self.directory / f"{name}.jsonl"
                # Record batches bound memory use for long governed sessions.
                parquet = self.directory / f"{name}.parquet"
                with pq.ParquetWriter(parquet, schema, compression="zstd") as writer:
                    rows = []
                    with journal.open(encoding="utf-8") as stream:
                        for line in stream:
                            rows.append(_columnar(json.loads(line)))
                            if len(rows) >= 8192:
                                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                                rows = []
                    if rows:
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                self._artifacts.append(
                    ArtifactRecord(
                        artifact_id=str(uuid.uuid4()),
                        kind=f"{name}_parquet",
                        run_id=self.run_id,
                        path=parquet.name,
                        size_bytes=parquet.stat().st_size,
                        sha256=sha256_file(parquet),
                        created_at=_now(),
                        metadata={"canonical": True},
                    )
                )
            for artifact in self._artifacts:
                if sha256_file(self.directory / artifact.path) != artifact.sha256:
                    raise ValueError(
                        f"Registered artifact changed before finalization: {artifact.path}"
                    )
            preliminary["artifact_manifest"] = self._artifacts
            self._record = RunRecord.model_validate(preliminary)
            finalized = self._record.model_dump(mode="json")
            _write_new(self.directory / "run.json", finalized)
            seal = {
                path.relative_to(self.directory).as_posix(): {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(self.directory.rglob("*"))
                if path.is_file()
            }
            _write_new(
                self.directory / "seal.json",
                {
                    "schema_version": "1.0.0",
                    "run_id": self.run_id,
                    "created_at": _now().isoformat(),
                    "files": seal,
                },
            )
            return finalized


def _run_directory(root: str | Path, run_id: str) -> Path:
    # Prevent path traversal through CLI/dashboard run identifiers.
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("run_id must be a canonical UUID")
    return Path(root).resolve() / "runs" / run_id


def load_run(root: str | Path, run_id: str) -> dict[str, Any]:
    directory = _run_directory(root, run_id)
    path = directory / "run.json"
    if not path.exists():
        path = directory / "run-start.json"
    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8")).model_dump(mode="json")
    if record["status"] == "running":
        latest = read_progress(directory)
        if latest:
            for key in ("environment_steps", "training_steps", "episodes", "optimizer_updates"):
                record[key] = latest[key]
            if latest.get("checkpoint"):
                record["checkpoint_hash"] = latest["checkpoint"]["sha256"]
            record["summary"] = {
                **record["summary"],
                "accounting": {
                    **latest,
                    "counter_basis": "persisted_completed_operations_lower_bound",
                },
            }
    return record


def read_progress(directory: str | Path) -> dict[str, Any] | None:
    """Read through the last complete progress line without repairing or sealing it."""
    path = Path(directory) / "progress.jsonl"
    if not path.exists():
        return None
    latest = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            latest = json.loads(line)
    return latest


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    runs = Path(root).resolve() / "runs"
    if not runs.exists():
        return []
    result = [
        load_run(root, path.name)
        for path in runs.iterdir()
        if path.is_dir() and (path / "run-start.json").exists()
    ]
    return sorted(result, key=lambda record: record["start_time"], reverse=True)


def verify_run(root: str | Path, run_id: str) -> dict[str, Any]:
    directory = _run_directory(root, run_id)
    if not (directory / "seal.json").exists():
        return {
            "run_id": run_id,
            "valid": False,
            "errors": ["Run is not sealed (running or interrupted)"],
        }
    seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in seal["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            errors.append(f"Unsafe manifest path: {name}")
        elif not path.is_file():
            errors.append(f"Missing file: {name}")
        elif (
            path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]
        ):
            errors.append(f"Hash/size mismatch: {name}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() != "seal.json"
    }
    errors.extend(f"Unregistered file: {name}" for name in sorted(actual - set(seal["files"])))
    try:
        record = load_run(root, run_id)
        if record["config_sha256"] != hash_config(record["configuration"]):
            errors.append("Configuration hash mismatch")
        for artifact in record["artifact_manifest"]:
            path = (directory / artifact["path"]).resolve()
            if (
                not path.is_relative_to(directory)
                or not path.is_file()
                or sha256_file(path) != artifact["sha256"]
            ):
                errors.append(f"Artifact integrity failure: {artifact['artifact_id']}")
    except (ValueError, OSError) as error:
        errors.append(f"Invalid run record: {error}")
    return {"run_id": run_id, "valid": not errors, "errors": errors}


def connect_database(root: str | Path):
    """Return an independent DuckDB snapshot; caller must close it.

    Tables: runs, metrics, telemetry, evaluations, trajectories. Dictionaries in
    SQL are serialized JSON columns (e.g. configuration_json, dimensions_json).
    Finalized data comes from Parquet; active journals are read to the last
    complete line, allowing live dashboard queries without shared writer locks.
    """
    import duckdb

    connection = duckdb.connect(":memory:")
    records = list_runs(root)
    if records:
        flat = []
        for record in records:
            flat.append(
                {
                    key + "_json" if isinstance(value, (dict, list)) else key: canonical_json(value)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in record.items()
                }
            )
        connection.register("runs", pa.Table.from_pylist(flat))
    else:
        connection.register(
            "runs",
            pa.table(
                {
                    "run_id": pa.array([], type=pa.string()),
                    "experiment_id": pa.array([], type=pa.string()),
                    "status": pa.array([], type=pa.string()),
                }
            ),
        )
    for name, schema in PARQUET_SCHEMAS.items():
        paths = []
        active = []
        for record in records:
            directory = _run_directory(root, record["run_id"])
            if record["status"] != "running" and (directory / f"{name}.parquet").exists():
                paths.append(str(directory / f"{name}.parquet"))
            elif (directory / f"{name}.jsonl").exists():
                with (directory / f"{name}.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        active.append(_columnar(json.loads(line)))
        if paths:
            connection.read_parquet(paths, union_by_name=True).create_view(f"_{name}_sealed")
        connection.register(f"_{name}_active", pa.Table.from_pylist(active, schema=schema))
        connection.execute(
            f"CREATE VIEW {name} AS SELECT * FROM _{name}_active"
            + (f" UNION ALL SELECT * FROM _{name}_sealed" if paths else "")
        )
    return connection


def query_metrics(root: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    if run_id is not None:
        _run_directory(root, run_id)
    with connect_database(root) as connection:
        cursor = connection.execute(
            "SELECT * FROM metrics"
            + (" WHERE run_id = ?" if run_id else "")
            + " ORDER BY run_id, elapsed_seconds, step, name",
            [run_id] if run_id else [],
        )
        names = [column[0] for column in cursor.description]
        result = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    for row in result:
        row["dimensions"] = json.loads(row.pop("dimensions_json"))
    return result
