"""Versioned scientific record contracts; JSON schemas live in ``schemas/``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ScientificRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0.0"


class ExperimentRecord(ScientificRecord):
    experiment_id: str = Field(min_length=1)
    hypothesis_id: str | None = None
    description: str
    independent_variables: list[str] = Field(default_factory=list)
    dependent_variables: list[str] = Field(default_factory=list)
    controls: dict[str, Any] = Field(default_factory=dict)
    success_criterion: str | None = None
    governed: bool = False


class ArtifactRecord(ScientificRecord):
    artifact_id: str
    kind: str = Field(min_length=1)
    run_id: str
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricRecord(ScientificRecord):
    run_id: str
    name: str = Field(min_length=1)
    value: FiniteFloat
    step: int = Field(ge=0)
    elapsed_seconds: FiniteFloat = Field(ge=0)
    timestamp: datetime
    dimensions: dict[str, Any] = Field(default_factory=dict)


class SystemTelemetryRecord(ScientificRecord):
    run_id: str
    timestamp: datetime
    elapsed_seconds: FiniteFloat = Field(ge=0)
    cpu_percent: FiniteFloat | None = Field(default=None, ge=0, le=100)
    per_core_cpu_percent: list[FiniteFloat] = Field(default_factory=list)
    ram_used_bytes: int | None = Field(default=None, ge=0)
    process_rss_bytes: int | None = Field(default=None, ge=0)
    gpu_percent: FiniteFloat | None = Field(default=None, ge=0, le=100)
    vram_used_bytes: int | None = Field(default=None, ge=0)
    measurements: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(ScientificRecord):
    evaluation_id: str
    run_id: str
    timestamp: datetime
    elapsed_seconds: FiniteFloat = Field(ge=0)
    environment: str
    seed: int | None = None
    episodes: int = Field(default=1, ge=1)
    results: dict[str, Any]
    protocol: str = "exploratory"
    checkpoint_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(ScientificRecord):
    run_id: str
    episode_id: str
    step: int = Field(ge=0)
    timestamp: datetime
    elapsed_seconds: FiniteFloat = Field(ge=0)
    gas: bool
    brake: bool
    action_duration_seconds: FiniteFloat = Field(ge=0)
    observation: dict[str, Any] = Field(default_factory=dict)
    reward: FiniteFloat | None = None
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    frame_artifact_id: str | None = None


class ModelLineageRecord(ScientificRecord):
    run_id: str
    checkpoint_artifact_id: str
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_run: str | None = None
    parent_checkpoint: str | None = None
    derivation: Literal["training", "adaptation", "distillation", "initialization"]
    policy_architecture: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRecord(ScientificRecord):
    benchmark_id: str
    version: str
    run_ids: list[str]
    protocol: dict[str, Any]
    environment: str
    status: Literal["planned", "running", "completed", "failed", "blocked"]
    results: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class CalibrationRecord(ScientificRecord):
    calibration_id: str
    version: str
    run_id: str
    source_trajectory_artifacts: list[str]
    simulator_version: str
    parameters: dict[str, FiniteFloat]
    held_out_trajectory_artifacts: list[str]
    fit_metrics: dict[str, FiniteFloat]
    validation_metrics: dict[str, FiniteFloat]
    limitations: list[str] = Field(default_factory=list)


class RunRecord(ScientificRecord):
    run_id: str
    experiment_id: str = Field(min_length=1)
    status: Literal["running", "completed", "failed", "cancelled"]
    start_time: datetime
    end_time: datetime | None = None
    duration: FiniteFloat | None = Field(default=None, ge=0)
    git_sha: str | None = None
    dirty_worktree: bool | None = None
    source_diff_sha256: str | None = None
    project_version: str
    machine_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpu: str
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    ram: int | None = Field(default=None, ge=0)
    driver: str | None = None
    cuda: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    python_version: str
    framework_versions: dict[str, str] = Field(default_factory=dict)
    algorithm: str
    algorithm_version: str = "1"
    policy_architecture: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment: str
    environment_version: str = "1"
    simulator_version: str | None = None
    calibration_version: str | None = None
    vehicle_profile: str | None = None
    map_profile: str | None = None
    seed: int = Field(ge=0)
    parent_checkpoint: str | None = None
    parent_run: str | None = None
    training_steps: int = Field(default=0, ge=0)
    environment_steps: int = Field(default=0, ge=0)
    episodes: int = Field(default=0, ge=0)
    optimizer_updates: int = Field(default=0, ge=0)
    wall_clock_seconds: FiniteFloat = Field(default=0, ge=0)
    environment_steps_per_second: FiniteFloat = Field(default=0, ge=0)
    evaluation_results: list[EvaluationRecord] = Field(default_factory=list)
    artifact_manifest: list[ArtifactRecord] = Field(default_factory=list)
    checkpoint_hash: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


SCHEMA_MODELS = {
    "experiments": ExperimentRecord,
    "runs": RunRecord,
    "metrics": MetricRecord,
    "system_telemetry": SystemTelemetryRecord,
    "evaluations": EvaluationRecord,
    "trajectories": TrajectoryRecord,
    "artifacts": ArtifactRecord,
    "model_lineage": ModelLineageRecord,
    "benchmarks": BenchmarkRecord,
    "simulator_calibration": CalibrationRecord,
}
