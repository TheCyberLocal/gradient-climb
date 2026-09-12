"""Incremental learner accounting independent of a successful return from train()."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingProgress:
    """Count completed operations; an interrupted in-flight operation remains unknown.

    The runner retains this object when the learner raises. Periodic durable
    snapshots are lower bounds after process termination, never exact resumes.
    """

    sink: Callable[[dict], None] | None = None
    interval: float = 1.0
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    publication_seconds: float = 0.0
    environment_steps: int = 0
    episodes: int = 0
    optimizer_updates: int = 0
    command_started: float | None = None
    experience: dict = field(default_factory=dict)
    phase: str = "initialization"
    model: Any = None
    config: dict = field(default_factory=dict)
    metrics: list[dict] = field(default_factory=list)
    prepare_checkpoint: Callable[[], None] | None = None
    checkpoint_safe: bool = False
    _last_published: float = float("-inf")

    def snapshot(self) -> dict:
        return {
            "environment_steps": self.environment_steps,
            "training_steps": self.environment_steps,
            "episodes": self.episodes,
            "optimizer_updates": self.optimizer_updates,
            "training_clock_seconds": max(0.0, (self.finished or time.monotonic()) - self.started),
            "accounting_publication_seconds": self.publication_seconds,
            "phase": self.phase,
            "counter_basis": "observed_completed_operations",
            "in_flight_operation": self.phase
            if self.phase in {"environment_step", "optimizer_step"}
            else None,
            "checkpoint_safe": self.checkpoint_safe,
            "experience": dict(self.experience),
        }

    def configure_simulator(self, env) -> None:
        """Declare known measurement scope, without assigning unknown work zero."""
        self.experience = {
            "schema_version": "experience-3.0",
            "scope": "learner_environment_only_excludes_observer_and_independent_evaluation",
            "simulator_transitions": 0,
            "simulator_episodes": 0,
            "simulator_seconds": 0.0,
            "physics_steps": 0,
            "policy_decisions": 0,
            "actor_rendered_frames": 0,
            "real_game_interaction_seconds": 0.0,
            "action_duration_seconds": env.action_duration,
            "physics_substeps_per_transition": env.substeps,
        }

    def decisions_completed(self, count: int) -> None:
        self.experience["policy_decisions"] += count

    def simulator_step_completed(self, count: int, episodes: int) -> None:
        self.experience["simulator_transitions"] += count
        self.experience["simulator_episodes"] += episodes
        self.experience["physics_steps"] += (
            count * self.experience["physics_substeps_per_transition"]
        )
        self.experience["simulator_seconds"] = (
            self.experience["simulator_transitions"] * self.experience["action_duration_seconds"]
        )

    def publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.sink and (force or now - self._last_published >= self.interval):
            try:
                self.sink(self.snapshot())
                self._last_published = now
            finally:
                self.publication_seconds += time.monotonic() - now
