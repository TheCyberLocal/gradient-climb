"""Cross-entropy method: inexpensive, gradient-free linear policy competitor."""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from gradientclimb.environments import make_environment
from gradientclimb.simulation import SIMULATOR_VERSION
from gradientclimb.simulation.hill import FEATURE_NAMES

from .policies import LinearPolicy
from .ppo import DEFAULT_CHECKPOINT_TIMES, TrainingResult
from .progress import TrainingProgress


def train_cem(
    seconds: float,
    seed: int = 0,
    num_envs: int = 64,
    callback=None,
    checkpoint_times=DEFAULT_CHECKPOINT_TIMES,
    profile: str = "default",
    terrain: str = "train",
    randomization: bool = False,
    stack: int = 4,
    max_steps: int = 1000,
    population: int = 32,
    episodes_per_candidate: int = 2,
    elite_fraction: float = 0.2,
    initial_std: float = 0.6,
    log_interval: float = 5.0,
    progress: TrainingProgress | None = None,
) -> TrainingResult:
    """Governed search with common seeds per candidate and a fixed validation set.

    Population times episodes_per_candidate determines actual parallelism;
    num_envs is accepted for CLI compatibility and recorded separately. Complete
    generations update the search distribution. A truncated final generation is
    discarded and its real experience cost is still counted. Best-model selection
    uses fixed *training validation* seeds, separate from external evaluation.
    """
    if seconds <= 0 or population < 4 or episodes_per_candidate < 1 or not 0 < elite_fraction < 1:
        raise ValueError("Invalid CEM budget/population/elite fraction")
    start = time.monotonic()
    progress = progress if progress is not None else TrainingProgress()
    progress.started = start
    deadline = start + seconds
    rng = np.random.default_rng(seed)
    width = len(FEATURE_NAMES)
    mean = np.zeros((width + 1, 4), dtype=np.float32)
    deviation = np.full_like(mean, initial_std)
    model = LinearPolicy(mean.copy())
    actual_envs = population * episodes_per_candidate
    env = make_environment(
        num_envs=actual_envs,
        seed=seed,
        profile=profile,
        terrain=terrain,
        randomization=randomization,
        stack=stack,
        max_steps=max_steps,
    )
    config = {
        "algorithm": "cem",
        "algorithm_version": "original-0.1.0",
        "seconds": seconds,
        "seed": seed,
        "num_envs": actual_envs,
        "requested_num_envs": num_envs,
        "profile": profile,
        "terrain": terrain,
        "randomization": randomization,
        "stack": stack,
        "max_steps": max_steps,
        "population": population,
        "episodes_per_candidate": episodes_per_candidate,
        "elite_fraction": elite_fraction,
        "initial_std": initial_std,
        "log_interval": log_interval,
        "checkpoint_times": list(checkpoint_times),
        "simulator_version": SIMULATOR_VERSION,
        "calibration_version": "uncalibrated",
        "observation_source": "idealized_simulator_state",
        "action_space": "four_joint_pedal_states",
    }
    model.config = config
    metrics = []
    progress.model, progress.config, progress.metrics = model, config, metrics
    progress.checkpoint_safe = True
    progress.phase = "ready"
    progress.publish(force=True)
    targets = deque(sorted({float(t) for t in checkpoint_times if 0 < t <= seconds}))
    next_log = log_interval
    generation = total_steps = episodes = evaluated_candidate_episodes = 0
    last_mean = best_score = 0.0
    elite_count = max(2, int(population * elite_fraction))
    # Same fixed seeds each generation make incumbent comparisons meaningful;
    # external held-out seeds are required to detect selection overfitting.
    candidate_seeds = [seed * 10007 + 101 + i for i in range(episodes_per_candidate)]

    def emit(final=False, target=None):
        elapsed = time.monotonic() - start
        row = {
            "wall_clock_seconds": elapsed,
            "environment_steps": total_steps,
            "env_steps": total_steps,
            "training_steps": total_steps,
            "episodes": episodes,
            "evaluated_candidate_episodes": evaluated_candidate_episodes,
            "optimizer_updates": generation,
            "iterations": generation,
            "environment_steps_per_second": total_steps / max(elapsed, 1e-9),
            "mean_episode_distance": last_mean,
            "best_training_distance": best_score,
            "search_std": float(deviation.mean()),
            "accounting_publication_seconds": progress.publication_seconds,
            "checkpoint": target is not None or final,
            "checkpoint_target_seconds": target,
            "requested_checkpoint_seconds": target,
            "final": final,
        }
        metrics.append(row)
        if callback:
            callback(row, model, elapsed)

    def boundaries():
        nonlocal next_log
        elapsed = time.monotonic() - start
        emitted = False
        while targets and elapsed >= targets[0]:
            emit(target=targets.popleft())
            emitted = True
            elapsed = time.monotonic() - start
        if elapsed >= next_log:
            if not emitted:
                emit()
            next_log = elapsed + log_interval

    while time.monotonic() < deadline:
        candidates = (
            mean + rng.standard_normal((population, width + 1, 4)).astype(np.float32) * deviation
        )
        candidates[0] = model.weights  # Incumbent is evaluated on exactly the same seeds.
        candidates[1] = mean
        weights = np.repeat(candidates, episodes_per_candidate, axis=0)
        observations, _ = env.reset(candidate_seeds * population)
        finished = np.zeros(actual_envs, dtype=bool)
        distances = np.zeros(actual_envs)
        for _ in range(max_steps):
            boundaries()
            if time.monotonic() >= deadline:
                break
            logits = (
                np.einsum("ni,nij->nj", observations[:, -width:], weights[:, :-1]) + weights[:, -1]
            )
            progress.phase = "environment_step"
            observations, _, _terminated, _truncated, info = env.step(logits.argmax(-1))
            total_steps += actual_envs
            episodes += len(info["episodes"])
            progress.environment_steps, progress.episodes = total_steps, episodes
            progress.phase = "candidate_evaluation"
            progress.publish()
            for episode in info["episodes"]:
                index = episode["env_index"]
                if not finished[index]:
                    finished[index] = True
                    distances[index] = episode["distance"]
                    evaluated_candidate_episodes += 1
            if finished.all():
                break
        if not finished.all():
            break
        scores = distances.reshape(population, episodes_per_candidate).mean(-1)
        elite = np.argsort(scores)[-elite_count:]
        progress.phase = "optimizer_step"
        progress.checkpoint_safe = False
        mean = 0.25 * mean + 0.75 * candidates[elite].mean(0)
        deviation = np.maximum(0.04, 0.25 * deviation + 0.75 * candidates[elite].std(0))
        winner = int(np.argmax(scores))
        best_score = float(scores[winner])
        model.weights = candidates[winner].copy()
        last_mean = float(scores.mean())
        generation += 1
        progress.optimizer_updates = generation
        progress.checkpoint_safe = True
        progress.phase = "candidate_evaluation"
        progress.publish()
        boundaries()
    boundaries()
    emit(final=True)
    progress.phase = "completed"
    progress.publish(force=True)
    return TrainingResult(
        model, metrics, config, time.monotonic() - start, total_steps, episodes, generation
    )
