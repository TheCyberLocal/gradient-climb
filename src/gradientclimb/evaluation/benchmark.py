"""Episode-level simulator assessment, with explicit scope and fixed seeds."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv


def summarize(values, bootstrap_seed: int = 0, bootstrap_samples: int = 2000) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Summary needs at least one finite observation")
    rng = np.random.default_rng(bootstrap_seed)
    if len(values) > 1:
        means = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(1)
        lo, hi = np.quantile(means, [0.025, 0.975])
    else:
        lo = hi = float("nan")
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "ci95_low": float(lo) if np.isfinite(lo) else None,
        "ci95_high": float(hi) if np.isfinite(hi) else None,
        "ci_method": "episode percentile bootstrap; conditional on one trained policy",
    }


def evaluate(
    policy,
    seeds=tuple(range(1000, 1020)),
    profile: str = "default",
    terrain: str = "train",
    max_steps: int = 1000,
    stack: int | None = None,
    deterministic: bool = True,
) -> dict:
    """Evaluate exactly one initial episode for each explicit seed in parallel.

    Same-step automatic resets occur internally, but post-reset episodes never
    enter results. CIs describe episode variation, not training-seed uncertainty.
    Training and test seeds must be disjoint; this function records, not guesses,
    that experimental responsibility. No in-game qualification is implied.
    """
    start = time.monotonic()
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires nonempty unique seeds")
    if stack is None:
        stack = getattr(policy, "config", {}).get("stack", 4)
    env = VectorHillEnv(len(seeds), seeds[0], profile, terrain, False, stack, max_steps)
    observations, _ = env.reset(seeds)
    active = np.ones(len(seeds), dtype=bool)
    episodes = []
    action_counts = np.zeros(4, dtype=int)
    transitions = np.zeros((4, 4), dtype=int)
    previous = np.zeros(len(seeds), dtype=int)
    steps = 0
    for _ in range(max_steps):
        actions = policy.act(observations, deterministic=deterministic)
        np.add.at(action_counts, actions[active], 1)
        np.add.at(transitions, (previous[active], actions[active]), 1)
        steps += int(active.sum())
        observations, _, _, _, info = env.step(actions)
        previous = actions.copy()
        for episode in info["episodes"]:
            index = episode["env_index"]
            if active[index]:
                active[index] = False
                episodes.append(
                    {**episode, "seed": seeds[index], "profile": profile, "terrain": terrain}
                )
        if not active.any():
            break
    episodes.sort(key=lambda row: seeds.index(row["seed"]))
    if len(episodes) != len(seeds):
        raise RuntimeError("Incomplete episode evaluation")
    reasons = Counter(row["termination"] for row in episodes)
    distance = summarize([row["distance"] for row in episodes])
    return {
        "benchmark_version": "surrogate-evaluation-0.1.0",
        "scope": "uncalibrated_simulator",
        "simulator_version": SIMULATOR_VERSION,
        "calibration_version": "uncalibrated",
        "policy_config": getattr(policy, "config", {}),
        "seeds": seeds,
        "profile": profile,
        "terrain": terrain,
        "max_steps": max_steps,
        "deterministic": deterministic,
        "episodes": episodes,
        "summary": {
            "distance": distance,
            "return": summarize([e["return_"] for e in episodes]),
            "survival_seconds": summarize([e["seconds"] for e in episodes]),
            "failure_rates": {k: v / len(episodes) for k, v in sorted(reasons.items())},
        },
        "mean_distance": distance["mean"],
        "median_distance": distance["median"],
        "action_counts": action_counts.tolist(),
        "action_transition_counts": transitions.tolist(),
        "environment_steps": steps,
        "wall_clock_seconds": time.monotonic() - start,
    }


def generalization_suite(policy, seeds=tuple(range(2000, 2020)), max_steps: int = 1000) -> dict:
    conditions = {
        "in_distribution": ("default", "train"),
        "new_map": ("default", "rough"),
        "new_vehicle": ("heavy", "train"),
        "new_vehicle_and_map": ("heavy", "rough"),
    }
    return {
        name: evaluate(policy, seeds, profile, terrain, max_steps)
        for name, (profile, terrain) in conditions.items()
    }


def compare_paired(first: dict, second: dict, bootstrap_seed: int = 0) -> dict:
    """Paired distance deltas on matching scenario seeds; positive favors first."""
    if (first["profile"], first["terrain"], first["max_steps"]) != (
        second["profile"],
        second["terrain"],
        second["max_steps"],
    ):
        raise ValueError("Paired comparisons require identical scenarios and limits")
    left = {e["seed"]: e["distance"] for e in first["episodes"]}
    right = {e["seed"]: e["distance"] for e in second["episodes"]}
    if left.keys() != right.keys():
        raise ValueError("Paired comparisons require matching evaluation seeds")
    differences = np.array([left[s] - right[s] for s in sorted(left)])
    summary = summarize(differences, bootstrap_seed)
    std = differences.std(ddof=1) if len(differences) > 1 else 0
    return {
        "distance_difference": summary,
        "paired_cohen_dz": float(differences.mean() / std) if std > 0 else None,
        "win_fraction": float(np.mean(differences > 0)),
        "limitation": "Episode-seed pairing does not substitute for independent training replicates",
    }
