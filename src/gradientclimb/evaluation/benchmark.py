"""Episode-level simulator assessment, with explicit scope and fixed seeds."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from gradientclimb.environments import LEGACY_ENVIRONMENT_ID, environment_from_config


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
    profile: str | None = None,
    terrain: str | None = None,
    max_steps: int | None = None,
    stack: int | None = None,
    deterministic: bool = True,
) -> dict:
    """Evaluate exactly one initial episode for each explicit seed in parallel.

    Same-step automatic resets occur internally, but post-reset episodes never
    enter results. CIs describe episode variation, not training-seed uncertainty.
    Omitted scenario settings come from the checkpoint; explicit arguments can
    select a transfer condition. Training and test seeds must be disjoint; this
    function records, not guesses, that responsibility. No game qualification
    is implied. Randomization is disabled for frozen evaluation.
    """
    start = time.monotonic()
    seeds = list(seeds)
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Evaluation requires nonempty unique nonnegative integer seeds")
    overrides = {"profile": profile, "terrain": terrain, "max_steps": max_steps, "stack": stack}
    env = environment_from_config(
        getattr(policy, "config", {}),
        num_envs=len(seeds),
        seed=seeds[0],
        randomization=False,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    profile, terrain, max_steps = env.profile, env.terrain, env.max_steps
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
        "benchmark_version": "simulator-evaluation-3.0",
        "scope": env.environment_spec.evidence_domain,
        "simulator_version": env.environment_spec.environment_id,
        "calibration_version": env.scenario.calibration_version,
        "distance_unit": env.environment_spec.distance_unit,
        "action_duration_seconds": env.action_duration,
        "dt": env.dt,
        "substeps": env.substeps,
        "randomization": env.randomization,
        "scenario": env.scenario.model_dump(mode="json"),
        "scenario_hash": env.scenario.sha256,
        "vehicle_profile_hash": env.scenario.vehicle.sha256,
        "map_profile_hash": env.scenario.map.sha256,
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
    if _comparison_scope(first) != _comparison_scope(second):
        raise ValueError(
            "Paired comparisons require identical domains, units, scenarios and limits"
        )
    left = {e["seed"]: e["distance"] for e in first["episodes"]}
    right = {e["seed"]: e["distance"] for e in second["episodes"]}
    if (
        not left
        or len(left) != len(first["episodes"])
        or len(right) != len(second["episodes"])
        or left.keys() != right.keys()
    ):
        raise ValueError("Paired comparisons require matching unique evaluation seeds")
    left_distances = np.asarray([left[s] for s in sorted(left)], dtype=float)
    right_distances = np.asarray([right[s] for s in sorted(left)], dtype=float)
    if not np.isfinite(left_distances).all() or not np.isfinite(right_distances).all():
        raise ValueError("Paired distances must all be finite measurements")
    differences = left_distances - right_distances
    summary = summarize(differences, bootstrap_seed)
    std = differences.std(ddof=1) if len(differences) > 1 else 0
    return {
        "distance_difference": summary,
        "paired_cohen_dz": float(differences.mean() / std) if std > 0 else None,
        "win_fraction": float(np.mean(differences > 0)),
        "limitation": "Episode-seed pairing does not substitute for independent training replicates",
    }


def _comparison_scope(result: dict) -> tuple:
    required = (
        "scope",
        "simulator_version",
        "calibration_version",
        "profile",
        "terrain",
        "max_steps",
        "deterministic",
    )
    if any(key not in result or result[key] is None for key in required):
        raise ValueError("Paired result is missing its evaluation scope")
    # The sealed legacy protocol fixes these units and cadence. Do not infer
    # them for unknown generations or relabel historical artifact bytes.
    legacy = (
        result.get("benchmark_version") == "surrogate-evaluation-0.1.0"
        and result["simulator_version"] == LEGACY_ENVIRONMENT_ID
        and result["scope"] == "uncalibrated_simulator"
    )
    unit = result.get("distance_unit", "surrogate_unit" if legacy else None)
    duration = result.get("action_duration_seconds", 0.06 if legacy else None)
    if (
        not isinstance(unit, str)
        or not unit
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not np.isfinite(duration)
    ):
        raise ValueError("Paired results require declared distance units and action cadence")
    if duration <= 0:
        raise ValueError("Action cadence must be positive")
    return (
        *(result[key] for key in required),
        unit,
        duration,
        result.get("dt", 0.02 if legacy else None),
        result.get("substeps", 3 if legacy else None),
        result.get("randomization", False if legacy else None),
        result.get("vehicle_profile_hash"),
        result.get("map_profile_hash"),
    )
