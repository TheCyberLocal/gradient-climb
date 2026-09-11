from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run


def run_training(
    root,
    algorithm,
    seconds,
    seed,
    config=None,
    experiment="surrogate-pilot",
    benchmark_class="cold_start",
    parent_run=None,
):
    from gradientclimb.algorithms import train_cem, train_ppo
    from gradientclimb.simulation import SIMULATOR_VERSION

    config = dict(config or {})
    if algorithm not in {"ppo", "cem"}:
        raise ValueError("Unknown learning algorithm")
    if algorithm == "cem" and config.get("parent_checkpoint"):
        raise ValueError("CEM currently supports cold starts only")
    if benchmark_class == "cold_start" and config.get("parent_checkpoint"):
        raise ValueError("A cold-start run cannot load trained parent weights")
    if benchmark_class != "cold_start" and not config.get("parent_checkpoint"):
        raise ValueError("Non-cold-start training requires declared parent weights")
    checkpoint_source = config.get("parent_checkpoint")
    parent_hash = sha256_file(Path(checkpoint_source)) if checkpoint_source else None
    run_config = {
        "algorithm": algorithm,
        "seconds": seconds,
        "seed": seed,
        **config,
        "benchmark_class": benchmark_class,
        "parent_hash": parent_hash,
    }
    with RunRecorder(
        Path(root),
        experiment,
        run_config,
        seed,
        algorithm,
        "uncalibrated_hill_surrogate",
        simulator_version=SIMULATOR_VERSION,
        calibration_version="uncalibrated",
        parent_run=parent_run,
        parent_checkpoint=parent_hash,
        vehicle_profile=config.get("profile", "default"),
        map_profile=config.get("terrain", "train"),
        benchmark_class=benchmark_class,
        qualifies_real_game=False,
    ) as run:
        last_checkpoint = None

        def callback(metrics, model, elapsed):
            nonlocal last_checkpoint
            step = int(metrics.get("environment_steps", 0))
            for name, value in metrics.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    run.metric(
                        name,
                        value,
                        step,
                        training_elapsed_seconds=elapsed,
                        episodes=metrics.get("episodes", 0),
                    )
            run.telemetry(include_gpu=True)
            if metrics.get("checkpoint"):
                target = metrics.get("checkpoint_target_seconds")
                label = f"at-{target:g}s" if target is not None else "final"
                path = run.directory / f"policy-{label}.pt"
                model.save(path)
                last_checkpoint = run.register_artifact(
                    path,
                    "checkpoint",
                    {
                        "training_elapsed_seconds": elapsed,
                        "target_seconds": target,
                        "model_config": model.config,
                        "final": metrics.get("final", False),
                        "parent_run": parent_run,
                        "parent_checkpoint": parent_hash,
                    },
                )
                print(
                    json.dumps(
                        {
                            "run_id": run.run_id,
                            "checkpoint": label,
                            "elapsed": elapsed,
                            "steps": step,
                        }
                    ),
                    flush=True,
                )

        train = train_ppo if algorithm == "ppo" else train_cem
        result = train(seconds=seconds, seed=seed, callback=callback, **config)
        config_path = run.directory / "resolved-training-config.json"
        config_path.write_text(json.dumps(result.config, indent=2), encoding="utf-8")
        run.register_artifact(config_path, "resolved_configuration")
        # Final model evaluation is separate from the training clock and explicitly timed.
        from gradientclimb.evaluation import evaluate

        evaluation = evaluate(
            result.model,
            range(10000, 10020),
            profile=config.get("profile", "default"),
            terrain=config.get("terrain", "train"),
            max_steps=config.get("max_steps", 1000),
        )
        run.evaluation(
            {
                "results": evaluation,
                "episodes": len(evaluation["episodes"]),
                "checkpoint_hash": last_checkpoint["sha256"],
                "protocol": "held-out-surrogate-validation-0.1",
            }
        )
        run.finalize(
            environment_steps=result.environment_steps,
            training_steps=result.environment_steps,
            episodes=result.episodes,
            optimizer_updates=result.optimizer_updates,
            checkpoint_hash=last_checkpoint["sha256"],
            training_clock_seconds=result.elapsed,
            median_distance=evaluation["median_distance"],
            mean_distance=evaluation["mean_distance"],
            evaluation_seconds=evaluation["wall_clock_seconds"],
            final_checkpoint=last_checkpoint["path"],
            model_config=result.config,
            scope="uncalibrated_simulator",
            qualifies_real_game=False,
        )
    return load_run(Path(root), run.run_id)


def run_evaluation(
    root,
    checkpoint=None,
    baseline="random",
    episodes=20,
    seed_start=20000,
    profile="default",
    terrain="train",
    generalization=False,
):
    import torch

    from gradientclimb.algorithms import AlwaysGasPolicy, RandomPolicy, load_policy
    from gradientclimb.evaluation import evaluate, generalization_suite

    torch.set_num_threads(1)
    if episodes < 1:
        raise ValueError("At least one evaluation episode is required")
    policy = (
        load_policy(checkpoint)
        if checkpoint
        else (RandomPolicy(seed_start) if baseline == "random" else AlwaysGasPolicy())
    )
    config = {
        "checkpoint_hash": sha256_file(checkpoint) if checkpoint else None,
        "baseline": None if checkpoint else baseline,
        "episodes": episodes,
        "seed_start": seed_start,
        "profile": profile,
        "terrain": terrain,
        "generalization": generalization,
    }
    with RunRecorder(
        Path(root),
        "surrogate-evaluation",
        config,
        seed_start,
        policy.config["algorithm"],
        "uncalibrated_hill_surrogate",
        parent_checkpoint=config["checkpoint_hash"],
        vehicle_profile=profile,
        map_profile=terrain,
    ) as run:
        seeds = range(seed_start, seed_start + episodes)
        results = (
            generalization_suite(policy, seeds)
            if generalization
            else {"target": evaluate(policy, seeds, profile, terrain)}
        )
        for condition, result in results.items():
            run.evaluation(
                {
                    "results": result,
                    "episodes": episodes,
                    "checkpoint_hash": config["checkpoint_hash"],
                    "protocol": "held-out-surrogate-test-0.1",
                    "metadata": {"condition": condition},
                }
            )
            run.metric("median_distance", result["median_distance"], condition=condition)
        result_file = run.directory / "evaluation.json"
        result_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        run.register_artifact(result_file, "evaluation")
        run.finalize(scope="uncalibrated_simulator", results=results, qualifies_real_game=False)
    return load_run(Path(root), run.run_id)


def throughput(root, env_counts=(32, 64, 128, 256, 512, 1024), seconds=3):
    from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv

    if seconds <= 0:
        raise ValueError("Benchmark duration must be positive")
    results = []
    with RunRecorder(
        Path(root),
        "simulation-throughput",
        {"env_counts": list(env_counts), "seconds_per_condition": seconds},
        algorithm="uniform-random",
        environment="uncalibrated_hill_surrogate",
        simulator_version=SIMULATOR_VERSION,
    ) as run:
        for n in env_counts:
            env = VectorHillEnv(n, seed=42)
            rng = np.random.default_rng(42)
            for _ in range(10):
                env.step(rng.integers(0, 4, n))
            start = time.perf_counter()
            steps = 0
            while time.perf_counter() - start < seconds:
                env.step(rng.integers(0, 4, n))
                steps += n
            elapsed = time.perf_counter() - start
            row = {
                "envs": n,
                "steps": steps,
                "seconds": elapsed,
                "steps_per_second": steps / elapsed,
            }
            results.append(row)
            run.metric("environment_steps_per_second", row["steps_per_second"], num_envs=n)
            run.telemetry(include_gpu=True)
        run.finalize(results=results, includes_policy_inference=False)
    return {"run_id": run.run_id, "results": results}


def evaluate_checkpoints(root, parent_run, episodes=20):
    """Post-session evaluations retain actual and requested training times separately."""
    import torch

    from gradientclimb.algorithms import load_policy
    from gradientclimb.evaluation import evaluate
    from gradientclimb.experiments import verify_run

    torch.set_num_threads(1)
    root = Path(root)
    if episodes < 1:
        raise ValueError("Episode count must be positive")
    if not verify_run(root, parent_run)["valid"]:
        raise ValueError("Parent run failed artifact verification")
    source = load_run(root, parent_run)
    artifacts = [
        a
        for a in source["artifact_manifest"]
        if a["kind"] == "checkpoint" and a["metadata"].get("target_seconds") is not None
    ]
    if not artifacts:
        raise ValueError("Parent has no scheduled checkpoints")
    config = {
        "parent_run": parent_run,
        "episodes": episodes,
        "seeds": list(range(10000, 10000 + episodes)),
        "checkpoint_hashes": [a["sha256"] for a in artifacts],
        "clock": "offline_evaluation",
    }
    with RunRecorder(
        root,
        "timed-checkpoint-evaluation",
        config,
        10000,
        source["algorithm"],
        source["environment"],
        parent_run=parent_run,
        simulator_version=source["simulator_version"],
        evidence_domain="simulation",
        benchmark_class=source["metadata"].get("benchmark_class"),
    ) as run:
        for artifact in artifacts:
            path = root / "runs" / parent_run / artifact["path"]
            policy = load_policy(path)
            result = evaluate(
                policy,
                config["seeds"],
                source.get("vehicle_profile") or "default",
                source.get("map_profile") or "train",
            )
            meta = artifact["metadata"]
            actual = meta["training_elapsed_seconds"]
            run.evaluation(
                {
                    "results": result,
                    "episodes": episodes,
                    "checkpoint_hash": artifact["sha256"],
                    "protocol": "one-hour-v1",
                    "metadata": {
                        "training_minutes": actual / 60,
                        "requested_minutes": meta["target_seconds"] / 60,
                        "training_elapsed_seconds": actual,
                        "evidence_domain": "simulation",
                    },
                }
            )
            run.metric(
                "evaluation/median_distance",
                result["median_distance"],
                training_elapsed_seconds=actual,
                requested_seconds=meta["target_seconds"],
            )
        run.finalize(
            qualifies_real_game=False,
            checkpoint_count=len(artifacts),
            scope="uncalibrated_simulator",
        )
    return load_run(root, run.run_id)
