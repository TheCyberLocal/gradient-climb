from __future__ import annotations

import inspect
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run

OBSERVER_SNAPSHOT_KEYS = ("snapshot_count", "snapshot_copy_seconds", "snapshot_callback_seconds")
OBSERVER_ENV_KEYS = ("profile", "terrain", "randomization", "stack", "max_steps", "hidden_size")
LOG = logging.getLogger(__name__)


def observer_training_config(config, train_ppo):
    """The environment/model shape the observer must mirror, using train_ppo defaults."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(train_ppo).parameters.items()
    }
    return {key: config.get(key, defaults[key]) for key in OBSERVER_ENV_KEYS}


def run_training(
    root,
    algorithm,
    seconds,
    seed,
    config=None,
    experiment="surrogate-pilot",
    benchmark_class="cold_start",
    parent_run=None,
    observer=None,
):
    """Govern one training run; ``observer`` (mapping or ObserverConfig) adds a live view.

    Enabling the observer changes no training hyperparameter: the recorded
    configuration is identical except for a separate ``observer`` block, and the
    resolved training configuration artifact is byte-identical.
    """
    from gradientclimb.algorithms import train_cem, train_ppo
    from gradientclimb.algorithms.progress import TrainingProgress
    from gradientclimb.simulation import SIMULATOR_VERSION
    from gradientclimb.visualization.observer import ObserverConfig, TrainingObserver, build_sinks

    config = dict(config or {})
    if algorithm not in {"ppo", "cem"}:
        raise ValueError("Unknown learning algorithm")
    if algorithm == "cem" and config.get("parent_checkpoint"):
        raise ValueError("CEM currently supports cold starts only")
    if benchmark_class == "cold_start" and config.get("parent_checkpoint"):
        raise ValueError("A cold-start run cannot load trained parent weights")
    if benchmark_class != "cold_start" and not config.get("parent_checkpoint"):
        raise ValueError("Non-cold-start training requires declared parent weights")
    if {"snapshot", "snapshot_interval", "callback", "progress"} & set(config):
        raise ValueError(
            "Snapshotting and callbacks are wired by the runner, not the training config"
        )
    observer = ObserverConfig.from_mapping(observer)
    live = None
    video_skipped_reason = None
    if observer is not None:
        if algorithm != "ppo":
            raise ValueError("The live observer requires PPO")
        observer.validate(seed)
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
    if observer is not None:
        # Built on the calling thread before train_ppo seeds Torch; construction forks the RNG.
        sinks, video_skipped_reason = build_sinks(observer)
        live = TrainingObserver(observer, observer_training_config(config, train_ppo), sinks)
        run_config["observer"] = observer.to_record(live.env.action_duration)
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
        progress = TrainingProgress()
        config_path = run.directory / "resolved-training-config.json"
        initialization_recorded = False

        def save_checkpoint(model, elapsed, label, target=None, final=False):
            nonlocal last_checkpoint
            previous_phase = progress.phase
            progress.phase = "checkpoint_serialization"
            path = run.directory / f"policy-{label}.pt"
            model.save(path)
            progress.phase = "checkpoint_registration"
            last_checkpoint = run.register_artifact(
                path,
                "checkpoint",
                {
                    "training_elapsed_seconds": elapsed,
                    "target_seconds": target,
                    "model_config": model.config,
                    "final": final,
                    "stage": label,
                    "parent_run": parent_run,
                    "parent_checkpoint": parent_hash,
                    "continuation": "warm_start" if algorithm == "ppo" else "inference_only",
                    "exact_resume": False,
                },
            )
            progress.phase = previous_phase
            run.annotate(
                last_valid_checkpoint=last_checkpoint["path"],
                last_valid_checkpoint_hash=last_checkpoint["sha256"],
            )
            run.record_progress({**progress.snapshot(), "checkpoint": last_checkpoint})

        def persist_progress(snapshot):
            nonlocal initialization_recorded
            run.record_progress(
                {**snapshot, **({"checkpoint": last_checkpoint} if last_checkpoint else {})}
            )
            if snapshot["phase"] == "ready" and not initialization_recorded:
                initialization_recorded = True
                config_path.write_text(json.dumps(progress.config, indent=2), encoding="utf-8")
                run.register_artifact(config_path, "resolved_configuration")
                if progress.prepare_checkpoint:
                    progress.prepare_checkpoint()
                save_checkpoint(progress.model, snapshot["training_clock_seconds"], "initial")

        progress.sink = persist_progress

        def callback(metrics, model, elapsed):
            nonlocal last_checkpoint
            previous_phase = progress.phase
            progress.phase = "metric_journal"
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
            progress.phase = previous_phase
            if metrics.get("checkpoint"):
                target = metrics.get("checkpoint_target_seconds")
                label = f"at-{target:g}s" if target is not None else "final"
                save_checkpoint(
                    model, elapsed, label, target=target, final=metrics.get("final", False)
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
        observer_report = None
        observer_summary = None
        training_error = None
        cleanup_errors = []
        try:
            if live is not None:
                live.start()
            result = train(
                seconds=seconds,
                seed=seed,
                callback=callback,
                progress=progress,
                **config,
                **(
                    {"snapshot_interval": observer.snapshot_interval, "snapshot": live.snapshot}
                    if live is not None
                    else {}
                ),
            )
            progress.finished = progress.started + result.elapsed
        except BaseException as error:
            progress.finished = time.monotonic()
            training_error = error
            run.annotate(failure_phase=progress.phase)
            raise
        finally:
            if live is not None:
                try:
                    observer_report = live.stop(timeout=10.0)
                    observer_summary = _record_observer(
                        run, run_config["observer"], observer_report, progress, video_skipped_reason
                    )
                    run.annotate(observer=observer_summary)
                except BaseException as error:  # noqa: BLE001 - preserve learner and cleanup evidence
                    observer_summary = {
                        "error": f"{type(error).__name__}: {error}",
                        "stopped_cleanly": False,
                        "record_error": "observer cleanup or recording failed",
                    }
                    run.annotate(observer=observer_summary)
                    if training_error is not None:
                        training_error.add_note(f"Observer cleanup also failed: {error}")
            try:
                progress.publish(force=True)
                if training_error is not None and progress.model is not None:
                    if progress.checkpoint_safe:
                        if progress.prepare_checkpoint:
                            progress.prepare_checkpoint()
                        save_checkpoint(
                            progress.model,
                            progress.snapshot()["training_clock_seconds"],
                            "interrupted",
                        )
                    else:
                        run.annotate(
                            checkpoint_recovery="refused: optimizer operation may be partial"
                        )
            except BaseException as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
                if training_error is not None:
                    training_error.add_note(f"Progress/checkpoint recovery also failed: {error}")
                else:
                    raise
            run.annotate(
                learner_status="interrupted"
                if isinstance(training_error, KeyboardInterrupt)
                else "failed"
                if training_error is not None
                else "completed",
                cleanup_errors=cleanup_errors,
                **({"checkpoint_hash": last_checkpoint["sha256"]} if last_checkpoint else {}),
            )
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
            **({"observer": observer_summary} if observer_summary is not None else {}),
        )
    return load_run(Path(root), run.run_id)


def _record_observer(run, observer_block, report, result, video_skipped_reason):
    """Write observer evidence after training stops, including its exception paths.

    Nothing here may fail the finished training run: a missing or unregistrable
    video is recorded as ``video_skipped_reason`` and any other recording error
    as ``record_error`` in the observer summary.
    """
    final = result.metrics[-1] if result.metrics else {}
    summary = {key: value for key, value in report.items() if key != "episodes"}
    summary.update({key: final.get(key) for key in (*OBSERVER_SNAPSHOT_KEYS, "snapshot_errors")})
    summary["video_skipped_reason"] = video_skipped_reason or report.get("video_skipped_reason")
    summary["record_error"] = None
    video = report.get("video")
    if observer_block.get("video") and not video and summary["video_skipped_reason"] is None:
        # A video was requested but no encoder finished: name the cause instead of
        # leaving the consumer to search the per-sink reports.
        sink_errors = [
            f"{sink.get('kind')}: {sink.get('error')}"
            for sink in report.get("sinks", [])
            if isinstance(sink, dict) and sink.get("error")
        ]
        summary["video_skipped_reason"] = (
            "; ".join(sink_errors)
            or (f"observer error: {report['error']}" if report.get("error") else None)
            or "no encoder report; observer did not finish"
        )
    if video and summary["video_skipped_reason"] is None:
        try:
            if not Path(video).is_file():
                raise FileNotFoundError(f"encoder reported {video} but no file exists")
            run.register_artifact(
                video,
                "video",
                {
                    "frames": report["frames_rendered"],
                    "fps": report["fps"],
                    "observer": observer_block,
                },
            )
        except Exception as error:  # Evidence about the video, not a failed run.
            summary["video"] = None
            summary["video_skipped_reason"] = f"video not registered: {error}"
            LOG.warning("Observer video could not be registered", exc_info=True)
    elif video:
        summary["video"] = None
    try:
        episodes_path = run.directory / "observer-episodes.json"
        episodes_path.write_text(
            json.dumps(
                {"observer": observer_block, "episodes": report["episodes"]},
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        run.register_artifact(episodes_path, "observer_episodes")
        for episode in report["episodes"]:
            run.metric(
                "observer/episode_distance",
                episode["distance"],
                step=int(episode.get("snapshot_environment_steps") or 0),
                snapshot_index=episode["snapshot_index"],
                snapshot_elapsed=episode["snapshot_elapsed"],
                optimizer_updates=episode["snapshot_optimizer_updates"],
                termination=episode["termination"],
                policy=episode["policy"],
            )
        run.metric(
            "observer/frames_rendered", report["frames_rendered"], step=result.environment_steps
        )
        run.metric(
            "observer/snapshots_received",
            report["snapshots_received"],
            step=result.environment_steps,
        )
    except Exception as error:  # Observer evidence must never fail the training run.
        summary["record_error"] = f"{type(error).__name__}: {error}"
        LOG.warning("Observer evidence could not be recorded", exc_info=True)
    summary["status"] = (
        "degraded"
        if summary.get("error")
        or summary.get("record_error")
        or summary.get("sink_errors")
        or summary.get("stopped_cleanly") is False
        or summary.get("video_skipped_reason")
        else "completed"
    )
    return summary


def run_evaluation(
    root,
    checkpoint=None,
    baseline="random",
    episodes=20,
    seed_start=20000,
    profile=None,
    terrain=None,
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
    profile = profile if profile is not None else policy.config.get("profile", "default")
    terrain = terrain if terrain is not None else policy.config.get("terrain", "train")
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
