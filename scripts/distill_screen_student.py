"""Prepare or explicitly execute an uncalibrated body-feature distillation run.

Default mode is a provenance/configuration dry run. This script never opens a
game, captures a screen or sends input. Real-game dispatch is a separate task.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gradientclimb.artifacts import sha256_file


def resolve_plan(args):
    from gradientclimb.algorithms.screen_distillation import BODY_FEATURES, BODY_SCALES
    from gradientclimb.experiments import load_run, verify_run
    from gradientclimb.perception.screen_features import ScreenFeatureBridge

    if isinstance(args.seconds, bool) or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("A finite positive distillation budget is required")
    if not args.schema_id or args.seed < 0 or args.num_envs < 1 or not 1 <= args.history <= 16:
        raise ValueError("Invalid schema, training seed, environment count or history")
    if args.validation_seed_start < 0 or args.validation_episodes < 1:
        raise ValueError("Validation requires nonnegative seeds and positive episode count")
    bridge = ScreenFeatureBridge(
        None, None, history=args.history, max_interval_seconds=args.max_interval_seconds
    )
    if args.schema_id != bridge.schema_id:
        raise ValueError(
            "Destination schema must match this bridge's history and interval settings"
        )
    validation_seeds = list(
        range(args.validation_seed_start, args.validation_seed_start + args.validation_episodes)
    )
    if args.seed in validation_seeds:
        raise ValueError("Training seed must be excluded from held-out episode seeds")
    parent = load_run(Path(args.root), args.parent_run)
    verification = verify_run(Path(args.root), args.parent_run)
    if parent["status"] != "completed" or not verification["valid"]:
        raise ValueError("Teacher parent must be a completed, validly sealed canonical run")
    digest = sha256_file(args.teacher)
    # Artifact identity, not a guessed filename, establishes the teacher lineage.
    artifacts = parent["artifact_manifest"]
    if not any(row["sha256"] == digest and row["kind"] == "checkpoint" for row in artifacts):
        raise ValueError("Teacher checkpoint is not registered to the declared parent run")
    source = json.loads(Path(args.feature_definition).read_text(encoding="utf-8"))
    if source["features"] != list(BODY_FEATURES) or source["feature_scales"] != list(BODY_SCALES):
        raise ValueError(
            "Screen feature definition differs from implemented body-feature semantics"
        )
    return {
        "protocol_version": "screen-body-distillation-1",
        "status": "prepared_not_executed",
        "benchmark_class": "distillation",
        "scope": "uncalibrated_analytic_surrogate_projection",
        "qualifies_real_game": False,
        "parent_run": args.parent_run,
        "parent_checkpoint": digest,
        "parent_training_clock_seconds": parent.get("summary", {}).get("training_clock_seconds"),
        "teacher_path": str(Path(args.teacher).resolve()),
        "schema_id": args.schema_id,
        "history": args.history,
        "max_interval_seconds": args.max_interval_seconds,
        "features": list(BODY_FEATURES),
        "feature_scales": list(BODY_SCALES),
        "feature_definition_sha256": sha256_file(args.feature_definition),
        "training_seconds": args.seconds,
        "training_seed": args.seed,
        "num_envs": args.num_envs,
        "validation_seeds": validation_seeds,
        "training_profile": "default",
        "training_terrain": "train",
        "training_occupancy": "frozen_deterministic_teacher",
        "loss": "mean binary cross entropy against detached teacher pedal probabilities",
        "validation": "teacher and student occupancies, same held-out seeds, no gradient updates",
        "clock_excludes": [
            "Python imports",
            "parent seal verification",
            "teacher loading",
            "held-out evaluation",
            "final checkpoint serialization",
        ],
        "clock_includes": [
            "student/environment initialization",
            "projected observations",
            "frozen teacher inference",
            "student optimization",
            "metric persistence",
        ],
        "calibration": "none; schema-compatible analytic counterparts are not validated pixels",
        "real_dispatch": "separate explicit dispatch; native UI and freshness guards remain required",
    }


def execute(args, plan):
    import torch

    from gradientclimb.algorithms import load_policy
    from gradientclimb.algorithms.screen_distillation import evaluate_student, train_student
    from gradientclimb.experiments import RunRecorder, load_run
    from gradientclimb.simulation import SIMULATOR_VERSION

    torch.set_num_threads(1)
    teacher = load_policy(args.teacher)
    if teacher.config.get("algorithm") != "ppo":
        raise ValueError("Distillation requires an independently parameterized PPO teacher")
    with RunRecorder(
        Path(args.root),
        "screen-body-distillation",
        {**plan, "status": "dispatched"},
        args.seed,
        "screen_body_distillation",
        "uncalibrated_hill_surrogate",
        parent_run=args.parent_run,
        parent_checkpoint=plan["parent_checkpoint"],
        simulator_version=SIMULATOR_VERSION,
        calibration_version="uncalibrated",
        benchmark_class="distillation",
        qualifies_real_game=False,
    ) as run:

        def callback(metrics, model, elapsed):
            for name, value in metrics.items():
                if value is not None:
                    run.metric(
                        name, value, metrics["environment_steps"], training_elapsed_seconds=elapsed
                    )

        student, metrics = train_student(
            teacher,
            schema_id=args.schema_id,
            seconds=args.seconds,
            seed=args.seed,
            num_envs=args.num_envs,
            history=args.history,
            max_interval_seconds=args.max_interval_seconds,
            callback=callback,
        )
        student.config.update(
            parent_run=args.parent_run,
            parent_checkpoint=plan["parent_checkpoint"],
            parent_training_clock_seconds=plan["parent_training_clock_seconds"],
        )
        checkpoint_path = run.directory / "screen-body-student-final.pt"
        student.save(checkpoint_path)
        checkpoint = run.register_artifact(
            checkpoint_path,
            "checkpoint",
            {
                "training_elapsed_seconds": metrics["training_clock_seconds"],
                "final": True,
                "parent_run": args.parent_run,
                "parent_checkpoint": plan["parent_checkpoint"],
                "derivation": "distillation",
                "model_config": student.config,
            },
        )
        evaluations = {}
        for occupancy in ("teacher", "student"):
            result = evaluate_student(
                student, teacher, plan["validation_seeds"], occupancy=occupancy
            )
            evaluations[occupancy] = result
            run.evaluation(
                {
                    "results": result,
                    "episodes": len(result["episodes"]),
                    "checkpoint_hash": checkpoint["sha256"],
                    "protocol": "analytic-body-distillation-validation-1",
                }
            )
        run.finalize(
            **metrics,
            checkpoint_hash=checkpoint["sha256"],
            final_checkpoint=checkpoint["path"],
            evaluation_seconds=sum(e["wall_clock_seconds"] for e in evaluations.values()),
            mean_distance=evaluations["student"]["mean_distance"],
            teacher_occupancy_joint_agreement=evaluations["teacher"]["joint_action_agreement"],
            student_occupancy_joint_agreement=evaluations["student"]["joint_action_agreement"],
            scope=plan["scope"],
            qualifies_real_game=False,
        )
    return load_run(Path(args.root), run.run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--schema-id", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--max-interval-seconds", type=float, default=0.5)
    parser.add_argument("--validation-seed-start", type=int, default=30000)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--feature-definition",
        type=Path,
        default=Path("experiments/definitions/real-screen-cem.json"),
    )
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = resolve_plan(args)
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(execute(args, plan) if args.execute else plan, indent=2))


if __name__ == "__main__":
    main()
