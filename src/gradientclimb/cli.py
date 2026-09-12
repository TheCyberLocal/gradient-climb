"""Public research commands. Every advertised command has an executable path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def output(value):
    print(json.dumps(value, indent=2, default=str, allow_nan=False), flush=True)


def main(argv=None, *, command_clock=None):
    parser = argparse.ArgumentParser(prog="gradientclimb")
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Read machine and framework capabilities")
    exp = commands.add_parser("experiment").add_subparsers(dest="operation", required=True)
    run = exp.add_parser("run", help="Execute the synthetic provenance experiment")
    run.add_argument("--gain", type=float, default=0.2)
    run.add_argument("--seed", type=int, default=0)
    exp.add_parser("list")
    for name in ("show", "verify"):
        exp.add_parser(name).add_argument("run_id")
    compare = exp.add_parser("compare")
    compare.add_argument("run_ids", nargs="+")
    train = commands.add_parser("train", help="Train in the versioned research surrogate")
    train.add_argument("--algorithm", choices=["ppo", "cem"], default="ppo")
    train.add_argument("--seconds", type=float, default=60)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--envs", type=int, default=64)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--config", type=Path)
    train.add_argument("--experiment", default="surrogate-pilot")
    train.add_argument("--parent-checkpoint", type=Path)
    train.add_argument("--parent-run")
    train.add_argument(
        "--benchmark-class",
        choices=["cold_start", "simulator_pretrained", "generalist_adaptation", "fine_tuning"],
        default="cold_start",
    )
    live = train.add_mutually_exclusive_group()
    live.add_argument(
        "--headed",
        action="store_true",
        help="Show one observer environment driven by policy snapshots in a live Tk window",
    )
    live.add_argument(
        "--headless-record",
        action="store_true",
        help="Run the observer without a window (frames counted; optional --observer-video)",
    )
    train.add_argument(
        "--observer-policy",
        choices=["current", "best"],
        default="current",
        help=(
            "Follow the newest snapshot, or the best-so-far snapshot ranked by the mean distance "
            "of the last <=100 training episodes completed BEFORE the snapshot (earlier weights): "
            "a lagging training signal, not held-out evaluation"
        ),
    )
    train.add_argument(
        "--observer-interval", type=float, default=5.0, help="Snapshot refresh in seconds"
    )
    train.add_argument(
        "--observer-seed",
        type=int,
        default=41000,
        help=(
            "Observer environment seed; rejected inside the held-out ranges 1000-1019, "
            "2000-2019, 10000-10019, 20000-20019, 30000-30019 or when equal to --seed"
        ),
    )
    train.add_argument(
        "--observer-fps",
        type=float,
        help="Rendered frames (= simulator actions) per second; default real time (16.667)",
    )
    train.add_argument(
        "--observer-video", type=Path, help="Optional MP4; skipped with a reason without ffmpeg"
    )
    train.add_argument("--observer-mode", choices=["thread"], default="thread")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--baseline", choices=["random", "always_gas"], default="random")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed-start", type=int, default=20000)
    evaluate.add_argument("--profile", help="Override the checkpoint vehicle profile")
    evaluate.add_argument("--terrain", help="Override the checkpoint terrain profile")
    evaluate.add_argument("--generalization", action="store_true")
    checkpoints = commands.add_parser(
        "evaluate-checkpoints", help="Evaluate saved timed checkpoints offline"
    )
    checkpoints.add_argument("run_id")
    checkpoints.add_argument("--episodes", type=int, default=20)
    watch = commands.add_parser(
        "watch", help="Render one simulator environment with an optional learned policy"
    )
    watch.add_argument("--checkpoint", type=Path)
    watch.add_argument("--seconds", type=float, default=30)
    watch.add_argument("--seed", type=int, default=20000)
    watch.add_argument("--video", type=Path)
    capture = commands.add_parser("capture", help="Record or benchmark a foreground game window")
    capture.add_argument("--backend", choices=["mss", "pillow", "dxcam"], default="mss")
    capture.add_argument("--frames", type=int, default=120)
    capture.add_argument("--fps", type=float)
    capture.add_argument("--record", action="store_true")
    calibrate = commands.add_parser(
        "calibrate", help="Fit effective dynamics with disjoint trajectories"
    )
    calibrate.add_argument("--train", type=Path, nargs="+", required=True)
    calibrate.add_argument("--heldout", type=Path, nargs="+", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("kind", choices=["throughput"])
    benchmark.add_argument("--seconds", type=float, default=3)
    benchmark.add_argument("--envs", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    dash = commands.add_parser("dashboard")
    dash.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    from .experiments import list_runs, load_run, run_synthetic, verify_run

    if args.command == "doctor":
        from .telemetry import capture_provenance

        facts = capture_provenance(Path.cwd())
        try:
            import torch

            facts["pytorch_runtime"] = {
                "version": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                facts["pytorch_runtime"].update(
                    device=torch.cuda.get_device_name(0),
                    capability=torch.cuda.get_device_capability(0),
                )
        except ImportError:
            facts["pytorch_runtime"] = None
        output(facts)
    elif args.command == "experiment":
        if args.operation == "run":
            output(run_synthetic(args.root, args.seed, args.gain))
        elif args.operation == "list":
            output(list_runs(args.root))
        elif args.operation == "verify":
            result = verify_run(args.root, args.run_id)
            output(result)
            if not result["valid"]:
                raise SystemExit(1)
        elif args.operation == "compare":
            output(
                [
                    {
                        k: load_run(args.root, rid)[k]
                        for k in (
                            "run_id",
                            "algorithm",
                            "seed",
                            "duration",
                            "environment_steps",
                            "summary",
                        )
                    }
                    for rid in args.run_ids
                ]
            )
        else:
            output(load_run(args.root, args.run_id))
    elif args.command == "train":
        from .benchmarks.runner import run_training

        config = json.loads(args.config.read_text()) if args.config else {}
        config.setdefault("num_envs", args.envs)
        if args.algorithm == "ppo":
            config.setdefault("device", args.device)
        if args.parent_checkpoint:
            config["parent_checkpoint"] = str(args.parent_checkpoint.resolve())
        observer = None
        if args.headed or args.headless_record:
            # A separate block: the training config above is built exactly as without it.
            observer = {
                "mode": args.observer_mode,
                "display": "window" if args.headed else "none",
                "policy": args.observer_policy,
                "snapshot_interval": args.observer_interval,
                "seed": args.observer_seed,
                "fps": args.observer_fps,
                "video": str(args.observer_video.resolve()) if args.observer_video else None,
            }
        output(
            run_training(
                args.root,
                args.algorithm,
                args.seconds,
                args.seed,
                config,
                args.experiment,
                args.benchmark_class,
                args.parent_run,
                observer=observer,
                **({"command_clock": command_clock} if command_clock is not None else {}),
            )
        )
    elif args.command == "evaluate":
        from .benchmarks.runner import run_evaluation

        output(
            run_evaluation(
                args.root,
                args.checkpoint,
                args.baseline,
                args.episodes,
                args.seed_start,
                args.profile,
                args.terrain,
                args.generalization,
            )
        )
    elif args.command == "benchmark":
        from .benchmarks.runner import throughput

        output(throughput(args.root, args.envs, args.seconds))
    elif args.command == "evaluate-checkpoints":
        from .benchmarks.runner import evaluate_checkpoints

        output(evaluate_checkpoints(args.root, args.run_id, args.episodes))
    elif args.command == "watch":
        from .artifacts import sha256_file
        from .experiments import RunRecorder, list_runs
        from .visualization.replay import watch

        checkpoint_hash = sha256_file(args.checkpoint) if args.checkpoint else None
        parents = [
            record["run_id"]
            for record in list_runs(args.root)
            if checkpoint_hash
            and any(
                artifact["kind"] == "checkpoint" and artifact["sha256"] == checkpoint_hash
                for artifact in record.get("artifact_manifest", [])
            )
        ]
        with RunRecorder(
            args.root,
            "surrogate-render",
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_hash": checkpoint_hash,
                "simulation_seconds": args.seconds,
                "video": str(args.video),
            },
            args.seed,
            "policy-render",
            "uncalibrated_hill_surrogate",
            parent_checkpoint=checkpoint_hash,
            parent_run=parents[0] if len(parents) == 1 else None,
            evidence_domain="simulation_render",
            qualifies_real_game=False,
        ) as run:
            result = watch(args.checkpoint, args.seconds, args.seed, args.video)
            if args.video:
                run.register_artifact(args.video, "video", result)
            run.finalize(**result)
        output({"run_id": run.run_id, **result})
    elif args.command == "capture":
        from .capture.screen import WindowCapture
        from .capture.windows import WindowGuard, discover_windows
        from .experiments import RunRecorder

        targets = discover_windows()
        if len(targets) != 1:
            raise RuntimeError(f"Expected exactly one visible game window; found {len(targets)}")
        with RunRecorder(
            args.root,
            "screen-capture-benchmark",
            {
                "backend": args.backend,
                "frames": args.frames,
                "target_fps": args.fps,
                "record": args.record,
            },
            algorithm="capture",
            environment="actual_hill_climb_racing",
            evidence_domain="real_game",
        ) as run:
            with WindowCapture(WindowGuard(targets[0]), backend=args.backend) as stream:
                directory = run.directory / "frames" if args.record else None
                result = stream.benchmark(
                    frame_count=args.frames, target_fps=args.fps, output_dir=directory
                )
            if directory:
                for path in directory.iterdir():
                    if path.is_file():
                        run.register_artifact(
                            path, "frame" if path.suffix == ".png" else "capture_metadata"
                        )
            run.finalize(
                status="completed" if result.get("status") == "completed" else "failed",
                capture=result,
            )
        output({"run_id": run.run_id, **result})
    elif args.command == "calibrate":
        from .artifacts import sha256_file
        from .calibration import fit_calibration, load_trajectory
        from .experiments import RunRecorder

        config = {
            "training_sources": {str(p): sha256_file(p) for p in args.train},
            "heldout_sources": {str(p): sha256_file(p) for p in args.heldout},
        }
        with RunRecorder(
            args.root,
            "effective-dynamics-calibration",
            config,
            algorithm="least-squares",
            environment="trajectory_dataset",
        ) as run:
            train_data = [load_trajectory(p) for p in args.train]
            heldout_data = [load_trajectory(p) for p in args.heldout]
            for path in args.train + args.heldout:
                run.register_artifact(path, "source_trajectory")
            fit = fit_calibration(train_data, heldout_data, run_id=run.run_id)
            destination = fit.save(run.directory / "calibration")
            for path in destination.iterdir():
                run.register_artifact(path, "calibration")
            run.finalize(calibration_id=fit.record.calibration_id, report=fit.report)
        output({"run_id": run.run_id, "calibration": fit.record.model_dump(mode="json")})
    elif args.command == "dashboard":
        import uvicorn

        from .dashboard import create_app

        uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
