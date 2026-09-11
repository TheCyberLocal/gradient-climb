"""Bounded post-benchmark jobs. Dry-run is the default; no implicit budgets.

Plan format: {"after_run": UUID, "actions": [{"id": "checkpoints", "kind":
"evaluate_checkpoints", "source": "after_run", "episodes": 20}, ...]}.
Kinds: evaluate_checkpoints, generalization, train. Train requires explicit
seconds, seeds, algorithm, experiment, config and benchmark_class. Source is
after_run or action:<earlier-id>. Cold starts have no source. Every training job
runs in a fresh Python process. --resume skips only verified completed jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("after_run"), str)
        or not plan["after_run"]
        or not isinstance(plan.get("actions"), list)
        or not plan["actions"]
    ):
        raise ValueError("A plan needs after_run and a nonempty actions list")
    seen, counts = {}, {"after_run": 1}
    jobs, training_seconds = 0, 0
    for action in plan["actions"]:
        if not isinstance(action, dict):
            raise TypeError("Each action must be an object")
        identifier = action.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.replace("_", "").replace("-", "").isalnum()
            or identifier in seen
        ):
            raise ValueError("Action IDs must be unique simple names")
        kind = action.get("kind")
        if kind not in {"evaluate_checkpoints", "generalization", "train"}:
            raise ValueError(f"Unknown action kind: {kind}")
        source = action.get("source")
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("A source must be a nonempty string")
        if (
            source
            and source != "after_run"
            and (not source.startswith("action:") or source[7:] not in seen)
        ):
            raise ValueError("Source must be after_run or a preceding action")
        if source and source.startswith("action:") and seen[source[7:]] != "train":
            raise ValueError("Only training actions produce source checkpoints")
        if kind == "train":
            if action.get("algorithm") not in {"ppo", "cem"} or not isinstance(
                action.get("config"), dict
            ):
                raise ValueError("Training requires algorithm and explicit config")
            if (
                type(action.get("seconds")) not in (int, float)
                or not math.isfinite(action["seconds"])
                or not 0 < action["seconds"] <= 3600
            ):
                raise ValueError(
                    "Each post-benchmark training budget must be explicit and within (0,3600] seconds"
                )
            seeds = action.get("seeds")
            if (
                not isinstance(seeds, list)
                or not seeds
                or len(seeds) > 10
                or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError("Training needs 1–10 unique explicit nonnegative seeds")
            if not isinstance(action.get("experiment"), str) or not action["experiment"]:
                raise ValueError("Training requires an experiment ID")
            category = action.get("benchmark_class")
            if category not in {
                "cold_start",
                "simulator_pretrained",
                "generalist_adaptation",
                "fine_tuning",
            }:
                raise ValueError("Training requires a declared benchmark class")
            if (
                bool(source) != (category != "cold_start")
                or "parent_checkpoint" in action["config"]
            ):
                raise ValueError(
                    "Declare warm-start lineage using source; cold starts must have no parent"
                )
            if action["algorithm"] == "cem" and source:
                raise ValueError("The implemented CEM supports cold starts only")
            reserved = {"algorithm", "seconds", "seed", "callback", "parent_checkpoint"}
            if reserved.intersection(action["config"]):
                raise ValueError("Training config must not override job identity, clock or lineage")
            if type(action.get("require_clean_source", False)) is not bool:
                raise ValueError("require_clean_source must be a boolean")
        elif (
            not source
            or type(action.get("episodes")) is not int
            or not 1 <= action["episodes"] <= 100
        ):
            raise ValueError("Evaluation needs an explicit source and 1–100 episodes")
        if kind == "generalization" and (
            type(action.get("seed_start")) is not int or action["seed_start"] < 0
        ):
            raise ValueError("Generalization requires an explicit evaluation seed_start")
        seen[identifier] = kind
        count = counts[source] if source else 1
        if kind == "train":
            count *= len(action["seeds"])
            training_seconds += count * action["seconds"]
        counts[f"action:{identifier}"] = count
        jobs += count
    if "total_requested_learning_seconds" in plan and (
        type(plan["total_requested_learning_seconds"]) not in (int, float)
        or plan["total_requested_learning_seconds"] != training_seconds
    ):
        raise ValueError("Declared total learning budget disagrees with expanded jobs")
    return {"expanded_jobs": jobs, "requested_learning_seconds": training_seconds}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def journal_event(directory, event, **details):
    record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **details}
    with (directory / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def assert_finished(source):
    if source.get("status") != "completed":
        raise RuntimeError(
            "The governed source run must be completed before post-benchmark execution"
        )


def ensure_ready(root, after_run):
    from gradientclimb.experiments import list_runs, load_run, verify_run

    source = load_run(root, after_run)
    assert_finished(source)
    active = [
        r["run_id"]
        for r in list_runs(root)
        if r["status"] == "running" and r["algorithm"] in {"ppo", "cem"}
    ]
    if active:
        raise RuntimeError(
            f"Learning runs remain active; refusing concurrent post-benchmark work: {active}"
        )
    checked = verify_run(root, after_run)
    if not checked["valid"]:
        raise ValueError(f"Source verification failed: {checked}")
    return source


def checkpoint_path(root, run):
    from gradientclimb.artifacts import sha256_file

    relative = run.get("summary", {}).get("final_checkpoint")
    if not relative or not run.get("checkpoint_hash"):
        raise ValueError(f"Source {run['run_id']} has no declared final checkpoint")
    directory = (root / "runs" / run["run_id"]).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or sha256_file(path) != run["checkpoint_hash"]:
        raise ValueError("Checkpoint path or digest failed validation")
    return str(path)


def validate_job_record(record, action, source, seed):
    """A valid seal alone does not prove that a result belongs to this job."""
    assert_finished(record)
    config = record.get("configuration", {})
    if action["kind"] == "train":
        if (
            record.get("algorithm") != action["algorithm"]
            or record.get("seed") != seed
            or record.get("experiment_id") != action["experiment"]
            or record.get("parent_run") != (source["run_id"] if source else None)
            or config.get("seconds") != action["seconds"]
            or config.get("benchmark_class") != action["benchmark_class"]
            or any(config.get(key) != value for key, value in action["config"].items())
        ):
            raise ValueError("Stored training result does not match the planned job")
    elif action["kind"] == "evaluate_checkpoints":
        if (
            record.get("parent_run") != source["run_id"]
            or config.get("episodes") != action["episodes"]
        ):
            raise ValueError("Stored checkpoint evaluation does not match its planned parent")
    elif (
        record.get("parent_checkpoint") != source["checkpoint_hash"]
        or config.get("episodes") != action["episodes"]
        or config.get("seed_start") != action["seed_start"]
        or config.get("generalization") is not True
    ):
        raise ValueError("Stored generalization result does not match the planned checkpoint")


def run_job(job_path):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    root = Path(job["root"])
    ensure_ready(root, job["after_run"])
    from gradientclimb.benchmarks.runner import evaluate_checkpoints, run_evaluation, run_training
    from gradientclimb.experiments import load_run

    action = job["action"]
    source = load_run(root, job["source_run"]) if job.get("source_run") else None
    if action["kind"] == "evaluate_checkpoints":
        result = evaluate_checkpoints(root, source["run_id"], action["episodes"])
    elif action["kind"] == "generalization":
        result = run_evaluation(
            root,
            checkpoint_path(root, source),
            episodes=action["episodes"],
            seed_start=action["seed_start"],
            generalization=True,
        )
    else:
        if action.get("require_clean_source"):
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                check=True,
            )
            if status.stdout.strip():
                raise RuntimeError("This reproduction requires a clean source worktree")
        config = dict(action["config"])
        if source:
            config["parent_checkpoint"] = checkpoint_path(root, source)
        result = run_training(
            root,
            action["algorithm"],
            action["seconds"],
            job["seed"],
            config,
            action["experiment"],
            action["benchmark_class"],
            source["run_id"] if source else None,
        )
    write_json(Path(job["result_path"]), {"run_id": result["run_id"], "status": result["status"]})


def execute(plan, root, resume=False):
    from gradientclimb.experiments import load_run, verify_run

    validate_plan(plan)
    ensure_ready(root, plan["after_run"])
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    directory = root / "post-benchmark" / digest
    if directory.exists() and not resume:
        raise FileExistsError(
            "This plan has execution state; use --resume to skip verified completed jobs"
        )
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan.json"
    if not plan_path.exists():
        write_json(plan_path, plan)
    outcomes = {"after_run": [plan["after_run"]]}
    for action in plan["actions"]:
        sources = outcomes[action["source"]] if action.get("source") else [None]
        seeds = action["seeds"] if action["kind"] == "train" else [None]
        completed = []
        for source_index, source in enumerate(sources):
            for seed in seeds:
                name = f"{action['id']}-{source_index}-{seed if seed is not None else 'eval'}"
                result_path = directory / f"{name}.result.json"
                source_record = load_run(root, source) if source else None
                if resume and result_path.exists():
                    result = json.loads(result_path.read_text())
                    record = load_run(root, result["run_id"])
                    if (
                        record["status"] != "completed"
                        or not verify_run(root, result["run_id"])["valid"]
                    ):
                        raise ValueError(
                            "An existing job result is incomplete or failed verification"
                        )
                    validate_job_record(record, action, source_record, seed)
                    journal_event(directory, "resume_verified", job=name, **result)
                else:
                    job = {
                        "root": str(root),
                        "after_run": plan["after_run"],
                        "action": action,
                        "source_run": source,
                        "seed": seed,
                        "result_path": str(result_path),
                    }
                    job_path = directory / f"{name}.job.json"
                    write_json(job_path, job)
                    print(json.dumps({"starting": name, "plan_sha256": digest}), flush=True)
                    journal_event(directory, "job_started", job=name, source_run=source, seed=seed)
                    started = time.perf_counter()
                    try:
                        with (directory / f"{name}.console.log").open("a", encoding="utf-8") as log:
                            subprocess.run(
                                [
                                    sys.executable,
                                    str(Path(__file__).resolve()),
                                    "--run-job",
                                    str(job_path),
                                ],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    except Exception as error:
                        journal_event(
                            directory,
                            "job_failed",
                            job=name,
                            process_seconds=time.perf_counter() - started,
                            error=repr(error),
                        )
                        raise
                    result = json.loads(result_path.read_text())
                    if (
                        result["status"] != "completed"
                        or not verify_run(root, result["run_id"])["valid"]
                    ):
                        raise ValueError(
                            "Completed subprocess did not produce a valid completed run"
                        )
                    validate_job_record(
                        load_run(root, result["run_id"]), action, source_record, seed
                    )
                    journal_event(
                        directory,
                        "job_completed",
                        job=name,
                        process_seconds=time.perf_counter() - started,
                        **result,
                    )
                    print(json.dumps({"completed": name, **result}), flush=True)
                completed.append(result["run_id"])
        outcomes[f"action:{action['id']}"] = completed
        write_json(directory / "outcomes.json", outcomes)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        for status in ("running", "failed", "cancelled"):
            try:
                assert_finished({"status": status})
            except RuntimeError:
                continue
            raise AssertionError("Unfinished source was accepted")
        assert_finished({"status": "completed"})
        print("Post-benchmark guard invariants passed; no jobs executed.")
        return
    if args.run_job:
        run_job(args.run_job)
        return
    if not args.plan:
        parser.error("--plan is required")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expanded = validate_plan(plan)
    if not args.execute:
        print(
            json.dumps({"mode": "dry-run", "plan": plan, "jobs_executed": 0, **expanded}, indent=2)
        )
        return
    print(json.dumps(execute(plan, args.root.resolve(), args.resume), indent=2))


if __name__ == "__main__":
    main()
