"""Paired measurement of the live-observer cost on governed PPO training.

Protocol ``headed-overhead-paired-0.1``: for each seed, train twice with the same
seed, budget and configuration — once without an observer and once with the
headless-record observer (no window unless ``--window``) — in a counterbalanced
order (even seed positions run off then on; odd positions on then off). Each arm
is a canonical ``run_training`` record; every arm runs in a fresh Python process
by default (``--in-process`` runs them in this interpreter for tests). The
per-seed differences (on minus off; negative = headed cost) of
``environment_steps`` and ``environment_steps_per_second`` (steps over the
training clock) are summarized with a percentile bootstrap over the paired seeds
(each pair an independently trained policy; no episodes are resampled) and
sealed in one protocol run through RunRecorder under the algorithm label
``headed-overhead-protocol`` — it is a measurement record, never a training run.
The default is a dry-run plan print; ``--execute`` trains. The script refuses to
start while a PPO/CEM run is active under the root, and it never touches the
training hyperparameters: the paired records must have identical configurations
except for ``observer``, and the on arm's observer must have run to completion
(no error, stopped cleanly, frames rendered, snapshots received) for the pair
to count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

PROTOCOL = "headed-overhead-paired-0.1"
PROTOCOL_ALGORITHM = "headed-overhead-protocol"
SIGN_CONVENTION = "on minus off; negative = headed cost"
CI_METHOD = (
    "percentile bootstrap over paired seeds (n = pairs), each pair an independently "
    "trained policy; no episodes resampled; undefined for a single pair"
)
DEFAULT_ENVS = 64
ARMS = ("off", "on")
RESERVED_CONFIG_KEYS = {
    "algorithm",
    "seconds",
    "seed",
    "callback",
    "parent_checkpoint",
    "snapshot",
    "snapshot_interval",
    "observer",
}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def build_plan(args) -> dict:
    from gradientclimb.visualization.observer import ObserverConfig

    seeds = [int(seed) for seed in args.seeds]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if not args.seconds > 0:
        raise ValueError("The per-arm training budget must be positive seconds")
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise TypeError("The training config must be a JSON object of train_ppo keywords")
    if RESERVED_CONFIG_KEYS & set(config):
        raise ValueError(f"Config must not set {sorted(RESERVED_CONFIG_KEYS & set(config))}")
    # An explicit --envs must never be silently overridden by the config file.
    if args.envs is not None and "num_envs" in config and config["num_envs"] != args.envs:
        raise ValueError(
            f"--envs {args.envs} conflicts with num_envs {config['num_envs']} in the config"
        )
    envs = args.envs if args.envs is not None else config.get("num_envs", DEFAULT_ENVS)
    if not isinstance(envs, int) or isinstance(envs, bool) or envs < 1:
        raise ValueError("At least one training environment is required")
    config["num_envs"] = envs
    observer = {
        "mode": "thread",
        "display": "window" if args.window else "none",
        "policy": "current",
        "snapshot_interval": float(args.observer_interval),
        "seed": int(args.observer_seed),
        "fps": float(args.observer_fps) if args.observer_fps else None,
        "video": None,
    }
    resolved = ObserverConfig.from_mapping(observer)
    for seed in seeds:
        resolved.validate(seed)
    return {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "seconds": float(args.seconds),
        "num_envs": int(config["num_envs"]),
        "config": config,
        "experiment": args.experiment,
        "benchmark_class": "cold_start",
        "algorithm": "ppo",
        "observer": observer,
        "order": {
            str(seed): list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
            for index, seed in enumerate(seeds)
        },
        "sign_convention": SIGN_CONVENTION,
        "rate_definition": "environment_steps / summary.training_clock_seconds (training clock)",
        "ci_method": CI_METHOD,
    }


def ensure_idle(root: Path) -> None:
    from gradientclimb.experiments import list_runs

    active = [
        record["run_id"]
        for record in list_runs(root)
        if record["status"] == "running" and record["algorithm"] in {"ppo", "cem"}
    ]
    if active:
        raise RuntimeError(f"Learning runs remain active; refusing paired measurement: {active}")


def train_arm(root: Path, plan: dict, seed: int, arm: str) -> dict:
    from gradientclimb.benchmarks.runner import run_training

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    return run_training(
        root,
        plan["algorithm"],
        plan["seconds"],
        seed,
        plan["config"],
        plan["experiment"],
        plan["benchmark_class"],
        None,
        observer=plan["observer"] if arm == "on" else None,
    )


def run_arm(root: Path, plan: dict, seed: int, arm: str, in_process: bool, directory: Path) -> dict:
    """One arm as a canonical run record; fresh process unless ``in_process``."""
    from gradientclimb.experiments import load_run

    name = f"seed-{seed}-{arm}"
    started = time.perf_counter()
    if in_process:
        record = train_arm(root, plan, seed, arm)
    else:
        result_path = directory / f"{name}.result.json"
        job_path = directory / f"{name}.job.json"
        write_json(
            job_path,
            {
                "root": str(root),
                "plan": plan,
                "seed": seed,
                "arm": arm,
                "result_path": str(result_path),
            },
        )
        with (directory / f"{name}.console.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--run-job", str(job_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record = load_run(root, json.loads(result_path.read_text(encoding="utf-8"))["run_id"])
    print(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "run_id": record["run_id"],
                "status": record["status"],
                "process_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    return record


def run_job(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    root = Path(job["root"])
    ensure_idle(root)
    record = train_arm(root, job["plan"], int(job["seed"]), job["arm"])
    write_json(Path(job["result_path"]), {"run_id": record["run_id"], "status": record["status"]})


OBSERVER_HEALTH = (
    "the on arm's observer must have run to completion: no error, stopped cleanly, "
    "no sink errors, frames rendered and snapshots received"
)


def validate_observer_ran(on: dict) -> None:
    """A pair whose observer crashed or stopped early is not an overhead measurement."""
    observer = on.get("summary", {}).get("observer")
    problems = []
    if not isinstance(observer, dict):
        observer, problems = {}, ["no observer summary"]
    if observer.get("error") is not None:
        problems.append(f"error={observer['error']!r}")
    if observer.get("record_error") is not None:
        problems.append(f"record_error={observer['record_error']!r}")
    if observer.get("stopped_cleanly") is not True:
        problems.append("stopped_cleanly is not true")
    if observer.get("sink_errors"):
        problems.append(f"sink_errors={observer['sink_errors']!r}")
    if not (isinstance(observer.get("frames_rendered"), int) and observer["frames_rendered"] > 0):
        problems.append("frames_rendered is not positive")
    received = observer.get("snapshots_received")
    if not (isinstance(received, int) and received > 0):
        problems.append("snapshots_received is not positive")
    if problems:
        joined = "; ".join(problems)
        raise ValueError(
            f"Run {on.get('run_id')} observer did not run to completion ({joined}); "
            f"{OBSERVER_HEALTH}"
        )


def validate_pair(off: dict, on: dict, root: Path | None = None) -> None:
    """Both arms must be completed twins that differ only in the observer block."""
    from gradientclimb.experiments import verify_run

    if off.get("seed") != on.get("seed"):
        raise ValueError("Paired arms must share the training seed")
    if off.get("experiment_id") != on.get("experiment_id"):
        raise ValueError("Paired arms must belong to the same experiment")
    off_config, on_config = dict(off.get("configuration", {})), dict(on.get("configuration", {}))
    if "observer" in off_config or "observer" not in on_config:
        raise ValueError("The off arm must have no observer block and the on arm must have one")
    on_config.pop("observer")
    if on_config != off_config:
        raise ValueError("Paired arms differ beyond the observer block")
    if off_config.get("seconds") != on_config.get("seconds"):
        raise ValueError("Paired arms must share the training budget")
    for record in (off, on):
        if record.get("status") != "completed":
            raise ValueError(f"Run {record.get('run_id')} is not completed")
        if "training_clock_seconds" not in record.get("summary", {}):
            raise ValueError(f"Run {record.get('run_id')} lacks a training clock summary")
        if root is not None and not verify_run(root, record["run_id"])["valid"]:
            raise ValueError(f"Run {record['run_id']} failed artifact verification")
    validate_observer_ran(on)


def _rate(record: dict) -> float:
    return record["environment_steps"] / record["summary"]["training_clock_seconds"]


def paired_summary(values: list[float]) -> dict:
    """``summarize`` over per-seed paired differences with a truthful CI label.

    ``summarize`` bootstraps whatever observations it is given; here those are
    one difference per seed, not episodes of one policy, so its default
    ``ci_method`` label would be false and is replaced.
    """
    from gradientclimb.evaluation.benchmark import summarize

    summary = summarize(values)
    summary["ci_method"] = CI_METHOD
    summary["unit_of_analysis"] = "paired seed (on minus off)"
    return summary


def paired_differences(pairs: list[dict]) -> dict:
    """Summaries of on-minus-off differences across seeds plus per-pair rows."""
    rows = []
    for pair in pairs:
        off, on = pair["off"], pair["on"]
        validate_pair(off, on)
        steps_off, steps_on = off["environment_steps"], on["environment_steps"]
        rate_off, rate_on = _rate(off), _rate(on)
        observer = on.get("summary", {}).get("observer", {})
        rows.append(
            {
                "seed": off["seed"],
                "off_run": off["run_id"],
                "on_run": on["run_id"],
                "environment_steps_off": steps_off,
                "environment_steps_on": steps_on,
                "environment_steps_difference": steps_on - steps_off,
                "environment_steps_per_second_off": rate_off,
                "environment_steps_per_second_on": rate_on,
                "environment_steps_per_second_difference": rate_on - rate_off,
                "relative_environment_steps_change": (steps_on - steps_off) / steps_off
                if steps_off
                else None,
                "observer": {
                    key: observer.get(key)
                    for key in (
                        "frames_rendered",
                        "snapshots_received",
                        "snapshots_loaded",
                        "observer_episodes",
                        "snapshot_count",
                        "snapshot_copy_seconds",
                        "snapshot_callback_seconds",
                        "snapshot_errors",
                        "error",
                        "record_error",
                        "stopped_cleanly",
                        "sink_errors",
                        "video_skipped_reason",
                    )
                },
            }
        )
    if not rows:
        raise ValueError("At least one completed pair is required")
    if any(row["relative_environment_steps_change"] is None for row in rows):
        raise ValueError("An off arm recorded zero environment steps; the pair is uninformative")
    return {
        "protocol": PROTOCOL,
        "sign_convention": SIGN_CONVENTION,
        "n_pairs": len(rows),
        "ci_method": CI_METHOD,
        "environment_steps_difference": paired_summary(
            [row["environment_steps_difference"] for row in rows]
        ),
        "environment_steps_per_second_difference": paired_summary(
            [row["environment_steps_per_second_difference"] for row in rows]
        ),
        "relative_environment_steps_change": paired_summary(
            [row["relative_environment_steps_change"] for row in rows]
        ),
        "pairs": rows,
    }


def seal_protocol(
    root: Path, plan: dict, pairs: list[dict], summary: dict, directory: Path
) -> dict:
    from gradientclimb.experiments import RunRecorder, load_run
    from gradientclimb.simulation import SIMULATOR_VERSION

    # A measurement record: its algorithm label must never read as a PPO/CEM
    # training run (analysis ingestion, idle checks and dashboards key on those).
    with RunRecorder(
        root,
        plan["experiment"],
        config=plan,
        seed=plan["seeds"][0],
        algorithm=PROTOCOL_ALGORITHM,
        environment="uncalibrated_hill_surrogate",
        simulator_version=SIMULATOR_VERSION,
        evidence_domain="simulation",
        protocol=PROTOCOL,
        record_kind="paired_overhead_measurement",
        pair_seeds=list(plan["seeds"]),
        arm_algorithm=plan["algorithm"],
        arm_benchmark_class=plan["benchmark_class"],
        state_directory=str(directory),
    ) as run:
        for row in summary["pairs"]:
            dimensions = {"seed": row["seed"], "off_run": row["off_run"], "on_run": row["on_run"]}
            run.metric(
                "environment_steps_difference", row["environment_steps_difference"], **dimensions
            )
            run.metric(
                "environment_steps_per_second_difference",
                row["environment_steps_per_second_difference"],
                **dimensions,
            )
            run.metric(
                "relative_environment_steps_change",
                row["relative_environment_steps_change"],
                **dimensions,
            )
        pairs_path = directory / "pairs.json"
        write_json(
            pairs_path,
            [{"seed": pair["seed"], "off": pair["off"], "on": pair["on"]} for pair in pairs],
        )
        run.register_artifact(pairs_path, "paired_runs")
        run.finalize(
            protocol=PROTOCOL,
            sign_convention=SIGN_CONVENTION,
            rate_definition=plan["rate_definition"],
            ci_method=CI_METHOD,
            n_pairs=summary["n_pairs"],
            environment_steps_difference=summary["environment_steps_difference"],
            environment_steps_per_second_difference=summary[
                "environment_steps_per_second_difference"
            ],
            relative_environment_steps_change=summary["relative_environment_steps_change"],
            pairs=[{key: value for key, value in row.items()} for row in summary["pairs"]],
            scope="uncalibrated_simulator",
            qualifies_real_game=False,
            limitation=(
                f"{summary['n_pairs']} paired seed(s) on one machine; other machine load "
                "uncontrolled; the on arm counts snapshot copy and callback time inside its clock"
            ),
        )
    return load_run(root, run.run_id)


def execution_directory(root: Path, digest: str) -> Path:
    """A fresh state directory per execution: re-running a plan never overwrites."""
    base = root / "headed-overhead"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{attempt + 1}"
        candidate = base / f"{stamp}-{digest[:12]}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique protocol state directory")


def execute(plan: dict, root: Path, in_process: bool) -> dict:
    ensure_idle(root)
    digest = plan_digest(plan)
    directory = execution_directory(root, digest)
    write_json(directory / "plan.json", plan)
    print(
        json.dumps(
            {"starting": PROTOCOL, "plan_sha256": digest, "state_directory": str(directory)}
        ),
        flush=True,
    )
    pairs = []
    for seed in plan["seeds"]:
        arms = {}
        for arm in plan["order"][str(seed)]:
            arms[arm] = run_arm(root, plan, seed, arm, in_process, directory)
        validate_pair(arms["off"], arms["on"], root)
        pairs.append({"seed": seed, "off": arms["off"], "on": arms["on"]})
    summary = paired_differences(pairs)
    protocol_run = seal_protocol(root, plan, pairs, summary, directory)
    write_json(
        directory / "outcome.json",
        {
            "protocol_run": protocol_run["run_id"],
            "plan_sha256": digest,
            "pairs": summary["pairs"],
        },
    )
    return {
        "mode": "executed",
        "plan_sha256": digest,
        "state_directory": str(directory),
        "protocol_run": protocol_run["run_id"],
        "n_pairs": summary["n_pairs"],
        "ci_method": CI_METHOD,
        "environment_steps_difference": summary["environment_steps_difference"],
        "environment_steps_per_second_difference": summary[
            "environment_steps_per_second_difference"
        ],
        "relative_environment_steps_change": summary["relative_environment_steps_change"],
        "pairs": [
            {key: row[key] for key in ("seed", "off_run", "on_run", "environment_steps_difference")}
            for row in summary["pairs"]
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument(
        "--envs",
        type=int,
        default=None,
        help=f"Training environments per arm (default {DEFAULT_ENVS}; must agree with --config)",
    )
    parser.add_argument("--config", type=Path, help="JSON object of extra train_ppo keywords")
    parser.add_argument("--experiment", default="headed-overhead-pilot")
    parser.add_argument("--observer-interval", type=float, default=5.0)
    parser.add_argument("--observer-fps", type=float)
    parser.add_argument("--observer-seed", type=int, default=41000)
    parser.add_argument("--window", action="store_true", help="Measure the Tk window arm")
    parser.add_argument("--in-process", action="store_true", help="No child processes (tests)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-job", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_job:
        run_job(args.run_job)
        return None
    plan = build_plan(args)
    if not args.execute:
        result = {"mode": "dry-run", "plan": plan, "plan_sha256": plan_digest(plan), "runs": 0}
        print(json.dumps(result, indent=2))
        return result
    result = execute(plan, args.root.resolve(), args.in_process)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
