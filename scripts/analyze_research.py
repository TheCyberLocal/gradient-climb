"""Refresh a source-backed research snapshot; no training or model inference.

Usage: python scripts/analyze_research.py --root artifacts [--verify]
Full verification checks sealed artifacts. The default hashes only the inspected
source snapshots, keeping refresh work small during an active training session.
Reports are derived views; canonical run records are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
CHECKPOINT_MINUTES = (5, 10, 20, 30, 45, 60)
DISCOVERY_CORRECTION = {
    "excluded_run": "b5a1748c-0fb1-4787-9d05-a077fd2c5443",
    "corrected_run": "b978c7e7-a4ff-40ae-a1bd-2577ea935010",
    "reason": "The prior label-construction record incorrectly used episode metadata for a non-episodic self-match check. The corrected record covers the same five source images, not five additional observations.",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=2):
    return f"{value:,.{digits}f}" if finite(value) else "not yet measured"


def snapshot(path, sources):
    """Hash precisely the bytes inspected, including a live file's read prefix."""
    data = path.read_bytes()
    relative = os.path.relpath(path, PROJECT).replace("\\", "/")
    sources[relative] = {
        "path": relative,
        "snapshot_sha256": hashlib.sha256(data).hexdigest(),
        "snapshot_bytes": len(data),
    }
    return data.decode("utf-8-sig")


def journal(path, sources):
    if not path.exists():
        return []
    text = snapshot(path, sources)
    # A writer can be part-way through its last append. Never parse a partial row.
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith("\n")]


def distance_result(evaluation):
    result = evaluation.get("results", {})
    episodes = result.get("episodes", [])
    if any(not finite(e.get("distance")) for e in episodes):
        raise ValueError("Episode distance is missing or nonfinite; refusing silent exclusion")
    values = [e["distance"] for e in episodes]
    if not values:
        return None
    recorded = result.get("summary", {}).get("distance", {})
    average = statistics.mean(values)
    if finite(recorded.get("mean")) and not math.isclose(recorded["mean"], average, abs_tol=1e-7):
        raise ValueError(
            f"Recorded mean disagrees with episodes: {evaluation.get('evaluation_id')}"
        )
    return {
        "n": len(values),
        "mean": average,
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "best": max(values),
        "ci95_low": recorded.get("ci95_low"),
        "ci95_high": recorded.get("ci95_high"),
    }


def capture_records(runs, benchmark_id):
    """Capture throughput is an API-loop measurement, never a source-frame rate."""
    benchmark = next((r for r in runs if r["run_id"] == benchmark_id), None)
    rows = []
    for run in runs:
        measured = run.get("summary", {}).get("capture")
        if run.get("algorithm") != "capture" or not isinstance(measured, dict):
            continue
        frames, elapsed = measured.get("captured_frames"), measured.get("elapsed_seconds")
        fps = measured.get("end_to_end_fps")
        if (
            finite(frames)
            and finite(elapsed)
            and elapsed > 0
            and finite(fps)
            and not math.isclose(frames / elapsed, fps, rel_tol=1e-6)
        ):
            raise ValueError(f"Capture FPS contradicts frame count/time: {run['run_id']}")
        concurrent = None
        if benchmark:
            start = datetime.fromisoformat(benchmark["start_time"])
            end = (
                datetime.fromisoformat(benchmark["end_time"])
                if benchmark.get("end_time")
                else datetime.now(UTC)
            )
            cap_start = datetime.fromisoformat(run["start_time"])
            cap_end = (
                datetime.fromisoformat(run["end_time"])
                if run.get("end_time")
                else datetime.now(UTC)
            )
            concurrent = cap_start < end and cap_end > start
        rows.append(
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "backend": measured.get("backend"),
                "recording_enabled": run.get("configuration", {}).get("record"),
                "captured_frames": frames,
                "recorded_frames": measured.get("recorded_frames"),
                "recorded_bytes": measured.get("recorded_bytes"),
                "elapsed_seconds": elapsed,
                "end_to_end_capture_loop_fps": fps,
                "capture_api_ms_mean": measured.get("capture_latency_ms_mean"),
                "capture_api_ms_p95": measured.get("capture_latency_ms_p95"),
                "output_resolution": measured.get("output_resolution"),
                "source_new_frame_rate": None,
                "source_dropped_frames": measured.get("source_dropped_frames"),
                "scene": run.get("metadata", {}).get("scene"),
                "overlaps_selected_training": concurrent,
                "limitations": measured.get("limitations", []),
            }
        )
    return rows


def checkpoint_rows(runs, evaluations, benchmark_id):
    """A final score never substitutes for a scheduled checkpoint measurement."""
    rows = {
        m: {
            "requested_minutes": m,
            "actual_training_seconds": None,
            "status": "not yet measured",
            "distance": None,
            "evaluation_run": None,
            "checkpoint_hash": None,
        }
        for m in CHECKPOINT_MINUTES
    }
    source = next((r for r in runs if r["run_id"] == benchmark_id), None)
    if source is None:
        return list(rows.values())
    artifacts = {}
    for artifact in source.get("artifact_manifest", []):
        meta = artifact.get("metadata", {})
        target = meta.get("target_seconds")
        if artifact.get("kind") == "checkpoint" and finite(target) and target / 60 in rows:
            minute = target / 60
            artifacts[artifact["sha256"]] = minute
            rows[minute].update(
                status="checkpoint saved; evaluation not yet measured",
                actual_training_seconds=meta.get("training_elapsed_seconds"),
                checkpoint_hash=artifact["sha256"],
            )
    parents = {
        r["run_id"]: r.get("parent_run") or r.get("configuration", {}).get("parent_run")
        for r in runs
    }
    for evaluation in sorted(evaluations, key=lambda e: e.get("timestamp", "")):
        if parents.get(evaluation["run_id"]) != benchmark_id:
            continue
        meta = evaluation.get("metadata", {})
        minute = meta.get("requested_minutes")
        digest = evaluation.get("checkpoint_hash")
        if not digest or digest not in artifacts:
            continue
        if minute is None:
            minute = artifacts.get(digest)
        if minute not in rows:
            continue
        if digest in artifacts and artifacts[digest] != minute:
            raise ValueError("Checkpoint hash and requested timestamp disagree")
        actual = meta.get("training_elapsed_seconds")
        if actual is None and finite(meta.get("training_minutes")):
            actual = meta["training_minutes"] * 60
        if actual is None:
            actual = rows[minute]["actual_training_seconds"]
        source_actual = rows[minute]["actual_training_seconds"]
        if (
            finite(source_actual)
            and finite(actual)
            and not math.isclose(source_actual, actual, abs_tol=1e-6)
        ):
            raise ValueError("Checkpoint evaluation and saved training timestamps disagree")
        distance = distance_result(evaluation)
        if distance is not None and finite(actual):
            rows[minute].update(
                status="measured in uncalibrated simulator",
                actual_training_seconds=actual,
                distance=distance,
                evaluation_run=evaluation["run_id"],
                checkpoint_hash=digest,
            )
    return list(rows.values())


def paired_evaluations(before, after):
    """Pair recorded episodes only; no policy inference or training takes place."""
    left, right = before["results"], after["results"]
    scope = (
        "scope",
        "simulator_version",
        "calibration_version",
        "profile",
        "terrain",
        "max_steps",
        "deterministic",
    )
    if any(left.get(key) != right.get(key) for key in scope):
        raise ValueError("Paired results must share the complete evaluation scope")
    left_seeds = [row["seed"] for row in left.get("episodes", [])]
    right_seeds = [row["seed"] for row in right.get("episodes", [])]
    if (
        not left_seeds
        or len(set(left_seeds)) != len(left_seeds)
        or set(left_seeds) != set(right_seeds)
        or len(right_seeds) != len(left_seeds)
    ):
        raise ValueError("Paired results require identical unique episode seeds")
    from gradientclimb.evaluation import compare_paired

    return {
        "before_evaluation_run": before["run_id"],
        "after_evaluation_run": after["run_id"],
        "before_checkpoint_hash": before["checkpoint_hash"],
        "after_checkpoint_hash": after["checkpoint_hash"],
        "profile": left["profile"],
        "terrain": left["terrain"],
        "seeds": sorted(left_seeds),
        "before_mean": distance_result(before)["mean"],
        "after_mean": distance_result(after)["mean"],
        "difference_after_minus_before": compare_paired(right, left),
    }


def adaptation_comparisons(runs, evaluations):
    """Require declared weight lineage and matching generalization measurements."""
    conditions = {"in_distribution", "new_map", "new_vehicle", "new_vehicle_and_map"}
    available = [e for e in evaluations if e.get("metadata", {}).get("condition") in conditions]
    rows = []
    for run in runs:
        if (
            run["status"] != "completed"
            or not run.get("parent_run")
            or not run.get("parent_checkpoint")
            or not run.get("checkpoint_hash")
        ):
            continue
        for after in available:
            if after.get("checkpoint_hash") != run["checkpoint_hash"]:
                continue
            candidates = [
                e
                for e in available
                if e.get("checkpoint_hash") == run["parent_checkpoint"]
                and e["metadata"]["condition"] == after["metadata"]["condition"]
                and e["results"].get("seeds") == after["results"].get("seeds")
            ]
            if not candidates:
                continue
            before = max(candidates, key=lambda e: e.get("timestamp", ""))
            comparison = paired_evaluations(before, after)
            rows.append(
                {
                    "parent_training_run": run["parent_run"],
                    "child_training_run": run["run_id"],
                    "condition": after["metadata"]["condition"],
                    **comparison,
                }
            )
    return rows


def ablation_results(training, evaluations):
    prefix = "post-hour-ablation-"
    runs = [
        r
        for r in training
        if r["status"] == "completed" and r["experiment"].startswith(prefix) and r["distance"]
    ]
    groups = defaultdict(list)
    for run in runs:
        groups[run["experiment"][len(prefix) :]].append(run)
    summaries = []
    baseline = {r["seed"]: r for r in groups.get("baseline", [])}
    for condition, members in groups.items():
        values = [r["distance"]["mean"] for r in members]
        pairs = []
        for member in members:
            reference = baseline.get(member["seed"])
            if reference is None or condition == "baseline":
                continue
            before = [
                e
                for e in evaluations
                if e["run_id"] == reference["run_id"]
                and e.get("checkpoint_hash") == reference["checkpoint_hash"]
            ]
            after = [
                e
                for e in evaluations
                if e["run_id"] == member["run_id"]
                and e.get("checkpoint_hash") == member["checkpoint_hash"]
            ]
            if before and after:
                pairs.append(
                    {"training_seed": member["seed"], **paired_evaluations(before[-1], after[-1])}
                )
        differences = [p["after_mean"] - p["before_mean"] for p in pairs]
        summaries.append(
            {
                "condition": condition,
                "training_seeds": [r["seed"] for r in members],
                "run_ids": [r["run_id"] for r in members],
                "mean_validation_distance": statistics.mean(values),
                "std_between_training_seeds": statistics.stdev(values) if len(values) > 1 else None,
                "actual_training_seconds": [r["actual_training_seconds"] for r in members],
                "paired_difference_vs_baseline_mean": statistics.mean(differences)
                if differences
                else None,
                "paired_difference_vs_baseline_training_seed_sd": statistics.stdev(differences)
                if len(differences) > 1
                else None,
                "paired_training_seed_count": len(pairs),
                "paired_comparisons": pairs,
            }
        )
    return summaries


def collect(root, benchmark_id=None, verify=False):
    from gradientclimb.experiments import list_runs, verify_run

    runs = list_runs(root)
    sources, metrics, evaluations, integrity, source_seal_checks = {}, {}, [], {}, {}
    evidence_times = []
    for run in runs:
        directory = root / "runs" / run["run_id"]
        record = directory / ("run.json" if (directory / "run.json").exists() else "run-start.json")
        inspected_run = json.loads(snapshot(record, sources))
        run.clear()
        run.update(inspected_run)
        metrics[run["run_id"]] = journal(directory / "metrics.jsonl", sources)
        recorded_evaluations = journal(directory / "evaluations.jsonl", sources)
        if run["status"] == "completed":
            evaluations.extend(recorded_evaluations)
        seal_path = directory / "seal.json"
        if run["status"] != "running" and seal_path.exists():
            seal = json.loads(snapshot(seal_path, sources))
            checked = []
            for name in (record.name, "metrics.jsonl", "evaluations.jsonl"):
                key = os.path.relpath(directory / name, PROJECT).replace("\\", "/")
                if key in sources:
                    expected = seal["files"].get(name)
                    if not expected or sources[key]["snapshot_sha256"] != expected["sha256"]:
                        raise ValueError(
                            f"Inspected source disagrees with seal: {run['run_id']}/{name}"
                        )
                    checked.append(name)
            source_seal_checks[run["run_id"]] = {
                "valid": True,
                "files": checked,
                "scope": "inspected source files only",
            }
        evidence_times.extend(m["timestamp"] for m in metrics[run["run_id"]] if m.get("timestamp"))
        evidence_times.append(run.get("end_time") or run["start_time"])
        if verify and run["status"] != "running":
            integrity[run["run_id"]] = verify_run(root, run["run_id"])
            if not integrity[run["run_id"]]["valid"]:
                raise ValueError(f"Artifact verification failed: {integrity[run['run_id']]}")
    if benchmark_id is None:
        candidates = [
            r
            for r in runs
            if r.get("configuration", {}).get("seconds") == 3600
            and not r.get("parent_checkpoint")
            and r.get("algorithm") == "ppo"
        ]
        benchmark_id = (
            max(candidates, key=lambda r: r["start_time"])["run_id"] if candidates else None
        )
    elif not any(r["run_id"] == benchmark_id for r in runs):
        raise ValueError("Requested benchmark run is absent from the canonical store")
    # Keep the defective record in the provenance index, but exclude its evidence.
    evaluations = [e for e in evaluations if e["run_id"] != DISCOVERY_CORRECTION["excluded_run"]]
    measurements = []
    for evaluation in evaluations:
        distance = distance_result(evaluation)
        if distance is not None:
            result = evaluation["results"]
            measurements.append(
                {
                    "run_id": evaluation["run_id"],
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "checkpoint_hash": evaluation.get("checkpoint_hash"),
                    "protocol": evaluation.get("protocol"),
                    "metadata": evaluation.get("metadata", {}),
                    "profile": result.get("profile"),
                    "terrain": result.get("terrain"),
                    "seeds": result.get("seeds"),
                    "max_steps": result.get("max_steps"),
                    "scope": result.get("scope"),
                    "distance": distance,
                    "failure_rates": result.get("summary", {}).get("failure_rates", {}),
                    "survival_seconds": result.get("summary", {}).get("survival_seconds", {}),
                    "action_counts": result.get("action_counts"),
                }
            )
    training, groups = [], defaultdict(list)
    for run in runs:
        config = run.get("summary", {}).get("model_config") or run.get("configuration", {})
        if run.get("algorithm") not in {"ppo", "cem"} or not finite(config.get("seconds")):
            continue
        latest = {m["name"]: m["value"] for m in metrics[run["run_id"]]}
        matches = [m for m in measurements if m["run_id"] == run["run_id"]]
        result = matches[-1] if matches else None
        elapsed = run.get("summary", {}).get(
            "training_clock_seconds", latest.get("wall_clock_seconds")
        )
        steps = latest.get("environment_steps", run.get("environment_steps", 0))
        row = {
            "run_id": run["run_id"],
            "experiment": run["experiment_id"],
            "status": run["status"],
            "algorithm": run["algorithm"],
            "seed": run["seed"],
            "device": config.get("device", "cpu"),
            "num_envs": config.get("num_envs"),
            "requested_seconds": config["seconds"],
            "actual_training_seconds": elapsed,
            "environment_steps": steps,
            "steps_per_second": steps / elapsed if finite(elapsed) and elapsed > 0 else None,
            "parent_run": run.get("parent_run"),
            "parent_checkpoint": run.get("parent_checkpoint"),
            "benchmark_class": run.get("metadata", {}).get("benchmark_class", "unspecified"),
            "config": config,
            "distance": result["distance"] if result else None,
            "checkpoint_hash": run.get("checkpoint_hash"),
            "evaluation_scope": {
                key: result.get(key)
                for key in ("profile", "terrain", "seeds", "max_steps", "scope")
            }
            if result
            else None,
            "timing": {
                k: latest.get(k)
                for k in ("environment_seconds", "inference_seconds", "optimizer_seconds")
            },
        }
        training.append(row)
        if run["experiment_id"] == "runtime-screen" and row["distance"]:
            key = f"{row['algorithm']} / {row['device']} / {row['num_envs']} envs"
            groups[key].append(row)
    pilots = []
    for name, members in groups.items():
        means = [r["distance"]["mean"] for r in members]
        pilots.append(
            {
                "condition": name,
                "training_seeds": [r["seed"] for r in members],
                "runs": [r["run_id"] for r in members],
                "n_training_seeds": len(members),
                "mean_validation_distance": statistics.mean(means),
                "std_between_training_seeds": statistics.stdev(means) if len(means) > 1 else None,
                "mean_steps_per_second": statistics.mean(r["steps_per_second"] for r in members),
            }
        )
    curves = {}
    for run in training:
        points = [
            [m.get("dimensions", {}).get("training_elapsed_seconds"), m["value"]]
            for m in metrics[run["run_id"]]
            if m["name"] == "mean_episode_distance"
        ]
        curves[run["run_id"]] = [p for p in points if all(finite(v) for v in p)]
    cold = [
        r
        for r in training
        if r["status"] == "completed"
        and r["distance"]
        and not r["parent_checkpoint"]
        and finite(r["actual_training_seconds"])
    ]
    frontier = [
        r["run_id"]
        for r in cold
        if not any(
            other["evaluation_scope"] == r["evaluation_scope"]
            and other["actual_training_seconds"] <= r["actual_training_seconds"]
            and other["environment_steps"] <= r["environment_steps"]
            and other["distance"]["mean"] >= r["distance"]["mean"]
            and (
                other["actual_training_seconds"],
                other["environment_steps"],
                other["distance"]["mean"],
            )
            != (r["actual_training_seconds"], r["environment_steps"], r["distance"]["mean"])
            for other in cold
        )
    ]
    support = [
        "docs/operations/game-discovery.md",
        "docs/operations/workstation.md",
        "docs/methodology/qualification.md",
        "docs/methodology/cycle-1-pause-directive.md",
        "research/experiments/cycle-1-plan-status.json",
        "research/experiments/cycle-1-integrity.json",
        "docs/research/screen-body-distillation.md",
        "experiments/definitions/ablations.json",
        "experiments/definitions/adaptation-pending.json",
        "experiments/definitions/post-hour-battery.json",
        "docs/operations/post-benchmark.md",
        "scripts/analyze_research.py",
        "scripts/report_plots.py",
        "src/gradientclimb/simulation/hill.py",
        "src/gradientclimb/algorithms/ppo.py",
        "src/gradientclimb/algorithms/cem.py",
    ]
    for name in support:
        if (PROJECT / name).exists():
            snapshot(PROJECT / name, sources)
    cycle_path = PROJECT / "research/experiments/cycle-1-plan-status.json"
    cycle = json.loads(cycle_path.read_text()) if cycle_path.exists() else None
    audit_path = PROJECT / "research/experiments/cycle-1-integrity.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit:
        # Reuse the full audit only when its complete run set and seal bytes match.
        current_ids = {r["run_id"] for r in runs}
        audit_current = current_ids == {r["run_id"] for r in audit["runs"]}
        for checked in audit["runs"]:
            key = os.path.relpath(root / "runs" / checked["run_id"] / "seal.json", PROJECT).replace(
                "\\", "/"
            )
            audit_current &= sources.get(key, {}).get("snapshot_sha256") == checked["seal_sha256"]
        if not audit_current:
            audit = None
    reference = next((r for r in runs if r["run_id"] == benchmark_id), runs[0] if runs else {})
    cutoff = max((datetime.fromisoformat(t) for t in evidence_times), default=None)
    return {
        "schema_version": "research-summary-1.0",
        "generation_config": {
            "artifact_root": os.path.relpath(root, PROJECT).replace("\\", "/"),
            "benchmark_run": benchmark_id,
            "verify_sealed_artifacts": verify,
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_as_of": cutoff.isoformat() if cutoff else None,
        "research_status": "PAUSED_CYCLE_1_REAL_GAME_QUALIFICATION_INCOMPLETE"
        if cycle
        else "INCOMPLETE_REAL_GAME_QUALIFICATION",
        "cycle_status": cycle,
        "matching_full_integrity_audit": audit,
        "qualifies_real_game": False,
        "evidence_corrections": [DISCOVERY_CORRECTION]
        if any(r["run_id"] == DISCOVERY_CORRECTION["excluded_run"] for r in runs)
        else [],
        "benchmark_run": benchmark_id,
        "benchmark_status": reference.get("status") if benchmark_id else "not started",
        "checkpoints": checkpoint_rows(runs, evaluations, benchmark_id),
        "checkpoints_by_training_run": {
            r["run_id"]: checkpoint_rows(runs, evaluations, r["run_id"])
            for r in training
            if r["requested_seconds"] >= 300
        },
        "paired_adaptation": adaptation_comparisons(runs, evaluations),
        "ablations": ablation_results(training, evaluations),
        "pilots": pilots,
        "training_runs": training,
        "evaluations": measurements,
        "training_curves": curves,
        "cold_start_observed_frontier": frontier,
        "hardware": reference.get("hardware", {}),
        "framework_versions": reference.get("framework_versions", {}),
        "analysis_dependencies": {"matplotlib": version("matplotlib")},
        "python_version": reference.get("python_version"),
        "source_records": [
            {
                "run_id": r["run_id"],
                "experiment": r["experiment_id"],
                "algorithm": r["algorithm"],
                "status": r["status"],
                "git_sha": r.get("git_sha"),
                "dirty_worktree": r.get("dirty_worktree"),
                "source_diff_sha256": r.get("source_diff_sha256"),
                "configuration_sha256": r.get("config_sha256"),
                "checkpoint_sha256": r.get("checkpoint_hash"),
                "parent_run": r.get("parent_run"),
                "summary": r.get("summary", {}),
                "evidence_excluded": r["run_id"] == DISCOVERY_CORRECTION["excluded_run"],
            }
            for r in runs
        ],
        "discovery_evaluations": [
            e
            for e in evaluations
            if e.get("metadata", {}).get("evidence_domain") == "real_game_discovery"
        ],
        "capture_measurements": capture_records(runs, benchmark_id),
        "control_probes": [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "summary": r.get("summary", {}),
                "config": r.get("configuration", {}),
            }
            for r in runs
            if r["experiment_id"] == "real-control-probe"
        ],
        "native_training_attempts": [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "duration_seconds": r.get("duration"),
                "optimizer_updates": r.get("optimizer_updates"),
                "requested_seconds": r.get("summary", {}).get("requested_training_seconds"),
                "eligible_episodes": r.get("summary", {}).get("eligible_episodes"),
                "stop_reason": r.get("summary", {}).get("stop_reason"),
                "governed_elapsed_at_stop": r.get("summary", {}).get("governed_elapsed_at_stop"),
                "episodes": [
                    {
                        k: ep.get(k)
                        for k in (
                            "distance",
                            "observed_hud_max",
                            "score_decision",
                            "reason",
                            "park_error",
                        )
                    }
                    for ep in r.get("summary", {}).get("episode_summaries", [])
                ],
            }
            for r in runs
            if r.get("algorithm") == "sequential_episode_cem"
        ],
        "pixel_measurements": [
            {"run_id": r["run_id"], "status": r["status"], "summary": r.get("summary", {})}
            for r in runs
            if r.get("metadata", {}).get("evidence_domain") == "real_game_pixel_measurement"
        ],
        "throughput": [
            {"run_id": r["run_id"], "results": r.get("summary", {}).get("results", [])}
            for r in runs
            if r["experiment_id"] == "simulation-throughput" and r["status"] == "completed"
        ],
        "sources": list(sources.values()),
        "artifact_verification": integrity,
        "source_seal_checks": source_seal_checks,
        "verification_mode": "full sealed artifact check"
        if verify
        else "inspected sealed source files verified against seals; full artifact recheck not requested",
        "snapshot_consistency": "Per-file read snapshots; active journals can have different evidence cutoffs.",
    }


def svg_plot(series, x_label, y_label, title, log_x=False, points_only=False):
    from report_plots import scientific_plot

    return scientific_plot(series, x_label, y_label, title, log_x, points_only)


def render(summary, output_dir, root):
    sections = []
    figures = {}

    def paragraph(title, text):
        sections.append((title, text, None))

    def table(title, headers, rows, note=""):
        sections.append((title, note, (headers, rows)))

    pilots = summary["pilots"]
    text = "GradientClimb measures how control quality changes with training time, experience, compute and prior knowledge. "
    text += "The current evidence establishes learning in an original uncalibrated simulator. Real-game qualification remains incomplete. "
    text += f"The selected one-hour record is {summary['benchmark_run'] or 'not yet available'} ({summary['benchmark_status']})."
    paragraph("Abstract and research status", text)
    cycle = summary.get("cycle_status")
    if cycle:
        battery = cycle["post_hour_executor"]
        paragraph(
            "Cycle 1 pause and evidence cutoff",
            f"Research is paused by explicit user direction. All {battery['actions_completed']} registered post-hour actions completed, including {battery['learning_runs_completed']} learning runs: {fmt(battery['requested_learning_seconds'], 0)} requested training seconds and {fmt(battery['actual_learning_seconds'], 3)} actual training seconds. This total excludes the original hour and separate runtime pilots; process setup and offline evaluation have separate records. No queued experiment is authorized to start. Canonical evidence ends at {summary['evidence_as_of']}; report generation is a later read-only analysis. Source: research/experiments/cycle-1-plan-status.json. Historical preregistrations retain their original wording; the separate status record governs their terminal cycle disposition.",
        )
        table(
            "Registered plan disposition",
            ["Plan", "Cycle status", "Execution status", "Interpretation"],
            [
                [p["plan"]["path"], p["cycle_status"], p["execution_status"], p["note"]]
                for p in cycle["plans"]
            ],
        )
    primary = next(
        (r for r in summary["training_runs"] if r["run_id"] == summary["benchmark_run"]), None
    )
    if primary and primary["status"] == "completed" and primary["distance"]:
        measured = [row for row in summary["checkpoints"] if row["distance"]]
        checkpoint_note = ""
        if len(measured) > 1 and any(
            right["distance"]["mean"] < left["distance"]["mean"]
            for left, right in pairwise(measured)
        ):
            checkpoint_note = " Measured checkpoint means were not monotonic; a later policy did not outperform every earlier policy on this validation set."
        paragraph(
            "Completed primary observation",
            f"Run {primary['run_id']} trained for an actual {fmt(primary['actual_training_seconds'], 3)} seconds and collected {primary['environment_steps']:,.0f} transitions. Its recorded final validation mean was {fmt(primary['distance']['mean'])} nominal m and median {fmt(primary['distance']['median'])} nominal m across {primary['distance']['n']} episodes. This is one independent training seed, followed by conditional episode evaluation."
            + checkpoint_note,
        )
    replicated_ppo = next(
        (
            p
            for p in pilots
            if p["condition"] == "ppo / cpu / 256 envs" and p["n_training_seeds"] >= 3
        ),
        None,
    )
    replicated_cem = next(
        (
            p
            for p in pilots
            if p["condition"] == "cem / cpu / 64 envs" and p["n_training_seeds"] >= 3
        ),
        None,
    )
    if replicated_ppo and replicated_cem:
        paragraph(
            "Provisional surrogate trainer",
            "PPO on CPU with 256 environments is the provisional choice from the replicated runtime screen. It had higher observed average validation distance and lower between-training-seed variability than the implemented CEM comparator. The table below retains all conditions and source runs. Three training seeds do not establish broad algorithm superiority, and this choice does not select a qualified real-game model.",
        )
    paragraph(
        "Experimental environment and methods",
        "Hill Climb Racing is the first intended real testbed. Simulator metres are nominal surrogate units. The two-contact vehicle surrogate models suspension, traction, braking/reverse, airborne pitch, fuel and termination. Four joint pedal states 00/10/01/11 remain available. PPO uses independent Bernoulli pedals and a 24-feature, four-frame stacked MLP by default (96→64→64; 10,563 actor/critic parameters). CEM searches a 100-parameter linear joint-state controller on two fixed training seeds. Source: src/gradientclimb/simulation/hill.py and src/gradientclimb/algorithms/. No proprietary dynamics or game assets are redistributed.",
    )
    paragraph(
        "Training clock and statistical protocol",
        "Actual monotonic training time includes environment/model/optimizer initialization, collection, updates, logging and checkpoint callbacks. Python imports/provenance setup and offline evaluation are outside that clock. Final serialization can create a small recorded overshoot. Validation episodes use seeds 10000–10019 in the runtime screen. Between-seed SD is computed across independently trained policies; episode bootstrap intervals condition on one policy and cannot substitute for training replicates. Model/runtime selection used validation results, so these are not untouched final-test results. Cold starts and runs with parent weights remain separate.",
    )
    hardware = summary["hardware"]
    table(
        "Recorded workstation",
        ["Field", "Value"],
        [
            ["CPU", hardware.get("cpu", "unavailable")],
            [
                "Physical / logical cores",
                f"{hardware.get('physical_cores', 'unknown')} / {hardware.get('logical_cores', 'unknown')}",
            ],
            [
                "RAM GiB",
                fmt(hardware.get("ram_bytes", 0) / 2**30)
                if hardware.get("ram_bytes")
                else "unavailable",
            ],
            [
                "GPU",
                "; ".join(g.get("name", "unknown") for g in hardware.get("gpus", []))
                or "unavailable",
            ],
            ["Python", summary.get("python_version") or "unavailable"],
            ["Frameworks", json.dumps(summary["framework_versions"], sort_keys=True)],
        ],
        "Detailed machine inventory: docs/operations/workstation.md. Device capability alone is not evidence of faster learning.",
    )
    table(
        "Runtime and algorithm pilots",
        ["Condition", "Seeds", "Mean distance m", "Seed SD m", "Transitions/s", "Source runs"],
        [
            [
                p["condition"],
                ", ".join(map(str, p["training_seeds"])),
                fmt(p["mean_validation_distance"]),
                fmt(p["std_between_training_seeds"]),
                fmt(p["mean_steps_per_second"], 0),
                ", ".join(p["runs"]),
            ]
            for p in pilots
        ],
        "Exploratory 60-second budgets. Three seeds support a provisional engineering choice, not a broad superiority claim. CPU64 and CUDA256 each have one runtime-screen seed.",
    )
    pilot_runs = [r for r in summary["training_runs"] if r["experiment"] == "runtime-screen"]
    paragraph(
        "Development/search compute",
        f"Runtime-screen requested budgets total {fmt(sum(r['requested_seconds'] for r in pilot_runs), 1)} s; actual recorded training clocks total {fmt(sum(r['actual_training_seconds'] or 0 for r in pilot_runs), 2)} s. These are prior development/search costs, not part of the governed one-hour cold-start session. Baseline evaluation, lightweight reporting and screenshot labeling occurred on the shared workstation during the main session; concurrent activity and thermal state limit causal hardware comparisons.",
    )
    table(
        "One-hour scheduled checkpoint measurements",
        [
            "Requested min",
            "Actual training s",
            "Mean m",
            "Median m",
            "Best m",
            "Status",
            "Evaluation run / checkpoint SHA-256",
        ],
        [
            [
                str(r["requested_minutes"]),
                fmt(r["actual_training_seconds"]),
                fmt((r["distance"] or {}).get("mean")),
                fmt((r["distance"] or {}).get("median")),
                fmt((r["distance"] or {}).get("best")),
                r["status"],
                f"{r['evaluation_run'] or '—'} / {r['checkpoint_hash'] or '—'}",
            ]
            for r in summary["checkpoints"]
        ],
        "Scheduled rows require a matching declared parent run and checkpoint evaluation. Missing measurements stay missing. A final checkpoint score is never copied into an earlier time point. These results, when present, concern the surrogate and do not qualify real-game competence.",
    )
    target = next(
        (r for r in summary["training_runs"] if r["run_id"] == summary["benchmark_run"]), None
    )
    selected = [target] if target else [r for r in pilot_runs if r["algorithm"] == "ppo"][:3]
    figures["learning-curve.svg"] = svg_plot(
        [
            (
                f"{r['algorithm']} seed {r['seed']} · {r['run_id'][:8]}",
                summary["training_curves"].get(r["run_id"], []),
            )
            for r in selected
        ],
        "Actual training seconds",
        "Rolling training distance (m)",
        "Training diagnostics",
    )
    paragraph(
        "Learning curves",
        "The plotted rolling mean covers the last 100 completed training episodes collected by changing stochastic policies. It is a training diagnostic, distinct from fixed-policy offline checkpoint evaluation. The source metric is mean_episode_distance, with training_elapsed_seconds from each canonical metric row.",
    )
    sections.append(("Training diagnostics", "", "learning-curve.svg"))
    evaluated = [row for row in summary["checkpoints"] if row["distance"]]
    figures["checkpoint-quality.svg"] = svg_plot(
        [
            (
                statistic.capitalize(),
                [[row["actual_training_seconds"], row["distance"][statistic]] for row in evaluated],
            )
            for statistic in ("mean", "median")
        ],
        "Actual training seconds",
        "Validation distance (nominal m)",
        "Fixed-policy checkpoint evaluation",
    )
    sections.append(
        (
            "Measured checkpoint quality",
            "Each point uses the saved policy at its recorded actual training time and the same 20 validation seeds. Mean and median describe episode variation for one trained policy; these six time points are not six independent training replicates. No point is interpolated from a final evaluation.",
            "checkpoint-quality.svg",
        )
    )
    cold = [
        r
        for r in summary["training_runs"]
        if r["status"] == "completed"
        and r["distance"]
        and not r["parent_checkpoint"]
        and finite(r["actual_training_seconds"])
    ]
    figures["compute-frontier.svg"] = svg_plot(
        [
            (
                f"{r['algorithm']} {r['device']} n{r['num_envs']} s{r['seed']} {r['run_id'][:8]}",
                [[r["actual_training_seconds"], r["distance"]["mean"]]],
            )
            for r in cold
        ],
        "Actual training seconds (log scale)",
        "Validation mean distance (m)",
        "Observed cold-start quality and time",
        log_x=True,
        points_only=True,
    )
    paragraph(
        "Observed efficiency frontier",
        "The plot compares completed cold-start policies by actual training duration and validation quality. Experience counts remain separately inspectable in the JSON snapshot. The observed nondominated set minimizes time and transitions while maximizing mean distance; it ignores uncertainty and is descriptive, not a statistically established frontier. Parent-weight adaptation and extended runs are excluded from this cold-start comparison.",
    )
    sections.append(("Quality versus training time", "", "compute-frontier.svg"))
    throughput = [
        (f"run {r['run_id'][:8]}", [[v["envs"], v["steps_per_second"]] for v in r["results"]])
        for r in summary["throughput"]
    ]
    figures["throughput.svg"] = svg_plot(
        throughput,
        "Parallel environments",
        "Simulator transitions / second",
        "Simulator-only throughput",
    )
    sections.append(
        (
            "Simulator throughput",
            "Includes environment stepping and random action generation; excludes policy inference and optimization. Each condition follows 10 warm-up steps. Source runs: "
            + ", ".join(r["run_id"] for r in summary["throughput"]),
            "throughput.svg",
        )
    )
    baselines = [
        e
        for e in summary["evaluations"]
        if not e["checkpoint_hash"]
        and any(
            r["run_id"] == e["run_id"] and r["experiment"] == "surrogate-evaluation"
            for r in summary["source_records"]
        )
    ]
    table(
        "Recorded simulator baselines",
        ["Baseline", "Run", "Mean m", "Median m", "Episodes", "Protocol"],
        [
            [
                next(
                    r["algorithm"] for r in summary["source_records"] if r["run_id"] == e["run_id"]
                ),
                e["run_id"],
                fmt(e["distance"]["mean"]),
                fmt(e["distance"]["median"]),
                e["distance"]["n"],
                e["protocol"],
            ]
            for e in baselines
        ],
        "Random and always-gas policies are explicit baselines. Discovery-only real-game observations are excluded.",
    )
    generalization = [
        e
        for e in summary["evaluations"]
        if e["metadata"].get("condition")
        in {"in_distribution", "new_map", "new_vehicle", "new_vehicle_and_map"}
    ]
    table(
        "Vehicle/map generalization",
        ["Condition", "Profile / terrain", "Mean m", "Median m", "Source run", "Checkpoint hash"],
        [
            [
                e["metadata"]["condition"],
                f"{e['profile']} / {e['terrain']}",
                fmt(e["distance"]["mean"]),
                fmt(e["distance"]["median"]),
                e["run_id"],
                e["checkpoint_hash"],
            ]
            for e in generalization
        ],
        "Not yet measured when no rows are present. Synthetic heavy/rough shifts do not establish generalization to commercial-game vehicles/maps.",
    )
    table(
        "Generalization outcomes and pedal use",
        [
            "Evaluation run / condition",
            "Termination fractions",
            "Mean survival s",
            "Joint pedal counts 00 / 10 / 01 / 11",
        ],
        [
            [
                f"{row['run_id']} / {row['metadata']['condition']}",
                ", ".join(
                    f"{reason}: {value:.1%}" for reason, value in row["failure_rates"].items()
                ),
                fmt(row["survival_seconds"].get("mean")),
                " / ".join(map(str, row["action_counts"]))
                if row["action_counts"]
                else "not measured",
            ]
            for row in generalization
        ],
        "Time-limit truncation is the fixed 60-second evaluation horizon, not proof of indefinite survival. Counts cover recorded policy decisions during the selected episodes. They establish which joint states were used in the simulator; they do not establish the causal value of each state or real-game input acknowledgement.",
    )
    parented = [r for r in summary["training_runs"] if r["parent_checkpoint"]]
    table(
        "Adaptation and extended training",
        ["Experiment / class", "Additional seconds", "Mean m", "Parent run", "Child run"],
        [
            [
                f"{r['experiment']} / {r['benchmark_class']}",
                fmt(r["actual_training_seconds"]),
                fmt((r["distance"] or {}).get("mean")),
                r["parent_run"] or "not recorded",
                r["run_id"],
            ]
            for r in parented
        ],
        "Both completed ten-minute children load the fixed original parent independently. The source extension is not the heavy/rough child's parent. Paired source retention and target changes appear below. The earlier replicated 5/10/30/60-minute proposal remains deferred, distinct from these completed single-seed ten-minute trials.",
    )
    table(
        "Replicated short component screen",
        [
            "Condition",
            "Training seeds",
            "Mean m",
            "Seed SD m",
            "Mean paired change vs baseline m",
            "Paired training seeds",
            "Runs",
        ],
        [
            [
                row["condition"],
                ", ".join(map(str, row["training_seeds"])),
                fmt(row["mean_validation_distance"]),
                fmt(row["std_between_training_seeds"]),
                "reference"
                if row["condition"] == "baseline"
                else fmt(row["paired_difference_vs_baseline_mean"]),
                row["paired_training_seed_count"],
                ", ".join(row["run_ids"]),
            ]
            for row in summary["ablations"]
        ],
        "The completed registered screen uses 60 requested seconds and training seeds 101/102/103. Each component shares its baseline training seed and validation seeds 10000–10019. These short runs measure source-condition validation, not a domain-randomization generalization benefit. Removing history also changes first-layer size and cost; reducing epochs reallocates wall time between optimization and experience. Three training seeds are exploratory. The earlier 300-second screen remains a separate unexecuted protocol.",
    )
    other_scheduled = [
        {"training_run": run_id, **row}
        for run_id, rows in summary["checkpoints_by_training_run"].items()
        if run_id != summary["benchmark_run"]
        for row in rows
    ]
    table(
        "Reproduction and child checkpoint measurements",
        [
            "Training run",
            "Requested min",
            "Actual s",
            "Mean m",
            "Median m",
            "Status",
            "Evaluation run",
        ],
        [
            [
                row["training_run"],
                row["requested_minutes"],
                fmt(row["actual_training_seconds"]),
                fmt((row["distance"] or {}).get("mean")),
                fmt((row["distance"] or {}).get("median")),
                row["status"],
                row["evaluation_run"] or "—",
            ]
            for row in other_scheduled
        ],
        "Each row belongs to the named training run. For a child, the clock measures additional exposure after its declared parent. Checkpoint evaluation uses the child's training vehicle/map. Unscheduled longer child measurements remain missing: a ten-minute run cannot establish 20/30/45/60-minute adaptation. Parent training costs remain separate.",
    )
    table(
        "Paired target improvement and source retention",
        [
            "Child / parent",
            "Condition",
            "Before mean m",
            "After mean m",
            "Paired change m [episode CI95%]",
            "Paired episodes",
            "Evaluation runs",
        ],
        [
            [
                f"{row['child_training_run']} / {row['parent_training_run']}",
                row["condition"],
                fmt(row["before_mean"]),
                fmt(row["after_mean"]),
                f"{fmt(row['difference_after_minus_before']['distance_difference']['mean'])} [{fmt(row['difference_after_minus_before']['distance_difference']['ci95_low'])}, {fmt(row['difference_after_minus_before']['distance_difference']['ci95_high'])}]",
                len(row["seeds"]),
                f"{row['before_evaluation_run']} / {row['after_evaluation_run']}",
            ]
            for row in summary["paired_adaptation"]
        ],
        "Positive changes favor the child. Default/train (in_distribution) measures retention on the original source condition; a negative change is observed forgetting there. Condition names refer to the original parent: rough terrain is a trained condition for the adapted child. Every pair requires the declared parent checkpoint and identical scenario, horizon, simulator/calibration version, deterministic setting and episode seeds. Bootstrap intervals concern paired episode variation for these fixed policies; one adaptation seed cannot establish training-seed reliability. No matched cold-start heavy/rough baseline exists, so a warm-start speed advantage remains unmeasured.",
    )
    if summary["paired_adaptation"]:
        parts = []
        for row in summary["paired_adaptation"]:
            if row["condition"] in {"in_distribution", "new_vehicle_and_map"}:
                parts.append(
                    f"Child {row['child_training_run']} on {row['condition']}: parent mean {fmt(row['before_mean'])} to child mean {fmt(row['after_mean'])} nominal m."
                )
        paragraph(
            "Additional training outcome",
            " ".join(parts)
            + " The source extension does not establish broad improvement; heavy/rough fine-tuning shows a target improvement with source forgetting. These are single-child results with no measured 20/30/45/60-minute continuation.",
        )
    table(
        "Native learning attempt at the pause",
        [
            "Run",
            "Status",
            "Requested s",
            "Record duration s",
            "Governed clock at stop s",
            "Eligible episodes",
            "Optimizer updates",
        ],
        [
            [
                r["run_id"],
                r["status"],
                fmt(r["requested_seconds"]),
                fmt(r["duration_seconds"], 3),
                fmt(r["governed_elapsed_at_stop"], 3),
                r["eligible_episodes"] if r["eligible_episodes"] is not None else "not recorded",
                r["optimizer_updates"],
            ]
            for r in summary.get("native_training_attempts", [])
        ],
        "Run 51d2527e-9274-40ec-a04d-309117de107d recorded a stable terminal reading of 289 m and 131 observation frames, then parking failed on an unrecognized advertisement. Its eligibility gate rejected the episode: fitness is null and no learner update occurred. The reading is retained as diagnostic native evidence, not a learned-policy score. The shortened pilot did not complete its 600-second budget; the planned native hour and its six checkpoints were never run.",
    )
    table(
        "Actual-window capture measurements",
        [
            "Backend / recording",
            "Resolution",
            "Frames",
            "Elapsed s",
            "Capture-loop FPS",
            "API mean / p95 ms",
            "Concurrent training",
            "Run",
        ],
        [
            [
                f"{c['backend']} / {'PNG' if c['recording_enabled'] else 'unrecorded'}",
                " × ".join(map(str, c["output_resolution"] or [])),
                c["captured_frames"],
                fmt(c["elapsed_seconds"]),
                fmt(c["end_to_end_capture_loop_fps"]),
                f"{fmt(c['capture_api_ms_mean'])} / {fmt(c['capture_api_ms_p95'])}",
                str(c["overlaps_selected_training"]),
                c["run_id"],
            ]
            for c in summary["capture_measurements"]
        ],
        "These are exploratory API capture-loop measurements, not the game's new-frame rate or display-to-observation latency. The PNG-recording pipeline includes encoding and writing work absent from unrecorded runs. Menu and paused scenes were not matched across the initial trials, per-run scene labels were not recorded, and CPU training was active. One trial per backend/condition does not establish backend superiority. Source frame identifiers, source-dropped frames and GPU impact remain unavailable.",
    )
    table(
        "Recorded input attempts and pixel trajectories",
        ["Run", "Status", "Recorded frames", "Observed s", "Reason"],
        [
            [
                r["run_id"],
                r["status"],
                r["summary"].get("frames", "not yet recorded"),
                fmt(r["summary"].get("observed_seconds")),
                r["summary"].get("reason", "not yet recorded"),
            ]
            for r in summary["control_probes"]
        ],
        "These are scripted diagnostic probes, not learned-policy episodes. A completed status means the bounded input schedule finished. Windows input delivery does not establish game acknowledgement; captured coasting does not confirm controlled dynamics. Requested input traces and pixels alone establish neither control competence nor transfer.",
    )
    table(
        "Pixel measurement pipeline",
        [
            "Run",
            "Source frames",
            "Wheel-supported frames",
            "Camera-supported intervals",
            "Held-out evaluation recorded",
        ],
        [
            [
                r["run_id"],
                r["summary"].get("source_frames", "not yet measured"),
                r["summary"].get("wheel_supported_frames", "not yet measured"),
                r["summary"].get("camera_supported_intervals", "not yet measured"),
                str(r["summary"].get("heldout_accuracy_measured", False)),
            ]
            for r in summary["pixel_measurements"]
        ],
        "Heuristic support counts describe output availability. They are not localization, orientation or generalization accuracy; those require independent labels and their recorded evaluation.",
    )
    paragraph(
        "Real-game integration, perception and transfer",
        "Real-game qualification remains incomplete. An observed no-deliberate-input episode ended at 26 m; it was discovery only, with imprecise timing, and is not a random-policy baseline. Five actual screenshots were used to construct five templates; self-matching those same images is a construction check, not held-out accuracy. The corrected non-episodic source is b978c7e7-a4ff-40ae-a1bd-2577ea935010. Prior record b5a1748c-0fb1-4787-9d05-a077fd2c5443 had an episode-metadata defect and is excluded from analytical evidence. Both refer to the same five images, not ten. See docs/operations/game-discovery.md. Capture and probe measurements above update with canonical records; neither alone establishes calibrated dynamics, unattended real-game competence, real adaptation or a sim-to-real performance ratio. Input-tool interruptions are operational states, not permanent scientific conclusions.",
    )
    paragraph(
        "Negative results and limitations",
        "The CUDA256 pilot executed fewer transitions than CPU256 for this small policy and CPU simulator. CEM seed0 looked competitive, but its replicated results were more variable; fitting two fixed training seeds is a material limitation. Architecture, observation history and optimizer all differ between PPO and CEM, so this comparison does not isolate a single causal component. PPO epoch work can vary due to the approximate-KL stop. Uncalibrated state observations, simple terrain families and finite episode horizons limit transfer claims. Native-game safety and perception must be validated before interpreting simulator scores as real competence.",
    )
    paragraph(
        "Implemented student comparator, never trained",
        "The teacher-to-screen-body student implementation and pure semantic tests are present in src/gradientclimb/algorithms/screen_distillation.py and docs/research/screen-body-distillation.md. Its eight body features, validity masks and temporal history exclude privileged velocity, fuel and contact inputs. The analytic projection is uncalibrated. The queued 60-second seed500 pilot was cancelled before dispatch at the user-directed pause. There is no learned student artifact, teacher-agreement measurement, student simulator score or native student evaluation. Implementation does not establish transfer.",
    )
    audit = summary.get("matching_full_integrity_audit")
    if audit:
        paragraph(
            "Finalized integrity audit",
            f"All {audit['run_count']} canonical runs passed full artifact verification at {audit['checked_at_utc']}, covering {audit['total_bytes']:,} bytes with no unfinished run. This report rechecked that the audit's run set and seal-file hashes match its sources. Source: research/experiments/cycle-1-integrity.json. Hash integrity does not establish measurement validity, scientific success or off-machine backup.",
        )
    paragraph(
        "Reproducibility and source integrity",
        f"Generated from canonical run records and journals. Evidence cutoff: {summary['evidence_as_of']}. Verification: {summary['verification_mode']}. Every source snapshot, configuration/source identifier and checkpoint hash is retained in results-summary.json. Refresh the fixed primary with python scripts/analyze_research.py --root artifacts --benchmark-run {summary['benchmark_run']}; --verify rechecks all sealed artifacts. Literature: research/literature/README.md. Governing protocol: docs/methodology/qualification.md. The cycle pause and resumption plan govern any future research dispatch.",
    )
    table(
        "Source run index",
        ["Run", "Experiment", "Status", "Git SHA", "Dirty", "Checkpoint SHA-256"],
        [
            [
                r["run_id"],
                r["experiment"],
                r["status"],
                r["git_sha"],
                str(r["dirty_worktree"]),
                r["checkpoint_sha256"] or "—",
            ]
            for r in summary["source_records"]
        ],
    )
    markdown = [
        "# GradientClimb research report",
        "",
        "**Status: Cycle 1 paused; real-game qualification incomplete.**"
        if cycle
        else "**Status: real-game qualification incomplete.**",
        "",
    ]
    body = [
        '<header><p class="eyebrow">GRADIENTCLIMB · RESEARCH RECORD</p><h1>Learning is measurable.<br>Game transfer remains unqualified.</h1><p class="status">Cycle 1 paused · real-game qualification incomplete</p></header>'
    ]
    for title, note, content in sections:
        markdown.extend([f"## {title}", "", note, ""])
        body.append(f"<section><h2>{html.escape(title)}</h2>")
        if note:
            body.append(f"<p>{html.escape(note)}</p>")
        if isinstance(content, tuple):
            headers, rows = content
            if not rows:
                markdown.extend(["Not yet measured.", ""])
                body.append('<p class="missing">Not yet measured.</p>')
            else:
                escape_md = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
                markdown.extend(
                    [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join("---" for _ in headers) + " |",
                    ]
                )
                markdown.extend("| " + " | ".join(escape_md(v) for v in row) + " |" for row in rows)
                markdown.append("")
                body.append(
                    '<div class="table-wrap"><table><thead><tr>'
                    + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
                    + "</tr></thead><tbody>"
                )
                for row in rows:
                    body.append(
                        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                    )
                body.append("</tbody></table></div>")
        elif isinstance(content, str):
            markdown.extend([f"![{title}](figures/{content})", ""])
            body.append(figures[content])
        body.append("</section>")
    embedded = json.dumps(summary, allow_nan=False).replace("<", "\\u003c")
    css = "body{margin:0;background:#f0f0e9;color:#192b35;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:54px 28px 80px}header{border-bottom:3px solid #0b8587;padding-bottom:30px}h1{font-size:clamp(34px,5vw,62px);line-height:1.08;letter-spacing:-.04em}.eyebrow{letter-spacing:.18em;color:#087f8c;font-size:12px;font-weight:750}.status,.missing{color:#865215}.status{display:inline-block;background:#f7e7c4;padding:7px 13px;border-radius:6px}section{margin-top:40px}h2{font-size:23px;line-height:1.25}p{max-width:95ch}.table-wrap{overflow:auto;background:#fffdf8;border:1px solid #d8ded7;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px;text-align:left;border-bottom:1px solid #e0e4df;vertical-align:top}th{background:#e7ede6}td{overflow-wrap:anywhere;min-width:70px}svg{width:100%;height:auto;border:1px solid #d8ded7;border-radius:8px;font-family:system-ui,sans-serif}details{margin-top:40px}pre{overflow:auto;background:white;padding:16px;font-size:11px}@media print{body{background:white}main{padding:0}section{break-inside:avoid}h1{font-size:34px}details{display:none}}"
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GradientClimb research report</title><style>'
        + css
        + "</style></head><body><main>"
        + "".join(body)
        + "<details><summary>Inspect the embedded evidence snapshot</summary><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + '</pre></details></main><script type="application/json" id="research-evidence">'
        + embedded
        + "</script></body></html>"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    for name, svg in figures.items():
        (output_dir / "figures" / name).write_text(svg, encoding="utf-8")
    (output_dir / "gradientclimb-research-report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    (output_dir / "gradientclimb-research-report.html").write_text(document, encoding="utf-8")


def self_test():
    source = {
        "run_id": "parent",
        "artifact_manifest": [
            {
                "kind": "checkpoint",
                "sha256": "test-checkpoint",
                "metadata": {"target_seconds": 300, "training_elapsed_seconds": 302.7},
            }
        ],
    }
    child = {"run_id": "child", "parent_run": "parent"}
    final = {"run_id": "child", "metadata": {}, "results": {"episodes": [{"distance": 900}]}}
    assert all(r["distance"] is None for r in checkpoint_rows([source, child], [final], "parent"))
    measured = {
        **final,
        "checkpoint_hash": "test-checkpoint",
        "metadata": {"requested_minutes": 5, "training_elapsed_seconds": 302.7},
    }
    rows = checkpoint_rows([source, child], [measured], "parent")
    assert rows[0]["distance"]["mean"] == 900 and rows[0]["actual_training_seconds"] == 302.7
    assert all(r["distance"] is None for r in rows[1:])
    unknown = {**measured, "checkpoint_hash": "unregistered-checkpoint"}
    assert all(r["distance"] is None for r in checkpoint_rows([source, child], [unknown], "parent"))
    contradictory = {
        **measured,
        "metadata": {"requested_minutes": 5, "training_elapsed_seconds": 300},
    }
    try:
        checkpoint_rows([source, child], [contradictory], "parent")
    except ValueError:
        pass
    else:
        raise AssertionError("Checkpoint timestamps must match saved artifact provenance")
    other_parent = {"run_id": "child", "parent_run": "different"}
    assert all(
        r["distance"] is None for r in checkpoint_rows([source, other_parent], [measured], "parent")
    )
    try:
        distance_result(
            {"results": {"episodes": [{"distance": 10}], "summary": {"distance": {"mean": 11}}}}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A contradictory published mean must be rejected")
    capture = {
        "run_id": "capture-test",
        "algorithm": "capture",
        "status": "completed",
        "summary": {
            "capture": {
                "captured_frames": 60,
                "elapsed_seconds": 5,
                "end_to_end_fps": 12,
                "source_dropped_frames": None,
            }
        },
    }
    captured = capture_records([capture], None)[0]
    assert captured["end_to_end_capture_loop_fps"] == 12
    assert captured["source_new_frame_rate"] is None and captured["source_dropped_frames"] is None
    capture["summary"]["capture"]["end_to_end_fps"] = 13
    try:
        capture_records([capture], None)
    except ValueError:
        pass
    else:
        raise AssertionError("A contradictory capture FPS must be rejected")
    print(
        "Report invariants passed: missing checkpoints, exact timestamps, parent isolation, mean/FPS consistency, unknown source frame rate."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts")
    parser.add_argument("--benchmark-run")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "research/reports")
    parser.add_argument(
        "--summary", type=Path, default=PROJECT / "research/experiments/results-summary.json"
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    summary = collect(args.root.resolve(), args.benchmark_run, args.verify)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(summary, args.output_dir, args.root)
    print(
        json.dumps(
            {
                "summary": str(args.summary),
                "report": str(args.output_dir / "gradientclimb-research-report.html"),
                "benchmark_run": summary["benchmark_run"],
                "benchmark_status": summary["benchmark_status"],
                "measured_scheduled_checkpoints": sum(
                    r["distance"] is not None for r in summary["checkpoints"]
                ),
                "source_run_count": len(summary["source_records"]),
                "qualifies_real_game": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
