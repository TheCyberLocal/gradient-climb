"""Acquire one preregistered public owner-video excerpt for local construction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

from gradientclimb.artifacts import sha256_file
from gradientclimb.artifacts.archive import _inside, _walk
from gradientclimb.experiments.recorder import RunRecorder, verify_run


def bounded_process(command, directory, deadline, maximum_bytes, log_name, environment=None):
    """Retain output and sampled child CPU lower bounds, including failed commands."""
    cpu = {}
    owned_processes = {}
    sample_errors = 0
    path = directory / log_name
    receipt_path = directory / f"{log_name}.receipt.json"
    started = time.perf_counter()
    process = None
    failure = None
    returncode = None

    def sample(processes):
        nonlocal sample_errors
        for child in processes:
            try:
                identity = child.pid, child.create_time()
                owned_processes[identity] = child
                times = child.cpu_times()
                cpu[identity] = max(cpu.get(identity, 0), times.user + times.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                sample_errors += 1

    try:
        with path.open("xb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            while True:
                try:
                    parent = psutil.Process(process.pid)
                    sample([parent])
                    sample(parent.children(recursive=True))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    sample_errors += 1
                if time.perf_counter() > deadline:
                    raise TimeoutError("Registered acquisition wall budget exhausted")
                if sum(p.stat().st_size for p in _walk(directory)) > maximum_bytes:
                    raise RuntimeError("Registered acquisition output budget exhausted")
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(0.2)
            if returncode:
                raise RuntimeError(
                    f"Acquisition command failed with exit code {returncode}; see retained log"
                )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_errors = []
        terminated = 0
        live_descendants_after_launcher_exit = 0
        if process is not None:
            # Keep the psutil Process objects (which include creation identity),
            # even when the launcher has exited and children were reparented.
            try:
                parent = psutil.Process(process.pid)
                sample([parent])
                sample(parent.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                sample_errors += 1
            launcher_exited = process.poll() is not None
            sample(list(owned_processes.values()))
            waiting = []
            for child in reversed(list(owned_processes.values())):
                try:
                    if child.is_running():
                        if launcher_exited and child.pid != process.pid:
                            live_descendants_after_launcher_exit += 1
                        child.kill()
                        terminated += 1
                        waiting.append(child)
                except psutil.NoSuchProcess:
                    pass
                except BaseException as error:  # noqa: BLE001 - continue cleanup of other children
                    cleanup_errors.append(error)
            try:
                _, alive = psutil.wait_procs(waiting, timeout=10)
                if alive:
                    cleanup_errors.append(
                        RuntimeError("Owned child processes remain alive after cleanup")
                    )
            except BaseException as error:  # noqa: BLE001 - receipt must survive cleanup errors
                cleanup_errors.append(error)
            try:
                if process.poll() is None:
                    process.kill()  # Popen owns the launch handle even if psutil access failed.
                final_returncode = process.wait(timeout=10)
                if returncode is None:
                    returncode = final_returncode
            except BaseException as error:  # noqa: BLE001 - preserve primary guard failure
                cleanup_errors.append(error)
            if live_descendants_after_launcher_exit and failure is None:
                cleanup_errors.append(RuntimeError("Launcher exited with live owned descendants"))
        try:
            receipt_path.write_text(
                json.dumps(
                    {
                        "elapsed_seconds": time.perf_counter() - started,
                        "exit_code": returncode,
                        "error": f"{type(failure).__name__}: {failure}" if failure else None,
                        "cleanup_errors": [f"{type(e).__name__}: {e}" for e in cleanup_errors],
                        "terminated_owned_processes": terminated,
                        "live_descendants_after_launcher_exit": live_descendants_after_launcher_exit,
                        "sampled_process_tree_cpu_core_seconds_lower_bound": sum(cpu.values())
                        if cpu
                        else None,
                        "cpu_measurement_status": "sampled_lower_bound"
                        if cpu
                        else "no_process_counter_observed",
                        "sampled_process_identities": len(cpu),
                        "process_sample_errors": sample_errors,
                        "cpu_scope": "Maximum observed lifetime CPU for each launched process identity, sampled about every 0.2s; short-lived children and work after last sample may be missing. Separate from recorder-process CPU, not energy.",
                        "cleanup_scope": "Observed process identities retained across launcher exit. Descendants that start and detach entirely between samples are not discovered by this polling implementation.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        except BaseException as error:  # noqa: BLE001 - preserve primary error if receipt storage fails
            cleanup_errors.append(error)
        if cleanup_errors:
            details = "; ".join(f"{type(e).__name__}: {e}" for e in cleanup_errors)
            if failure is not None:
                failure.add_note(f"Acquisition cleanup/receipt also failed: {details}")
            else:
                raise RuntimeError(
                    f"Acquisition cleanup/receipt failed: {details}"
                ) from cleanup_errors[0]


def acquire(root, protocol_path):
    root = root.absolute()
    protocol_file = _inside(root, protocol_path)
    plan = json.loads(protocol_file.read_bytes())
    if plan["schema_version"] != "owner-video-acquisition-3.0" or plan["status"] != "registered":
        raise ValueError("A registered acquisition protocol is required")
    # Deliberately bounded to the owner's chosen public archive and one video.
    if plan["video_url"] != "https://www.youtube.com/watch?v=wx_cI59vFX0":
        raise ValueError("This construction pilot supports only its frozen owner source")
    if not 0 < plan["end_seconds"] <= 300 or plan["start_seconds"] != 0:
        raise ValueError("This pilot is limited to the first five minutes")
    if (
        not 0 < plan["wall_budget_seconds"] <= 300
        or not 0 < plan["output_budget_bytes"] <= 256 * 1024**2
    ):
        raise ValueError("Acquisition exceeds supported bounds")
    protocol_sha = sha256_file(protocol_file)
    artifact_root = root / "artifacts"
    for start in artifact_root.glob("runs/*/run-start.json"):
        if (
            json.loads(start.read_bytes()).get("configuration", {}).get("protocol_sha256")
            == protocol_sha
        ):
            raise ValueError(
                "This protocol already has an acquisition attempt; register a successor"
            )
    config = {**plan, "protocol_path": protocol_path, "protocol_sha256": protocol_sha}
    with RunRecorder(
        artifact_root,
        plan["experiment_id"],
        config,
        algorithm="external-video-acquisition",
        environment="owner-public-video",
        evidence_domain="construction_external_video",
        source_root=root,
        qualifies_real_game=False,
    ) as run:
        deadline = time.perf_counter() + plan["wall_budget_seconds"]
        run.register_artifact(protocol_file, "owner_video_acquisition_protocol")
        run.annotate(
            environment_steps=0,
            episodes=0,
            optimizer_updates=0,
            policy_decisions=0,
            new_real_interaction_seconds=0,
            synchronized_action_labels=0,
            training_eligible=False,
            whole_source_purpose="construction",
            inherited_human_play_seconds=None,
            inherited_editing_seconds=None,
            inherited_human_skill_training_seconds=None,
            prior_scope="Original play, editing and skill acquisition are inherited, unknown and nonzero where applicable; excerpt duration is not new real interaction or independent episodes.",
        )
        payload = run.directory / "acquisition"
        payload.mkdir()
        # Leave space for canonical retained copies, metadata and final sealing.
        payload_budget = plan["output_budget_bytes"] // 3
        failure = None
        try:
            wheel = payload / plan["tool"]["filename"]
            with (
                urllib.request.urlopen(plan["tool"]["url"], timeout=20) as response,
                wheel.open("xb") as out,
            ):
                count = 0
                while block := response.read(1024**2):
                    count += len(block)
                    if count > plan["tool"]["bytes"] or time.perf_counter() > deadline:
                        raise RuntimeError(
                            "Tool package exceeds frozen size or acquisition deadline"
                        )
                    out.write(block)
            if count != plan["tool"]["bytes"] or sha256_file(wheel) != plan["tool"]["sha256"]:
                raise ValueError("Official tool wheel does not match frozen digest/size")
            tool_dir = payload / "tool"
            bounded_process(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-compile",
                    "--disable-pip-version-check",
                    "--target",
                    str(tool_dir),
                    str(wheel),
                ],
                payload,
                deadline,
                payload_budget,
                "install.log",
            )
            ffmpeg = Path(plan["ffmpeg"]["path"])
            if sha256_file(ffmpeg) != plan["ffmpeg"]["sha256"]:
                raise ValueError("Installed ffmpeg changed after registration")
            env = dict(os.environ, PYTHONPATH=str(tool_dir))
            command = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--ignore-config",
                "--no-plugin-dirs",
                "--no-cache-dir",
                "--no-playlist",
                "--retries",
                "0",
                "--fragment-retries",
                "0",
                "--socket-timeout",
                "20",
                "--no-progress",
                "--no-overwrites",
                "--ffmpeg-location",
                str(ffmpeg),
                "--download-sections",
                f"*0-{plan['end_seconds']}",
                "--format",
                "bestvideo[height<=720]/best[height<=720]",
                "--output",
                str(payload / "owner-excerpt.%(ext)s"),
                plan["video_url"],
            ]
            bounded_process(command, payload, deadline, payload_budget, "download.log", env)
            videos = [
                p for p in payload.glob("owner-excerpt.*") if p.suffix in {".mp4", ".webm", ".mkv"}
            ]
            if len(videos) != 1:
                raise ValueError("Expected exactly one retained video excerpt")
            run.annotate(
                video_sha256=sha256_file(videos[0]),
                requested_media_seconds=plan["end_seconds"],
                actual_media_seconds=None,
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            # Retain all partial acquisition files and tool bytes under the final seal.
            retention_errors = []
            try:
                for path in _walk(payload):
                    if "tool" not in path.relative_to(payload).parts:
                        try:
                            run.register_artifact(path, "owner_video_acquisition_evidence")
                        except BaseException as error:  # noqa: BLE001 - retain remaining evidence
                            retention_errors.append(error)
            except BaseException as error:  # noqa: BLE001 - preserve acquisition failure
                retention_errors.append(error)
            if retention_errors:
                details = "; ".join(f"{type(e).__name__}: {e}" for e in retention_errors)
                if failure is not None:
                    failure.add_note(f"Acquisition artifact retention also failed: {details}")
                else:
                    raise RuntimeError(
                        f"Acquisition artifact retention failed: {details}"
                    ) from retention_errors[0]
        run.finalize(acquisition_completed=True, training_steps=0, episodes=0, optimizer_updates=0)
    return {"run_id": run.run_id, "verification": verify_run(artifact_root, run.run_id)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.project_root, args.protocol), indent=2))
