"""Capture observable hardware facts; unavailable measurements remain null."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gradientclimb.artifacts import canonical_json

try:
    import psutil
except ImportError:  # Measurements remain explicitly unavailable on minimal installations.
    psutil = None


def _command(
    arguments: list[str], cwd: Path | None = None, *, strip_output: bool = True
) -> str | None:
    try:
        result = subprocess.run(arguments, cwd=cwd, capture_output=True, timeout=4, check=False)
        if result.returncode != 0:
            return None
        # Decode synchronously: Windows' subprocess text-reader thread otherwise
        # loses a UTF-8 Git diff when the host's default code page is CP1252.
        output = result.stdout.decode("utf-8", errors="strict")
        return output.strip() if strip_output else output
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return None


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvidia-smi.exe"
    return str(candidate) if candidate.is_file() else None


def _finite_measurement(value: Any, *, maximum: float | None = None) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return (
        number
        if math.isfinite(number) and number >= 0 and (maximum is None or number <= maximum)
        else None
    )


def gpu_sample(*, include_identity: bool = False) -> dict[str, Any]:
    """GPU utilization and VRAM from ``nvidia-smi`` (one subprocess); empty when absent."""
    result: dict[str, Any] = {}
    executable = _nvidia_smi()
    started = time.perf_counter()
    devices = []
    error = None
    if executable:
        output = _command(
            [
                executable,
                "--query-gpu=uuid,index,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            for parts in csv.reader(output.splitlines(), skipinitialspace=True):
                if len(parts) != 5:
                    error = "malformed nvidia-smi row"
                    continue
                uuid_value, index, name, utilization, memory = [part.strip() for part in parts]
                percent = _finite_measurement(utilization, maximum=100)
                used = _finite_measurement(memory)
                devices.append(
                    {
                        "uuid": uuid_value if uuid_value not in {"", "N/A", "[N/A]"} else None,
                        "index": int(index) if index.isdecimal() else None,
                        "name": name or None,
                        "gpu_percent": percent,
                        "vram_used_bytes": int(used * 1024**2) if used is not None else None,
                        "scope": "device_wide_all_processes; not_policy_attributed",
                    }
                )
            if devices:
                result.update({key: devices[0][key] for key in ("gpu_percent", "vram_used_bytes")})
        else:
            error = "nvidia-smi query unavailable or failed"
    else:
        error = "nvidia-smi unavailable"
    if include_identity:
        completed = time.perf_counter()
        result["gpu_measurements"] = {
            "requested": True,
            "devices": devices,
            "error": error,
            "query_started_monotonic_seconds": started,
            "sampled_monotonic_seconds": completed,
            "query_seconds": completed - started,
            "utilization_semantics": "vendor_sampled_busy_percentage; sampling_window_not_measured",
        }
    return result


def resource_sample(include_gpu: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    measurement: dict[str, Any] = {
        "version": "resources-sample-3.0",
        "sample_started_monotonic_seconds": time.perf_counter(),
        "process": {
            "pid": os.getpid(),
            "create_time_unix_seconds": None,
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
            "scope": "current_process_all_threads; excludes_child_processes",
        },
        "gpu": {"requested": include_gpu, "devices": [], "error": None},
        "errors": {},
    }
    if psutil is not None:
        try:
            result["cpu_percent"] = _finite_measurement(
                psutil.cpu_percent(interval=None), maximum=100
            )
            result["per_core_cpu_percent"] = [
                value
                for item in psutil.cpu_percent(interval=None, percpu=True)
                if (value := _finite_measurement(item, maximum=100)) is not None
            ]
        except Exception as error:  # noqa: BLE001 - optional provider faults are missing evidence
            measurement["errors"]["host_cpu"] = f"{type(error).__name__}: {error}"
        try:
            used = _finite_measurement(psutil.virtual_memory().used)
            result["ram_used_bytes"] = int(used) if used is not None else None
            measurement["host_sampled_monotonic_seconds"] = time.perf_counter()
        except Exception as error:  # noqa: BLE001 - retain independent surviving resource fields
            measurement["errors"]["host_ram"] = f"{type(error).__name__}: {error}"
        try:
            process = psutil.Process()
            measurement["process"]["create_time_unix_seconds"] = _finite_measurement(
                process.create_time()
            )
            cpu = process.cpu_times()
            measurement["process"].update(
                cpu_user_seconds=_finite_measurement(cpu.user),
                cpu_system_seconds=_finite_measurement(cpu.system),
                sampled_monotonic_seconds=time.perf_counter(),
            )
        except Exception as error:  # noqa: BLE001 - OS counters can be unavailable independently
            measurement["errors"]["process_cpu"] = f"{type(error).__name__}: {error}"
        try:
            rss = _finite_measurement(psutil.Process().memory_info().rss)
            result["process_rss_bytes"] = int(rss) if rss is not None else None
            measurement["process_memory_sampled_monotonic_seconds"] = time.perf_counter()
        except Exception as error:  # noqa: BLE001 - optional RSS query failure is recorded
            measurement["errors"]["process_rss"] = f"{type(error).__name__}: {error}"
    else:
        measurement["errors"]["psutil"] = "psutil unavailable"
    if include_gpu:
        try:
            gpu = gpu_sample(include_identity=True)
            measurement["gpu"] = gpu.pop("gpu_measurements")
            result.update(gpu)
        except Exception as error:  # noqa: BLE001 - optional GPU query failure is recorded
            measurement["gpu"]["error"] = f"{type(error).__name__}: {error}"
    measurement["sample_completed_monotonic_seconds"] = time.perf_counter()
    result["measurements"] = {"resources_sample": measurement}
    return result


def capture_provenance(
    source_root: Path | None = None, *, source_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    source_root = source_root or Path.cwd()
    source_root = source_root.resolve()
    source_paths = (
        "src",
        "tests",
        "scripts",
        "configs",
        "schemas",
        "pyproject.toml",
        "requirements-lock.txt",
    )
    git_sha = _command(["git", "rev-parse", "HEAD"], source_root)
    status = _command(["git", "status", "--porcelain"], source_root)
    diff = _command(
        ["git", "diff", "HEAD", "--binary", "--", *source_paths], source_root, strip_output=False
    )
    # Include untracked source files: a dirty SHA by itself cannot reproduce new code.
    untracked_output = _command(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "tests",
            "scripts",
            "configs",
        ],
        source_root,
    )
    untracked_hashes = {}
    for name in (untracked_output or "").split("\0"):
        candidate = source_root / name
        if (
            name
            and candidate.is_file()
            and not candidate.is_symlink()
            and candidate.resolve().is_relative_to(source_root)
        ):
            from gradientclimb.artifacts import sha256_file

            untracked_hashes[name] = sha256_file(candidate)
    source_hash = (
        hashlib.sha256(
            canonical_json({"diff": diff, "untracked": untracked_hashes}).encode()
        ).hexdigest()
        if git_sha and diff is not None and untracked_output is not None
        else None
    )
    if source_state is not None:
        source_state.update(
            {
                "schema_version": "1.0.0",
                "git_sha": git_sha,
                "tracked_source_paths": list(source_paths),
                "tracked_diff": diff,
                "untracked_source_sha256": untracked_hashes,
                "source_diff_sha256": source_hash,
                "untracked_contents_included": False,
                "tracked_diff_available": diff is not None,
                "untracked_manifest_available": untracked_output is not None,
                "collection_is_atomic": False,
                "reproduction_note": "Apply tracked diff to git SHA; untracked source requires matching external files or a later verified commit. Ignored files and artifact/personal directories are excluded.",
            }
        )
    hardware: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
        "ram_bytes": psutil.virtual_memory().total if psutil else None,
    }
    gpus = []
    executable = _nvidia_smi()
    if executable:
        output = _command(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in (output or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                try:
                    gpus.append(
                        {
                            "name": parts[0],
                            "driver": parts[1],
                            "vram_bytes": int(parts[2]) * 1024**2,
                        }
                    )
                except ValueError:
                    continue
    hardware["gpus"] = gpus
    # Imported framework state describes this process, not a different machine.
    # Retain it below for diagnosis without allowing lazy imports to change identity.
    hardware["machine_fingerprint_basis"] = "hardware-observation-v2"
    machine_fingerprint = hashlib.sha256(canonical_json(hardware).encode()).hexdigest()
    torch = sys.modules.get("torch")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if torch is not None and hasattr(torch, "cuda"):
        hardware["pytorch_cuda_available"] = torch.cuda.is_available()
    versions = {}
    for package in (
        "gradientclimb",
        "numpy",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "duckdb",
        "pyarrow",
        "pydantic",
        "psutil",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "git_sha": git_sha,
        "dirty_worktree": bool(status) if status is not None else None,
        "source_diff_sha256": source_hash,
        "project_version": versions.get("gradientclimb", "0.1.0+uninstalled"),
        "machine_fingerprint": machine_fingerprint,
        "cpu": hardware["cpu"],
        "gpu": gpus,
        "ram": hardware["ram_bytes"],
        "driver": gpus[0]["driver"] if gpus else None,
        "cuda": cuda,
        "hardware": hardware,
        "python_version": platform.python_version(),
        "framework_versions": versions,
    }
