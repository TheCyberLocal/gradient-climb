"""Capture observable hardware facts; unavailable measurements remain null."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from gradientclimb.artifacts import canonical_json

try:
    import psutil
except ImportError:  # Measurements remain explicitly unavailable on minimal installations.
    psutil = None


def _command(arguments: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            arguments, cwd=cwd, capture_output=True, text=True, timeout=4, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvidia-smi.exe"
    return str(candidate) if candidate.is_file() else None


def resource_sample(include_gpu: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if psutil is not None:
        result.update(
            cpu_percent=psutil.cpu_percent(interval=None),
            per_core_cpu_percent=psutil.cpu_percent(interval=None, percpu=True),
            ram_used_bytes=psutil.virtual_memory().used,
            process_rss_bytes=psutil.Process().memory_info().rss,
        )
    executable = _nvidia_smi() if include_gpu else None
    if executable:
        output = _command(
            [executable, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        )
        if output:
            try:
                first = output.splitlines()[0].split(",")
                result["gpu_percent"] = float(first[0])
                result["vram_used_bytes"] = int(float(first[1]) * 1024**2)
            except (ValueError, IndexError):
                pass
    return result


def capture_provenance(source_root: Path | None = None) -> dict[str, Any]:
    source_root = source_root or Path.cwd()
    git_sha = _command(["git", "rev-parse", "HEAD"], source_root)
    status = _command(["git", "status", "--porcelain"], source_root)
    diff = _command(["git", "diff", "HEAD", "--binary"], source_root)
    # Include untracked source files: a dirty SHA by itself cannot reproduce new code.
    untracked_output = _command(["git", "ls-files", "--others", "--exclude-standard"], source_root)
    untracked_hashes = {}
    for name in (untracked_output or "").splitlines():
        candidate = source_root / name
        if candidate.is_file():
            from gradientclimb.artifacts import sha256_file

            untracked_hashes[name] = sha256_file(candidate)
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
        "source_diff_sha256": hashlib.sha256(
            canonical_json({"diff": diff, "untracked": untracked_hashes}).encode()
        ).hexdigest()
        if git_sha
        else None,
        "project_version": versions.get("gradientclimb", "0.1.0+uninstalled"),
        "machine_fingerprint": hashlib.sha256(canonical_json(hardware).encode()).hexdigest(),
        "cpu": hardware["cpu"],
        "gpu": gpus,
        "ram": hardware["ram_bytes"],
        "driver": gpus[0]["driver"] if gpus else None,
        "cuda": cuda,
        "hardware": hardware,
        "python_version": platform.python_version(),
        "framework_versions": versions,
    }
