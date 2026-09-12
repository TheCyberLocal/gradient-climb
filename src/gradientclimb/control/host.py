"""Fail-closed host budget and exclusive native input ownership.

Disk margins are conservative workstation policy, not emulator vendor thresholds.
An OS-held lock is released on process exit; no stale PID file authorizes input.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class HostLimits:
    system_free_gib: float = 12.0
    artifact_free_gib: float = 20.0
    max_drive_growth_gib: float = 8.0
    check_seconds: float = 1.0

    def __post_init__(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Host limits must be finite and positive")


class HostBudgetGuard:
    def __init__(
        self,
        artifact_root,
        *,
        limits=None,
        disk_usage=shutil.disk_usage,
        clock=time.perf_counter,
        target=None,
        process_factory=psutil.Process,
    ):
        self.limits = limits or HostLimits()
        self.roots = {
            "system": Path(os.environ.get("SystemDrive", "C:") + "\\")
            if os.name == "nt"
            else Path("/"),
            "artifacts": Path(artifact_root).resolve(),
        }
        self.disk_usage, self.clock = disk_usage, clock
        self.target, self.process_factory = target, process_factory
        self.initial = None
        self.last_checked = -math.inf
        self.fault = None
        self.trace = []

    def check(self, *, force=False):
        if self.fault:
            raise RuntimeError(self.fault)
        now = self.clock()
        if not force and now - self.last_checked < self.limits.check_seconds:
            return
        self.last_checked = now
        row = {"monotonic_seconds": now, "free_bytes": {}, "process_ok": None}
        try:
            for key, path in self.roots.items():
                row["free_bytes"][key] = self.disk_usage(path).free
            if self.initial is None:
                self.initial = dict(row["free_bytes"])
            for key, minimum in (
                ("system", self.limits.system_free_gib),
                ("artifacts", self.limits.artifact_free_gib),
            ):
                free = row["free_bytes"][key]
                if free < minimum * 1024**3:
                    raise RuntimeError(
                        f"Host disk margin: {key} has less than {minimum:g} GiB free"
                    )
                if self.initial[key] - free > self.limits.max_drive_growth_gib * 1024**3:
                    raise RuntimeError(f"Host disk growth exceeded budget on {key}")
            if self.target is not None:
                process = self.process_factory(self.target.pid)
                row["process_ok"] = process.is_running() and (
                    process.create_time() == self.target.process_created_at
                )
                if not row["process_ok"]:
                    raise RuntimeError("Native process identity/health changed")
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            row["fault"] = self.fault
            raise RuntimeError(self.fault) from exc
        finally:
            self.trace.append(row)


class NativeInputLease:
    """One native collection process per Windows logon session; no gameplay input.

    Mutex existence, rather than recursive mutex ownership, excludes a second
    collector in the same process too. ERROR_ALREADY_EXISTS fails closed even
    after a previous owner died while another handle remains open.
    """

    def __init__(self):
        self.handle = None

    def __enter__(self):
        if os.name != "nt":
            raise OSError("Native input ownership requires Windows")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_bool
        self.kernel = kernel
        ctypes.set_last_error(0)
        handle = kernel.CreateMutexW(None, False, "Local\\GradientClimb.NativeInput.v1")
        error = ctypes.get_last_error()
        if not handle:
            raise OSError(error, "Cannot acquire native input ownership")
        if error == 183:
            kernel.CloseHandle(handle)
            raise RuntimeError("Another GradientClimb native collector owns input")
        self.handle = handle
        return self

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __exit__(self, *_):
        self.close()
