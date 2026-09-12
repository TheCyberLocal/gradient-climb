"""Prospective resources-3.0 accounting from measured cumulative counters and gauges.

Process CPU is user+system time (https://psutil.readthedocs.io/stable/).
NVIDIA utilization is vendor-sampled device activity, not per-policy work or
energy (https://docs.nvidia.com/deploy/nvidia-smi/). Instantaneous gauges are
integrated only between valid adjacent readings within a declared maximum gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _number(value: Any, *, maximum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return (
        float(value)
        if math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum)
        else None
    )


@dataclass
class _Gauge:
    max_gap: float
    previous: tuple[float, float, Any] | None = None
    valid_samples: int = 0
    missing_samples: int = 0
    covered_seconds: float = 0.0
    integral: float = 0.0
    peak: float | None = None
    long_gaps: int = 0
    nonmonotonic_intervals: int = 0
    identity_changes: int = 0

    def add(self, value: Any, elapsed: float, identity: Any = "host", *, maximum=None) -> None:
        value = _number(value, maximum=maximum)
        if value is None:
            self.missing_samples += 1
            self.previous = None
            return
        self.valid_samples += 1
        self.peak = value if self.peak is None else max(self.peak, value)
        if self.previous is not None:
            previous_time, previous_value, previous_identity = self.previous
            delta = elapsed - previous_time
            if previous_identity != identity:
                self.identity_changes += 1
            elif delta <= 0:
                self.nonmonotonic_intervals += 1
            elif delta > self.max_gap:
                self.long_gaps += 1
            else:
                self.integral += delta * (previous_value + value) / 2
                self.covered_seconds += delta
        self.previous = elapsed, value, identity

    def report(self, span: float) -> dict:
        return {
            "sampled_peak": self.peak,
            "time_weighted_sampled_mean": self.integral / self.covered_seconds
            if self.covered_seconds
            else None,
            "covered_seconds": self.covered_seconds,
            "coverage_fraction": min(1.0, self.covered_seconds / span) if span > 0 else None,
            "uncovered_seconds": max(0.0, span - self.covered_seconds),
            "valid_samples": self.valid_samples,
            "missing_samples": self.missing_samples,
            "long_gap_intervals_excluded": self.long_gaps,
            "nonmonotonic_intervals_excluded": self.nonmonotonic_intervals,
            "identity_change_intervals_excluded": self.identity_changes,
            "interpolation": "trapezoid_between_valid_adjacent_samples_only",
        }


class ResourceAccumulator:
    """Bounded-memory integrals; old telemetry is never retrospectively relabeled."""

    def __init__(self, *, monotonic_origin: float, max_gap_seconds: float = 30.0):
        if _number(max_gap_seconds) is None or max_gap_seconds == 0:
            raise ValueError("A positive finite resource interpolation gap is required")
        self.origin, self.max_gap = monotonic_origin, max_gap_seconds
        self.first: float | None = None
        self.last: float | None = None
        self.last_row: float | None = None
        self.samples = self.nonmonotonic_samples = 0
        self.cpu_previous = None
        self.cpu_delta = self.cpu_covered = 0.0
        self.cpu_valid = self.cpu_missing = self.cpu_resets = self.cpu_identity_changes = 0
        self.cpu_long_gaps = self.cpu_intervals = 0
        self.process_identities: set[tuple[int, float]] = set()
        self.rss, self.ram = _Gauge(self.max_gap), _Gauge(self.max_gap)
        self.gpus: dict[str, dict] = {}
        self.gpu_requested = self.gpu_failed = 0
        self.error_samples = 0

    def _elapsed(self, value, fallback):
        measured = _number(value)
        return max(0.0, measured - self.origin) if measured is not None else fallback

    def add(self, row: dict) -> None:
        sample = row.get("measurements", {}).get("resources_sample", {})
        if sample.get("version") != "resources-sample-3.0":
            return
        elapsed = _number(row.get("elapsed_seconds"))
        if elapsed is None:
            self.nonmonotonic_samples += 1
            return
        if self.last_row is not None and elapsed < self.last_row:
            self.nonmonotonic_samples += 1
            return
        self.last_row = elapsed
        started = self._elapsed(sample.get("sample_started_monotonic_seconds"), elapsed)
        completed = self._elapsed(sample.get("sample_completed_monotonic_seconds"), elapsed)
        self.first = started if self.first is None else min(self.first, started)
        self.last = max(completed, self.last or 0.0)
        self.samples += 1
        self.error_samples += bool(sample.get("errors"))
        process = sample.get("process", {})
        pid, created = process.get("pid"), _number(process.get("create_time_unix_seconds"))
        identity = (pid, created) if type(pid) is int and pid > 0 and created is not None else None
        user, system = (
            _number(process.get("cpu_user_seconds")),
            _number(process.get("cpu_system_seconds")),
        )
        cpu_time = self._elapsed(process.get("sampled_monotonic_seconds"), elapsed)
        if identity is not None and user is not None and system is not None:
            self.process_identities.add(identity)
            self.cpu_valid += 1
            if self.cpu_previous is not None:
                previous_time, previous_user, previous_system, previous_identity = self.cpu_previous
                interval = cpu_time - previous_time
                if identity != previous_identity:
                    self.cpu_identity_changes += 1
                elif user < previous_user or system < previous_system:
                    self.cpu_resets += 1
                elif interval > 0:
                    self.cpu_delta += user - previous_user + system - previous_system
                    self.cpu_covered += interval
                    self.cpu_intervals += 1
                    self.cpu_long_gaps += interval > self.max_gap
            self.cpu_previous = cpu_time, user, system, identity
        else:
            self.cpu_missing += (
                1  # Cumulative counters can bridge missing polls when identity agrees.
            )
        self.rss.add(
            row.get("process_rss_bytes") if identity is not None else None,
            self._elapsed(sample.get("process_memory_sampled_monotonic_seconds"), elapsed),
            identity,
        )
        self.ram.add(
            row.get("ram_used_bytes"),
            self._elapsed(sample.get("host_sampled_monotonic_seconds"), elapsed),
        )
        gpu = sample.get("gpu", {})
        if gpu.get("requested"):
            self.gpu_requested += 1
            self.gpu_failed += bool(gpu.get("error"))
            gpu_time = self._elapsed(gpu.get("sampled_monotonic_seconds"), elapsed)
            current = {
                device["uuid"]: device for device in gpu.get("devices", []) if device.get("uuid")
            }
            for uuid, device in current.items():
                if uuid not in self.gpus:
                    self.gpus[uuid] = {
                        "identity": {key: device.get(key) for key in ("uuid", "index", "name")},
                        "utilization": _Gauge(self.max_gap),
                        "vram": _Gauge(self.max_gap),
                    }
            for uuid, values in self.gpus.items():
                device = current.get(uuid, {})
                values["utilization"].add(device.get("gpu_percent"), gpu_time, uuid, maximum=100)
                values["vram"].add(device.get("vram_used_bytes"), gpu_time, uuid)

    def report(self) -> dict:
        span = (
            max(0.0, self.last - self.first)
            if self.first is not None and self.last is not None
            else 0.0
        )
        discontinuity = bool(self.cpu_resets or self.cpu_identity_changes)
        gpus = []
        for uuid in sorted(self.gpus):
            device = self.gpus[uuid]
            utilization = device["utilization"]
            gpus.append(
                {
                    **device["identity"],
                    "scope": "device_wide_all_processes; not_policy_attributed",
                    "utilization_equivalent_seconds": utilization.integral / 100
                    if utilization.covered_seconds
                    else None,
                    "utilization_percent": utilization.report(span),
                    "vram_bytes": device["vram"].report(span),
                }
            )
        return {
            "version": "resources-3.0",
            "clock_boundary": "initial_resource_sample_start_to_final_resource_sample_completion",
            "initial_sample_elapsed_seconds": self.first,
            "final_sample_elapsed_seconds": self.last,
            "sample_span_seconds": span,
            "sample_count": self.samples,
            "boundary_exclusions": "before_initial_sample_and_after_final_sample; not_full_command_cost",
            "max_interpolation_gap_seconds": self.max_gap,
            "nonmonotonic_samples_excluded": self.nonmonotonic_samples,
            "samples_with_errors": self.error_samples,
            "process_cpu": {
                "core_seconds": self.cpu_delta
                if self.cpu_intervals and not discontinuity
                else None,
                "observed_segment_delta_seconds": self.cpu_delta if self.cpu_intervals else None,
                "scope": "recorder_process_all_threads; excludes_child_processes",
                "method": "measured_cumulative_user_plus_system_counter_differences",
                "status": "counter_discontinuity"
                if discontinuity
                else "observed"
                if self.cpu_intervals
                else "unavailable",
                "process_identities": [
                    {"pid": pid, "create_time_unix_seconds": created}
                    for pid, created in sorted(self.process_identities)
                ],
                "covered_seconds": self.cpu_covered,
                "coverage_fraction": min(1.0, self.cpu_covered / span) if span > 0 else None,
                "uncovered_seconds": max(0.0, span - self.cpu_covered),
                "valid_samples": self.cpu_valid,
                "missing_samples": self.cpu_missing,
                "counter_resets": self.cpu_resets,
                "identity_changes": self.cpu_identity_changes,
                "long_gap_intervals_bridged_by_cumulative_counters": self.cpu_long_gaps,
            },
            "memory": {
                "process_rss_bytes": self.rss.report(span),
                "host_ram_used_bytes": self.ram.report(span),
            },
            "gpus": gpus,
            "gpu_query_count": self.gpu_requested,
            "gpu_query_failures": self.gpu_failed,
            "gpu_attribution": "host_device_activity_includes_other_processes; no_policy_attribution",
            "energy_measured": False,
            "peak_semantics": "sampled_peaks_only; unsampled_peaks_may_be_higher",
        }

    def snapshot(self) -> dict:
        """Detach the latest measured costs; cutoff is last sample, not a checkpoint clock."""
        return self.report()
