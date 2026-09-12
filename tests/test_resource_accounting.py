"""Measured CPU deltas and sampled memory/GPU integration with explicit missingness."""

import json
from types import SimpleNamespace

import pytest

from gradientclimb.experiments import RunRecorder, load_run, recorder, verify_run
from gradientclimb.telemetry import provenance
from gradientclimb.telemetry.resources import ResourceAccumulator

ORIGIN = 1000.0


def sample(
    elapsed,
    *,
    user=10.0,
    system=5.0,
    rss=100,
    ram=1000,
    gpu=50,
    vram=2000,
    requested=True,
    pid=7,
    created=123.0,
    gpu_uuid="GPU-reference",
    errors=None,
):
    clock = ORIGIN + elapsed
    return {
        "elapsed_seconds": elapsed,
        "process_rss_bytes": rss,
        "ram_used_bytes": ram,
        "measurements": {
            "resources_sample": {
                "version": "resources-sample-3.0",
                "sample_started_monotonic_seconds": clock,
                "sample_completed_monotonic_seconds": clock,
                "host_sampled_monotonic_seconds": clock,
                "process_memory_sampled_monotonic_seconds": clock,
                "errors": errors or {},
                "process": {
                    "pid": pid,
                    "create_time_unix_seconds": created,
                    "cpu_user_seconds": user,
                    "cpu_system_seconds": system,
                    "sampled_monotonic_seconds": clock,
                },
                "gpu": {
                    "requested": requested,
                    "sampled_monotonic_seconds": clock,
                    "error": None,
                    "devices": [
                        {
                            "uuid": gpu_uuid,
                            "index": 0,
                            "name": "Measured device",
                            "gpu_percent": gpu,
                            "vram_used_bytes": vram,
                        }
                    ],
                },
            }
        },
    }


def test_resource_integrals_use_cumulative_cpu_and_trapezoid_gauges():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add(sample(0, user=10, system=5, rss=100, ram=1000, gpu=0, vram=1000))
    accounting.add(sample(2, user=11, system=5.5, rss=300, ram=2000, gpu=50, vram=2000))
    accounting.add(sample(4, user=12, system=6, rss=500, ram=3000, gpu=100, vram=3000))
    result = accounting.snapshot()
    assert result["version"] == "resources-3.0" and result["sample_span_seconds"] == 4
    assert result["process_cpu"]["core_seconds"] == 3
    assert result["process_cpu"]["coverage_fraction"] == 1
    assert result["memory"]["process_rss_bytes"]["time_weighted_sampled_mean"] == 300
    assert result["memory"]["host_ram_used_bytes"]["sampled_peak"] == 3000
    device = result["gpus"][0]
    assert device["uuid"] == "GPU-reference"
    assert device["utilization_equivalent_seconds"] == 2
    assert device["vram_bytes"]["time_weighted_sampled_mean"] == 2000
    assert "not_policy_attributed" in device["scope"] and result["energy_measured"] is False
    result["process_cpu"]["process_identities"].clear()
    assert accounting.snapshot()["process_cpu"]["process_identities"]


def test_missing_cpu_polls_can_bridge_cumulative_delta_but_gauges_cannot():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add(sample(0, user=0, system=0))
    accounting.add(sample(2, user=None, system=None, rss=None, ram=None, gpu=None, vram=None))
    accounting.add(sample(4, user=2, system=1))
    result = accounting.report()
    assert result["process_cpu"]["core_seconds"] == 3
    assert result["process_cpu"]["covered_seconds"] == 4
    assert result["process_cpu"]["missing_samples"] == 1
    assert result["memory"]["process_rss_bytes"]["time_weighted_sampled_mean"] is None
    assert result["gpus"][0]["utilization_equivalent_seconds"] is None
    assert result["gpus"][0]["utilization_percent"]["coverage_fraction"] == 0


def test_long_gaps_are_reported_and_excluded_for_instantaneous_gauges():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN, max_gap_seconds=3)
    accounting.add(sample(0, user=0))
    accounting.add(sample(10, user=8))
    result = accounting.report()
    assert result["process_cpu"]["core_seconds"] == 8
    assert result["process_cpu"]["long_gap_intervals_bridged_by_cumulative_counters"] == 1
    assert result["memory"]["process_rss_bytes"]["long_gap_intervals_excluded"] == 1
    assert result["memory"]["process_rss_bytes"]["time_weighted_sampled_mean"] is None
    assert result["gpus"][0]["utilization_equivalent_seconds"] is None


@pytest.mark.parametrize("fault", ["reset", "identity"])
def test_counter_discontinuity_never_becomes_a_valid_total(fault):
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add(sample(0, user=10))
    accounting.add(sample(1, user=12))
    accounting.add(
        sample(2, user=1 if fault == "reset" else 13, created=999 if fault == "identity" else 123)
    )
    accounting.add(sample(3, user=15, created=999 if fault == "identity" else 123))
    result = accounting.report()["process_cpu"]
    assert result["core_seconds"] is None and result["status"] == "counter_discontinuity"
    assert result["counter_resets"] + result["identity_changes"] == 1


def test_no_samples_one_sample_and_old_rows_preserve_missingness():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add({"elapsed_seconds": 1, "cpu_percent": 80, "gpu_percent": 100})
    assert accounting.report()["sample_count"] == 0
    assert accounting.report()["process_cpu"]["core_seconds"] is None
    accounting.add(sample(0, gpu=0, rss=0))
    report = accounting.report()
    assert report["process_cpu"]["core_seconds"] is None
    assert report["memory"]["process_rss_bytes"]["sampled_peak"] == 0
    assert report["memory"]["process_rss_bytes"]["time_weighted_sampled_mean"] is None
    assert report["gpus"][0]["utilization_equivalent_seconds"] is None
    accounting.add(sample(2, gpu=0, rss=0))
    assert accounting.report()["gpus"][0]["utilization_equivalent_seconds"] == 0
    assert accounting.report()["process_cpu"]["core_seconds"] == 0


def test_unscheduled_gpu_poll_is_not_a_missing_measurement_but_failed_query_is():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add(sample(0))
    accounting.add(sample(1, requested=False))
    accounting.add(sample(2))
    assert accounting.report()["gpus"][0]["utilization_equivalent_seconds"] == 1
    failed = sample(3)
    failed["measurements"]["resources_sample"]["gpu"].update(devices=[], error="query failed")
    accounting.add(failed)
    accounting.add(sample(4))
    report = accounting.report()
    assert report["gpu_query_failures"] == 1
    assert report["gpus"][0]["utilization_percent"]["covered_seconds"] == 2


def test_invalid_or_nonmonotonic_samples_do_not_create_costs():
    accounting = ResourceAccumulator(monotonic_origin=ORIGIN)
    accounting.add(sample(2, user=10, gpu=float("nan")))
    accounting.add(sample(1, user=10000, gpu=100))
    accounting.add(sample(3, user=11, gpu=101, rss=-1))
    report = accounting.report()
    assert report["nonmonotonic_samples_excluded"] == 1
    assert report["process_cpu"]["core_seconds"] == 1
    assert report["gpus"][0]["utilization_equivalent_seconds"] is None
    assert report["memory"]["process_rss_bytes"]["missing_samples"] == 1


def test_sampling_records_process_identity_cpu_counters_and_all_gpu_devices(monkeypatch):
    process = SimpleNamespace(
        create_time=lambda: 123.0,
        cpu_times=lambda: SimpleNamespace(user=5.0, system=2.0, children_user=9000),
        memory_info=lambda: SimpleNamespace(rss=4096),
    )
    monkeypatch.setattr(
        provenance,
        "psutil",
        SimpleNamespace(
            cpu_percent=lambda interval, percpu=False: [20.0, 30.0] if percpu else 25.0,
            virtual_memory=lambda: SimpleNamespace(used=8192),
            Process=lambda: process,
        ),
    )
    monkeypatch.setattr(provenance, "_nvidia_smi", lambda: "nvidia-smi")
    commands = []
    monkeypatch.setattr(
        provenance,
        "_command",
        lambda command: (
            commands.append(command)
            or "GPU-a, 0, First device, 25, 128\nGPU-b, 1, Second device, N/A, 256"
        ),
    )
    result = provenance.resource_sample(include_gpu=True)
    measured = result["measurements"]["resources_sample"]
    assert measured["process"]["pid"] > 0
    assert measured["process"]["cpu_user_seconds"] == 5
    assert "children_user" not in measured["process"]
    assert [device["uuid"] for device in measured["gpu"]["devices"]] == ["GPU-a", "GPU-b"]
    assert measured["gpu"]["devices"][1]["gpu_percent"] is None
    assert result["gpu_percent"] == 25 and result["vram_used_bytes"] == 128 * 1024**2
    assert "uuid,index,name" in commands[0][1]
    json.dumps(result, allow_nan=False)


def test_sampling_faults_keep_surviving_measurements_without_inventing_zero(monkeypatch):
    def unavailable():
        raise PermissionError("counter unavailable")

    monkeypatch.setattr(
        provenance,
        "psutil",
        SimpleNamespace(
            cpu_percent=lambda **kwargs: [20] if kwargs.get("percpu") else 20,
            virtual_memory=lambda: SimpleNamespace(used=123),
            Process=unavailable,
        ),
    )
    monkeypatch.setattr(provenance, "_nvidia_smi", lambda: None)
    sampled = provenance.resource_sample(True)
    assert sampled["ram_used_bytes"] == 123
    assert "process_rss_bytes" not in sampled
    measurement = sampled["measurements"]["resources_sample"]
    assert measurement["process"]["cpu_user_seconds"] is None
    assert measurement["gpu"]["devices"] == [] and measurement["gpu"]["error"]
    assert "process_cpu" in measurement["errors"]


def test_command_decodes_utf8_synchronously_and_leaves_invalid_source_unavailable(monkeypatch):
    payload = "diff --git a/file b/file\n+Δ → elapsed\n".encode()

    def captured(arguments, **kwargs):
        assert "text" not in kwargs and kwargs["capture_output"]
        return SimpleNamespace(returncode=0, stdout=payload)

    monkeypatch.setattr(provenance.subprocess, "run", captured)
    assert provenance._command(["git", "diff"], strip_output=False) == payload.decode("utf-8")
    payload = b"invalid utf8: \xff"
    assert provenance._command(["git", "diff"]) is None


@pytest.mark.parametrize(
    "field",
    ["telemetry_interval_seconds", "resource_gpu_interval_seconds", "resource_max_gap_seconds"],
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_sampling_configuration_rejected_before_artifacts(tmp_path, field, value):
    with pytest.raises(ValueError):
        RunRecorder(tmp_path, "invalid-resource-config", {}, **{field: value})
    assert not (tmp_path / "runs").exists()


def test_recorder_retains_resource_summary_for_failed_run_with_legacy_schema(tmp_path, monkeypatch):
    calls = 0

    def measured(include_gpu):
        nonlocal calls
        calls += 1
        row = sample(calls, user=10 + calls, system=2, requested=include_gpu)
        details = row["measurements"]["resources_sample"]
        # Recorder's actual monotonic origin is unavailable to this stub; its row clock
        # still supplies an honest test timestamp without synthetic absolute clocks.
        for key in list(details):
            if "monotonic" in key:
                details.pop(key)
        details["process"].pop("sampled_monotonic_seconds")
        details["gpu"].pop("sampled_monotonic_seconds")
        row.pop("elapsed_seconds")
        return row

    monkeypatch.setattr(recorder, "resource_sample", measured)
    with (
        pytest.raises(RuntimeError, match="learner failure"),
        RunRecorder(tmp_path, "resource-fault", {}, telemetry_interval_seconds=0) as run,
    ):
        snapshot = run.resource_snapshot()
        assert snapshot["sample_count"] == 1 and snapshot["process_cpu"]["core_seconds"] is None
        raise RuntimeError("learner failure")
    record = load_run(tmp_path, run.run_id)
    resources = record["summary"]["resources"]
    assert record["schema_version"] == "1.0.0"
    assert record["status"] == "failed" and resources["version"] == "resources-3.0"
    assert resources["sample_count"] == 2
    assert resources["process_cpu"]["core_seconds"] == 1
    assert record["metadata"]["resource_sampling"]["gpu_interval_seconds"] == 10
    assert verify_run(tmp_path, run.run_id)["valid"]
