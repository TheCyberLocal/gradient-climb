import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import load_run, verify_run

SPEC = importlib.util.spec_from_file_location(
    "owner_acquisition", Path(__file__).parents[1] / "scripts/acquire_owner_video.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_failed_command_keeps_log_and_cost_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="exit code 7"):
        MODULE.bounded_process(
            [
                sys.executable,
                "-c",
                "print('retained partial work', flush=True); raise SystemExit(7)",
            ],
            tmp_path,
            time.perf_counter() + 20,
            1024**2,
            "failed.log",
        )
    assert b"retained partial work" in (tmp_path / "failed.log").read_bytes()
    receipt = json.loads((tmp_path / "failed.log.receipt.json").read_bytes())
    assert receipt["exit_code"] == 7
    assert receipt["elapsed_seconds"] > 0
    assert receipt["error"] is not None


def test_wall_guard_terminates_owned_process_and_retains_receipt(tmp_path):
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="wall budget"):
        MODULE.bounded_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            started + 0.3,
            1024**2,
            "timeout.log",
        )
    assert time.perf_counter() - started < 15
    assert (
        "TimeoutError" in json.loads((tmp_path / "timeout.log.receipt.json").read_bytes())["error"]
    )


def test_output_guard_retains_partial_without_running_to_completion(tmp_path):
    with pytest.raises(RuntimeError, match="output budget"):
        MODULE.bounded_process(
            [sys.executable, "-c", "import time; print('a'*10000, flush=True); time.sleep(30)"],
            tmp_path,
            time.perf_counter() + 20,
            1000,
            "full.log",
        )
    assert (tmp_path / "full.log").stat().st_size >= 10000
    assert "output budget" in (tmp_path / "full.log.receipt.json").read_text()


def test_exited_launcher_does_not_leave_observed_child_running(tmp_path):
    child_pid = tmp_path / "child.pid"
    command = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(0.7); raise SystemExit(7)"
    )
    with pytest.raises(RuntimeError, match="exit code 7"):
        MODULE.bounded_process(
            [sys.executable, "-c", command, str(child_pid)],
            tmp_path,
            time.perf_counter() + 20,
            1024**2,
            "orphan.log",
        )
    pid = int(child_pid.read_text())
    assert not psutil.pid_exists(pid)
    receipt = json.loads((tmp_path / "orphan.log.receipt.json").read_bytes())
    # Windows virtual-environment launchers can add interpreter processes.
    assert receipt["live_descendants_after_launcher_exit"] >= 1
    assert receipt["terminated_owned_processes"] >= 1


def test_cleanup_errors_preserve_primary_guard_and_still_write_receipt(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 123

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("fake-owned-command", timeout)

    class FakeOwned:
        pid = 123

        def create_time(self):
            return 1

        def cpu_times(self):
            return SimpleNamespace(user=0.25, system=0.125)

        def children(self, recursive):
            return []

        def is_running(self):
            return True

        def kill(self):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(MODULE.psutil, "Process", lambda *args: FakeOwned())
    monkeypatch.setattr(MODULE.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    with pytest.raises(TimeoutError, match="wall budget") as failure:
        MODULE.bounded_process(["fake"], tmp_path, time.perf_counter() - 1, 1024, "cleanup.log")
    assert any(
        "AccessDenied" in note and "TimeoutExpired" in note for note in failure.value.__notes__
    )
    receipt = json.loads((tmp_path / "cleanup.log.receipt.json").read_bytes())
    assert "TimeoutError" in receipt["error"]
    assert len(receipt["cleanup_errors"]) == 2


def test_successful_command_has_final_receipt_without_cleanup_failure(tmp_path):
    MODULE.bounded_process(
        [sys.executable, "-c", "print('complete')"],
        tmp_path,
        time.perf_counter() + 20,
        1024**2,
        "success.log",
    )
    receipt = json.loads((tmp_path / "success.log.receipt.json").read_bytes())
    assert receipt["exit_code"] == 0 and receipt["error"] is None
    assert receipt["cleanup_errors"] == []


@pytest.fixture
def fake_acquisition(tmp_path, monkeypatch):
    wheel_bytes = b"synthetic pinned wheel; never executed"
    wheel = tmp_path / "fixture-wheel.whl"
    wheel.write_bytes(wheel_bytes)
    ffmpeg = tmp_path / "fixture-ffmpeg"
    ffmpeg.write_bytes(b"synthetic ffmpeg; never executed")
    plan = {
        "schema_version": "owner-video-acquisition-3.0",
        "status": "registered",
        "experiment_id": "synthetic-owner-acquisition",
        "video_url": "https://www.youtube.com/watch?v=wx_cI59vFX0",
        "start_seconds": 0,
        "end_seconds": 300,
        "wall_budget_seconds": 30,
        "output_budget_bytes": 1024**2,
        "tool": {
            "filename": "fixture.whl",
            "url": "https://example.invalid/fixture.whl",
            "bytes": len(wheel_bytes),
            "sha256": sha256_file(wheel),
        },
        "ffmpeg": {"path": str(ffmpeg), "sha256": sha256_file(ffmpeg)},
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(
        MODULE.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(wheel_bytes)
    )
    calls = []

    def fake_process(command, directory, deadline, maximum_bytes, log_name, environment=None):
        calls.append(log_name)
        (directory / log_name).write_text("synthetic command output", encoding="utf-8")
        (directory / f"{log_name}.receipt.json").write_text('{"synthetic":true}', encoding="utf-8")
        if log_name == "install.log":
            (directory / "tool").mkdir()
            (directory / "tool/fake.py").write_text("# synthetic", encoding="utf-8")
        else:
            (directory / "owner-excerpt.mp4").write_bytes(b"synthetic media placeholder")

    monkeypatch.setattr(MODULE, "bounded_process", fake_process)
    return tmp_path, calls, fake_process


def test_fake_acquisition_registers_payload_before_final_seal(fake_acquisition):
    root, calls, _ = fake_acquisition
    result = MODULE.acquire(root, "protocol.json")
    record = load_run(root / "artifacts", result["run_id"])
    assert calls == ["install.log", "download.log"]
    assert result["verification"]["valid"] and record["status"] == "completed"
    assert record["summary"]["acquisition_completed"]
    assert record["summary"]["actual_media_seconds"] is None
    assert not record["summary"]["training_eligible"]
    assert record["summary"]["synchronized_action_labels"] == 0
    assert record["summary"]["resources"]["version"] == "resources-3.0"
    assert any(a["path"].endswith("owner-excerpt.mp4") for a in record["artifact_manifest"])
    assert record["episodes"] == record["training_steps"] == record["environment_steps"] == 0
    with pytest.raises(ValueError, match="already has an acquisition attempt"):
        MODULE.acquire(root, "protocol.json")


def test_fake_download_and_retention_failures_preserve_primary_and_other_files(
    fake_acquisition, monkeypatch
):
    root, _, process = fake_acquisition

    def fail_download(*args, **kwargs):
        process(*args, **kwargs)
        if args[4] == "download.log":
            (args[1] / "z-after-failure.txt").write_text("retained later", encoding="utf-8")
            raise RuntimeError("synthetic primary download failure")

    original = MODULE.RunRecorder.register_artifact

    def fail_registration(run, path, kind, *args, **kwargs):
        if path.name == "download.log":
            raise OSError("synthetic retention failure")
        return original(run, path, kind, *args, **kwargs)

    monkeypatch.setattr(MODULE, "bounded_process", fail_download)
    monkeypatch.setattr(MODULE.RunRecorder, "register_artifact", fail_registration)
    with pytest.raises(RuntimeError, match="primary download failure") as failure:
        MODULE.acquire(root, "protocol.json")
    assert any("retention failure" in note for note in failure.value.__notes__)
    directory = next((root / "artifacts/runs").iterdir())
    record = load_run(root / "artifacts", directory.name)
    assert record["status"] == "failed"
    assert "primary download failure" in record["summary"]["error"]
    assert any(a["path"].endswith("z-after-failure.txt") for a in record["artifact_manifest"])
    assert verify_run(root / "artifacts", directory.name)["valid"]
