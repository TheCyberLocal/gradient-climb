"""Live headed/headless observer: isolation, snapshots, sinks and run records."""

import json
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from gradientclimb.algorithms import ActorCritic, train_ppo
from gradientclimb.benchmarks import runner as runner_module
from gradientclimb.benchmarks.runner import run_training
from gradientclimb.experiments import RunRecorder, load_run, query_metrics, verify_run
from gradientclimb.visualization import observer as observer_module
from gradientclimb.visualization.observer import (
    DEFAULT_OBSERVER_SEED,
    PROVENANCE_KEYS,
    ObserverConfig,
    ResourceSampler,
    TrainingObserver,
    build_sinks,
    compose_overlay,
    cpu_percent_between,
    format_overlay_lines,
)
from gradientclimb.visualization.replay import watch
from gradientclimb.visualization.sinks import FfmpegSink, NullSink, TkSink

TINY = {
    "num_envs": 8,
    "rollout_steps": 16,
    "hidden_size": 16,
    "minibatch_size": 64,
    "epochs": 2,
    "max_steps": 30,
}
EXISTING_ROW_KEYS = {
    "wall_clock_seconds",
    "environment_steps",
    "env_steps",
    "training_steps",
    "episodes",
    "optimizer_updates",
    "iterations",
    "environment_steps_per_second",
    "mean_episode_distance",
    "mean_episode_return",
    "policy_loss",
    "entropy",
    "approx_kl",
    "inference_seconds",
    "environment_seconds",
    "optimizer_seconds",
    "checkpoint",
    "checkpoint_target_seconds",
    "requested_checkpoint_seconds",
    "final",
}
SNAPSHOT_ROW_KEYS = {
    "snapshot_count",
    "snapshot_copy_seconds",
    "snapshot_callback_seconds",
    "snapshot_errors",
}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def payload(model, index, **overrides):
    base = {
        "index": index,
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "best_state_dict": None,
        "best_index": None,
        "best_snapshot": None,
        "is_best": False,
        "elapsed": 0.5 + index,
        "optimizer_updates": 3 * index,
        "environment_steps": 100 * index,
        "episodes": 2 * index,
        "iterations": index,
        "mean_episode_distance": None,
        "best_mean_episode_distance": None,
        "algorithm_version": "original-0.1.0",
        "seed": 0,
    }
    base.update(overrides)
    return base


def carry_best(item, best):
    """Make ``item`` carry ``best``'s weights and provenance exactly as train_ppo does."""
    item["best_state_dict"] = best["state_dict"]
    item["best_index"] = best["index"]
    item["best_snapshot"] = {key: best[key] for key in PROVENANCE_KEYS}
    item["is_best"] = item is best
    return item


def wait_until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def headless_observer(policy="current", max_steps=20):
    clock = FakeClock()
    config = ObserverConfig(
        policy=policy, fps=200.0, telemetry_interval=60.0, width=320, height=180
    )
    observer = TrainingObserver(
        config,
        {"max_steps": max_steps, "hidden_size": 16},
        [NullSink()],
        clock=clock,
        sleep=clock.sleep,
    )
    return observer, clock


def test_snapshot_payload_is_detached_cpu_copy_and_tracks_best():
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    payloads = []
    result = train_ppo(
        2.0,
        seed=9,
        checkpoint_times=[],
        snapshot_interval=0.25,
        snapshot=payloads.append,
        **TINY,
    )
    assert len(payloads) >= 2
    live = {name: parameter.data_ptr() for name, parameter in result.model.named_parameters()}
    best_seen = None
    first_with_mean = None
    for index, item in enumerate(payloads):
        assert item["index"] == index
        for name, tensor in item["state_dict"].items():
            assert tensor.device.type == "cpu"
            assert tensor.requires_grad is False
            assert tensor.data_ptr() != live[name]
        if item["mean_episode_distance"] is not None and first_with_mean is None:
            first_with_mean = item
        if item["is_best"]:
            best_seen = index
            assert item["best_index"] == index
            assert item["best_state_dict"] is item["state_dict"]
        else:
            assert item["best_index"] == best_seen
        assert item["seed"] == 9 and item["algorithm_version"] == "original-0.1.0"
    assert first_with_mean is not None and first_with_mean["is_best"]
    final = result.metrics[-1]
    assert EXISTING_ROW_KEYS <= set(final)
    assert SNAPSHOT_ROW_KEYS <= set(final)
    assert final["snapshot_count"] == len(payloads)
    assert final["snapshot_errors"] == 0
    assert final["snapshot_copy_seconds"] > 0 and final["snapshot_callback_seconds"] >= 0
    # Best-selector values live only in the payload: metric rows never depend on
    # the snapshot cadence, so an observed and an unobserved run share row keys.
    assert "best_mean_episode_distance" not in final
    assert payloads[-1]["best_mean_episode_distance"] is not None
    best = payloads[-1]["best_snapshot"]
    assert best["index"] == payloads[-1]["best_index"]
    assert set(PROVENANCE_KEYS) <= set(best)
    unobserved = train_ppo(0.5, seed=9, checkpoint_times=[], **TINY)
    assert set(unobserved.metrics[-1]) == set(final)
    assert unobserved.metrics[-1]["snapshot_count"] == 0
    assert "snapshot" not in result.config and "snapshot_interval" not in result.config


def test_snapshot_callback_errors_never_reach_learner():
    def explode(item):
        raise RuntimeError("observer exploded")

    result = train_ppo(
        1.0, seed=3, checkpoint_times=[], snapshot_interval=0.2, snapshot=explode, **TINY
    )
    assert result.metrics[-1]["final"] is True
    assert result.metrics[-1]["snapshot_errors"] > 0
    assert result.environment_steps > 0


def test_snapshot_without_interval_is_rejected():
    with pytest.raises(ValueError):
        train_ppo(1.0, snapshot=lambda item: None, **TINY)
    with pytest.raises(ValueError):
        train_ppo(1.0, snapshot=lambda item: None, snapshot_interval=0, **TINY)


def test_observer_construction_does_not_consume_torch_rng():
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    expected = torch.rand(3)
    torch.manual_seed(0)
    TrainingObserver(ObserverConfig(), {"hidden_size": 16}, [NullSink()])
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(torch.rand(3), expected)


def test_observer_thread_follows_snapshots_headless():
    observer, _ = headless_observer()
    observer.start()
    model = ActorCritic(observer.env.observation_dim, 16)
    observer.snapshot(payload(model, 0))
    assert wait_until(lambda: len(observer.episodes) >= 2)
    report = observer.stop()
    assert report["frames_rendered"] > 0
    assert report["observer_env_steps"] == report["frames_rendered"]
    assert report["error"] is None and report["stopped_cleanly"] and report["thread_joined"]
    assert report["snapshots_received"] == 1 and report["snapshots_loaded"] == 1
    assert report["observer_episodes"] == len(report["episodes"]) >= 2
    for row in report["episodes"]:
        assert row["snapshot_index"] == 0
        assert row["snapshot_elapsed"] == 0.5
        assert row["snapshot_optimizer_updates"] == 0
        assert row["length"] <= 20 and row["termination"]
        assert isinstance(row["distance"], float)
    assert report["sinks"][0]["kind"] == "null"
    assert report["sinks"][0]["frames"] == report["frames_rendered"]
    assert json.dumps(report, allow_nan=False)


def test_best_selector_only_reloads_on_best_snapshots():
    observer, _ = headless_observer(policy="best")
    observer.start()
    first = ActorCritic(observer.env.observation_dim, 16)
    a = payload(first, 0)
    carry_best(a, a)
    observer.snapshot(a)
    assert wait_until(lambda: observer.loaded_snapshot_index == 0)
    b = carry_best(payload(ActorCritic(observer.env.observation_dim, 16), 1), a)
    observer.snapshot(b)
    seen = len(observer.episodes)
    assert wait_until(lambda: len(observer.episodes) >= seen + 2)
    assert observer.loaded_snapshot_index == 0 and observer.snapshots_loaded == 1
    c = payload(ActorCritic(observer.env.observation_dim, 16), 2)
    carry_best(c, c)
    observer.snapshot(c)
    assert wait_until(lambda: observer.loaded_snapshot_index == 2)
    report = observer.stop()
    assert report["snapshots_received"] == 3 and report["snapshots_loaded"] == 2
    assert report["error"] is None and report["policy"] == "best"
    # Every episode driven by snapshot #0's weights carries #0's provenance, not
    # the provenance of the newer payloads (#1) that merely carried it along.
    by_index = {row["snapshot_index"]: row for row in report["episodes"]}
    assert set(by_index) <= {0, 2}
    zero = by_index[0]
    assert zero["snapshot_elapsed"] == 0.5 and zero["snapshot_optimizer_updates"] == 0
    assert zero["snapshot_environment_steps"] == 0
    for row in report["episodes"]:
        expected = {0: a, 2: c}[row["snapshot_index"]]
        assert row["snapshot_elapsed"] == expected["elapsed"]
        assert row["snapshot_optimizer_updates"] == expected["optimizer_updates"]
        assert row["snapshot_environment_steps"] == expected["environment_steps"]


def test_snapshot_age_and_provenance_follow_the_loaded_snapshot():
    observer, clock = headless_observer(policy="best")
    model = ActorCritic(observer.env.observation_dim, 16)
    a = payload(model, 0)  # elapsed 0.5
    carry_best(a, a)
    observer.snapshot(a)
    observer._load_if_newer()
    clock.now += 5.0
    b = carry_best(payload(model, 7), a)  # elapsed 7.5, arrives 5 s after a
    observer.snapshot(b)
    clock.now += 2.0
    observer._load_if_newer()  # weights stay #0: b carries a as best
    fields = observer._fields(0.0)
    assert observer.loaded_snapshot_index == 0
    assert fields["snapshot_index"] == 0 and fields["snapshot_elapsed"] == 0.5
    assert fields["snapshot_optimizer_updates"] == 0
    assert fields["training_elapsed_seconds"] == pytest.approx(7.5 + 2.0)
    # Age is measured on the training clock from the loaded snapshot, not from
    # the newest payload's arrival.
    assert fields["snapshot_age_seconds"] == pytest.approx(9.5 - 0.5)
    current, clock2 = headless_observer(policy="current")
    current.snapshot(payload(model, 0))
    current._load_if_newer()
    clock2.now += 3.0
    assert current._fields(0.0)["snapshot_age_seconds"] == pytest.approx(3.0)


def test_terminal_frame_describes_the_finished_episode():
    observer, _ = headless_observer()
    observer.snapshot(payload(ActorCritic(observer.env.observation_dim, 16), 0))
    observer._load_if_newer()
    episode = {
        "distance": 84.3,
        "length": 233,
        "termination": "crash",
        "gas": 1,
        "brake": 0,
    }
    fields = observer._fields(-1.0, episode)
    assert fields["observer_distance"] == 84.3 and fields["observer_step"] == 233
    assert fields["observer_termination"] == "crash" and fields["step_reward"] == -1.0
    assert fields["velocity"] is None and fields["theta"] is None
    assert fields["gas"] == 1 and fields["brake"] == 0
    line = format_overlay_lines(fields)[2]
    assert "distance 84.3 m" in line and "step 233" in line and "ended: crash" in line
    live = observer._fields(0.01)
    assert live["observer_termination"] is None and live["observer_distance"] == 0.0
    assert "ended" not in format_overlay_lines(live)[2]


def test_stop_interrupts_a_long_frame_wait():
    # 0.05 fps means a 20 s wait between frames; stop() must not wait for it.
    config = ObserverConfig(fps=0.05, telemetry_interval=60.0, width=320, height=180)
    observer = TrainingObserver(config, {"max_steps": 20, "hidden_size": 16}, [NullSink()])
    observer.start()
    observer.snapshot(payload(ActorCritic(observer.env.observation_dim, 16), 0))
    assert wait_until(lambda: observer.frames_rendered >= 1)
    started = time.monotonic()
    report = observer.stop(timeout=2.0)
    assert time.monotonic() - started < 2.0
    assert report["thread_joined"] and report["stopped_cleanly"] and report["error"] is None
    assert report["video_skipped_reason"] is None


def test_report_flags_a_thread_that_outlived_stop():
    observer, _ = headless_observer()
    observer.stop_timeout, observer.thread_joined = 1.0, False
    report = observer.report()
    assert not report["stopped_cleanly"]
    assert "still running" in report["video_skipped_reason"] and report["video"] is None


def test_observer_stops_cleanly_without_any_snapshot():
    observer, _ = headless_observer()
    observer.start()
    time.sleep(0.2)
    started = time.monotonic()
    report = observer.stop(timeout=5.0)
    assert time.monotonic() - started < 5.0
    assert report["frames_rendered"] == 0 and report["error"] is None
    assert report["stopped_cleanly"] and report["thread_joined"]


def test_compose_overlay_keeps_size_and_tolerates_missing_gpu():
    frame = Image.new("RGB", (320, 180), "black")
    fields = {
        "training_elapsed_seconds": 12.5,
        "environment_steps": 1234567,
        "gpu_percent": None,
        "vram_used_bytes": None,
        "cpu_percent": 37.0,
        "step_reward": -0.0123,
    }
    result = compose_overlay(frame, fields)
    assert result.size == (320, 180) and result is not frame
    assert compose_overlay(frame, {}).size == (320, 180)


def test_build_sinks_probes_tk_before_any_run_exists(tmp_path, monkeypatch):
    def no_display():
        raise RuntimeError("Tk cannot open a window: no display")

    monkeypatch.setattr(TkSink, "probe", staticmethod(no_display))
    with pytest.raises(RuntimeError, match="no display"):
        build_sinks(ObserverConfig(display="window"))
    with pytest.raises(RuntimeError, match="no display"):
        run_training(tmp_path, "ppo", 1.0, 0, {**TINY}, observer={"display": "window"})
    assert not (tmp_path / "runs").exists()
    sinks, reason = build_sinks(ObserverConfig(display="none"))
    assert [sink.kind for sink in sinks] == ["null"] and reason is None


def test_build_sinks_skips_video_when_ffmpeg_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    sinks, reason = build_sinks(ObserverConfig(video=str(tmp_path / "live.mp4")))
    assert [sink.kind for sink in sinks] == ["null"] and reason == "ffmpeg executable not found"
    with pytest.raises(RuntimeError):
        watch(None, 0.3, 20000, video=tmp_path / "rollout.mp4")
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"x")
    monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")
    with pytest.raises(FileExistsError):
        build_sinks(ObserverConfig(video=str(existing)))
    with pytest.raises(FileExistsError):
        watch(None, 0.3, 20000, video=existing)
    sinks, reason = build_sinks(ObserverConfig())
    assert [sink.kind for sink in sinks] == ["null"] and reason is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_sink_writes_video(tmp_path):
    path = tmp_path / "clip.mp4"
    sink = FfmpegSink(path)
    sink.open(96, 64, 10)
    for shade in (0, 128, 255):
        sink.write(Image.new("RGB", (96, 64), (shade, shade, shade)))
    report = sink.close()
    assert path.exists() and path.stat().st_size > 0
    assert report == {
        "kind": "ffmpeg",
        "frames": 3,
        "video": str(path),
        "size": (96, 64),
        "fps": 10.0,
    }


def test_tk_sink_smoke():
    pytest.importorskip("tkinter")
    try:
        TkSink.probe()
    except RuntimeError as error:
        pytest.skip(f"no display: {error}")
    sink = TkSink("test")
    sink.open(64, 48, 10)
    sink.write(Image.new("RGB", (64, 48), "red"))
    report = sink.close()
    assert report == {"kind": "tk", "frames": 1, "closed_by_user": False}


def test_watch_results_unchanged_with_null_sink():
    result = watch(None, 1.2, 20000, sink=NullSink())
    assert set(result) == {"frames", "simulation_seconds", "render_wall_seconds", "video", "scope"}
    assert result["frames"] == int(1.2 / 0.06)
    assert result["video"] is None and result["scope"] == "uncalibrated_simulator"


def test_headless_observer_changes_only_the_observer_block(tmp_path):
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    config = {**TINY, "checkpoint_times": []}
    observer = {"display": "none", "policy": "current", "snapshot_interval": 0.25, "fps": 200.0}
    off = run_training(tmp_path, "ppo", 2.0, 5, config, "headed-test")
    on = run_training(tmp_path, "ppo", 2.0, 5, config, "headed-test", observer=observer)
    assert "observer" not in off["configuration"] and "observer" not in off["summary"]
    on_config = dict(on["configuration"])
    block = on_config.pop("observer")
    assert on_config == off["configuration"]
    assert block["enabled"] and block["policy"] == "current" and block["fps"] == 200.0
    assert block["seed"] == DEFAULT_OBSERVER_SEED and block["learning_data"] == "excluded"

    def resolved(record):
        return [
            a["sha256"]
            for a in record["artifact_manifest"]
            if a["kind"] == "resolved_configuration"
        ]

    assert resolved(off) == resolved(on) and len(resolved(on)) == 1
    summary = on["summary"]["observer"]
    assert summary["frames_rendered"] > 0 and summary["snapshots_received"] >= 1
    assert summary["error"] is None and summary["stopped_cleanly"]
    assert summary["record_error"] is None and summary["sink_errors"] == []
    assert summary["snapshot_count"] >= 1 and summary["snapshot_copy_seconds"] > 0
    assert summary["video"] is None and summary["video_skipped_reason"] is None
    assert any(a["kind"] == "observer_episodes" for a in on["artifact_manifest"])
    names = {row["name"] for row in query_metrics(tmp_path, on["run_id"])}
    assert {
        "observer/episode_distance",
        "observer/frames_rendered",
        "snapshot_copy_seconds",
    } <= names
    assert "snapshot_count" in names
    # Training metric series are the same set with or without the observer.
    off_names = {row["name"] for row in query_metrics(tmp_path, off["run_id"])}
    assert {name for name in names if not name.startswith("observer/")} == off_names
    for record in (off, on):
        assert record["status"] == "completed"
        assert verify_run(tmp_path, record["run_id"])["valid"]


@pytest.mark.parametrize("seed", [1000, 1005, 2010, 2019, 10000, 20005, 30019])
def test_observer_seed_validation_rejects_reserved_ranges(seed):
    with pytest.raises(ValueError):
        ObserverConfig(seed=seed).validate(0)


@pytest.mark.parametrize("seed", [999, 1020, 2020, DEFAULT_OBSERVER_SEED])
def test_observer_seed_validation_accepts_unreserved_seeds(seed):
    ObserverConfig(seed=seed).validate(0)


def test_cpu_percent_between_uses_private_snapshots(monkeypatch):
    Times = namedtuple("Times", "user system idle")
    assert cpu_percent_between(Times(10, 5, 85), Times(20, 10, 170)) == pytest.approx(15.0)
    assert cpu_percent_between(Times(10, 5, 85), Times(10, 5, 85)) is None
    Linux = namedtuple("Linux", "user system idle iowait guest")
    assert cpu_percent_between(Linux(0, 0, 0, 0, 0), Linux(30, 10, 50, 10, 30)) == pytest.approx(
        40.0
    )
    if observer_module.psutil is None:
        pytest.skip("psutil not installed")

    def forbidden(*args, **kwargs):
        raise AssertionError("the observer must not touch psutil's shared cpu_percent state")

    monkeypatch.setattr(observer_module.psutil, "cpu_percent", forbidden)
    calls = []
    monkeypatch.setattr(observer_module, "gpu_sample", lambda: calls.append(1) or {})
    sampler = ResourceSampler(0.02, gpu_interval=None)
    sampler.start()
    assert wait_until(lambda: sampler.samples >= 3)
    sampler.stop()
    assert sampler.samples >= 3 and calls == [] and sampler.gpu_samples == 0
    assert set(sampler.latest) == {"cpu_percent", "gpu_percent", "vram_used_bytes"}
    paced = ResourceSampler(0.02, gpu_interval=0.05)
    paced.start()
    assert wait_until(lambda: paced.samples >= 6)
    paced.stop()
    assert 1 <= paced.gpu_samples < paced.samples


def test_record_observer_never_fails_a_finished_run(tmp_path):
    report = {
        "frames_rendered": 3,
        "fps": 10.0,
        "snapshots_received": 1,
        "video": str(tmp_path / "missing.mp4"),
        "video_skipped_reason": None,
        "episodes": [
            {
                "distance": 1.5,
                "snapshot_index": 0,
                "snapshot_elapsed": 0.5,
                "snapshot_optimizer_updates": 0,
                "snapshot_environment_steps": 0,
                "termination": "crash",
                "policy": "current",
            }
        ],
    }
    result = SimpleNamespace(
        metrics=[{"snapshot_count": 1, "snapshot_errors": 0}], environment_steps=10
    )
    with RunRecorder(tmp_path, "headed-test", {"seconds": 1.0}, algorithm="ppo") as run:
        summary = runner_module._record_observer(run, {"enabled": True}, report, result, None)
        run.finalize(observer=summary)
    record = load_run(tmp_path, run.run_id)
    assert record["status"] == "completed"
    assert summary["video"] is None
    assert summary["video_skipped_reason"].startswith("video not registered")
    assert summary["record_error"] is None and summary["snapshot_count"] == 1
    assert any(a["kind"] == "observer_episodes" for a in record["artifact_manifest"])
    assert not any(a["kind"] == "video" for a in record["artifact_manifest"])
    broken = {**report, "video": None, "episodes": [{"distance": float("nan")}]}
    with RunRecorder(tmp_path, "headed-test", {"seconds": 1.0}, algorithm="ppo") as run:
        summary = runner_module._record_observer(run, {"enabled": True}, broken, result, None)
        run.finalize(observer=summary)
    assert load_run(tmp_path, run.run_id)["status"] == "completed"
    assert summary["record_error"] and summary["video_skipped_reason"] is None


def test_observer_seed_and_algorithm_validation(tmp_path):
    with pytest.raises(ValueError):
        ObserverConfig(seed=7).validate(7)
    ObserverConfig(seed=7).validate(8)
    with pytest.raises(ValueError):
        ObserverConfig(policy="latest").validate(0)
    with pytest.raises(ValueError):
        ObserverConfig(mode="process").validate(0)
    with pytest.raises(ValueError):
        ObserverConfig.from_mapping({"unknown": 1})
    with pytest.raises(ValueError):
        run_training(tmp_path, "cem", 1.0, 0, {}, observer={"display": "none"})
    with pytest.raises(ValueError):
        run_training(tmp_path, "ppo", 1.0, 0, {}, observer={"display": "none", "seed": 0})
    with pytest.raises(ValueError):
        run_training(tmp_path, "ppo", 1.0, 0, {"snapshot_interval": 1.0})
    assert not (tmp_path / "runs").exists()


def test_cli_train_builds_observer_block(tmp_path, monkeypatch):
    from gradientclimb.benchmarks import runner
    from gradientclimb.cli import main

    calls = []

    def fake(
        root,
        algorithm,
        seconds,
        seed,
        config,
        experiment,
        benchmark_class,
        parent_run,
        observer=None,
    ):
        calls.append({"config": config, "observer": observer, "seed": seed})
        return {"run_id": "fake"}

    monkeypatch.setattr(runner, "run_training", fake)
    base = ["--root", str(tmp_path), "train", "--seconds", "1", "--seed", "4"]
    main(
        [
            *base,
            "--headed",
            "--observer-policy",
            "best",
            "--observer-interval",
            "2",
            "--observer-fps",
            "30",
        ]
    )
    assert calls[-1]["observer"] == {
        "mode": "thread",
        "display": "window",
        "policy": "best",
        "snapshot_interval": 2.0,
        "seed": 41000,
        "fps": 30.0,
        "video": None,
    }
    video = tmp_path / "live.mp4"
    main([*base, "--headless-record", "--observer-video", str(video), "--observer-seed", "5"])
    assert calls[-1]["observer"]["display"] == "none"
    assert calls[-1]["observer"]["video"] == str(video.resolve())
    assert calls[-1]["observer"]["seed"] == 5
    main(base)
    assert calls[-1]["observer"] is None
    assert all(call["config"] == {"num_envs": 64, "device": "cpu"} for call in calls)
    with pytest.raises(SystemExit):
        main([*base, "--headed", "--headless-record"])


@pytest.mark.parametrize(
    "module", ["gradientclimb.visualization.sinks", "gradientclimb.visualization.observer"]
)
def test_sinks_module_does_not_import_torch(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import sys, {module}; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
