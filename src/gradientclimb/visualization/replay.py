from __future__ import annotations

import time
from pathlib import Path

from .sinks import FfmpegSink, FrameSink, TkSink


def watch(checkpoint=None, seconds=30, seed=20000, video=None, sink: FrameSink | None = None):
    """Render just one environment, live in Tk or as a local FFmpeg video.

    No game assets are used. Simulation time and rendering time are distinct.
    This is a selected policy rollout, not a recording of training collection.
    ``sink`` overrides the default Tk/FFmpeg choice (for example ``NullSink`` in
    headless tests); the returned record is identical either way.
    """
    import torch

    from gradientclimb.algorithms import RandomPolicy, load_policy
    from gradientclimb.environments import environment_from_config

    if not 0 < seconds <= 600:
        raise ValueError("Visualization duration must be in (0,600] simulation seconds")
    torch.set_num_threads(1)
    policy = load_policy(checkpoint) if checkpoint else RandomPolicy(seed)
    env = environment_from_config(policy.config, num_envs=1, seed=seed)
    obs, _ = env.reset(seed)
    frames = int(seconds / env.action_duration)
    if sink is None:
        sink = FfmpegSink(video) if video else TkSink("GradientClimb | selected surrogate rollout")
    sink.open(960, 540, 1 / env.action_duration)
    start = time.perf_counter()
    count = 0
    try:
        for _ in range(frames):
            action = policy.act(obs)
            obs, *_ = env.step(action)
            sink.write(env.render())
            if sink.paced:
                time.sleep(env.action_duration)
            count += 1
    finally:
        report = sink.close()
    return {
        "frames": count,
        "simulation_seconds": count * env.action_duration,
        "render_wall_seconds": time.perf_counter() - start,
        "video": str(Path(video)) if video else report.get("video"),
        "scope": env.environment_spec.evidence_domain,
        "simulator_version": env.environment_spec.environment_id,
        "distance_unit": env.environment_spec.distance_unit,
        "scenario": env.scenario.model_dump(mode="json"),
        "scenario_hash": env.scenario.sha256,
        "environment_config": env.config,
        "qualification": False,
    }
