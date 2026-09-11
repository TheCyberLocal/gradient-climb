from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


def watch(checkpoint=None, seconds=30, seed=20000, video=None):
    """Render just one environment, live in Tk or as a local FFmpeg video.

    No game assets are used. Simulation time and rendering time are distinct.
    This is a selected policy rollout, not a recording of training collection.
    """
    import torch

    from gradientclimb.algorithms import RandomPolicy, load_policy
    from gradientclimb.simulation import VectorHillEnv

    if not 0 < seconds <= 600:
        raise ValueError("Visualization duration must be in (0,600] simulation seconds")
    torch.set_num_threads(1)
    policy = load_policy(checkpoint) if checkpoint else RandomPolicy(seed)
    env = VectorHillEnv(1, seed, stack=policy.config.get("stack", 4))
    obs, _ = env.reset(seed)
    frames = int(seconds / env.action_duration)
    process = None
    window = None
    if video:
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("FFmpeg is required for --video")
        video = Path(video)
        if video.exists():
            raise FileExistsError("Refusing to replace an existing research video")
        video.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [
                executable,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x540",
                "-r",
                str(1 / env.action_duration),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        import tkinter as tk

        from PIL import ImageTk

        window = tk.Tk()
        window.title("GradientClimb | selected surrogate rollout")
        label = tk.Label(window)
        label.pack()
    start = time.perf_counter()
    count = 0
    try:
        for _ in range(frames):
            action = policy.act(obs)
            obs, *_ = env.step(action)
            frame = env.render()
            if process:
                process.stdin.write(frame.convert("RGB").tobytes())
            else:
                photo = ImageTk.PhotoImage(frame)
                label.configure(image=photo)
                label.image = photo
                window.update()
                time.sleep(env.action_duration)
            count += 1
    finally:
        if process:
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed: {error}")
        if window:
            window.destroy()
    return {
        "frames": count,
        "simulation_seconds": count * env.action_duration,
        "render_wall_seconds": time.perf_counter() - start,
        "video": str(video) if video else None,
        "scope": "uncalibrated_simulator",
    }
