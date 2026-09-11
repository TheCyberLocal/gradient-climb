"""Tiny controlled experiments that exercise the entire record lifecycle."""

from __future__ import annotations

import random
from pathlib import Path

from .recorder import RunRecorder, load_run


def run_synthetic(root: Path, seed: int = 0, gain: float = 0.2, steps: int = 40) -> dict:
    """Fit a scalar target by noisy gradient updates; gain changes learning speed.

    This is an infrastructure demonstration, never evidence of game competence.
    """
    if not 0 < gain < 1 or steps < 1:
        raise ValueError("gain must be between 0 and 1 and steps must be positive")
    rng = random.Random(seed)
    config = {"gain": gain, "steps": steps, "target": 1.0, "noise_sigma": 0.01}
    with RunRecorder(
        root,
        "synthetic-convergence",
        config,
        seed,
        "scalar-gradient",
        "synthetic-quadratic",
        policy_architecture={"parameters": 1},
        simulator_version="none",
        telemetry_interval_seconds=0,
    ) as run:
        estimate = 0.0
        for step in range(steps):
            error = 1.0 - estimate
            estimate += gain * error + rng.gauss(0, 0.01)
            run.metric("loss", (1.0 - estimate) ** 2, step=step)
            run.metric("quality", 1.0 - abs(1.0 - estimate), step=step)
        checkpoint = run.directory / "scalar-checkpoint.json"
        checkpoint.write_text(f'{{"estimate": {estimate!r}, "seed": {seed}}}\n', encoding="utf-8")
        artifact = run.register_artifact(
            checkpoint, "checkpoint", {"format": "json", "parameters": 1}
        )
        run.evaluation(
            {
                "results": {
                    "absolute_error": abs(1.0 - estimate),
                    "quality": 1.0 - abs(1.0 - estimate),
                },
                "seed": seed,
                "protocol": "deterministic scalar holdout",
            }
        )
        run.finalize(
            training_steps=steps,
            environment_steps=steps,
            episodes=1,
            optimizer_updates=steps,
            checkpoint_hash=artifact["sha256"],
            final_quality=1.0 - abs(1.0 - estimate),
            demonstration_only=True,
        )
    return load_run(root, run.run_id)
