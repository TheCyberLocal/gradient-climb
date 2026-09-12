"""Atomic publication and explicit continuation capabilities for local checkpoints."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def continuation_manifest(algorithm: str, training_state: dict) -> dict:
    return {
        "manifest_version": "1.0.0",
        "continuation": "warm_start" if algorithm == "ppo" else "inference_only",
        "exact_resume": False,
        "model_weights": "stored",
        "optimizer": "stored" if "optimizer" in training_state else "unavailable",
        "normalization": "fixed_observation_transform; no learned normalization state",
        "recurrent_state": "not_applicable; feed_forward_actor",
        "torch_rng": "stored_not_restored" if "torch_rng" in training_state else "unavailable",
        "cuda_rng": "stored_not_restored" if "cuda_rng" in training_state else "unavailable",
        "sampler_rng": "stored_not_restored" if "sampler_rng" in training_state else "unavailable",
        "environment_state": "not_stored; reset_on_warm_start",
        "rollout_state": "not_stored; discard_on_warm_start",
        "counter_semantics": "completed_operations_at_checkpoint; new_run_counters_restart",
    }


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Keep the previous complete checkpoint if serialization or validation fails."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(temporary, map_location="cpu", weights_only=True)
        if restored.get("format_version") != payload["format_version"]:
            raise ValueError("Checkpoint readback failed")
        os.replace(temporary, path)
    except BaseException as error:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:  # noqa: BLE001 - retain the primary publication failure
            error.add_note(
                f"Checkpoint temporary cleanup failed: {type(cleanup).__name__}: {cleanup}"
            )
        raise
