"""Small policies with a shared batched observation -> joint action contract."""

from pathlib import Path

import numpy as np


class RandomPolicy:
    """Uniform distribution over all four pedal states."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.config = {"algorithm": "random", "seed": seed}

    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.rng.integers(0, 4, len(observations))


class AlwaysGasPolicy:
    """Explicitly scripted sanity baseline; never presented as a learned agent."""

    def __init__(self):
        self.config = {"algorithm": "always_gas"}

    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.ones(len(observations), dtype=np.int64)


class LinearPolicy:
    """Joint categorical linear controller optimized by cross-entropy search."""

    def __init__(self, weights: np.ndarray, config: dict | None = None):
        self.weights = np.asarray(weights, dtype=np.float32)
        if self.weights.ndim != 2 or self.weights.shape[1] != 4:
            raise ValueError("Linear weights must have four output states")
        self.config = config or {"algorithm": "cem"}

    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        width = len(self.weights) - 1
        return np.argmax(observations[:, -width:] @ self.weights[:-1] + self.weights[-1], axis=-1)

    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "algorithm": "cem",
                "config": self.config,
                "weights": torch.from_numpy(self.weights),
            },
            path,
        )


def load_policy(path: str | Path, device: str = "cpu"):
    """Load our tensor-only checkpoint format using PyTorch's restricted loader."""
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported GradientClimb checkpoint version")
    if checkpoint["algorithm"] == "cem":
        return LinearPolicy(checkpoint["weights"].cpu().numpy(), checkpoint["config"])
    if checkpoint["algorithm"] == "ppo":
        from .ppo import ActorCritic

        model = ActorCritic(checkpoint["observation_dim"], checkpoint["hidden_size"], device=device)
        model.load_state_dict(checkpoint["model_state"])
        model.config = checkpoint["config"]
        model.training_state = checkpoint.get("training_state", {})
        model.eval()
        return model
    raise ValueError(f"Unknown algorithm: {checkpoint['algorithm']}")
