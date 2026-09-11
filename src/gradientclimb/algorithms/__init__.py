"""Public algorithm APIs without eagerly initializing optional runtimes."""

from importlib import import_module

__all__ = [
    "ActorCritic",
    "AlwaysGasPolicy",
    "LinearPolicy",
    "RandomPolicy",
    "TrainingResult",
    "load_policy",
    "train_cem",
    "train_ppo",
]

_EXPORT_MODULES = {
    "ActorCritic": "ppo",
    "AlwaysGasPolicy": "policies",
    "LinearPolicy": "policies",
    "RandomPolicy": "policies",
    "TrainingResult": "ppo",
    "load_policy": "policies",
    "train_cem": "cem",
    "train_ppo": "ppo",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
