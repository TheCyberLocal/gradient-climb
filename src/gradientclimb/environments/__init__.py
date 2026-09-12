"""Versioned environment entry points, preserving the legacy constructor."""

from gradientclimb.simulation import VectorHillEnv

from .profiles import (
    ControlProfile,
    MapProfile,
    ObservationProfile,
    Scenario,
    UpgradeProfile,
    VehicleProfile,
)
from .registry import (
    ENVIRONMENTS,
    LEGACY_ENVIRONMENT_ID,
    RANDOMIZED_ENVIRONMENT_ID,
    REFERENCE_ENVIRONMENT_ID,
    EnvironmentUnavailableError,
    environment_from_config,
    environment_spec,
    make_environment,
)

__all__ = [
    "ENVIRONMENTS",
    "LEGACY_ENVIRONMENT_ID",
    "RANDOMIZED_ENVIRONMENT_ID",
    "REFERENCE_ENVIRONMENT_ID",
    "ControlProfile",
    "EnvironmentUnavailableError",
    "MapProfile",
    "ObservationProfile",
    "Scenario",
    "UpgradeProfile",
    "VectorHillEnv",
    "VehicleProfile",
    "environment_from_config",
    "environment_spec",
    "make_environment",
]
