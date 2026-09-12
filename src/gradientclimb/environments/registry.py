"""Explicit simulator generations and adapters around the frozen legacy engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv
from gradientclimb.simulation.hill import FEATURE_NAMES, VEHICLES

from .profiles import (
    Body,
    ControlProfile,
    EvidenceReference,
    MapProfile,
    ObservationProfile,
    ParameterEstimate,
    Scenario,
    UpgradeProfile,
    VehicleProfile,
)

LEGACY_ENVIRONMENT_ID = SIMULATOR_VERSION
REFERENCE_ENVIRONMENT_ID = "reference-0.1.0"
RANDOMIZED_ENVIRONMENT_ID = "reference-randomized-0.1.0"


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    available: bool
    evidence_domain: str
    distance_unit: str
    observation_source: str
    prerequisite: str | None = None


ENVIRONMENTS = MappingProxyType(
    {
        LEGACY_ENVIRONMENT_ID: EnvironmentSpec(
            LEGACY_ENVIRONMENT_ID,
            True,
            "uncalibrated_simulator",
            "surrogate_unit",
            "idealized_simulator_state",
        ),
        REFERENCE_ENVIRONMENT_ID: EnvironmentSpec(
            REFERENCE_ENVIRONMENT_ID,
            False,
            "reference_simulator",
            "game_metre",
            "rendered_pixels",
            "Requires an implemented physics engine, measured scenarios and a visual contract",
        ),
        RANDOMIZED_ENVIRONMENT_ID: EnvironmentSpec(
            RANDOMIZED_ENVIRONMENT_ID,
            False,
            "randomized_reference_simulator",
            "game_metre",
            "rendered_pixels",
            "Requires a reference simulator and declared measurement-derived uncertainty",
        ),
    }
)


class EnvironmentUnavailableError(NotImplementedError):
    """A reserved research generation has not been implemented or made available."""


def environment_spec(environment_id: str = LEGACY_ENVIRONMENT_ID) -> EnvironmentSpec:
    try:
        return ENVIRONMENTS[environment_id]
    except KeyError as error:
        raise ValueError(f"Unknown environment generation: {environment_id}") from error


def _legacy_scenario(env: VectorHillEnv) -> Scenario:
    evidence = EvidenceReference(
        source="src/gradientclimb/simulation/hill.py@surrogate-0.1.0",
        kind="synthetic",
        note="Historical uncalibrated numerical assumptions; no measured proprietary constants",
    )
    common = {
        "version": LEGACY_ENVIRONMENT_ID,
        "game_build": "not-applicable:synthetic",
        "provenance": (evidence,),
    }
    vehicle_id = f"legacy:{env.profile}"
    return Scenario(
        scenario_id=f"legacy:{env.profile}/{env.terrain}",
        version="scenario-contract-3.0",
        environment_id=LEGACY_ENVIRONMENT_ID,
        vehicle=VehicleProfile(
            profile_id=vehicle_id,
            **common,
            bodies=(Body(body_id="chassis", role="rigid_body_with_two_virtual_contacts"),),
            mechanisms=("two_virtual_suspension_contacts", "airborne_pedal_torque"),
            required_input_channels=("gas", "brake"),
            parameters=tuple(
                ParameterEstimate(
                    name=name,
                    value=value,
                    unit="legacy_nominal_coefficient",
                    lower=value * 0.8 if env.randomization else None,
                    upper=value * 1.2 if env.randomization else None,
                    evidence=evidence,
                )
                for name, value in vars(VEHICLES[env.profile]).items()
            ),
        ),
        upgrades=UpgradeProfile(
            profile_id=f"legacy:{env.profile}:no-upgrades", vehicle_profile_id=vehicle_id, **common
        ),
        map=MapProfile(
            profile_id=f"legacy:{env.terrain}",
            **common,
            geometry_kind="procedural_heightfield",
            distance_unit="surrogate_unit",
            meter_definition="Nominal simulator x displacement; not calibrated real-game metres",
        ),
        observation=ObservationProfile(
            profile_id=f"legacy:state-stack-{env.stack}",
            **common,
            source="idealized_simulator_state",
            schema_id=f"surrogate-24-feature-stack-{env.stack}",
            feature_names=FEATURE_NAMES,
            validity_masks=False,
            history_frames=env.stack,
            decision_seconds=env.action_duration,
            preprocessing_version="surrogate-state-scaling-0.1.0",
            deployable_from_game=False,
        ),
        controls=ControlProfile(
            profile_id="legacy:independent-pedals",
            **common,
            input_channels=("gas", "brake"),
            joint_states=((False, False), (True, False), (False, True), (True, True)),
            timing="fixed_decision",
            ordered_transitions=True,
        ),
        randomization=env.randomization,
        calibration_version="uncalibrated",
    )


def make_environment(
    environment_id: str = LEGACY_ENVIRONMENT_ID,
    *,
    num_envs: int = 64,
    seed: int = 0,
    profile: str = "default",
    terrain: str = "train",
    randomization: bool = False,
    stack: int = 4,
    max_steps: int = 1000,
    dt: float = 0.02,
    substeps: int = 3,
) -> VectorHillEnv:
    """Construct only an available engine, retaining legacy numerical behavior.

    Reference IDs are reserved, never aliases for the old surrogate. Metadata
    is attached separately so historical ``env.config`` values remain intact.
    """
    spec = environment_spec(environment_id)
    if not spec.available:
        raise EnvironmentUnavailableError(f"{environment_id}: {spec.prerequisite}")
    env = VectorHillEnv(
        num_envs, seed, profile, terrain, randomization, stack, max_steps, dt, substeps
    )
    env.environment_spec = spec
    env.scenario = _legacy_scenario(env)
    return env


def environment_from_config(config: Mapping[str, Any], **overrides: Any) -> VectorHillEnv:
    """Resolve checkpoint/training scenario settings with explicit caller overrides.

    Missing generation means the historical format, whose only simulator was
    surrogate-0.1.0. A declared unknown or contradictory generation is rejected.
    Learner-only settings are ignored; unknown override keys raise TypeError.
    """
    environment_id = config.get("environment_id", config.get("simulator_version"))
    if environment_id is None:
        environment_id = LEGACY_ENVIRONMENT_ID
    simulator_version = config.get("simulator_version")
    if simulator_version is not None and simulator_version != environment_id:
        raise ValueError("Environment ID and simulator version disagree")
    keys = (
        "num_envs",
        "seed",
        "profile",
        "terrain",
        "randomization",
        "stack",
        "max_steps",
        "dt",
        "substeps",
    )
    options = {key: config[key] for key in keys if key in config}
    options.update(overrides)
    spec = environment_spec(environment_id)
    if config.get("observation_source") not in (None, spec.observation_source):
        raise ValueError("Checkpoint observation source is incompatible with this environment")
    if environment_id == LEGACY_ENVIRONMENT_ID and config.get("calibration_version") not in (
        None,
        "uncalibrated",
    ):
        raise ValueError("The legacy surrogate cannot claim measured calibration")
    return make_environment(environment_id, **options)
