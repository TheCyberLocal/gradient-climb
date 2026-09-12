"""Immutable, content-addressed scenario contracts; no driving decision rules.

Profiles describe compatible mechanisms and observations. Their presence does
not establish calibration, game access, or implementation of those mechanisms.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class EvidenceReference(ImmutableRecord):
    source: str = Field(min_length=1)
    kind: Literal["measured", "inferred", "synthetic", "unknown"]
    sha256_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def measured_bytes_are_pinned(self):
        if self.kind == "measured" and self.sha256_digest is None:
            raise ValueError("Measured evidence requires a source artifact SHA-256")
        return self


class ParameterEstimate(ImmutableRecord):
    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    lower: float | None = None
    upper: float | None = None
    evidence: EvidenceReference

    @model_validator(mode="after")
    def coherent_uncertainty(self):
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Uncertainty requires both bounds or neither")
        if self.lower is not None and not self.lower <= self.value <= self.upper:
            raise ValueError("Parameter value must lie within its uncertainty bounds")
        return self


class Profile(ImmutableRecord):
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    game_build: str = Field(min_length=1)
    provenance: tuple[EvidenceReference, ...] = Field(min_length=1)
    rendering_assets: tuple[EvidenceReference, ...] = ()
    compatible_checkpoint_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def checkpoint_hashes(self):
        for value in self.compatible_checkpoint_hashes:
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Compatible checkpoints must be identified by SHA-256")
        return self


class Body(ImmutableRecord):
    body_id: str = Field(min_length=1)
    role: str = Field(min_length=1)


class Joint(ImmutableRecord):
    joint_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    body_a: str = Field(min_length=1)
    body_b: str = Field(min_length=1)


class VehicleProfile(Profile):
    bodies: tuple[Body, ...] = Field(min_length=1)
    joints: tuple[Joint, ...] = ()
    mechanisms: tuple[str, ...] = ()
    required_input_channels: tuple[str, ...] = ()
    parameters: tuple[ParameterEstimate, ...] = ()

    @model_validator(mode="after")
    def connected_topology(self):
        ids = [body.body_id for body in self.bodies]
        if len(set(ids)) != len(ids):
            raise ValueError("Body IDs must be unique")
        if len({joint.joint_id for joint in self.joints}) != len(self.joints):
            raise ValueError("Joint IDs must be unique")
        for joint in self.joints:
            if joint.body_a not in ids or joint.body_b not in ids or joint.body_a == joint.body_b:
                raise ValueError("Joint endpoints must name two distinct profile bodies")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("Parameter names must be unique")
        return self


class UpgradeLevel(ImmutableRecord):
    name: str = Field(min_length=1)
    observed_level: int | None = Field(default=None, ge=0)
    observed_maximum: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def level_within_observed_maximum(self):
        if (
            self.observed_level is not None
            and self.observed_maximum is not None
            and self.observed_level > self.observed_maximum
        ):
            raise ValueError("Upgrade level exceeds its observed maximum")
        return self


class UpgradeProfile(Profile):
    vehicle_profile_id: str = Field(min_length=1)
    levels: tuple[UpgradeLevel, ...] = ()

    @model_validator(mode="after")
    def unique_levels(self):
        if len({level.name for level in self.levels}) != len(self.levels):
            raise ValueError("Upgrade names must be unique")
        return self


class MapProfile(Profile):
    geometry_kind: Literal["procedural_heightfield", "collidable_structures"]
    distance_unit: Literal["surrogate_unit", "game_metre", "camera_compensated_pixel"]
    meter_definition: str = Field(min_length=1)
    measured_coverage: tuple[tuple[float, float], ...] = ()
    geometry_artifacts: tuple[EvidenceReference, ...] = ()

    @model_validator(mode="after")
    def coverage_requires_measurement(self):
        for start, end in self.measured_coverage:
            if start < 0 or end <= start:
                raise ValueError("Measured coverage must contain increasing nonnegative bounds")
        if self.measured_coverage and not any(
            e.kind == "measured" for e in self.geometry_artifacts
        ):
            raise ValueError("Measured coverage requires pinned measured geometry evidence")
        return self


class ObservationProfile(Profile):
    source: Literal["rendered_pixels", "extracted_screen_features", "idealized_simulator_state"]
    schema_id: str = Field(min_length=1)
    feature_names: tuple[str, ...] = ()
    validity_masks: bool
    history_frames: int = Field(ge=1)
    decision_seconds: float = Field(gt=0)
    preprocessing_version: str = Field(min_length=1)
    deployable_from_game: bool

    @model_validator(mode="after")
    def actor_access_is_explicit(self):
        if self.source == "idealized_simulator_state" and self.deployable_from_game:
            raise ValueError("Idealized simulator state cannot be declared a game observation")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Observation feature names must be unique")
        return self


class ControlProfile(Profile):
    input_channels: tuple[str, ...] = Field(min_length=1)
    joint_states: tuple[tuple[bool, ...], ...] = Field(min_length=1)
    timing: Literal["fixed_decision", "event_timed"]
    ordered_transitions: bool

    @model_validator(mode="after")
    def actions_match_channels(self):
        if len(set(self.input_channels)) != len(self.input_channels):
            raise ValueError("Input channels must be unique")
        if any(len(state) != len(self.input_channels) for state in self.joint_states):
            raise ValueError("Joint states must cover every independent input channel")
        if len(set(self.joint_states)) != len(self.joint_states):
            raise ValueError("Joint states must be unique")
        if tuple(False for _ in self.input_channels) not in self.joint_states:
            raise ValueError("Controls require an all-released state")
        return self


class Scenario(ImmutableRecord):
    scenario_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    vehicle: VehicleProfile
    upgrades: UpgradeProfile
    map: MapProfile
    observation: ObservationProfile
    controls: ControlProfile
    randomization: bool
    calibration_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def components_are_compatible(self):
        components = (self.vehicle, self.upgrades, self.map, self.observation, self.controls)
        if len({part.game_build for part in components}) != 1:
            raise ValueError("Scenario profiles must refer to the same game build")
        if self.upgrades.vehicle_profile_id != self.vehicle.profile_id:
            raise ValueError("Upgrade profile is for a different vehicle")
        if not set(self.vehicle.required_input_channels) <= set(self.controls.input_channels):
            raise ValueError("Control profile lacks required vehicle input channels")
        return self
