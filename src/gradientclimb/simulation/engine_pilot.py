"""Synthetic articulated-engine screen; not a registered training environment.

All geometry and parameters below are original engineering fixtures, not game
measurements. Box2D is optional and imported only when a fixture is constructed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

ENGINE_DISTRIBUTION = "Box2D"
ENGINE_VERSION = "2.3.10"
FIXTURE_VERSION = "articulated-engine-fixtures-3.1"
GROUND_ENVELOPE = {"speed_bound": 100.0, "horizon_seconds": 10.0, "margin": 20.0}


def ground_extent() -> tuple[float, float]:
    """Synthetic support covers the declared speed/horizon plus starting-body margin."""
    extent = GROUND_ENVELOPE["speed_bound"] * GROUND_ENVELOPE["horizon_seconds"]
    extent += GROUND_ENVELOPE["margin"]
    return -extent, extent


@dataclass
class GroundSupportDiagnostics:
    """Observe support coverage separately from penetration at every physics substep.

    Static floor exists where a wheel centre projects onto an actual segment.
    The outer-domain check includes the full horizontal wheel radius. A bridge
    gap is an explicit region, never fabricated static floor or missing data.
    """

    segments: list
    radius: float = 0.3
    checked_physics_substeps: int = 0
    outside_ground_domain_substeps: int = 0
    bridge_gap_substeps: int = 0
    first_ground_domain_exit: dict | None = None
    first_bridge_gap_entry: dict | None = None
    minima: dict = field(default_factory=dict)

    def observe(self, wheels, *, decision: int, physics_substep: int, simulated_seconds: float):
        lower, upper = ground_extent()
        outside = gap = False
        for index, (x, y) in enumerate(wheels):
            over_floor = any(first[0] <= x <= second[0] for first, second in self.segments)
            beyond_domain = x - self.radius < lower or x + self.radius > upper
            in_gap = not over_floor and lower <= x <= upper
            event = {
                "decision": decision,
                "physics_substep": physics_substep,
                "simulated_seconds": simulated_seconds,
                "wheel_index": index,
                "wheel_center": [float(x), float(y)],
                "wheel_bottom_y": float(y - self.radius),
            }
            regions = ["global"]
            if over_floor:
                regions.append("static_floor")
            if in_gap:
                regions.append("bridge_gap")
                gap = True
                if self.first_bridge_gap_entry is None:
                    self.first_bridge_gap_entry = event
            if beyond_domain:
                outside = True
                if self.first_ground_domain_exit is None:
                    self.first_ground_domain_exit = event
            for region in regions:
                if region not in self.minima or (
                    event["wheel_bottom_y"] < self.minima[region]["wheel_bottom_y"]
                ):
                    self.minima[region] = event
        self.checked_physics_substeps += 1
        self.outside_ground_domain_substeps += int(outside)
        self.bridge_gap_substeps += int(gap)

    def to_record(self) -> dict:
        def minimum(region):
            return self.minima.get(region, {}).get("wheel_bottom_y")

        return {
            "ground_domain_x": list(ground_extent()),
            "static_ground_segments": self.segments,
            "ground_envelope": dict(GROUND_ENVELOPE),
            "support_observation_cadence": "every_physics_substep",
            "checked_physics_substeps": self.checked_physics_substeps,
            "outside_ground_domain_substeps": self.outside_ground_domain_substeps,
            "ground_domain_contained": self.outside_ground_domain_substeps == 0
            if self.checked_physics_substeps
            else None,
            "first_ground_domain_exit": self.first_ground_domain_exit,
            "minimum_wheel_bottom_y": minimum("global"),
            "minimum_wheel_bottom_evidence": self.minima.get("global"),
            "minimum_wheel_bottom_over_static_floor_y": minimum("static_floor"),
            "minimum_wheel_bottom_over_static_floor_evidence": self.minima.get("static_floor"),
            "bridge_gap_substeps": self.bridge_gap_substeps,
            "first_bridge_gap_entry": self.first_bridge_gap_entry,
            "minimum_wheel_bottom_in_bridge_gap_y": minimum("bridge_gap"),
            "minimum_wheel_bottom_in_bridge_gap_evidence": self.minima.get("bridge_gap"),
        }


@dataclass(frozen=True)
class SolverSettings:
    decision_seconds: float = 1 / 60
    substeps: int = 2
    velocity_iterations: int = 8
    position_iterations: int = 3

    def __post_init__(self):
        if not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0:
            raise ValueError("Decision duration must be positive and finite")
        if any(
            type(value) is not int or value < 1
            for value in (self.substeps, self.velocity_iterations, self.position_iterations)
        ):
            raise ValueError("Substeps and solver iteration counts must be positive integers")


def engine_identity() -> dict:
    """Validate the installed binding and retain actual binary/license hashes."""
    from gradientclimb.artifacts import sha256_file

    distribution = importlib.metadata.distribution(ENGINE_DISTRIBUTION)
    if distribution.version != ENGINE_VERSION:
        raise RuntimeError(f"Engine pilot requires {ENGINE_DISTRIBUTION}=={ENGINE_VERSION}")
    files = []
    for relative in distribution.files or []:
        name = str(relative)
        if name.lower().endswith((".pyd", ".so", "license", "license.txt", "copying")):
            path = distribution.locate_file(relative)
            if path.is_file():
                files.append(
                    {"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
                )
    return {
        "distribution": ENGINE_DISTRIBUTION,
        "version": distribution.version,
        "engine_family": "Box2D 2.3 (SWIG binding; not the Box2D 3.x API)",
        "license_metadata": distribution.metadata.get("License"),
        "installed_files": files,
    }


class ArticulatedFixture:
    """One chassis, two independently rotating wheels, and optional moving planks."""

    def __init__(self, kind: str = "flat", seed: int = 0, settings: SolverSettings | None = None):
        if kind not in {"flat", "bridge_load", "bridge_traverse"}:
            raise ValueError("Unknown synthetic fixture")
        if type(seed) is not int or seed < 0:
            raise ValueError("Fixture seed must be a nonnegative integer")
        # Verify a pin before importing an optional native extension.
        if importlib.metadata.version(ENGINE_DISTRIBUTION) != ENGINE_VERSION:
            raise RuntimeError(f"Install {ENGINE_DISTRIBUTION}=={ENGINE_VERSION} for this pilot")
        from Box2D import b2World

        self.kind, self.seed = kind, seed
        self.settings = settings or SolverSettings()
        self.world = b2World(gravity=(0, -9.81), doSleep=True)
        self.ground = self.world.CreateStaticBody(userData="ground")
        self.planks, self.bridge_joints, self.wheel_joints = [], [], []
        self.render_polygons, self.render_circles = [], []
        self.surface_segments = []
        lower, upper = ground_extent()
        if kind == "flat":
            segments = [((lower, 0), (upper, 0))]
        else:
            segments = [((lower, 0), (6, 0)), ((14, 0), (upper, 0))]
            self._bridge()
        self.ground_support = GroundSupportDiagnostics(segments)
        for first, second in segments:
            self.ground.CreateEdgeFixture(vertices=(first, second), friction=0.9)
            self.surface_segments.append((first, second))
        rng = np.random.default_rng(seed)
        start_x = 10 if kind == "bridge_load" else 0
        start_y = 0.95 if kind == "bridge_load" else 1.4
        self.chassis = self.world.CreateDynamicBody(
            position=(start_x, start_y),
            angle=float(rng.uniform(-0.005, 0.005)),
            userData="chassis",
            bullet=True,
        )
        vertices = [(-0.95, -0.2), (0.95, -0.2), (0.7, 0.25), (-0.7, 0.25)]
        self.chassis.CreatePolygonFixture(vertices=vertices, density=2.0, friction=0.5)
        self.render_polygons.append((self.chassis, vertices, "#da945d"))
        self.wheels = []
        for offset in (-0.72, 0.72):
            wheel = self.world.CreateDynamicBody(
                position=(start_x + offset, start_y - 0.8),
                userData="wheel",
                bullet=True,
            )
            wheel.CreateCircleFixture(radius=0.3, density=1.0, friction=0.9, restitution=0)
            joint = self.world.CreateWheelJoint(
                bodyA=self.chassis,
                bodyB=wheel,
                anchor=wheel.position,
                axis=(0, 1),
                frequencyHz=4.0,
                dampingRatio=0.8,
                enableMotor=False,
                motorSpeed=0,
                maxMotorTorque=12.0,
                collideConnected=False,
            )
            self.wheels.append(wheel)
            self.wheel_joints.append(joint)
            self.render_circles.append((wheel, 0.3))
        self.decisions = 0
        self.peak_joint_error = self.peak_lateral_wheel_error = 0.0
        self.bridge_contact_steps = 0
        self.max_speed = self.max_abs_angular_speed = 0.0
        self.bridge_minimum_y = 0.0
        self.state_digest = hashlib.sha256()

    def _bridge(self):
        nodes = [(6 + i * 8 / 12, -0.4 * math.sin(math.pi * i / 12)) for i in range(13)]
        previous = self.ground
        for first, second in pairwise(nodes):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            plank = self.world.CreateDynamicBody(
                position=((first[0] + second[0]) / 2, (first[1] + second[1]) / 2 - 0.06),
                angle=math.atan2(dy, dx),
                angularDamping=0.4,
                userData="plank",
            )
            vertices = [
                (-length / 2, -0.06),
                (length / 2, -0.06),
                (length / 2, 0.06),
                (-length / 2, 0.06),
            ]
            plank.CreatePolygonFixture(vertices=vertices, density=2.0, friction=0.9)
            joint = self.world.CreateRevoluteJoint(
                bodyA=previous,
                bodyB=plank,
                anchor=(first[0], first[1] - 0.06),
                collideConnected=False,
            )
            self.bridge_joints.append(joint)
            self.planks.append(plank)
            self.render_polygons.append((plank, vertices, "#bfac83"))
            previous = plank
        self.bridge_joints.append(
            self.world.CreateRevoluteJoint(
                bodyA=previous,
                bodyB=self.ground,
                anchor=(14, -0.06),
                collideConnected=False,
            )
        )

    def state(self) -> np.ndarray:
        return np.array(
            [
                [
                    body.position.x,
                    body.position.y,
                    body.angle,
                    body.linearVelocity.x,
                    body.linearVelocity.y,
                    body.angularVelocity,
                ]
                for body in (self.chassis, *self.wheels, *self.planks)
            ],
            dtype=np.float64,
        )

    def step(self, action: int) -> None:
        if type(action) is not int or action not in range(4):
            raise ValueError("Fixture actions require the four joint pedal states")
        gas, brake = bool(action & 1), bool(action & 2)
        settings = self.settings
        for substep_index in range(settings.substeps):
            for wheel, joint in zip(self.wheels, self.wheel_joints, strict=True):
                # Separate drive torque and zero-speed brake motor act together.
                # The synthetic torque curve is not a game calibration.
                drive = -float(gas) * 3.0 * max(0.0, 1 - abs(wheel.angularVelocity) / 60)
                wheel.ApplyTorque(drive, True)
                self.chassis.ApplyTorque(-drive, True)
                joint.motorEnabled = brake
            self.world.Step(
                settings.decision_seconds / settings.substeps,
                settings.velocity_iterations,
                settings.position_iterations,
            )
            self.world.ClearForces()
            physics_substep = self.decisions * settings.substeps + substep_index + 1
            self.ground_support.observe(
                [(wheel.position.x, wheel.position.y) for wheel in self.wheels],
                decision=self.decisions + 1,
                physics_substep=physics_substep,
                simulated_seconds=physics_substep * settings.decision_seconds / settings.substeps,
            )
        self.decisions += 1
        state = self.state()
        if not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite articulated fixture state")
        self.state_digest.update(state.astype("<f8").tobytes())
        self.max_speed = max(self.max_speed, float(np.linalg.norm(state[:, 3:5], axis=1).max()))
        self.max_abs_angular_speed = max(self.max_abs_angular_speed, float(abs(state[:, 5]).max()))
        cs, sn = math.cos(self.chassis.angle), math.sin(self.chassis.angle)
        for joint in self.wheel_joints:
            delta = joint.anchorB - joint.anchorA
            self.peak_lateral_wheel_error = max(
                self.peak_lateral_wheel_error, abs(delta.x * cs + delta.y * sn)
            )
        for joint in self.bridge_joints:
            delta = joint.anchorB - joint.anchorA
            self.peak_joint_error = max(self.peak_joint_error, math.hypot(delta.x, delta.y))
        if self.planks:
            self.bridge_minimum_y = min(self.bridge_minimum_y, *(b.position.y for b in self.planks))
            if any(
                contact.touching
                and {contact.fixtureA.body.userData, contact.fixtureB.body.userData}
                == {"wheel", "plank"}
                for contact in self.world.contacts
            ):
                self.bridge_contact_steps += 1

    def diagnostics(self) -> dict:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "decisions": self.decisions,
            "simulated_seconds": self.decisions * self.settings.decision_seconds,
            "final_state": self.state().tolist(),
            "state_sha256": self.state_digest.hexdigest(),
            "peak_bridge_joint_anchor_error": self.peak_joint_error,
            "peak_wheel_lateral_constraint_error": self.peak_lateral_wheel_error,
            **self.ground_support.to_record(),
            "bridge_contact_steps": self.bridge_contact_steps,
            "minimum_bridge_body_y": self.bridge_minimum_y if self.planks else None,
            "max_body_speed": self.max_speed,
            "max_abs_angular_speed": self.max_abs_angular_speed,
            "body_count": int(self.world.bodyCount),
            "joint_count": int(self.world.jointCount),
        }

    def render(self, width: int = 320, height: int = 180):
        """Original geometric offscreen debug art; no game assets or perception claims."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (width, height), "#142432")
        draw = ImageDraw.Draw(image)
        scale = width / 18
        left = self.chassis.position.x - 5

        def pixel(point):
            return ((point[0] - left) * scale, height * 0.65 - point[1] * scale)

        for first, second in self.surface_segments:
            draw.line((pixel(first), pixel(second)), fill="#8cad77", width=3)
        for body, vertices, color in self.render_polygons:
            draw.polygon([pixel(body.GetWorldPoint(vertex)) for vertex in vertices], fill=color)
        for wheel, radius in self.render_circles:
            x, y = pixel(wheel.position)
            r = radius * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill="#17202a", outline="#d3d6d5")
            end = wheel.GetWorldPoint((radius, 0))
            draw.line(((x, y), pixel(end)), fill="#d3d6d5", width=1)
        draw.text((8, 8), "SYNTHETIC ENGINE FIXTURE", fill="#e4ece9")
        return image


def benchmark_arm(
    kind: str,
    worlds: int,
    decisions: int,
    seed: int,
    *,
    render: bool = False,
    render_every: int = 4,
    wall_limit_seconds: float = 30.0,
    settings: SolverSettings | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Bounded synchronous batch; raw physics and rendered throughput are separate."""
    if any(type(value) is not int or value < 1 for value in (worlds, decisions, render_every)):
        raise ValueError("Counts must be positive integers")
    if not math.isfinite(wall_limit_seconds) or wall_limit_seconds <= 0:
        raise ValueError("Wall limit must be positive and finite")
    settings = settings or SolverSettings()
    import psutil

    start = time.perf_counter()
    cpu_start = time.process_time()
    process = psutil.Process()
    memory_samples = [process.memory_info().rss]
    fixtures = [ArticulatedFixture(kind, seed + index, settings) for index in range(worlds)]
    initialization = time.perf_counter() - start
    physics_seconds = render_seconds = 0.0
    rendered = completed = 0
    for index in range(decisions):
        if time.perf_counter() - start >= wall_limit_seconds:
            break
        action = 0 if kind == "bridge_load" else (1 if index < decisions * 0.7 else 2)
        step_start = time.perf_counter()
        for fixture in fixtures:
            fixture.step(action)
        physics_seconds += time.perf_counter() - step_start
        completed += 1
        if render and index % render_every == 0:
            render_start = time.perf_counter()
            for fixture in fixtures:
                fixture.render()
                rendered += 1
            render_seconds += time.perf_counter() - render_start
        if completed % 60 == 0:
            memory_samples.append(process.memory_info().rss)
            if progress:
                progress(completed * worlds)
    if progress:
        progress(completed * worlds)
    memory_samples.append(process.memory_info().rss)
    elapsed = time.perf_counter() - start
    actual_decisions = completed * worlds
    return {
        "fixture_version": FIXTURE_VERSION,
        "kind": kind,
        "worlds": worlds,
        "seed": seed,
        "fidelity_level": "synthetic_articulated_geometry_with_debug_render",
        "real_fidelity_validated": False,
        "shared_perception_pipeline": False,
        "requested_decisions_per_world": decisions,
        "completed_decisions_per_world": completed,
        "completed": completed == decisions,
        "wall_limit_seconds": wall_limit_seconds,
        "wall_clock_seconds": elapsed,
        "initialization_seconds": initialization,
        "physics_and_diagnostics_seconds": physics_seconds,
        "render_seconds": render_seconds,
        "policy_decisions": 0,
        "optimizer_updates": 0,
        "scripted_decisions": actual_decisions,
        "simulated_seconds_total": actual_decisions * settings.decision_seconds,
        "simulated_seconds_per_world": completed * settings.decision_seconds,
        "physics_substeps": actual_decisions * settings.substeps,
        "rendered_observations": rendered,
        "physics_substeps_per_wall_second": actual_decisions * settings.substeps / elapsed,
        "scripted_decisions_per_wall_second": actual_decisions / elapsed,
        "rendered_observations_per_wall_second": rendered / elapsed,
        "render": render,
        "render_every": render_every,
        "render_size": [320, 180],
        "settings": vars(settings),
        "fixtures": [fixture.diagnostics() for fixture in fixtures],
        "process_cpu_core_seconds": time.process_time() - cpu_start,
        "max_sampled_process_rss_bytes": max(memory_samples),
        "process_memory_samples": len(memory_samples),
        "memory_sampling": "start, every 60 completed decisions per world, end; not a true peak",
        "rates_are_diagnostics_only": True,
        "qualifies_game_fidelity": False,
        "scope": "synthetic_articulated_engine_pilot",
    }
