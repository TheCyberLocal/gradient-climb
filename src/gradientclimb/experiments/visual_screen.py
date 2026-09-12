"""Canonical publication of one registered, four-pose construction screen."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links
from gradientclimb.datasets.demonstrations import (
    EvidenceRef,
    WindowPlan,
    _read_ref,
    assemble_windows,
)
from gradientclimb.datasets.partitions import partition_publication
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile
from gradientclimb.perception.screen_features import ScreenFeatureBridge
from gradientclimb.simulation.visual_labels import PoseReplay
from gradientclimb.simulation.visual_renderer import RENDERER_VERSION
from gradientclimb.simulation.visual_replay import _UnavailableHUD, replay_poses

from .recorder import RunRecorder, verify_run

VERSION = "visual-pose-construction-screen-3.0"
ZERO = {
    "environment_steps": 0,
    "training_steps": 0,
    "training_clock_seconds": 0,
    "episodes": 0,
    "optimizer_updates": 0,
    "policy_decisions": 0,
    "evaluation_episodes": 0,
    "real_game_evaluation_episodes": 0,
    "physics_substeps": 0,
    "simulator_seconds": 0,
    "new_real_interaction_seconds": 0,
    "qualifies_real_game": False,
}


def _covered(point, polygon):
    x, y = point
    inside = False
    for a, b in zip(polygon, (*polygon[1:], polygon[0])):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def geometry_discrepancies(geometry, measurement):
    """Pixel agreement with uncertain construction labels, never accuracy scores."""
    body = wheels = None
    if geometry.body is not None and measurement["body"].get("valid"):
        vertices = geometry.body.vertices
        cross = [a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, (*vertices[1:], vertices[0]))]
        centroid = [
            sum((a[k] + b[k]) * c for a, b, c in zip(vertices, (*vertices[1:], vertices[0]), cross))
            / (3 * sum(cross))
            for k in (0, 1)
        ]
        bbox = [min(p[k] for p in vertices) for k in (0, 1)] + [
            max(p[k] for p in vertices) for k in (0, 1)
        ]
        detected = measurement["body"]
        body = {
            "center_delta_xy_pixels": (np.asarray(detected["center_xy"]) - centroid).tolist(),
            "bbox_delta_xyxy_pixels": (np.asarray(detected["bbox"]) - bbox).tolist(),
            "label_polygon_area_centroid_xy": centroid,
            "label_continuous_vertex_bounds_xyxy": bbox,
            "label_uncertainty_pixels": geometry.body.uncertainty_pixels,
            "convention": "Label filled-polygon area centroid versus visible color-component pixel centroid; label continuous vertex extrema versus detector half-open pixel-edge bbox. Neither is center of mass.",
        }
    if len(geometry.wheels) == 2 and measurement["wheels"].get("valid"):
        detected = [measurement["wheels"][side]["center_xy"] for side in ("left", "right")]
        assignment = min(
            permutations(range(2)),
            key=lambda order: sum(
                math.dist(w.center, detected[i]) for w, i in zip(geometry.wheels, order)
            ),
        )
        wheels = [
            {
                "label_wheel_id": w.wheel_id,
                "detected_pair_slot": i,
                "center_delta_xy_pixels": (np.asarray(detected[i]) - w.center).tolist(),
                "center_distance_pixels": math.dist(w.center, detected[i]),
                "label_uncertainty_pixels": w.uncertainty_pixels,
            }
            for w, i in zip(geometry.wheels, assignment)
        ]
    terrain = []
    for point in measurement["terrain"].get("points", []):
        if not measurement["terrain"].get("valid") or not point.get("valid"):
            continue
        for strip in geometry.terrain:
            if strip.points[0][0] <= point["x"] <= strip.points[-1][0]:
                expected = float(np.interp(point["x"], *zip(*strip.points)))
                location = (point["x"], expected)
                polygons = (*geometry.occluders, *((geometry.body,) if geometry.body else ()))
                hidden = any(_covered(location, p.vertices) for p in polygons) or any(
                    math.dist(location, w.center) <= w.radius_pixels for w in geometry.wheels
                )
                if not hidden:
                    terrain.append(
                        {
                            "x": point["x"],
                            "label_y": expected,
                            "delta_y_pixels": point["y"] - expected,
                            "label_uncertainty_pixels": strip.uncertainty_pixels,
                        }
                    )
                break
    return {
        "coordinate_system": geometry.coordinate_system,
        "body": body,
        "wheels": wheels,
        "terrain": terrain or None,
        "terrain_comparable_points": len(terrain),
        "semantics": "Uncertain construction agreement only. Wheel association minimizes total center distance, without front/rear or temporal identity. Terrain compares the visible turf/soil boundary within explicit strips, excluding declared occlusion; no interpolation across gaps. Missing comparisons remain null.",
    }


def _protocol(root, protocol_path, replay_path):
    protocol_file, replay_file = _inside(root, protocol_path), _inside(root, replay_path)
    protocol = json.loads(protocol_file.read_bytes())
    replay = PoseReplay.model_validate_json(replay_file.read_bytes())
    reference = EvidenceRef(path=protocol_path, sha256=sha256_file(protocol_file))
    created = datetime.fromisoformat(protocol["created_at"])
    if created.utcoffset() is None or protocol["protocol_version"] != VERSION:
        raise ValueError("Unsupported protocol or registration timestamp")
    construction_reference, annotation_cutoff = reference, created
    if "construction_protocol" in protocol or "operational_successor" in protocol:
        construction_reference = EvidenceRef.model_validate(protocol.get("construction_protocol"))
        parent = json.loads(_read_ref(root, construction_reference).read_bytes())
        allowed = {
            "experiment_id",
            "status",
            "created_at",
            "construction_protocol",
            "operational_successor",
        }
        if "construction_protocol" in parent or "operational_successor" in parent:
            raise ValueError("Operational successor must pin the original construction protocol")
        if canonical_json(
            {k: v for k, v in protocol.items() if k not in allowed}
        ) != canonical_json({k: v for k, v in parent.items() if k not in allowed}):
            raise ValueError(
                "Operational successor changed substantive construction inputs or budgets"
            )
        annotation_cutoff = datetime.fromisoformat(parent["created_at"])
        if (
            annotation_cutoff.utcoffset() is None
            or created <= annotation_cutoff
            or protocol["experiment_id"] == parent["experiment_id"]
        ):
            raise ValueError(
                "Operational successor requires a later registration and new experiment identity"
            )
        successor = protocol.get("operational_successor")
        if (
            not isinstance(successor, dict)
            or set(successor) != {"frozen_replay", "failure_evidence", "reason"}
            or not isinstance(successor["reason"], str)
            or not successor["reason"].strip()
        ):
            raise ValueError(
                "Operational successor requires frozen labels and explicit failure evidence"
            )
        frozen_replay = EvidenceRef.model_validate(successor["frozen_replay"])
        if frozen_replay != EvidenceRef(path=replay_path, sha256=sha256_file(replay_file)):
            raise ValueError(
                "Operational successor must consume its exact frozen original replay bytes"
            )
        _read_ref(root, EvidenceRef.model_validate(successor["failure_evidence"]))
    budgets = protocol["budgets"]
    if (
        any(
            type(budgets[k]) is not int or budgets[k] != 4
            for k in ("maximum_poses", "maximum_rendered_frames", "maximum_source_image_reads")
        )
        or not math.isfinite(budgets["maximum_wall_seconds"])
        or not 0 < budgets["maximum_wall_seconds"] <= 120
        or type(budgets["maximum_output_bytes"]) is not int
        or not 0 < budgets["maximum_output_bytes"] <= 33554432
    ):
        raise ValueError("Only the registered four-pose, 120-second, 32-MiB bounds are supported")
    if (
        any(
            budgets[k] != 0
            for k in (
                "physics_steps",
                "policy_decisions",
                "optimizer_updates",
                "new_real_interaction_seconds",
            )
        )
        or protocol["renderer"]["version"] != RENDERER_VERSION
    ):
        raise ValueError("Protocol requires unsupported physics, actions or renderer")
    selected = protocol["selected_frames"]
    if (
        len(replay.frames) != 4
        or len(selected) != 4
        or len({p["frame_index"] for p in selected}) != 4
        or len({f.continuity_id for f in replay.frames}) != 4
    ):
        raise ValueError(
            "Exactly four unique selected poses with independent continuity IDs are required"
        )
    for frame, source in zip(replay.frames, selected, strict=True):
        g = frame.geometry
        if (
            g.source_image != EvidenceRef.model_validate(source["source_image"])
            or g.label_protocol != construction_reference
            or g.reviewed_at is None
            or g.reviewed_at <= annotation_cutoff
        ):
            raise ValueError(
                "Label source order, protocol hash or post-registration review receipt mismatch"
            )
        annotation = protocol["annotation"]
        if (
            list(g.image_size) != protocol["annotation"]["image_size"]
            or not 3 <= annotation["body_vertices_maximum"] <= 24
            or not 0 <= annotation["wheel_count_maximum"] <= 2
            or not 2 <= annotation["terrain_points_per_strip_maximum"] <= 24
            or (g.body and len(g.body.vertices) > annotation["body_vertices_maximum"])
            or len(g.wheels) > annotation["wheel_count_maximum"]
            or any(
                len(s.points) > annotation["terrain_points_per_strip_maximum"] for s in g.terrain
            )
        ):
            raise ValueError("Label geometry exceeds the registered annotation bounds")
    plan_ref = EvidenceRef.model_validate(protocol["window_plan"])
    plan = WindowPlan.model_validate_json(_read_ref(root, plan_ref).read_bytes())
    if replay.source_partition != plan.split or protocol["source_partition"] != plan.split:
        raise ValueError("Replay, protocol and window plan partitions must agree")
    return protocol, replay, plan, reference


@contextmanager
def _journal(run, check):
    path = run.directory / "visual-pose-operations.jsonl"
    progress = {
        "planned_poses": 4,
        "source_attempted": 0,
        "source_completed": 0,
        "render_attempted": 0,
        "rendered_frames": 0,
        "render_completed": 0,
    }
    stream = path.open("x", encoding="utf-8", newline="\n")

    def record(event):
        stream.write(
            canonical_json(
                {"schema_version": VERSION, "elapsed_seconds": run.elapsed_seconds, **event}
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        counter = {
            "source_started": "source_attempted",
            "source_completed": "source_completed",
            "frame_started": "render_attempted",
            "pixels_produced": "rendered_frames",
            "frame_completed": "render_completed",
        }.get(event["event"])
        if counter:
            progress[counter] += 1
        run.annotate(visual_progress=progress)
        run.record_progress(
            {
                **ZERO,
                "counter_provenance": "flushed_visual_operation_journal",
                "visual_progress": progress,
            }
        )
        if event["event"] not in {"screen_failed", "frame_failed"}:
            check()

    failure = None
    try:
        record({"event": "screen_started"})
        yield record, progress
    except BaseException as error:
        failure = error
        try:
            record(
                {
                    "event": "screen_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_uncompleted_pose_indices": list(range(progress["source_completed"], 4)),
                    "render_uncompleted_pose_indices": list(range(progress["render_completed"], 4)),
                    "source_failed_pose_index": progress["source_attempted"] - 1
                    if progress["source_attempted"] > progress["source_completed"]
                    else None,
                    "source_unattempted_pose_indices": list(range(progress["source_attempted"], 4)),
                    "render_failed_pose_index": progress["render_attempted"] - 1
                    if progress["render_attempted"] > progress["render_completed"]
                    else None,
                    "render_unattempted_pose_indices": list(range(progress["render_attempted"], 4)),
                }
            )
        except BaseException as journal_error:  # noqa: BLE001 - preserve the original operation failure
            error.add_note(f"Failure journal also failed: {journal_error}")
        raise
    finally:
        try:
            stream.close()
            run.register_artifact(path, "visual_pose_operation_journal")
        except BaseException as cleanup_error:
            if failure:
                failure.add_note(f"Journal retention also failed: {cleanup_error}")
            else:
                raise


def publish_visual_pose_screen(project_root, protocol_path, replay_path):
    """Consume frozen reviewed labels once; preserve failed attempts without retries."""
    started = time.perf_counter()
    root = Path(project_root).absolute()
    _no_links(root)
    protocol, replay, plan, protocol_ref = _protocol(root, protocol_path, replay_path)
    config = {
        "protocol_version": VERSION,
        "protocol": protocol_ref.model_dump(),
        "replay": {"path": replay_path, "sha256": sha256_file(_inside(root, replay_path))},
        "perception_profile": protocol["perception_profile"],
        "window_plan": protocol["window_plan"],
        "source_run_id": protocol["source_run_id"],
        "source_partition": plan.split,
        "accounting_scope": "Exactly four selected source RGB decodes/perception observations and four procedural renders. Mandatory integrity hashing additionally reads the whole sealed source and reviewed windows; these are not four total filesystem reads. Function-entry wall budget checked at operation boundaries, including preflight, excluding imports and final sealing; an indivisible operation can overrun before detection. Output budget covers retained run files, reserving 1 MiB for recorder finalization.",
        "prior_class": "human-demonstration, manual-construction-label and engineered-renderer/perception system priors; no policy prior or learning arm",
    }
    if "construction_protocol" in protocol:
        config.update(
            construction_protocol=protocol["construction_protocol"],
            operational_successor=protocol["operational_successor"],
        )
    with partition_publication(root, plan) as binding:
        for record in binding.artifact_root.glob("runs/*/run-start.json"):
            previous = json.loads(record.read_bytes()).get("configuration", {}).get("protocol")
            if isinstance(previous, dict) and previous.get("sha256") == protocol_ref.sha256:
                raise ValueError(
                    "This registered screen already has a canonical attempt; a successor protocol is required"
                )
        with RunRecorder(
            binding.artifact_root,
            protocol["experiment_id"],
            binding.configuration(config),
            algorithm="fixed-visual-construction",
            environment="declared-screen-pixel-poses",
            evidence_domain="construction_visual_pose_agreement",
            source_root=root,
            qualifies_real_game=False,
        ) as run:
            binding.register(run)
            maximum_payload_bytes = protocol["budgets"]["maximum_output_bytes"] - 1048576

            def check():
                elapsed = time.perf_counter() - started
                output_bytes = sum(
                    p.stat().st_size for p in run.directory.rglob("*") if p.is_file()
                )
                if (
                    elapsed > protocol["budgets"]["maximum_wall_seconds"]
                    or output_bytes > maximum_payload_bytes
                ):
                    raise RuntimeError(
                        "Visual screen wall or output budget exceeded; stop before further work"
                    )

            run.annotate(
                **ZERO,
                annotation_labor_seconds=None,
                inherited_development_seconds=None,
                prior_cost_scope="Capture and earlier attempts referenced below; annotation and detector/renderer development costs unmeasured, not zero",
                resource_scope="Recorder start through final resource sample; excludes imports and initial protocol/label/plan parse plus partition preflight; sealing follows final sample",
            )
            with _journal(run, check) as (record_event, progress):
                refs = [
                    (config["protocol"], "visual_screen_protocol"),
                    (config["replay"], "visual_geometry_labels"),
                    (protocol["window_plan"], "imitation_window_plan"),
                    (protocol["perception_profile"], "measurement_profile"),
                    (protocol["prior_diagnostic"], "prior_construction_diagnostic"),
                    *((r.model_dump(), "segmentation_review") for r in plan.reviews),
                    *((r.model_dump(), "prior_evidence") for r in plan.prior_evidence),
                    *(
                        (f.geometry.source_image.model_dump(), "selected_source_image")
                        for f in replay.frames
                    ),
                ]
                if "construction_protocol" in protocol:
                    refs.extend(
                        [
                            (protocol["construction_protocol"], "original_construction_protocol"),
                            (
                                protocol["operational_successor"]["failure_evidence"],
                                "prior_operational_failure",
                            ),
                        ]
                    )
                for ref, kind in refs:
                    expected = EvidenceRef.model_validate(ref)
                    artifact = run.register_artifact(_read_ref(root, expected), kind)
                    if artifact["sha256"] != expected.sha256:
                        raise ValueError("Evidence changed during registration")
                    check()
                profile = MeasurementProfile(
                    **json.loads(
                        _read_ref(
                            root, EvidenceRef.model_validate(protocol["perception_profile"])
                        ).read_bytes()
                    )
                )
                if tuple(profile.expected_size) != replay.frames[0].geometry.image_size:
                    raise ValueError("Frozen profile and label image geometry mismatch")
                dataset = assemble_windows(root, plan)
                run.annotate(
                    inherited_sources=dataset["sources"],
                    window_dataset_sha256=dataset["dataset_sha256"],
                )
                check()
                accepted = {}
                for window in dataset["windows"]:
                    if window["run_id"] == protocol["source_run_id"]:
                        for frame in window["frames"]:
                            if (
                                frame["frame_index"] in accepted
                                and frame != accepted[frame["frame_index"]]
                            ):
                                raise ValueError("Conflicting accepted source frame records")
                            accepted[frame["frame_index"]] = frame
                for pose, selected in zip(replay.frames, protocol["selected_frames"], strict=True):
                    source = accepted.get(selected["frame_index"])
                    if (
                        source is None
                        or pose.timestamp_ns != source["timestamp_ns"]
                        or source["path"] != pose.geometry.source_image.path
                        or source["file_sha256"] != pose.geometry.source_image.sha256
                    ):
                        raise ValueError(
                            "Selected label does not match an accepted canonical window frame and timestamp"
                        )
                source_rows, paired_rows, images = [], [], {}
                bridge = ScreenFeatureBridge(
                    HCRPixelMeasurer(profile), _UnavailableHUD(), history=4
                )
                for index, pose in enumerate(replay.frames):
                    record_event({"event": "source_started", "pose_index": index})
                    tick = time.perf_counter()
                    data = _inside(root, pose.geometry.source_image.path).read_bytes()
                    if hashlib.sha256(data).hexdigest() != pose.geometry.source_image.sha256:
                        raise ValueError("Selected source changed before decode")
                    with Image.open(io.BytesIO(data)) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    decoded = time.perf_counter()
                    if (rgb.shape[1], rgb.shape[0]) != pose.geometry.image_size:
                        raise ValueError("Decoded source and declared image size mismatch")
                    bridge.reset()
                    observation = bridge.observe(rgb, pose.timestamp_ns).as_dict()
                    measured = time.perf_counter()
                    row = {
                        "pose_index": index,
                        "source": pose.geometry.source_image.model_dump(),
                        "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "hash_decode_seconds": decoded - tick,
                        "perception_seconds": measured - decoded,
                        "observation": observation,
                        "discrepancies": geometry_discrepancies(
                            pose.geometry, observation["measurement"]
                        ),
                    }
                    record_event({"event": "source_completed", "result": row})
                    source_rows.append(row)

                def pixels(index, rgb):
                    record_event({"event": "pixels_produced", "pose_index": index})
                    tick = time.perf_counter()
                    stream = io.BytesIO()
                    Image.fromarray(rgb).save(stream, format="PNG")
                    payload = stream.getvalue()
                    check()
                    existing = sum(
                        p.stat().st_size for p in run.directory.rglob("*") if p.is_file()
                    )
                    if existing + 2 * len(payload) > maximum_payload_bytes:
                        raise RuntimeError("Exact rendered PNG would exceed output budget")
                    path = run.directory / f"actor-pose-{index:03d}.png"
                    with path.open("xb") as writer:
                        writer.write(payload)
                        writer.flush()
                        os.fsync(writer.fileno())
                    images[index] = {
                        "artifact": run.register_artifact(path, "exact_actor_rgb_png"),
                        "png_publication_seconds": time.perf_counter() - tick,
                    }

                def rendered_event(event):
                    if event["event"] == "frame_completed":
                        row = event["result"]
                        index = row["frame_index"]
                        pair = {
                            "pose_index": index,
                            "source_frame_index": protocol["selected_frames"][index]["frame_index"],
                            "label": replay.frames[index].geometry.model_dump(mode="json"),
                            "source": source_rows[index],
                            "rendered": row,
                            "retained_pixels": images[index],
                            "rendered_discrepancies": geometry_discrepancies(
                                replay.frames[index].geometry, row["observation"]["measurement"]
                            ),
                        }
                        record_event({**event, "paired_result": pair})
                        paired_rows.append(pair)
                    else:
                        record_event(event)

                replay_result = replay_poses(
                    replay, profile, project_root=root, on_event=rendered_event, on_pixels=pixels
                )
                summary = {
                    **ZERO,
                    "planned_poses": 4,
                    "retained_paired_poses": len(paired_rows),
                    "source_image_decodes": len(source_rows),
                    "rendered_observations": progress["rendered_frames"],
                    "fidelity_level": "declared_screen_pixel_pose_replay",
                    "measured_accuracy": None,
                    "publisher_elapsed_seconds_before_report": time.perf_counter() - started,
                    "source_hash_decode_seconds": sum(
                        r["hash_decode_seconds"] for r in source_rows
                    ),
                    "source_perception_seconds": sum(r["perception_seconds"] for r in source_rows),
                    "png_publication_seconds": sum(
                        r["png_publication_seconds"] for r in images.values()
                    ),
                }
                report = {
                    "schema_version": VERSION,
                    "input_contract": config,
                    "rows": paired_rows,
                    "summary": summary,
                    "render_replay_summary": replay_result["summary"],
                    "render_replay_identity": {
                        key: replay_result[key]
                        for key in (
                            "schema_version",
                            "replay_sha256",
                            "renderer_version",
                            "render_style_sha256",
                        )
                    },
                    "shared_observation_schema": replay_result["shared_observation_schema"],
                    "shared_observation_schema_id": replay_result["shared_observation_schema_id"],
                    "inherited_sources": dataset["sources"],
                    "window_dataset_sha256": dataset["dataset_sha256"],
                    "interpretation": "Purposefully selected construction agreement, not independent accuracy, real fidelity, learning or competence. Full directed orientation remains unresolved. Four source-image decodes; integrity verification additionally hashes the sealed source and reviewed spans. Source component timers exclude assembly, journaling and publication. Renderer receipt has its own nested scope; do not add it to whole publisher costs. Annotation/development labor is inherited and unmeasured.",
                }
                output = run.directory / "visual-pose-screen.json"
                payload = (canonical_json(report) + "\n").encode()
                if (
                    sum(p.stat().st_size for p in run.directory.rglob("*") if p.is_file())
                    + 2 * len(payload)
                    > maximum_payload_bytes
                ):
                    raise RuntimeError("Final report would exceed output budget")
                with output.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                run.register_artifact(output, "visual_pose_construction_report")
                record_event(
                    {"event": "screen_completed", "retained_paired_poses": len(paired_rows)}
                )
            check()
            run.finalize(**summary)
    return {
        "run_id": run.run_id,
        "report": str(output),
        "summary": summary,
        "verification": verify_run(binding.artifact_root, run.run_id),
    }
