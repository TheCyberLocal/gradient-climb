"""Supervised, bounded pixel/input probe; no menu clicks or automatic restart.

The operator starts/resumes the observed game after this process prints READY.
This discovery protocol uses a local UI reference, not validated general vision.
Raw game images remain local ignored artifacts. Escape aborts and releases pedals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from gradientclimb.capture.screen import WindowCapture
from gradientclimb.capture.windows import WindowGuard, discover_windows
from gradientclimb.control.pedals import PedalAction, PedalController
from gradientclimb.control.touch import (
    TouchPedalProfile,
    TouchPositionEvidence,
    WindowsTouchPedalBackend,
)
from gradientclimb.control.windows import WindowsPedalBackend
from gradientclimb.experiments import RunRecorder

BOXES = [(970, 44, 1023, 106), (70, 38, 115, 77)]
PEDAL_BOXES = [(130, 511, 218, 539), (900, 511, 955, 540)]  # BRAKE, GAS labels.
PEDAL_MAX_SHIFT = 8


def make_pedal_classifier(reference):
    """Bounded label matching grants positions only in the same fresh capture."""
    import cv2

    if reference.shape != (581, 1034, 3) or reference.dtype != np.uint8:
        raise ValueError("Expected locally inspected uint8 1034x581 pedal reference")
    templates = [reference[t:b, l:r].astype(np.float32) for l, t, r, b in PEDAL_BOXES]

    def positions(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.shape != reference.shape or rgb.dtype != np.uint8:
            return False, []
        matches = []
        for (left, top, right, bottom), template in zip(PEDAL_BOXES, templates, strict=True):
            margin = PEDAL_MAX_SHIFT
            search = rgb[top - margin : bottom + margin, left - margin : right + margin].astype(
                np.float32
            )
            squared, _, location, _ = cv2.minMaxLoc(
                cv2.matchTemplate(search, template, cv2.TM_SQDIFF)
            )
            score = 1 - math.sqrt(max(0, squared) / template.size) / 255
            matches.append(
                {"score": score, "offset_xy": [location[0] - margin, location[1] - margin]}
            )
        return all(match["score"] >= 0.96 for match in matches), matches

    return positions


def make_playing_classifier(reference):
    if reference.shape != (581, 1034, 3) or reference.dtype != np.uint8:
        raise ValueError("Expected locally inspected uint8 1034x581 reference")
    templates = [reference[t:b, l:r].astype(np.float32) for l, t, r, b in BOXES]

    def playing(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.shape != reference.shape or rgb.dtype != np.uint8:
            return False, []
        scores = [
            1 - float(np.sqrt(np.mean((rgb[t:b, l:r].astype(np.float32) - template) ** 2))) / 255
            for (l, t, r, b), template in zip(BOXES, templates, strict=True)
        ]
        sky = rgb[160:200, 850:930].mean(axis=(0, 1))
        return bool(min(scores) >= 0.96 and sky[1] > 160 and sky[2] > 180), scores

    return playing


def persist_probe(run, frames, entries, command_trace, os_trace, started):
    """Save complete raw traces first; one artifact failure does not skip the rest."""
    errors = []
    raw_paths = []
    for name, rows in (
        ("requested-pedal-leases", command_trace),
        ("os-pedal-transitions", os_trace),
        ("capture-observations", entries),
    ):
        path = run.directory / f"{name}.json"
        try:
            path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            raw_paths.append((path, name))
        except Exception as exc:  # noqa: BLE001 - retain all evidence persistence failures
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    for path, name in raw_paths:
        try:
            run.register_artifact(path, name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"register {name}: {type(exc).__name__}: {exc}")
    for index, event in enumerate(command_trace):
        try:
            run.trajectory(
                {
                    "episode_id": "supervised-probe",
                    "step": index,
                    "gas": event["gas"],
                    "brake": event["brake"],
                    "action_duration_seconds": event["duration_seconds"],
                    "elapsed_seconds": max(0, event["timestamp_ns"] / 1e9 - started)
                    if started
                    else 0,
                    "observation": {
                        "kind": "requested_pedal_lease",
                        **event,
                        "duration_semantics": "maximum requested lease; OS transitions logged separately",
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - raw command trace still includes every event
            errors.append(f"trajectory {index}: {type(exc).__name__}: {exc}")
    directory = run.directory / "frames"
    try:
        directory.mkdir()
        for index, (pixels, entry) in enumerate(zip(frames, entries, strict=True)):
            try:
                path = directory / f"frame-{index:06d}.png"
                Image.fromarray(pixels).save(path)
                entry.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                run.register_artifact(path, "real_game_frame", {"frame_index": index})
            except Exception as exc:  # noqa: BLE001 - attempt every remaining captured frame
                errors.append(f"frame {index}: {type(exc).__name__}: {exc}")
        manifest = directory / "frames.jsonl"
        manifest.write_text(
            "".join(json.dumps(e, allow_nan=False) + "\n" for e in entries), encoding="utf-8"
        )
        run.register_artifact(manifest, "timestamped_frames")
    except Exception as exc:  # noqa: BLE001 - full raw capture metadata was saved first
        errors.append(f"frame manifest: {type(exc).__name__}: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/game-discovery/playing-native-normalized.png"),
    )
    parser.add_argument("--wait-seconds", type=float, default=45)
    parser.add_argument("--input-mode", choices=["vk", "scancode", "touch"], default="vk")
    parser.add_argument(
        "--protocol", choices=["four-states", "gas", "neutral"], default="four-states"
    )
    args = parser.parse_args()
    if not math.isfinite(args.wait_seconds) or not 1 <= args.wait_seconds <= 120:
        raise ValueError("Wait must be 1..120 seconds")
    reference = np.asarray(Image.open(args.reference).convert("RGB"))
    playing = make_playing_classifier(reference)
    positions = make_pedal_classifier(reference) if args.input_mode == "touch" else None
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()

    targets = discover_windows()
    if len(targets) != 1:
        raise RuntimeError("Expected exactly one visible game window")
    target = targets[0]
    guard = WindowGuard(target)
    latest = {"playing": False, "positions": False, "time": 0.0, "frame": None}
    if args.input_mode == "touch":
        profile = TouchPedalProfile(
            target.client_rect, (1034, 581), (925, 480), (170, 480), reference_hash
        )

        def touch_evidence():
            snapshot = latest.copy()
            frame = snapshot["frame"]
            if frame is None:
                raise RuntimeError("No captured pedal-position evidence")
            return TouchPositionEvidence(
                profile.sha256,
                frame.source_rect,
                frame.started_ns,
                snapshot["playing"],
                snapshot["positions"],
            )

        backend = WindowsTouchPedalBackend(target, profile, touch_evidence, guard=guard)
    else:
        backend = WindowsPedalBackend(
            target, gas_vk=0x27, brake_vk=0x25, input_mode=args.input_mode
        )
    sequence = [(0, 0.5), (1, 1.5), (3, 1.0), (2, 1.0), (0, 0.5), (2, 1.0), (3, 1.0), (1, 1.0)]
    if args.protocol == "gas":
        sequence = [(1, 8.0)]
    elif args.protocol == "neutral":
        sequence = [(0, 8.0)]
    config = {
        "protocol": args.protocol,
        "schedule": sequence,
        "target": target.as_dict(),
        "control_bindings": {"gas": "visible GAS pedal", "brake": "visible BRAKE pedal"}
        if positions
        else {"gas": "Right", "brake": "Left"},
        "input_encoding": backend.input_encoding,
        "reference_sha256": reference_hash,
        "scope": "supervised system-identification pilot; not policy evaluation",
        "frame_size": [1034, 581],
        "max_observation_age_seconds": 0.45,
        "freshness_reference": "capture interval start; actual presentation time unavailable",
        "lease_seconds": 0.4,
        "max_recorded_frames": 500,
        "playing_anchor_boxes": BOXES,
        "anchor_threshold": 0.96,
        "pedal_label_boxes": PEDAL_BOXES if positions else None,
        "pedal_label_max_shift_pixels": PEDAL_MAX_SHIFT if positions else None,
        "pedal_label_threshold": 0.96 if positions else None,
    }
    frames, entries = [], []
    aborted = threading.Event()

    def allowed():
        if backend.sender.is_down(0x1B):
            aborted.set()
        if aborted.is_set():
            return False
        current = guard.validate()
        if positions and (not latest["positions"] or current.client_rect != profile.client_rect):
            return False
        return latest["playing"] and 0 <= time.perf_counter() - latest["time"] < 0.45

    with RunRecorder(
        "artifacts",
        "real-control-probe",
        config,
        algorithm="scripted-probe",
        environment="actual_hill_climb_racing",
        evidence_domain="real_game",
        telemetry_interval_seconds=1,
    ) as run:
        print(json.dumps({"ready": True, "run_id": run.run_id}), flush=True)
        run.register_artifact(args.reference, "local_playing_reference")
        status, reason, started = "failed", "no verified gameplay before deadline", None
        controller = None
        failure = None
        cleanup_errors = []
        waiting_observations = []
        waiting_frames = []
        boundaries = np.cumsum([duration for _, duration in sequence])
        previous_frame_ns = -1
        try:
            with (
                WindowCapture(guard, backend="dxcam", output_size=(1034, 581)) as capture,
                PedalController(backend, allowed) as controller,
            ):
                wait_deadline = time.perf_counter() + args.wait_seconds
                for _ in range(5000):
                    if backend.sender.is_down(0x1B):
                        aborted.set()
                    if aborted.is_set():
                        reason = "operator Escape"
                        break
                    if started is None and time.perf_counter() >= wait_deadline:
                        break
                    if started is not None and time.perf_counter() - started >= boundaries[-1]:
                        status, reason = "completed", "bounded schedule completed"
                        break
                    frame = capture.grab()
                    now = time.perf_counter()
                    if (
                        not 0 <= frame.started_ns <= frame.completed_ns <= int(now * 1e9)
                        or frame.completed_ns <= previous_frame_ns
                    ):
                        reason = "invalid or nonmonotonic capture timestamp"
                        break
                    previous_frame_ns = frame.completed_ns
                    yes, scores = playing(frame.rgb)
                    pedal_ok, pedal_matches = positions(frame.rgb) if positions else (True, None)
                    # Capture return time alone cannot establish observation freshness.
                    latest.update(
                        playing=yes, positions=pedal_ok, time=frame.started_ns / 1e9, frame=frame
                    )
                    if started is None:
                        waiting_observations.append(
                            {
                                **frame.metadata(),
                                "playing": bool(yes),
                                "anchor_scores": scores,
                                "pedal_positions_verified": bool(pedal_ok) if positions else None,
                                "pedal_label_matches": pedal_matches,
                            }
                        )
                        if len(waiting_frames) < 3:
                            waiting_frames.append(frame.rgb.copy())
                        if now >= wait_deadline:
                            break
                        if not yes or not pedal_ok:
                            if backend.sender.is_down(0x1B):
                                reason = "operator Escape"
                                break
                            time.sleep(0.04)
                            continue
                        started = now
                    elapsed = now - started
                    frames.append(frame.rgb.copy())
                    code = sequence[
                        min(
                            int(np.searchsorted(boundaries, elapsed, side="right")),
                            len(sequence) - 1,
                        )
                    ][0]
                    entry = {
                        "frame_index": len(frames) - 1,
                        **frame.metadata(),
                        "elapsed_seconds": elapsed,
                        "playing": bool(yes),
                        "anchor_scores": scores,
                        "pedal_positions_verified": bool(pedal_ok) if positions else None,
                        "pedal_label_matches": pedal_matches,
                        "requested_code": code,
                    }
                    entries.append(entry)
                    if elapsed >= boundaries[-1]:
                        status, reason = "completed", "bounded schedule completed"
                        break
                    if not yes or not allowed() or len(frames) >= 500:
                        reason = "playing/freshness/operator/recording guard stopped probe"
                        break
                    lease = min(0.4, float(boundaries[-1]) - (time.perf_counter() - started))
                    if lease <= 0:
                        status, reason = "completed", "bounded schedule completed"
                        break
                    controller.submit(PedalAction.from_code(code, lease))
                    if controller.fault:
                        reason = controller.fault
                        break
                    time.sleep(0.01)
                else:
                    reason = "capture iteration limit"
                controller.release()
        except KeyboardInterrupt:
            status, reason, failure = "cancelled", "operator interrupt", traceback.format_exc()
        except Exception:  # noqa: BLE001 - always persist partial frames and all control traces
            status, reason, failure = "failed", "capture/control failure", traceback.format_exc()
        finally:
            latest["playing"] = False
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure alongside original fault
                cleanup_errors.append(f"backend: {type(exc).__name__}: {exc}")
        command_trace = list(controller.trace) if controller is not None else []
        os_trace = list(backend.trace)
        persistence_errors = persist_probe(run, frames, entries, command_trace, os_trace, started)
        try:
            waiting_path = run.directory / "waiting-observations.json"
            waiting_path.write_text(
                json.dumps(waiting_observations, indent=2, allow_nan=False), encoding="utf-8"
            )
            run.register_artifact(waiting_path, "waiting_observations")
            for index, pixels in enumerate(waiting_frames):
                waiting_path = run.directory / f"waiting-frame-{index}.png"
                Image.fromarray(pixels).save(waiting_path)
                run.register_artifact(waiting_path, "waiting_frame")
        except Exception as exc:  # noqa: BLE001 - report diagnostics persistence failure
            persistence_errors.append(f"waiting diagnostics: {type(exc).__name__}: {exc}")
        if cleanup_errors or persistence_errors:
            status, reason = "failed", "cleanup or evidence persistence failed"
        directory = run.directory / "frames"
        run.finalize(
            status=status,
            reason=reason,
            frames=len(frames),
            schedule_seconds=float(boundaries[-1]),
            observed_seconds=entries[-1]["elapsed_seconds"] if entries else None,
            failure_traceback=failure,
            cleanup_errors=cleanup_errors,
            persistence_errors=persistence_errors,
            command_events=len(command_trace),
            os_transition_events=len(os_trace),
            episode_count=0,
            qualification_evidence=False,
        )
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "status": status,
                    "reason": reason,
                    "frames": len(frames),
                    "directory": str(directory),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
