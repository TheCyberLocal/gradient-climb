"""Record focused Hill Climb Racing pixels and human arrow controls; never inject input."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.demonstrations import (
    DEMONSTRATION_VERSION,
    DemonstrationLimits,
    DemonstrationWindowGuard,
    NativePedalReader,
    record_demonstration,
)
from gradientclimb.capture.screen import WindowCapture
from gradientclimb.capture.windows import discover_windows
from gradientclimb.control.game_adapter import GameUIRecognizer
from gradientclimb.control.host import HostBudgetGuard, NativeInputLease
from gradientclimb.experiments import RunRecorder


def wait_for_foreground(guard, seconds, *, clock=time.perf_counter, wait=time.sleep):
    """No capture or control sampling occurs while the user selects the game."""
    deadline = clock() + seconds
    while clock() < deadline:
        guard.validate(require_foreground=False)
        if guard.api.foreground() == guard.target.hwnd:
            guard.validate(require_foreground=True)
            return
        wait(0.1)
    raise TimeoutError("Game was not manually focused within the arming interval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--purpose", choices=["imitation", "human-benchmark", "development"], required=True
    )
    parser.add_argument("--demonstrator", default="owner", help="Local pseudonym, never uploaded")
    parser.add_argument(
        "--configuration-note",
        required=True,
        help="Vehicle/map/upgrades actually inspected by the operator",
    )
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--control-hz", type=float, default=100)
    parser.add_argument("--max-mib", type=int, default=512)
    parser.add_argument("--arm-seconds", type=float, default=30)
    parser.add_argument("--capture-backend", choices=["dxcam", "mss", "pillow"], default="dxcam")
    parser.add_argument(
        "--ui-profile", type=Path, default=Path("configs/perception/hcr-reset-ui.json")
    )
    args = parser.parse_args()
    if not 1 <= args.arm_seconds <= 120:
        raise ValueError("Arming interval must be within 1..120 seconds")
    limits = DemonstrationLimits(
        seconds=args.seconds,
        frames_per_second=args.fps,
        controls_per_second=args.control_hz,
        maximum_payload_bytes=args.max_mib * 1024**2,
    )
    with NativeInputLease():
        targets = discover_windows()
        if len(targets) != 1:
            raise RuntimeError("Expected exactly one visible Hill Climb Racing window")
        target = targets[0]
        guard = DemonstrationWindowGuard(target)
        host = HostBudgetGuard(args.root, target=target)
        host.check(force=True)
        recognizer = GameUIRecognizer.from_file(args.ui_profile, args.root)
        session_id = str(uuid.uuid4())
        config = {
            "version": DEMONSTRATION_VERSION,
            "session_id": session_id,
            "purpose": args.purpose,
            "demonstrator_alias": args.demonstrator,
            "operator_configuration_note": args.configuration_note,
            "configuration_independently_verified": False,
            "limits": asdict(limits),
            "target": target.as_dict(),
            "capture_backend": args.capture_backend,
            "ui_profile_sha256": sha256_file(args.ui_profile),
            "host_limits": asdict(host.limits),
            "input_injection": False,
            "controls": {"gas": "VK_RIGHT", "brake": "VK_LEFT"},
            "privacy_scope": "Selected game client pixels and focused Left/Right key states only; local files",
            "source_partition": args.purpose,
            "benchmark_excluded_from_training": args.purpose == "human-benchmark",
            "prior_knowledge": {
                "controller": "human",
                "human_practice_seconds": None,
                "system_prior": "engineered capture/guard and frozen UI construction references",
                "policy_weights_consumed": False,
                "imitation_training_source": args.purpose == "imitation",
            },
        }
        with RunRecorder(
            args.root,
            "human-demonstration",
            config,
            algorithm="human",
            environment="actual_hill_climb_racing",
            evidence_domain="real_game_demonstration",
            qualifies_real_game=False,
        ) as run:
            print(
                f"Read-only recorder armed for {args.arm_seconds:g} seconds. Manually focus the game. "
                "It records only game pixels and Left/Right keys, and stops when focus is lost. "
                f"Session: {session_id}",
                flush=True,
            )
            wait_for_foreground(guard, args.arm_seconds)

            def classify(frame):
                observed = recognizer.observe(frame)
                return {
                    "state": observed.state,
                    "variant": observed.variant,
                    "confidence": observed.confidence,
                    "scope": "frozen UI template hypothesis; independent segmentation required",
                }

            with WindowCapture(
                guard,
                backend=args.capture_backend,
                output_size=recognizer.profile.expected_size,
            ) as capture:
                summary = record_demonstration(
                    run,
                    capture=capture.grab,
                    controls=NativePedalReader(guard),
                    host_check=host.check,
                    limits=limits,
                    classify=classify,
                )
            report = run.directory / "demonstration-summary.json"
            report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            run.register_artifact(report, "human_demonstration_summary")
            preflight = run.directory / "demonstration-host-checks.json"
            preflight.write_text(json.dumps(host.trace, indent=2) + "\n", encoding="utf-8")
            run.register_artifact(preflight, "human_demonstration_host_checks")
            run.register_artifact(args.ui_profile, "human_demonstration_ui_profile")
            run.finalize(
                status=summary["status"],
                episode_count=summary["completed_segmented_episodes"],
                episode_count_scope="Completed segmented outcome records; unsegmented experience is unknown",
                qualification_evidence=False,
                demonstration=summary,
            )
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "session_id": session_id,
                        "directory": str(run.directory),
                        **summary,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
