"""Build an offline prototype dataset from five existing discovery screenshots.

No capture or input API is imported or called. Crops are label-derived observations,
not new real samples. All images and generated manifests stay under artifact storage.
The hard-coded boxes describe this one inspected 1034x581 discovery session only.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments.recorder import RunRecorder, verify_run
from gradientclimb.perception.pixels import TemplateRecognizer
from gradientclimb.perception.states import UIState


class ObservationLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str
    source_artifact_id: str
    source_path_relative_to_run: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_filename: str
    source_session_id: str
    state: UIState
    substate: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    captured_at: None = None
    action_state: None = None
    split: Literal["prototype_train"] = "prototype_train"
    legitimate_ad_close: Literal[False] = False
    restart_authorized: Literal[False] = False
    annotation_basis: str
    template_bbox_pixels: tuple[int, int, int, int]
    template_artifact_id: str
    notes: list[str]


class DiscoveryDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["discovery-observations-1"] = "discovery-observations-1"
    dataset_id: str
    run_id: str
    evidence_domain: Literal["real_game_discovery"] = "real_game_discovery"
    created_at: str
    source_session_count: Literal[1] = 1
    independent_heldout_images: Literal[0] = 0
    measured_generalization_accuracy: None = None
    observations: list[ObservationLabel] = Field(min_length=5, max_length=5)
    limitations: list[str]


SPECS = [
    ("playing", "playing.png", UIState.PLAYING, "playing", (867, 418, 989, 545)),
    ("paused", "paused.png", UIState.PAUSED, "pause_menu", (485, 83, 607, 115)),
    ("revive-offer", "second-chance.png", UIState.SELECTION, "revive_offer", (377, 91, 716, 166)),
    ("result", "result-no-input.png", UIState.RESULT, "out_of_fuel", (586, 366, 884, 404)),
    ("upgrades", "upgrades.png", UIState.SELECTION, "upgrade_menu", (400, 138, 699, 320)),
]


def build_dataset(source: Path, artifact_root: Path) -> dict:
    source, artifact_root = source.resolve(), artifact_root.resolve()
    source_hashes = {filename: sha256_file(source / filename) for _, filename, *_ in SPECS}
    for filename in source_hashes:
        with Image.open(source / filename) as image:
            if image.size != (1034, 581):
                raise ValueError(f"{filename}: expected inspected 1034x581 wrapper geometry")
    configuration = {
        "protocol": "offline five-image discovery label/prototype construction",
        "source_hashes": source_hashes,
        "source_session_count": 1,
        "independent_heldout_images": 0,
        "expected_size": [1034, 581],
        "state_labels_confirmed_during_discovery": True,
        "native_arrow_bindings_user_confirmed": {"gas": "right", "brake": "left"},
        "crop_specs": [
            {"id": name, "file": filename, "state": state.value, "substate": substate, "bbox": box}
            for name, filename, state, substate, box in SPECS
        ],
        "template_threshold": 0.98,
        "ambiguity_margin": 0.02,
    }
    with RunRecorder(
        artifact_root,
        "real-game-discovery-labels",
        configuration,
        algorithm="offline-template-labeling",
        environment="hill-climb-racing-discovery-screenshots",
        evidence_domain="real_game_discovery",
        vehicle_profile="Hill Climber",
        map_profile="CountrySide",
        telemetry_interval_seconds=0,
        data_role="prototype construction only; no controlled trajectories",
    ) as run:
        dataset_dir = run.directory / "discovery-dataset"
        template_dir = dataset_dir / "templates"
        template_dir.mkdir(parents=True, exist_ok=False)
        labels, templates, frames = [], [], []
        for name, filename, state, substate, box in SPECS:
            source_record = run.register_artifact(
                source / filename,
                "real_discovery_screenshot",
                {
                    "evidence_domain": "real_game_discovery",
                    "source_session_id": "discovery-2026-09-11",
                    "source_filename": filename,
                    "captured_at": None,
                    "state": state.value,
                    "substate": substate,
                    "redistribution": "local only; third-party game content",
                },
            )
            with Image.open(source / filename) as image:
                rgb = image.convert("RGB")
                frames.append(np.asarray(rgb).copy())
                target = template_dir / f"{name}.png"
                rgb.crop(box).save(target)
            template_record = run.register_artifact(
                target,
                "ui_template_crop",
                {
                    "evidence_domain": "derived_real_game_discovery",
                    "parent_artifact_id": source_record["artifact_id"],
                    "parent_sha256": source_record["sha256"],
                    "bbox_pixels": box,
                    "role": "state",
                    "control_authorization": False,
                },
            )
            left, top, right, bottom = box
            roi = [
                max(0, left - 4) / 1034,
                max(0, top - 4) / 581,
                min(1034, right + 4) / 1034,
                min(581, bottom + 4) / 581,
            ]
            templates.append(
                {
                    "label": name,
                    "state": state.value,
                    "file": f"templates/{name}.png",
                    "roi": roi,
                    "threshold": 0.98,
                    "role": "state",
                    "sha256": template_record["sha256"],
                }
            )
            notes = [
                "One discovery session; no independent held-out image.",
                "Includes wrapper chrome/sidebar; does not establish native-client alignment.",
                "Capture timestamp and contemporaneous pedal state were not recorded.",
            ]
            if name == "revive-offer":
                notes.append(
                    "Revive-video offer is not an advertisement; its X does not authorize ad closure."
                )
            labels.append(
                ObservationLabel(
                    observation_id=name,
                    source_artifact_id=source_record["artifact_id"],
                    source_path_relative_to_run=source_record["path"],
                    source_sha256=source_record["sha256"],
                    source_filename=filename,
                    source_session_id="discovery-2026-09-11",
                    state=state,
                    substate=substate,
                    width=1034,
                    height=581,
                    annotation_basis="Discovery-state labels confirmed during session; offline visual crop inspection.",
                    template_bbox_pixels=box,
                    template_artifact_id=template_record["artifact_id"],
                    notes=notes,
                )
            )
        dataset = DiscoveryDataset(
            dataset_id=f"discovery-prototype-{run.run_id}",
            run_id=run.run_id,
            created_at=datetime.now(UTC).isoformat(),
            observations=labels,
            limitations=[
                "Five images from one session; no generalization accuracy is measured.",
                "Templates are crops of these same images; resubstitution is a construction check only.",
                "No temporal trajectory, calibrated dynamics, OCR, restart, or advertisement policy is established.",
                "Native screenshot coordinate alignment remains unvalidated.",
                "All images retain third-party rights and stay in ignored local artifact storage.",
            ],
        )
        label_path = dataset_dir / "observations.json"
        label_path.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
        manifest_path = dataset_dir / "ui-templates.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "expected_size": [1034, 581],
                    "threshold": 0.97,
                    "ambiguity_margin": 0.02,
                    "templates": templates,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        recognizer = TemplateRecognizer.from_manifest(manifest_path)
        results = []
        for label, frame in zip(labels, frames):
            prediction = recognizer.classify(frame)
            results.append(
                {
                    "observation_id": label.observation_id,
                    "expected_state": label.state.value,
                    "predicted_state": prediction.state.value,
                    "similarity": prediction.confidence,
                    "matches_expected": prediction.state == label.state,
                    "legitimate_ad_close_visible": prediction.legitimate_close_visible,
                    "restart_visible": prediction.restart_visible,
                    "matches": [
                        {"label": m.label, "score": m.score, "accepted": m.accepted}
                        for m in recognizer.matches(frame)
                    ],
                }
            )
        correct = sum(result["matches_expected"] for result in results)
        report = {
            "protocol": "resubstitution construction check; not held-out validation",
            "source_images": 5,
            "correct_state_matches": correct,
            "results": results,
            "independent_heldout_images": 0,
            "measured_generalization_accuracy": None,
        }
        report_path = dataset_dir / "resubstitution-check.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        run.register_artifact(
            label_path, "labeled_observation_manifest", {"evidence_domain": "real_game_discovery"}
        )
        run.register_artifact(manifest_path, "ui_template_manifest", {"prototype_only": True})
        run.register_artifact(report_path, "resubstitution_check", {"heldout_validation": False})
        run.metric("labeled_discovery_images", 5)
        run.metric("independent_heldout_images", 0)
        run.metric("template_resubstitution_fraction", correct / 5)
        # EvaluationRecord models gameplay episodes (at least one). This five-image
        # construction check belongs in its registered report and metrics instead.
        run.finalize(
            episodes=0,
            training_steps=0,
            environment_steps=0,
            source_images=5,
            template_crops=5,
            evidence_domain="real_game_discovery",
            measured_generalization_accuracy=None,
            resubstitution_correct=correct,
        )
        result = {
            "run_id": run.run_id,
            "dataset_directory": str(dataset_dir),
            "resubstitution_correct": correct,
            "source_images": 5,
        }
    result["verification"] = verify_run(artifact_root, result["run_id"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/game-discovery"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--write-schema", type=Path)
    args = parser.parse_args()
    if args.write_schema:
        args.write_schema.write_text(
            json.dumps(DiscoveryDataset.model_json_schema(), indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(build_dataset(args.source, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
