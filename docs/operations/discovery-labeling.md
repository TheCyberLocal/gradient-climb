# Offline discovery labels and prototype UI templates

The initial dataset contains five screenshots from one real-game discovery session. It records observed UI states, source hashes, and state-only template crops. It does not contain controlled action trajectories or measure perception generalization.

The source screenshots already existed before offline labeling. The labeling script imports no capture or input API and performs no live computer interaction. All images, crops, generated manifests, and evaluation records remain in ignored local artifact storage; the repository contains only original code, this procedure, and a JSON schema.

## Recorded dataset

The current sealed run is `b978c7e7-a4ff-40ae-a1bd-2577ea935010`, with `evidence_domain=real_game_discovery`. Its base source checkpoint and captured working-tree diff are recorded in its run metadata; construction ran with uncommitted source changes. The local dataset directory is:

```text
artifacts/runs/b978c7e7-a4ff-40ae-a1bd-2577ea935010/discovery-dataset/
```

| Original screenshot | Label | Substate | Template region |
| --- | --- | --- | --- |
| `playing.png` | `playing` | `playing` | Gas-pedal graphic |
| `paused.png` | `paused` | `pause_menu` | PAUSED title |
| `second-chance.png` | `selection` | `revive_offer` | SECOND CHANCE header |
| `result-no-input.png` | `result` | `out_of_fuel` | TOUCH TO CONTINUE prompt |
| `upgrades.png` | `selection` | `upgrade_menu` | Upgrade-card region |

The revival screen offers a video; it is not itself an advertisement. Its X declines an offer and must not be labeled as an advertisement-close control. Similarly, the result prompt and pause-menu labels do not authorize a restart or click. All five templates have `role=state`; both control-visibility flags remain false for every checked source image.

These screenshots are 1034 by 581 pixels and include wrapper chrome/sidebar. The manifest requires exactly that input size and limits each template's search to its crop region plus four pixels. Native client capture alignment is unverified. A different size becomes unknown; an apparently matching size alone does not establish the same crop, DPI, layout, or game viewport.

## Files and provenance

`observations.json` validates against [the discovery schema](../../schemas/discovery_observations.schema.json). It identifies each original source artifact, source SHA-256, annotation basis, session ID, state, substate, and exact crop box. Capture timestamps and contemporaneous action states are null because they were not recorded; file modification times are not substituted for capture timestamps. Every image belongs to `prototype_train`.

`ui-templates.json` is loadable by `TemplateRecognizer.from_manifest`. It refers to local crop files by relative path and SHA-256. `resubstitution-check.json` records each state prediction and all candidate template scores against the five original source images. Source files, crops, manifests, and the check report are also registered in the run artifact index. Crops identify their parent source artifact/hash and are derived content, not additional real observations.

The run completed with five source images, five crops, zero episodes, zero environment steps, and zero training steps. Run seal verification passed without errors. The registered report and metrics describe a resubstitution construction check; there is no episodic evaluation record. This check must not enter a policy-distance leaderboard or held-out perception score.

The earlier sealed construction run `b5a1748c-0fb1-4787-9d05-a077fd2c5443` is retained unchanged for audit history. It used the episodic evaluation API, whose default incorrectly labeled the image check as one evaluation episode. Its run-level episode count was zero. The current run corrects that recording error by retaining the check as an artifact/metric rather than an episodic evaluation. Both runs use the same five source hashes; they do not represent ten distinct real observations. Prefer the current run for reporting and exclude the earlier evaluation row from episode totals.

## What was checked

All five source images matched their intended state. Other candidate templates fell below the acceptance threshold on those same images, and no source image granted advertisement-close or restart visibility. This checks crop coordinates, manifest loading, hash consistency, and basic matching behavior for the construction images.

There are **zero independent held-out images** and **no measured generalization accuracy**. The five self-matches cannot be reported as 100% real-game accuracy. Template scores are visual similarities, not calibrated confidence probabilities. No generated variations, additional crops, or repeats count as new real labels.

The playing template is a visible pedal graphic, not a validated detector of active physics. An unobserved overlay could retain that graphic. Unknown screens, animations, ads, revivals, focus loss, resizing, and stale observations still require independent evaluation before input can rely on these prototypes. The current prototypes cannot make unattended operation ready.

## Subsequent fixed-template native check

A later native capture, `artifacts/game-discovery/paused-native-normalized.png`, was visually labeled PAUSED before classification. Its source hash differs from all five construction images. The frozen discovery templates returned `unexpected`: the PAUSED template similarity was approximately 0.8813, below its 0.98 acceptance threshold. No playing, advertisement-close or restart evidence was emitted. Thresholds were not retuned on this image.

Sealed run `bf79e106-ec25-47f2-89ab-9f91184f5f55` stores the new image, capture metadata, frozen template files/training provenance, implementation and result. Seal verification passed. This is one correctly rejected unsafe-to-classify frame but an incorrect specific state classification: zero correct state matches out of one, zero false-playing matches, and one unknown prediction. It does not establish broad generalization accuracy. The original five-image construction dataset still has no held-out members; this later test is a separately recorded observation.

## Reproduce offline construction

From the repository root, with the five original PNGs already present:

```powershell
.venv/Scripts/python.exe scripts/label_discovery_screenshots.py --source artifacts/game-discovery --artifacts artifacts
```

Each execution creates a new run and leaves existing runs unchanged. It does not obtain missing screenshots. Regenerate only the source schema with:

```powershell
.venv/Scripts/python.exe scripts/label_discovery_screenshots.py --write-schema schemas/discovery_observations.schema.json
```

For a later explicitly resumed real session, collect complete independent sessions with capture timestamps, frame geometry, UI labels, and delivered-action traces. Allocate entire sessions to training versus held-out evaluation before selecting crops or thresholds. Include unknown/negative screens and report state confusion, false-playing rate, rejection coverage, action-control errors, and observation age. Keep any threshold tuning inside the training partition. Real vehicle/terrain estimation requires separate position, orientation, camera-motion, and terrain labels; this UI dataset supplies none of those measurements.
