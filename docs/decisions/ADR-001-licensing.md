# ADR-001: MIT for original GradientClimb source

Date: 2026-09-11
Status: Accepted for the initial source distribution
Scope: Original code and documentation; third-party dependencies and game content retain their own terms.

## Context and evidence

GradientClimb is intended to be a reusable research platform. The major proposed numerical, learning, data, API, image, telemetry, and development dependencies have permissive upstream terms, subject to their own attribution and distribution requirements. The [dependency audit](../../research/literature/dependency-licenses.md) links the inspected licenses and records the difference between top-level source licenses and bundled wheel notices.

Existing HCR simulators do not provide a simple licensed drop-in foundation for this project. [alexzh3's environment](https://github.com/alexzh3/hillclimbracing) and [0ql's clone](https://github.com/0ql/AI-Hill-Climb-Racing) declare GPL-3.0. Code Bullet and the reviewed visual-agent listing did not establish verified reuse rights. The project also cannot redistribute proprietary game assets under its own license.

## Decision

Use MIT for original GradientClimb source and documentation. Import major dependencies through their normal packages. Do not vendor the reviewed HCR clones or copy their implementation, sprites, checkpoints, or datasets. Implement required simulation and learning logic originally, informed by attributed published ideas, and use independently created visualization assets.

Retain all applicable upstream notices in any distributed dependency bundle. Apache Arrow remains Apache-2.0, Pillow remains MIT-CMU, and historical pybox2d remains Zlib if later selected. PyTorch and numerical wheels can carry additional component notices. The repository's MIT notice does not replace any of these terms.

The capture extra adds MSS under MIT and OpenCV with MIT packaging, Apache-2.0 core, and bundled third-party notices including LGPL-2.1 FFmpeg. An optional user-installed FFmpeg executable is a separately supplied GPL build. Neither binary is redistributed in this source repository. A later application bundle requires a renewed review of these binary distribution obligations; the source license remains MIT.

## Alternatives

| Alternative | Benefit | Reason not selected initially |
| --- | --- | --- |
| GPL-3.0 project with reused GPL simulator | Could reduce simulator construction work | Would introduce deliberate copyleft distribution obligations while still requiring action-space, calibration, and asset-provenance work. |
| Apache-2.0 for original source | Explicit patent provisions and broad reuse | Viable alternative; the initial project favors MIT simplicity, with no copied Apache source requiring a choice change. Dependency obligations still apply. |
| No project license | Avoid an immediate selection | Prevents clear reuse by researchers and does not solve dependency or game-content rights. |

## Consequences and verification

The dependency audit must track the actual chosen stack before releases. Capture installed distribution versions and available license-file hashes, and preserve notices if packaging those binaries. A new dependency or code/asset reuse decision can require revisiting this ADR. Local screenshots, videos, and game-derived datasets remain separately governed artifacts rather than implicitly MIT-licensed repository content.

This decision makes no claim that the third-party game or store authorizes every possible use. The user-defined screen/input-only and no-purchase/no-bypass boundaries remain requirements regardless of the research repository's license.
