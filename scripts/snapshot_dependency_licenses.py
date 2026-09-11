"""Record installed distribution versions and license-file hashes without installing anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("research/literature/installed-license-snapshot.json")
    )
    parser.add_argument("--lock", type=Path, default=Path("requirements-lock.txt"))
    parser.add_argument("--refresh-lock", action="store_true")
    args = parser.parse_args()
    if args.refresh_lock:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
        ).stdout
        rows = []
        for line in freeze.splitlines():
            if line.startswith("-e "):
                if "#egg=gradientclimb" in line:
                    continue  # Install this project separately; no self-pinning dependency.
                raise ValueError("Unexpected editable third-party dependency; pin it explicitly")
            if line.lower().startswith("gradientclimb"):
                continue
            if line.lower().split("==")[0] in ("dxcam", "comtypes"):
                line += "; sys_platform == 'win32'"
            rows.append(line)
        args.lock.write_text("\n".join(rows) + "\n", encoding="utf-8")
    packages = []
    for distribution in sorted(distributions(), key=lambda item: item.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        if name.lower() == "gradientclimb":
            continue
        files = []
        for relative in distribution.files or []:
            lower = Path(str(relative)).name.lower()
            if lower.endswith((".py", ".pyc", ".dll", ".so")) or not any(
                token in lower for token in ("license", "licence", "copying", "notice", "copyright")
            ):
                continue
            path = Path(distribution.locate_file(relative))
            if path.is_file():
                files.append(
                    {"path": str(relative), "sha256": digest(path), "bytes": path.stat().st_size}
                )
        legacy = (distribution.metadata.get("License") or "").splitlines()
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "declared_license_expression": distribution.metadata.get("License-Expression"),
                "legacy_license_first_line": legacy[0] if legacy else None,
                "license_files": files,
            }
        )
    output = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Installed Python distribution metadata and license-file hashes; not a complete binary-component legal audit.",
        "requirements_lock_sha256": digest(args.lock),
        "packages": packages,
        "generator": str(Path(__file__).as_posix()),
        "generator_sha256": digest(Path(__file__)),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "packages": len(packages),
                "notice_files": sum(len(row["license_files"]) for row in packages),
                "requirements_lock_sha256": output["requirements_lock_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
