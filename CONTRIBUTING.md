# Contributing

Develop on `dev`; `main` is stable. Keep implementation changes and scientific
claims reviewable. Do not commit checkpoints, raw frames, videos, databases, or
Parquet telemetry. Use immutable run directories and hashed artifact manifests.

Install with Python 3.11+ in an isolated environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train,capture]"
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests scripts
.venv/Scripts/python -m ruff format --check src tests scripts
```

For GPU experiments, install the matching official PyTorch CUDA wheel separately;
record the actual runtime with `gradientclimb doctor`. `requirements-lock.txt`
captures the research workstation, while `pyproject.toml` defines portable bounds.

Scientific changes require a simulator/calibration version when mechanics change,
disjoint evaluation seeds, declared prior training, and real elapsed clocks. Treat
failed runs as data. Add regressions for substantive bugs. Do not claim real-game
competence from simulator-only tests. Follow the boundary in the README and the
full [research protocol](docs/methodology/qualification.md).
