# Workstation assessment — 2026-09-11

Observed: Intel Core i9-14900HX, 24 physical / 32 logical cores, 63.71 GiB RAM,
NVIDIA GeForce RTX 4090 Laptop GPU with 16376 MiB VRAM, NVIDIA driver 616.56,
driver CUDA maximum 13.4. PyTorch CUDA visibility is recorded by `doctor` after
the isolated runtime installation; driver capability alone does not establish it.

Python 3.13.5 at C:/Python313/python.exe; Node 24.4.1; FFmpeg 9.0.1; Git and
GitHub CLI available. Project drive initially had ~95.8 GiB free. Use bounded
frame recording and avoid raw full-resolution continuous archives.

Existing machine software and drivers were left intact. The project uses `.venv`.
CUDA-enabled PyTorch is a project dependency, not a system CUDA reconfiguration.
The local Hill Climb Racing installation runs under Google Play Games (`crosvm.exe`)
and was launched using its registered application URI. No game files or private
memory are read. Window and package metadata are discovery evidence only.

Sandbox observations: Git ref creation and pip temporary build files required
the normal escalation mechanism. Hardware CIM queries also required read access
outside the sandbox. Subsequent runs should use the project's documented CLI.
