# Dependency and candidate-code license review

Review date: 2026-09-11. This inventory covers the major proposed Python stack, development tools, and inspected simulation/RL candidates. Linked upstream license files were inspected. It is not a claim that all optional packages are installed, that all releases have identical licenses, or that every transitive binary has a single top-level license.

No third-party game implementation, game asset, model, or dataset was copied during this review. Original GradientClimb code may use MIT while installed dependencies retain their own terms. See [ADR-001](../../docs/decisions/ADR-001-licensing.md).

## Major dependency families

| Package / role | Inspected upstream terms | Reuse and obligations |
| --- | --- | --- |
| Python runtime | [PSF license and incorporated notices](https://docs.python.org/3/license.html) | Runtime prerequisite. Preserve the runtime's license material when distributing a bundled interpreter. |
| NumPy / arrays | [BSD-3-Clause upstream](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Import dependency. The observed 2.5.3 wheel metadata declares `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`; retain wheel notices when distributing it. |
| PyTorch / learning | [BSD-style license and incorporated notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Import dependency; no copied implementation. CUDA-enabled binary distributions can include separately licensed components. Our MIT license does not relicense those components. |
| Gymnasium / environment API | [MIT](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) | Import and API contract; retain notices if redistributing library code. Optional environment extras require separate review. |
| DuckDB / analytical queries | [MIT](https://github.com/duckdb/duckdb/blob/main/LICENSE) | Import dependency. Extension and bundled-component terms remain their own. |
| PyArrow / Parquet | [Apache-2.0 plus bundled notices](https://github.com/apache/arrow/blob/main/LICENSE.txt) | Import dependency. Keep applicable LICENSE/NOTICE files; identify modifications if redistributing modified upstream code. |
| Pydantic / validation | [MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE) | Import dependency. Installed core and transitive distributions remain separately attributed. |
| FastAPI / local dashboard API | [MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE) | Import dependency. No upstream examples copied into project source. |
| Uvicorn / server | [BSD-3-Clause](https://github.com/Kludex/uvicorn/blob/main/LICENSE.md) | Import dependency. Preserve notices and no-endorsement condition for redistributed code/binaries. |
| psutil / system telemetry | [BSD-3-Clause](https://github.com/giampaolo/psutil/blob/master/LICENSE) | Import dependency with retained notices upon redistribution. |
| Pillow / image I/O and capture | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | Import dependency. This is a distinct permissive license; preserve its full notices and incorporated codec terms. |
| MSS 10.2.0 / native screen capture | [MIT](https://github.com/BoboTiG/python-mss/blob/main/LICENSE.txt) | Optional capture dependency. Installed license text inspected; no upstream source copied. |
| DXcam 0.3.0 / desktop duplication capture | [MIT](https://github.com/ra1nty/DXcam/blob/main/LICENSE) | Optional Windows capture dependency. Installed license text and metadata inspected; no library code or binary is vendored. Capture uses the public desktop graphics interface. |
| comtypes 1.4.16 / COM interface support | [MIT](https://github.com/enthought/comtypes/blob/main/LICENSE.txt) | DXcam dependency. Installed distribution notices remain applicable; no source copied. Windows system components retain their own licenses. |
| opencv-python-headless 5.0.0.93 / pixel matching | [MIT packaging](https://github.com/opencv/opencv-python/blob/5.x/LICENSE.txt), [Apache-2.0 OpenCV core](https://github.com/opencv/opencv/blob/5.x/LICENSE), [additional binary notices](https://github.com/opencv/opencv-python/blob/5.x/LICENSE-3RD-PARTY.txt) | Optional import dependency. The installed third-party license explicitly includes FFmpeg under LGPL-2.1 and further platform-specific codec/component terms. Do not label the whole wheel MIT or Apache-only. No wheel/DLL is bundled in this repository. |
| SciPy / fitting and statistics | [BSD-3-Clause upstream](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Candidate/import dependency. Wheels contain additional notices; do not reduce a binary inventory to this source-license label. |
| HTTPX / API validation | [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md) | Test/development dependency unless runtime use is added. |
| pytest / tests | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | Development dependency; not vendored. |
| Ruff / format and lint | [MIT with incorporated notices](https://github.com/astral-sh/ruff/blob/main/LICENSE) | Development binary; no implementation copied. |
| Browser HTML/CSS/JavaScript | Original project source; browser APIs | No third-party charting library, font, CDN code, or design asset selected by this review. Audit any later addition. |
| tkinter / optional simulation viewer | [Python standard-library wrapper](https://docs.python.org/3/library/tkinter.html); Tcl/Tk retains its incorporated runtime notices | Use the installed interpreter's optional module; no Tcl/Tk interpreter or binary is vendored. |
| FFmpeg / optional video encoding | [Build-dependent LGPL/GPL terms](https://ffmpeg.org/legal.html) | The workstation's external executable is reported as a GPL build. Invoke the user-installed executable as an optional tool only; no executable, libraries, or FFmpeg source is bundled. This is separate from OpenCV's bundled LGPL FFmpeg component. |
| Matplotlib 3.11.1 / exported research plots | [Matplotlib's PSF-based license and bundled notices](https://matplotlib.org/stable/project/license.html) | Optional analysis dependency, not MIT. Installed combined LICENSE and separate DejaVu/STIX font notices were hashed. No plotting-library or font binary is vendored. Preserve applicable component/font notices if redistributing those files or subsets. |
| contourpy 1.3.3 / contour computation | [BSD-3-Clause](https://github.com/contourpy/contourpy/blob/main/LICENSE) | Matplotlib dependency; installed license text inspected. No source copied. |
| cycler 0.12.1 / plot style cycles | [BSD-3-Clause](https://github.com/matplotlib/cycler/blob/main/LICENSE) | Matplotlib dependency; retained notice and no-endorsement conditions apply to redistribution. |
| fonttools 4.65.0 / font processing | [MIT source](https://github.com/fonttools/fonttools/blob/main/LICENSE) plus [external notices](https://github.com/fonttools/fonttools/blob/main/LICENSE.external) | Installed main and external notice files inspected and hashed. External notices include SIL Open Font License test-font material; do not label every incorporated font/resource MIT. |
| kiwisolver 1.5.1 / plot layout | [Modified BSD / BSD-3-Clause](https://github.com/nucleic/kiwi/blob/main/LICENSE) | Matplotlib dependency. Installed full license/policy text inspected; no source copied. |
| pyparsing 3.3.2 / expression parsing | [MIT](https://github.com/pyparsing/pyparsing/blob/master/LICENSE) | Matplotlib dependency; installed metadata expression and notice agree. |
| python-dateutil 2.9.0.post0 / dates | [BSD-3-Clause and Apache-2.0 contribution terms](https://github.com/dateutil/dateutil/blob/master/LICENSE) | The installed notice states BSD terms cover all code; newer/relicensed contributions additionally use Apache-2.0. Retain the complete notice rather than collapsing it to an unexplained single label. |
| six 1.17.0 / compatibility helpers | [MIT](https://github.com/benjaminp/six/blob/main/LICENSE) | dateutil dependency; installed notice inspected. No source copied. |

The dependency list above supports MIT for the project's own source, while dependencies retain their own terms. In particular, OpenCV wheels include an LGPL component, and the optional external FFmpeg executable is a GPL build; this source distribution republishes neither binary. Apache-2.0 components remain Apache-2.0; they are not converted to MIT. [Apache's own license](https://www.apache.org/licenses/LICENSE-2.0) defines its notice, modification, and patent terms.

## Candidate environments and learning systems

| Candidate | Inspected terms | Decision |
| --- | --- | --- |
| [alexzh3/hillclimbracing](https://github.com/alexzh3/hillclimbracing) | Repository declares GPL-3.0 and derivation from Code Bullet; LICENSE page retrieval failed, declaration visible in README | Literature only. Do not import or vendor code/models/assets into this MIT implementation. A future GPL component requires deliberate scope and distribution review. |
| [Code-Bullet/Hill-Climb-Racing-AI](https://github.com/Code-Bullet/Hill-Climb-Racing-AI) | No license identified on the inspected repository landing page | No source or assets reused. Public visibility is not a reuse grant; absence on the landing page is not an exhaustive file audit. |
| [0ql/AI-Hill-Climb-Racing](https://github.com/0ql/AI-Hill-Climb-Racing) | GPL-3.0 declared | Literature only; no code/assets reused. |
| [FahzainAhmad/agent-hill-climb-supervised](https://github.com/FahzainAhmad/agent-hill-climb-supervised) | No license identified in inspected listing; linked dataset terms not audited | No source, demonstrations, or checkpoint reuse. Collect our own local trajectories. |
| [Box2D](https://github.com/erincatto/box2d/blob/main/LICENSE) | Current upstream MIT | Permissive candidate engine; no source copied. Version, wrapper, and binary dependency inventory must match the actual selection. |
| [pybox2d](https://github.com/pybox2d/pybox2d/blob/master/LICENSE) | Zlib, including historical C++/Python notices | Permissive candidate binding. Preserve notices and mark altered sources; do not describe this older binding as current Box2D under MIT. |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) | MIT | Eligible external comparator; importing the package is preferable to copying implementations. |
| [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/LICENSE) | MIT | Eligible recurrent comparator; not vendored. |
| [PufferLib](https://github.com/PufferAI/PufferLib/blob/4.0/LICENSE) | MIT | Eligible framework candidate; inspect selected environment extras and native vendor dependencies if adopted. |
| [DreamerV3](https://github.com/danijar/dreamerv3) | MIT declared by official repository | Eligible later candidate; not installed or copied by this review; JAX/backend distribution needs its own inventory. |

## Distribution boundary and refresh rule

The repository license covers original code and original documentation. It grants no rights in Hill Climb Racing, Fingersoft branding, Google Play Games, screenshots, advertisements, or other third-party content. Local captures are experimental records and remain excluded from ordinary Git distribution. Using an ordinary screen/input interface does not change the game's license.

Keep package versions, wheel/source provenance, and license-file hashes with each release's environment record. Preserve dependency notices with any bundled binary release. For source-only distribution, dependencies are declared rather than republished inside the source tree. Recheck this inventory when a major dependency, framework extra, pretrained model, copied code fragment, font, or asset is added. The [upstream explanation of unlicensed repositories](https://choosealicense.com/no-permission/) supports the conservative no-reuse choice for candidates without a verified grant.

The [installed environment snapshot](installed-license-snapshot.json) records actual distribution versions, available license-file hashes, and the hash of `requirements-lock.txt`. Missing metadata stays null, not an inferred license. Mutable upstream links support this review's date; the snapshot improves traceability but is not a complete binary-component software bill of materials.

For the newly installed capture dependencies, the inspected MSS license SHA-256 is `479b38354134f96c0b5f84e5449ae3497c1c585594bcca48a1164e8b819c8c38`; OpenCV's MIT packaging license is `edef0fac1eb08d29d34563f724742e078da2513e196133a12c6ad9ff01e26107`, and its complete third-party notice file is `c1d60169b55cee56452b227c2fd2a7b9ddda3dd3741065f1c6201072be73fadb`. These hashes identify installed license files, not downloaded wheel hashes.

The subsequent DXcam 0.3.0 license-file SHA-256 is `4fe6baee928b96d2cf0f6a238275acfd86182cdaec6e8146654f34cf08c1c9b3`, and comtypes 1.4.16 is `3b1767f010980b46926b23bf0afce5d72f3359ee5e2b27baca71b9b4209ab383`. Both installed metadata expressions and inspected license texts identify MIT.

The analysis-extra supplement adds eight installed distributions and eleven notice files to the snapshot. Matplotlib's combined LICENSE hash is `822e8e528147569a41975592aee19c11992ab667ba50451cd929031d5fc74491`; its separate DejaVu/STIX notices and FontTools' external notices are included individually. This remains an installed-notice inventory, not a full binary-component audit or a grant over third-party fonts.

## Notebook environment supplement, 2026-09-11

The optional analysis extra now declares nbformat 5.11+, nbclient 0.11+,
ipykernel 7.3+ and nbconvert 7.17+, alongside Matplotlib. The installed versions
are nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.3.0 and nbconvert 7.17.1.
The lock captures those versions and their installed dependencies; no interpreter,
kernel, debugger, browser library or notebook dependency is vendored here.
NumPy and PyTorch versions were not changed by this addition.

The inspected [nbconvert license](https://github.com/jupyter/nbconvert/blob/main/LICENSE)
and [ipykernel license](https://github.com/ipython/ipykernel/blob/main/LICENSE)
are BSD-3-Clause, matching the installed notices. The installed nbformat,
nbclient, Jupyter client/core, IPython and traitlets notices also use BSD-3-Clause.
`nest-asyncio2`'s installed full notice is BSD-2-Clause; its short metadata value
"BSD" is not the complete license identification. Auxiliary Python packages
retain their MIT/BSD/Apache/PSF notices as recorded in the snapshot.

Two binary/component boundaries deserve explicit identification. The
[PyZMQ package license](https://github.com/zeromq/pyzmq/blob/main/LICENSE.md)
is BSD-3-Clause, while its installed wheel separately includes ZeroMQ under
MPL-2.0, libsodium under ISC, and a Tornado notice. The installed debugpy package
declares MIT, but `debugpy/ThirdPartyNotices.txt` identifies incorporated
PyDev.Debugger under EPL-1.0 and additional components. Neither whole binary
distribution should be labeled solely BSD or MIT. Preserve the complete shipped
notices if redistributing those dependencies; this repository declares optional
imports and does not redistribute either binary.

Other incorporated notices inspected include Bleach's MIT html5lib component,
Jedi's Apache-2.0 typeshed material and MIT Django stubs, and the JupyterLab
Pygments extension's separate frontend third-party-license inventory. Package
metadata can omit the expression even when an installed license file exists;
those omissions remain null in the machine-readable snapshot.

`scripts/snapshot_dependency_licenses.py --refresh-lock` reproduces the installed
inventory without installing software. The refreshed snapshot contains 93
distributions and 201 discovered notice files, including incorporated-component
notices. Its current lock SHA-256 is
`4af9e34a9541217167fa97886535cd4b958c9804a67cdbe847540c6b06dd8f8e`.
This is a notice inventory, not a complete binary-component audit. The project
source remains original MIT code; dependency and game-asset licenses stay separate.
