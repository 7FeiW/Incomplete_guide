
# Python Environments and Packaging


This document explains practical, beginner-friendly ways to manage Python environments and install your project while developing. It separates the common choices (`pip` vs `conda`) and environment tools (`venv` vs `virtualenv`), shows how to use `requirements.txt` and `environment.yml`, points out cross-platform pitfalls, and gives notes for working on HPC systems.

## What is a Python environment?

- A Python environment is an isolated runtime that contains a specific Python interpreter, installed libraries, and environment-specific settings (PATH, environment variables). It determines which Python executable and which package versions your code uses.
- Why use one: isolation — different projects can use different package versions without conflict; it makes testing, development, and deployment reproducible and safer.

Common environment types:

- `venv` (built-in): lightweight, included with Python 3. Use `python -m venv .venv`.
- `virtualenv` (third-party): older tool that supports more Python versions and extra features.
- `conda` environments: provided by Anaconda/Miniconda; manage both environments and binary packages.


---

## Pip and Conda

`pip` and `conda` are two commonly used Python package managers. `pip` is the Python packaging installer (standard tool). It installs packages from PyPI and other sources. `conda` is an environment manager + package manager (from Anaconda/Miniconda) that installs prebuilt binary packages and manages environments. It is commonly used in data-science stacks.

### Which to choose?

- Use `pip` (with `venv` or `virtualenv`) for most Python projects and when you rely on PyPI packages that install cleanly. Not all packages are available as PyPI wheels — some, like `rdkit`, are commonly installed via `conda`.
- Use `conda` when you need reliable binary packages (e.g., `numpy`, `pandas`, or other compiled C extensions), or when your stack depends heavily on native libraries.
- You can mix `pip` and `conda`, but do so carefully (see "Mixing pip and conda" below).

---

## Venv and Virtualenv

### `venv` (built into Python)

- Available in Python 3.3+ as `python -m venv`.
- Good default for new projects.

Create and activate (macOS / Linux, zsh):

```bash
python -m venv .venv
source .venv/bin/activate
# deactivate when done
deactivate
```

Windows PowerShell (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### `virtualenv` (third-party)

- Older, more feature-complete tool that supports more Python versions and some extra options.
- Install with `pip install virtualenv` and use `virtualenv .venv`.

When to use `virtualenv`? Mostly for legacy projects or when you need features not offered by `venv`. For modern projects, `venv` is typically fine.

---

## Using `pip` and `requirements.txt`

A `requirements.txt` file lists packages for `pip` to install. Common formats:

- `package==1.2.3` (exact/pinned)
- `package>=1.2` (minimum version)
- Git installs: `git+https://github.com/owner/repo.git@branch#egg=package`

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Create a pinned list for deployment:

```bash
pip freeze > requirements.txt
```

Dev dependencies:

- Keep development/test tools in a separate file like `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

Advanced: use `pip-tools` (`pip-compile`) or `poetry` to maintain a clear separation between human-edited dependency lists and pinned deployment lists.

Constraints files

- Use `-c constraints.txt` to constrain versions without listing packages in that file.

### Editable installs (during development)

```bash
pip install -e .
# or with extras
pip install -e '.[dev]'
```

This requires packaging metadata (in `pyproject.toml`, `setup.cfg`, or `setup.py`). Editable installs let you edit source without reinstalling.

---

## `pyproject.toml` vs `requirements.txt`

Quick guidance:

- Prefer `pyproject.toml` for new projects: modern standard (PEP 518/621), keeps project metadata and dependencies together, works with tools like uv/Poetry/PDM/Hatch.
- Keep `requirements.txt` for legacy flows, simple Docker base installs, or when ops expects a flat list.
- Use a lock file from your chosen tool (`uv.lock`, `poetry.lock`, `pdm.lock`) for reproducible builds; export to `requirements.txt` only when needed for deployment images.

Minimal `pyproject.toml` (setuptools backend):

```toml
[project]
name = "myapp"
version = "0.1.0"
description = "Example app"
requires-python = ">=3.10"
dependencies = [
	"requests>=2.32",
	"pydantic>=2.8",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

Common workflows:

- Install from `pyproject.toml` with plain pip (no lock):

```bash
python -m pip install .
```

- Add deps and sync with uv (creates `uv.lock`):

```bash
uv add requests
uv sync
```

- Export to `requirements.txt` for environments that only consume pip requirement files:

```bash
uv export --format requirements-txt > requirements.txt
# or with Poetry: poetry export -f requirements.txt --output requirements.txt --without-hashes
```

- Editable install with extras (works for both setups):

```bash
pip install -e '.[dev]'
```

Rule of thumb: author and lock dependencies in `pyproject.toml` + tool-specific lock; generate `requirements.txt` only when an external system requires it.

---

## Using `conda` and `environment.yml`

Create an environment with a specific Python version:

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

Install packages with conda (prefer for binary/C dependencies):

```bash
conda install numpy pandas -y
```

Export/restore an environment:

```bash
conda env export > environment.yml
conda env create -f environment.yml
```

### Editable installs (during development)
Since `conda develop` not longer avabiale, use `pip` within conda
```bash
pip install -e .
# or with extras
pip install -e '.[dev]'
```

Notes:

- Prefer `conda install` for packages that require compiled binaries (e.g., NumPy, SciPy) to avoid local builds.
- Use the `conda-forge` channel for many community packages: `conda install -c conda-forge <pkg>`.

Mixing `conda` and `pip`:

- If you must use `pip` inside a conda env, install conda packages first, then `pip install` the remaining packages.
- Exported `environment.yml` may include pip-installed packages under a `pip:` section.

---
## UV
[https://docs.astral.sh/uv/]

## Cross-platform issues (pip and conda)

Some things to watch for when supporting Windows, macOS, and Linux:

- Binary wheels: many packages publish prebuilt wheels for common platforms (Linux x86_64, macOS, Windows). On platforms with no wheel, `pip` builds from source, which can require compilers and system libraries.
- Path separators and executable scripts: scripts installed by packages behave differently on Windows (no shebang) vs Unix (shebang lines). Use entry points in `setup.cfg`/`pyproject.toml` to ensure cross-platform behavior.
- Activation commands differ: `source .venv/bin/activate` (Unix) vs `.\\.venv\\Scripts\\Activate.ps1` (PowerShell).
- Permissions: Unix may require `chmod +x` for scripts; Windows uses different execution policies.
- Line endings: CRLF vs LF can cause scripts to fail if the wrong line endings are used. Configure `.gitattributes` or your editor to use LF for scripts.
- Conda availability: `conda` isn't always available on all platforms or in some CI/HPC environments; prefer `venv`/`pip` when `conda` isn't an option.
- Locale and encoding differences can surface in tests; set `PYTHONUTF8=1` or normalize I/O in your code.


---

## Working on HPC / Clusters

High-performance computing (HPC) environments often have site-specific setups (module systems, restricted network, shared filesystems). Important points:

- Check your cluster's documentation before creating environments. Many clusters use Environment Modules (`module load python`) or Lmod.
- You may not be able to install system-wide packages or run `conda` installers without approval.
- Use `--user` installs or create environments in your home directory if allowed:

```bash
pip install --user -r requirements.txt
```

- Consider using containers (Singularity / Apptainer) for reproducible environments on HPC where Docker is not available.
- If using conda, prefer Miniconda installed in your home directory (follow site rules).
- Some clusters provide prebuilt Python stacks; prefer those when they match your needs.

Always consult the cluster's user guide — site-specific instructions override general guidance.

---

## Troubleshooting common problems

- "No wheel available" → try `conda install` (if available) or install system build tools (`build-essential`, compilers).
- Mixing `pip` and `conda` gave strange errors → recreate the environment and install conda packages first, then pip packages.
- SSL / proxy / offline issues → configure `pip` with a trusted-host or use an internal index mirror.
- PATH issues where scripts aren't found → ensure the environment's `bin`/`Scripts` directory is on `PATH` after activation.

---

## Quick command cheatsheet (copy-paste examples)

venv (macOS / Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

venv (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

conda:

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
conda install -c conda-forge numpy pandas -y
conda env export > environment.yml
```

Install from Git (pip):

```bash
pip install git+https://github.com/<owner>/<repo>.git
```

Freeze exact versions for deployment:

```bash
pip freeze > requirements.txt
```

---

## Further resources

- pip documentation: [https://pip.pypa.io/](https://pip.pypa.io/)
- virtualenv project: [https://virtualenv.pypa.io/](https://virtualenv.pypa.io/)
- conda docs: [https://docs.conda.io/](https://docs.conda.io/)
- poetry (alternative packaging): [https://python-poetry.org/](https://python-poetry.org/)
- uv (astral-sh): [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
