# PIP and Conda (If you have a src directory)

Use PIP or Conda editable package for package development. This is very useful to **avoid *absolute imports***.

* If you use pip as main package manager: `pip install -e .`
* If you use conda as main package manager: `conda develop install .`. If this is the first time, you will have to install conda develop.

If you wish to let users install this package from git:

```bash
# For HTTP
pip install git+https://bitbucket.org/<project_owner>/<project_name>
# Example: pip install git+https://bitbucket.org/egemsoft/esefpy-web
```

```bash
# For SSH
pip install git+ssh://git@bitbucket.org/<project_owner>/<project_name>.git/
# Example: pip install git+ssh://git@bitbucket.org/egemsoft/esefpy-web.git
```

```bash
# For Local Git Repository
pip install git+file///path/to/your/git/project/
# Example: pip install git+file:///Users/ahmetdal/workspace/celery/
```

Python packaging, development installs, and environments
-----------------------------------------------------

When developing Python projects it's important to separate the project environment from system Python and to make your package installable for development.

Virtual environments (recommended):

```
python -m venv .venv
source .venv/bin/activate   # Unix / macOS
.venv\Scripts\Activate    # Windows PowerShell
```

Editable installs (local development):

- With legacy `setup.py` or `setup.cfg`:

```
pip install -e .
```

- With modern `pyproject.toml` / `poetry` / `flit`, follow the tool's dev-install instructions, or use `pip install -e .` if the project supports editable installs.

Extra tips for packaging and dependencies:

- Use `requirements.txt` for simple dependency lists for runtime, and `requirements-dev.txt` for development/test tools.
- For reproducible environments, prefer `conda` or `poetry` lock files (`poetry.lock`, `environment.yml`, or `pip-tools` generated `requirements.txt`).
- Use `pip-tools` (pip-compile / pip-sync) or `poetry` to manage pinned dependencies.

Installing directly from a Git repository:

```
# HTTP
pip install git+https://github.com/<owner>/<repo>.git

# SSH
pip install git+ssh://git@github.com/<owner>/<repo>.git

# Specific branch, tag, or commit
pip install git+https://github.com/<owner>/<repo>.git@branch-or-tag
```

Editable extras (dev dependencies):

```
pip install -e '.[dev]'
```

Best practices for local development and CI:

- Keep tests runnable locally and in CI; run `pytest` from the project root.
- Include a simple `Makefile` or `task` scripts for common tasks: `make test`, `make lint`, `make build`.
- Use `pre-commit` hooks to enforce formatting (Black), linting (Flake8, ruff), and commit message standards.
- Avoid committing credentials or large datasets; use environment variables and `.gitignore`.

Useful one-liners:

- Freeze exact versions for deployment:

```
pip freeze > requirements.txt
```

- Recreate environment from `requirements.txt`:

```
python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt
```

Further reading and tools:

- Git official docs: https://git-scm.com/docs
- Pro Git book: https://git-scm.com/book/en/v2
- pip-tools: https://github.com/jazzband/pip-tools
- Poetry: https://python-poetry.org/
- pre-commit: https://pre-commit.com/


