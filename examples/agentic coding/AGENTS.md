# Project context

This illustrative project trains image classifiers from sample manifests.

## Canonical knowledge

- Architecture and entry points: [docs/architecture.md](docs/architecture.md)
- Plans and their statuses: [docs/plans/](docs/plans/)
- Shared constraints: [docs/rules/](docs/rules/)

## Environment and validation

- Use the uv project environment; setup is documented in README.md.
- Focused tests: `uv run pytest tests/test_dataset.py -q`
- Full tests: `uv run pytest tests -q`
- Lint for this task: `uv run ruff check src/project/dataset.py tests/test_dataset.py`

## Workflow

- Read the architecture, shared constraints, and relevant plan before editing.
- State assumptions and distinguish evidence from hypotheses.
- Make only changes required by the task.
- Do not commit, submit cluster jobs, or modify datasets unless requested.
