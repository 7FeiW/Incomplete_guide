# Addendum: Agent Environment Rules & Per-file Testing

This addendum highlights two concise recommendations to include in your repository's agent instructions.

1. Agent Environment Rules

- Include explicit environment-selection rules in `.github/copilot-instructions.md` so agents know which environment to use (examples: `dev` venv, `ci` container, or `gpu` machine).
- Provide exact setup/activation commands, required environment variables (e.g., `PYTHONPATH`), and any constraints on secrets/credential access.
- Example:

```
# Preferred environments
- dev: python -m venv venv && venv\Scripts\activate.ps1
- ci: docker run --rm -v $PWD:/work -w /work python:3.11
- gpu: use host with CUDA available, set CUDA_VISIBLE_DEVICES
```

2. Per-file Tests

- Encourage agents to create and run tests separated per module/file (naming convention `test_<module>.py`) so CI can run targeted tests.
- Example commands for running tests:

```
# Run a single test file
pytest tests/test_utils.py -q

# Run all tests
pytest tests/ -q
```

- This approach makes it easier to run focused tests during iterative agent work and speeds up CI by enabling parallelization.
