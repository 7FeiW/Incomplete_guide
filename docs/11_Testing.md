# Testing and Packaging

Tests help detect unintended changes and make research code safer to reuse.
Start with small tests for deterministic behavior, then add integration tests
for important workflows.

## Unit Tests with pytest

<!-- TODO: Add a small, runnable pytest example and explain its prerequisites. -->

For now, see the [pytest documentation](https://docs.pytest.org/en/stable/).

## Package Releases for `src/` Layouts

References:

- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [Packaging Python Projects tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PyPA sample project](https://github.com/pypa/sampleproject)

From the Python project's root, activate its development environment. The
project must have valid package metadata and build configuration, as described
in [Python Environments and Packaging](04_Python_Env.md#pyprojecttoml-vs-requirementstxt).
Install the build frontend and create both a source distribution and a wheel
in `dist/` (Bash or PowerShell):

```bash
python -m pip install build
python -m build
```

These commands build the configured Python project, not this documentation
repository. See [PyPA's replacement for setup.py commands](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/#what-commands-should-be-used-instead).

## What to Ask an LLM AGNET

Avoid: “Add tests.”

Ask instead:

```text
Inspect <changed code>, its existing tests, and the documented scientific or
behavioral requirements. Report missing cases, boundary conditions, error paths,
and invariants that the current tests do not establish. Propose a focused test
plan with expected behavior and the smallest relevant commands to run. Distinguish
unit tests from checks that need data or scientific review. Do not edit tests,
run expensive jobs, or claim coverage or passing results without observed output.
```
