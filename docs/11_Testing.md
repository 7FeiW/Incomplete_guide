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

Build Source Package:

```bash
python setup.py sdist
```

Build Binary Package (Optional):

```bash
python setup.py bdist_wheel
```
