## Documentation Placement

- **Docstrings**: In-code documentation using Google-style docstrings (see formatting.md)
- **Project docs**: Place in `docs/` directory (e.g., `docs/feature_name.md`)
- **Config docs**: Inline comments in YAML files in `config/`
- **README files**: Only at project root; do not create README.md in subdirectories unless explicitly requested
- **Do NOT create**: Markdown files in `src/` or alongside code files

## Coding Standards
- **Type hints**: All functions must have type hints where applicable
- **Naming**: Snake case for functions/vars, PascalCase for classes
- **Device handling**: Always check for CUDA availability; support CPU fallback
- **Random seeds**: Set seeds (torch, numpy, random) for reproducibility
- **Logging**: Use Python logging module, not print statements
- **Tests**: Write unit tests for models and utilities; use pytest fixtures. Prefer assertions on expected values/shapes/edge cases instead of len/non-null smoke checks. When adding features, include at least one test that validates a known molecule/example and one edge case.
- **Test file organization**: Place each related set of tests in its own test file (for example, group tests by module, feature, or class into separate `tests/test_*.py` files). This makes tests easier to run, review, and maintain.
- **Configs**: Use YAML for all experiment configs; never hardcode hyperparameters
- **Error handling**: Always use `raise` statements for explicit failure; never silently fail or use assert for validation
- **Assertions**: Only use `assert` for debugging/development; prefer `raise` with descriptive exceptions for production code
 - **Dictionary access**: Never use `.get(key, ...)` with a default value for expected fields; always use direct access `dict[key]` so the program fails fast (crashes) when a required key is missing. This enforces explicit failure rather than silently continuing with defaults.
- **Imports**: Organize imports in three groups (separated by blank lines): (1) Python standard library, (2) third-party packages, (3) local package imports. Within each group, sort imports alphabetically by library name
- **Indentation**: Use spaces (not tabs); use 4 spaces per indentation level (PEP 8 standard)

## Important Constraints

⚠️ **Never modify**:
- Model input/output shapes without updating all downstream code
- The config schema without deprecation warnings
- The preprocessing steps (breaks reproducibility of past experiments)

✓ **Always do**:
- Save model checkpoints with seed and config info
- Add tests for new model architectures
- Update configs/ when adding new hyperparameters
- Include docstrings explaining mathematical operations
- Run full test suite and validate on validation set before submitting PR
- Respect .gitattributes settings for line endings and file handling