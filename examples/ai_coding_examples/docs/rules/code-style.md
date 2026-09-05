## Documentation Placement

- **Docstrings**: In-code documentation using Google-style docstrings (see formatting.md)
- **Project docs**: Place in the correct `docs/` subdirectory — see "Docs Layout" below
- **Config docs**: Inline comments in YAML files in `config/`
- **README files**: Only at project root and `docs/README.md` (the docs index); do not create
  README.md in other subdirectories unless explicitly requested
- **Do NOT create**: Markdown files in `src/` or alongside code files

## Docs Layout

`docs/` is organized by *what a document is for*, not by topic. Put a new doc in the subdirectory
matching its kind:

| Directory | Holds | Example |
|-----------|-------|---------|
| `docs/guides/` | How to run something | `run_inference.md` |
| `docs/reference/` | How the system works today | `feature_set_audit.md` |
| `docs/findings/` | A measurement and its conclusion | `ce_extent_headroom_finding.md` |
| `docs/plans/` | Something proposed, in progress, or stopped | `model_improvement_plan.md` |
| `docs/notes/` | Scratch and literature notes | `lz_molecular_generation.md` |
| `docs/figures/` | PNGs referenced by docs | — |

The append-only analysis logs (`missing_peak_findings.md`, `missing_peak_findings_m-h.md`) stay at
`docs/` top level. Add sections to them with `/analyze-subclass` and `/analyze-subclass-mh` rather
than editing by hand.

### Every doc needs a status banner

Insert directly under the H1, before any other content:

```markdown
# Some Plan

> **Status:** ACTIVE &nbsp;·&nbsp; **Updated:** 2026-08-02
>
> One line on where this stands and what happens next.
```

Valid statuses: `GUIDE`, `REFERENCE`, `ACTIVE`, `PROPOSED`, `LANDED`, `FINDING`, `CLOSED`,
`CLOSED NULL`, `SUPERSEDED`, `STOPPED`, `CURATED ANALYSIS`. Update the banner whenever the doc's
conclusion changes — a stale `ACTIVE` on a dead line is worse than no banner.

### Closed and negative results are kept, not deleted

A plan that failed stays in `docs/plans/` marked `CLOSED NULL` / `STOPPED` / `SUPERSEDED`, with the
reason it stopped. These prevent re-running a line that was already measured. Do not archive or
delete them.

### When adding, moving, or renaming a doc

1. Add a row for it in [docs/README.md](../../docs/README.md) — every doc must be indexed.
2. Update any `docs/<path>.md` references in `src/`, `tests/`, `scripts/`, `config/`, `slurm_scripts/`,
   and `INSTALL.md`. Search **both** `*.yaml` and `*.yml`, plus `*.sh` and `*.pyx`.
3. Check relative links inside the doc itself — a link out of `docs/` needs `../../`, and a link to
   `docs/figures/` from a subdirectory needs `../figures/`.

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