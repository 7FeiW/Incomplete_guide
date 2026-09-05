# Plan: Reject duplicate sample IDs

> **Status:** PROPOSED · **Updated:** 2026-09-04
>
> Sample plan only. Repository inspection, implementation, and validation are pending.

This plan follows the image-classifier example in
[chapter 14](../../../../docs/14_Programming_with_LLM_Agents.md#sample-project)
and the task-record workflow in
[chapter 15](../../../../docs/15_Agentic_Workflow.md#task-state).
The Python paths below belong to that illustrative project; its source code and
environment are not included in this documentation repository. The neighboring
[architecture example](../architecture.md) describes the same sample project;
the other rules examples originate from a separate mass-spectrometry project.

## Objective and scope

Reject duplicate `sample_id` values during manifest validation, before dataset
construction proceeds to image loading. Report every duplicated ID once in a
single `ValueError` so the caller can correct the manifest in one pass.

- Preserve the public `Dataset` constructor, manifest schema, and existing path checks.
- Preserve valid rows in their original order, with their existing split assignments.
- Compare IDs as exact strings, without case or Unicode normalization.
- Validate without reading image contents; use synthetic rows and temporary path fixtures.
- Do not modify research datasets, change dependencies, or start training.

These identity rules are choices for this example. Confirm them before adapting
the plan to a project that permits repeated measurements under one sample ID.

## Files and context

All paths in this table are relative to the target Python project's root.

| File | Planned use or change |
| --- | --- |
| `src/project/dataset.py` | Inspect the validation path, then add the duplicate-ID check. |
| `tests/test_dataset.py` | Add regression cases and checks for preserved behavior. |
| `docs/rules/data-validation.md` | Read the agreed constraints created in chapter 15's setup. |
| `docs/plans/duplicate-sample-ids.md` | Save this plan and update its evidence and next action. |
| `docs/state/project-state.md` | Link this task and keep its status current. |

Before editing, read the target repository's instructions, architecture, and
environment setup. Verify the listed paths and constructor behavior against the
code. Record the starting Git revision, working-tree changes, environment or
lock-file identity, and baseline focused-test result here.

## Acceptance criteria

The following are expected outcomes, not observed test results.

| Synthetic IDs in manifest order | Required behavior |
| --- | --- |
| `s1`, `s2` | Accept; preserve row order and split assignments. |
| `s1`, `s1` | Raise one `ValueError` identifying `s1` once. |
| `s2`, `s1`, `s2`, `s1`, `s1` | Raise one `ValueError` identifying both `s1` and `s2`, each once. |
| `s1`, `S1` | Treat the IDs as distinct. |

Also check that validation does not call the image-content loader and that
existing path-validation tests retain their behavior. Cover distinct Unicode
spellings that normalization could otherwise merge. Do not impose an ordering
on the error's ID list unless the target project's contract requires one.

## Implementation steps

1. Inspect the constructor, manifest iteration, path validation, and existing
   fixtures. Run the baseline focused tests and record any pre-existing failure.
2. Add the synthetic regression cases. Run them before changing implementation;
   confirm that failures demonstrate missing duplicate validation rather than
   fixture or environment errors. If duplicates are already rejected, revise the
   plan to address the actual gap.
3. Add the smallest validation change that collects all repeated IDs and raises
   one descriptive error before image loading. Preserve existing iteration and
   path-check behavior.
4. Run the focused tests, full test suite, and lint checks from the
   [sample task's validation instructions](../../../../docs/14_Programming_with_LLM_Agents.md#task-requests)
   in the configured target project environment, from its root. Those commands
   assume uv, pytest, and Ruff are already configured as described in the guide.
5. Review the diff against the acceptance criteria. Record actual commands,
   exit codes, result summaries, and remaining limitations below. Update the
   active-work index with the next action.

## Risks and open questions

- Can manifest rows be consumed only once? Inspect this before adding a separate
  validation pass that could exhaust an iterator.
- Does a caller depend on which validation error appears first? Preserve the
  established path-check contract and resolve any conflict before implementation.
- How are missing or non-string IDs currently handled? Preserve that behavior;
  this task does not define a new schema policy.
- Rejecting previously accepted manifests is intentional here. Report affected
  callers without automatically repairing or deduplicating their data.

## Evidence and current state

| Record | Current value |
| --- | --- |
| Baseline revision and working tree | Not recorded; target project not inspected. |
| Environment and lock-file identity | Not recorded. |
| Baseline focused tests | Not run. |
| Regression failure before implementation | Not run. |
| Implementation changes | None. |
| Focused tests after implementation | Not run. |
| Full test suite and lint | Not run. |
| Diff review | Pending. |

Replace pending entries with observed evidence as work proceeds. Keep failures
and checks that could not run explicit; passing software checks does not establish
improved classifier performance.

## Next action and completion

Next action: inspect the target project and record its baseline, then confirm
the regression cases against its validation contract.

Mark the task complete only after the acceptance criteria, required checks, and
review are satisfied. Record the final revision or uncommitted diff, unresolved
limitations, and outcome here; remove the completed task from the active-work
index. Retain the plan so a fresh session can recover the evidence.
