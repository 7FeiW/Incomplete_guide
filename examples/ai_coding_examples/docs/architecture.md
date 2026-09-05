# Sample project architecture

> **Status:** REFERENCE · **Updated:** 2026-09-04
>
> Illustrative image-classifier project used in chapters 14 and 15.

This project trains image classifiers from sample manifests. This document maps
the dataset-validation boundary used by the
[programming example](../../../docs/14_Programming_with_LLM_Agents.md#sample-project)
and [workflow setup](../../../docs/15_Agentic_Workflow.md#step-by-step-setup).
The Python source and environment are illustrative and are not included in this
documentation repository. Verify these assumptions against your project before
using the example.

## Project layout

Paths below are relative to the sample Python project's root. The source files
and environment are assumed to exist; the shared documents are created during
chapter 15's setup.

```text
sample-project/
├── README.md                       # Purpose, environment setup, and navigation
├── pyproject.toml                  # Project and development dependencies
├── uv.lock                         # Recorded dependency resolution
├── src/
│   └── project/
│       └── dataset.py              # Dataset constructor and manifest validation
├── tests/
│   └── test_dataset.py             # Dataset and manifest-validation tests
└── docs/
    ├── architecture.md             # Stable component map and data boundaries
    ├── rules/
    │   └── data-validation.md       # Agreed manifest-validation constraints
    ├── plans/
    │   └── duplicate-sample-ids.md  # Task scope, status, progress, and validation evidence
    ├── knowledge/                  # Reusable procedures as they are established
    └── findings/                   # Supported conclusions as work proceeds
```

The guide supplies this architecture and the
[sample duplicate-ID plan](plans/duplicate-sample-ids.md). Create the validation
rules using the chapter 15 template. The other files in
this example directory's `rules/` folder originate from a mass-spectrometry
project and need adaptation before use here.

## Components and responsibilities

| Component | Input | Responsibility and output |
| --- | --- | --- |
| `Dataset` in `src/project/dataset.py` | Manifest rows containing `sample_id`, `path`, and `split` | Validate the manifest and construct a dataset for subsequent image loading. |
| Manifest validation | Row identifiers, paths, and split assignments | Apply the agreed checks; reject invalid input without reading image contents. |
| Image loading | Paths associated with accepted rows | Read image contents after manifest validation. The example does not prescribe an image library or decoding interface. |
| `tests/test_dataset.py` | Synthetic rows and temporary path fixtures | Check validation behavior and preservation of valid rows and splits. |

The training loop, model architecture, label representation, and evaluation
pipeline are outside this example's scope. Inspect their actual interfaces if a
future task needs to change them.

## Data flow and boundaries

```text
Manifest rows
    → manifest validation
    → dataset construction
    → image loading
```

The `Dataset` constructor is the entry point for this task. Validation operates
on manifest metadata before image contents are loaded. Invalid input raises an
error to the caller; valid input retains its row order and split assignments.

The manifest fields used in the example are:

| Field | Meaning | Boundary for this task |
| --- | --- | --- |
| `sample_id` | Identifier for one manifest row | Compare exact strings without case or Unicode normalization. |
| `path` | Location of the row's image | Preserve existing path checks; do not read image contents during validation. |
| `split` | Existing dataset split assignment | Preserve the assignment; validation does not repartition samples. |

The duplicate-ID task requires one `ValueError` listing every duplicated ID
once. This is planned behavior, not a verified implementation. The public
constructor and manifest schema remain unchanged. Missing or non-string ID
handling must be checked against the existing implementation before editing.

## Environment and verification

The sample assumes uv with pytest and Ruff declared as development dependencies,
and imports configured so tests can import `project`. The target project's
`README.md` should document setup. Run checks from that project's root using its
configured environment.

Use the [task's validation instructions](../../../docs/14_Programming_with_LLM_Agents.md#task-requests)
for focused tests, the full suite, and lint checks. Record the actual commands,
exit codes, and results in the plan. No application tests have been run in this
documentation repository.

## Plans and reproducibility

Keep changing progress and test evidence in the
[duplicate-ID plan](plans/duplicate-sample-ids.md), together with its status and
next action. Record the code revision,
working-tree changes, and environment or lock-file identity before implementation.

Use synthetic test inputs without modifying research datasets or starting
training. If later work runs an experiment, record its input-data provenance,
configuration, random seeds where applicable, and output locations separately
from this architecture document.
