# Paper Project Context Example

> This is an illustrative layout for a computational research paper. Adapt paths
> and requirements to the actual project. No paper sources are included here.

## Example Layout

```text
paper-project/
├── AGENTS.md
└── manuscript/
    ├── paper.tex
    ├── sections/
    │   ├── introduction.tex
    │   ├── methods.tex
    │   ├── results.tex
    │   └── discussion.tex
    ├── references.bib
    ├── figures/
    ├── supplementary/
    ├── evidence/
    ├── project-context.md
    ├── notation.md
    ├── style.md
    └── workflows/
```

`paper.tex` is the example LaTeX entry point; `sections/` contains its prose.
`references.bib` holds verified metadata and citation keys. `figures/` holds
publication figures; `supplementary/` holds additional methods and results.
`evidence/` contains compact claim and source notes or links to canonical
experiment records. Preserve code revisions, environments, configurations,
seeds where applicable, data provenance, and result locations in those records.
Keep large outputs and restricted data in their established storage locations.

`notation.md` is the definition of record for symbols, abbreviations, units, and
notation conventions. Define a symbol there before it enters the draft, and keep
renames in their own revision pass.

Markdown or Word projects can use their existing manuscript entry point instead.
The review procedures do not require LaTeX or this exact directory structure.

## Manuscript Requirements

Record the research question, intended contribution, audience, target venue,
current draft stage, terminology, and author-approved style samples. Verify the
venue's current formatting, length, supplementary-material, and AI-disclosure
requirements before treating them as constraints.

<!-- TODO: record the target venue and verified submission requirements. -->

## Build and Export

Use the actual project's documented build or export procedure. Verify the entry
file, compiler, bibliography backend, packages, and reference commands rather
than assuming a fixed compilation sequence. No build command is prescribed by
this example, and no manuscript build has been performed here.

<!-- TODO: document the actual manuscript build or export command and prerequisites. -->

Inspect the rendered manuscript for unresolved citations, references, equations,
figure placement, and formatting. For build failures, inspect logs and source
configuration before proposing auxiliary-file cleanup. Identify the diagnosis
and proposed cleanup files before changing them. Enable TeX shell escape only
when the project requires it and trusted source inputs have been inspected.
