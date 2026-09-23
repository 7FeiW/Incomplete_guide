# Additional Research Code Considerations

This section collects practical concerns that apply across a research codebase
but do not belong to one of the more focused sections.

<!-- TODO: Define and document the remaining considerations. -->

## What to Ask an LLM AGNET

Avoid: “Review my research code.”

Ask instead:

```text
Inspect this repository for cross-cutting research-code risks not already
covered by its project, environment, configuration, logging, testing, and
scaling documentation. Report concrete findings under data provenance,
reproducibility, scientific assumptions, privacy or access restrictions,
failure recovery, and maintenance. Cite the relevant paths and distinguish
observed evidence from questions for the project owner. Do not alter data,
experiments, access controls, or source files; return a prioritized review only.
```
