# Guide Writing Rules

These rules supplement the repository-level instructions in
[`AGENTS.md`](../AGENTS.md). The repository instructions take precedence if the
two documents conflict.

## Scope and Audience

- Treat the named numbered chapter as the writing scope unless the task names
  other files.
- Write for research programmers with varied software-engineering experience.
- Preserve the guide's practical Python research-workflow focus. Present
  project-specific paths, commands, policies, and resource limits only as
  clearly labelled examples.

## Evidence and Technical Meaning

- Do not invent citations, URLs, commands, software behavior, platform policy,
  benchmark results, or successful test output.
- Read authoritative sources before adding or materially revising time-sensitive
  software, service, or platform guidance. Link to the specific supporting page.
- Keep commands executable and state their shell, prerequisites, scope, and
  risks when they are not portable or harmless.
- Preserve qualifications, versions, placeholders, units, paths, and the
  distinction between observed results, assumptions, examples, and hypotheses.
- Mark unavailable evidence with a descriptive HTML TODO comment; do not fill a
  citation gap from memory.

## Editing and Review

- Read the complete target chapter, related chapters, and existing duplicated
  guidance before editing.
- Make the smallest change that resolves the task. Preserve chapter filenames,
  reading order, and existing technical meaning unless a task explicitly changes
  them.
- For a language edit, check the diff for altered claims, commands, links,
  placeholders, and Markdown structure.
- For a review, report precise locations, the evidence inspected, the issue or
  question, and a suggested next action. Keep the review read-only.
- State checks actually performed and unresolved questions in the handoff.
