# Research Project Guide Repository

*FW's Incomplete Guide to Python Research Codebase* is a work-in-progress
Markdown guide to practical Python computational-research workflows.

## Layout

- `README.md`: guide entry point and chapter list.
- `docs/`: numbered chapters; filenames define reading order.
- `examples/`: supporting examples.
- `editorial/`: writing rules, style profile, source record, and review runbook.

## Working Rules

- Read the complete target chapter, nearby related chapters, and existing
  duplicated guidance before editing. Prefer improving the canonical section.
- Preserve numbered chapter filenames. When adding, removing, or renaming a
  chapter, update `README.md` and every affected relative link.
- For documentation edits, read
  [`editorial/writing-rules.md`](editorial/writing-rules.md) and
  [`editorial/style.md`](editorial/style.md). For material external claims,
  commands, software behavior, or policy, also read
  [`editorial/sources.md`](editorial/sources.md). Use the
  [`documentation-review` runbook](editorial/workflows/documentation-review.md)
  for read-only reviews.
- Do not fabricate citations, URLs, behavior, command output, benchmarks, or
  policies. Use an authoritative source or a descriptive HTML TODO comment.
- Keep the guide focused on adaptable Python research practices. Clearly label
  project-specific or environment-specific examples.
- Treat shell commands as executable instructions: state prerequisites, shell,
  scope, and risks when relevant. Do not claim a command was tested unless it
  was executed in an appropriate environment.

## Markdown and Validation

- Use GitHub-Flavored Markdown, one level-1 title, nested headings without
  skips, fenced blocks with language tags, descriptive links, and blank lines
  around headings, lists, and code blocks.
- Keep examples small, copyable, and explicit about placeholders and platform
  assumptions. Do not expose credentials, private paths, or unpublished data.
- Review the diff for accidental changes, malformed fences, and heading errors;
  check changed relative links and image paths; run safe self-contained examples
  when practical.
- In the handoff, name the documents changed, substantive effect, checks run,
  and anything not verified.
