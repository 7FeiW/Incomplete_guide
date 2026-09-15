# Research Project Guide Repository

*FW's Incomplete Guide to Python Research Codebase* is a work-in-progress
Markdown guide to practical Python computational-research workflows.

## Layout

- `README.md`: guide entry point and chapter list.
- `docs/`: numbered chapters; filenames define reading order.
- `examples/`: supporting examples.
- `editorial/`: writing rules, style profile, source record, and review runbook.

## Working Rules

### Non-negotiable Writing Rules

1. Never fabricate citations. If a required citation is unknown, add a
   descriptive HTML TODO comment that identifies the claim requiring support.
   Do not insert an empty citation or invent authors, titles, venues, or
   publication years.
2. Ask when scientific content, the intended argument, or the correct technical
   phrasing cannot be established from the repository or supplied evidence.
3. Do not create thesis figures unless the user explicitly overrides this rule.
   During review-only tasks, report the missing figure without changing the
   file. When an edit requires a figure placeholder, preserve an existing
   placeholder or add a blank Markdown or HTML figure placeholder with a
   descriptive TODO comment, then ask what the figure should contain.

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
