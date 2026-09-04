# Research Project Guide Repository

This repository contains *FW's Incomplete Guide to Python Research Codebase*, a
work-in-progress collection of Markdown documents about building and operating
Python-based computational research projects. It is a documentation repository,
not a LaTeX thesis or an application codebase.

## Repository layout

- `README.md` introduces the guide and links to its main documents.
- `docs/` contains the numbered guide chapters. Preserve the numeric filename
  prefix because it defines the intended reading order.
- `examples/` contains supporting instruction and configuration examples that
  may be referenced by the guide.
- `.github/` contains repository and GitHub configuration.

There is currently no documentation build system or required test command.
Validate changes by inspecting the rendered Markdown structure, checking links,
and testing any commands or code examples when practical.

## Non-negotiable rules

1. Never fabricate citations, URLs, software behavior, command output,
   benchmarks, or platform policies. If a statement cannot be verified from the
   repository or an authoritative source, add a descriptive HTML comment such
   as `<!-- TODO: verify the supported Python versions. -->` or ask the user.
2. Treat shell commands as executable instructions. Do not include destructive,
   privileged, or cluster-specific commands without explaining their scope,
   assumptions, and risks.
3. Preserve the guide's focus on practical Python research workflows. Do not
   turn general guidance into project-specific requirements unless the text
   explicitly presents it as an example.
4. Do not claim that a command or example was tested unless it was actually
   executed in an appropriate environment.

## Editing workflow

Before editing, read the complete target document and any nearby chapters that
cover the same subject. Search the repository for duplicated explanations,
terminology, commands, and cross-references. Prefer improving the canonical
section rather than adding a competing version elsewhere.

When adding, removing, or renaming a chapter:

- preserve the numbered ordering convention in `docs/`;
- update the document list in `README.md`;
- update all relative links and references to the chapter; and
- distinguish intentional filename changes from incidental spelling cleanup,
  because external links may rely on existing filenames.

Keep changes focused on the requested topic. Do not rewrite an entire chapter
solely to impose a new voice, and do not silently change the technical meaning
while correcting grammar.

## Markdown conventions

- Use GitHub-Flavored Markdown.
- Use one level-1 heading for the document title, then nest headings without
  skipping levels.
- Put a blank line around headings, lists, block quotes, and fenced code blocks.
- Use fenced code blocks with an appropriate language tag, such as `python`,
  `bash`, `json`, `yaml`, or `text`.
- Keep commands directly copyable. Put explanatory prose outside the code block
  unless a comment is part of the example.
- Use relative links for repository files and descriptive link text for external
  resources.
- Keep manual tables of contents and navigation links synchronized with their
  headings. Prefer stable, concise heading text.
- Use standard Markdown constructs instead of raw HTML unless Markdown cannot
  express the required result. HTML comments are appropriate for TODO notes.
- Avoid trailing whitespace except where a deliberate Markdown line break is
  required.

## Writing style

Write for research programmers who may understand their scientific domain but
have varying software-engineering experience.

- Lead with the practical problem, then explain the recommendation, its reason,
  and its limitations.
- Explain what a tool or command consumes, what it does, and what it produces.
- Prefer concrete, runnable examples over abstract prescriptions.
- Define specialized terms and abbreviations on first use.
- Separate universal practices from optional choices and environment-specific
  advice.
- Use cautious language for tradeoffs. Prefer formulations such as "use this
  when" and "this may help" over unsupported absolutes such as "always faster"
  or "best."
- State the comparison axis when comparing tools, for example reproducibility,
  portability, installation speed, memory use, or maintenance burden.
- Keep paragraphs focused and use lists for procedures, checklists, alternatives,
  and sets of constraints.
- Use consistent terminology and capitalization within and across chapters.
- Correct grammar and spelling in edited passages while preserving the author's
  direct, practical tone.

## Technical examples

Examples should be small enough to understand in isolation and realistic enough
to adapt to a research project.

- Include prerequisites, assumed working directory, and relevant platform when
  they affect whether a command succeeds.
- Use placeholders such as `<project-root>`, `<environment-name>`, and
  `<job-id>` for values readers must replace, and explain each placeholder.
- Do not expose credentials, access tokens, private hosts, personal paths, or
  unpublished data locations. Use obvious synthetic values.
- Pin versions only when reproducibility requires it. Explain whether a version
  is an example, a tested constraint, or a minimum requirement.
- For Python examples, prefer readable code with explicit inputs and outputs.
  Account for errors where omission would teach an unsafe pattern.
- For configuration examples, keep comments valid for the shown format. JSON
  does not support comments.
- For shell examples, identify whether the syntax targets Bash, PowerShell, a
  SLURM script, or another environment. Do not mix shell dialects in one block.
- For HPC and SLURM examples, avoid presenting site-specific partitions,
  modules, paths, account names, or resource limits as portable defaults.
- For performance advice, distinguish measurements from hypotheses and recommend
  profiling before optimization.
- For research workflows, emphasize reproducibility: record code versions,
  environments, configurations, random seeds where applicable, input-data
  provenance, and output locations.

## Sources and links

Prefer primary and authoritative sources, including official project
documentation, standards, and platform documentation. Link to the specific page
that supports the claim rather than a search page or a project home page when a
more precise page exists.

Software commands, APIs, service policies, and HPC instructions can change.
Verify time-sensitive details before adding or materially revising them. When a
source is required but unavailable, leave a descriptive TODO rather than
inventing a reference.

## Validation

For every documentation change:

1. Review the diff for unintended edits, malformed fences, heading-level errors,
   and accidental changes inside commands.
2. Check changed relative links and image paths from the location of the edited
   file.
3. Verify that code blocks use the correct language and shell dialect.
4. Run safe, self-contained examples when the local environment supports them;
   otherwise report that they were not executed.
5. If a Markdown linter or link checker is added later, use the repository's
   documented command and do not silently reformat unrelated files.

In the final response, summarize the documents changed, the substantive effect,
and any validation that was or was not performed.
