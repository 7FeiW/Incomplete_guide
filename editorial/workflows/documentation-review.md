# Documentation Review

## Inputs

- The target numbered section or sections.
- The task's requested review scope and intended audience.
- Relevant entries in [`../sources.md`](../sources.md) when the text contains
  material external claims.

## Procedure

1. Read `AGENTS.md`, [`../writing-rules.md`](../writing-rules.md),
   [`../style.md`](../style.md), the complete target section, and nearby
   sections on the same topic.
2. Inspect the existing terminology, cross-references, commands, examples, and
   source links relevant to the requested scope.
3. Check the target for unsupported or overbroad claims, missing prerequisites,
   unsafe or non-portable commands, unclear placeholders, inconsistent terms,
   broken Markdown structure, and duplicated guidance.
4. For each finding, report the guide section, the issue, its consequence,
   evidence inspected or unavailable, and a suggested next action. Distinguish
   demonstrated errors from questions and optional style preferences.
5. Do not edit files. Report the review scope, files read, checks performed, and
   anything that could not be verified.

## Outputs and Stopping Conditions

Return a read-only, location-specific report. Stop after reporting findings; an
editor applies any selected changes in a separate task and checks the resulting
diff, links, fenced blocks, and safe examples as required by `AGENTS.md`.
