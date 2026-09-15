# Editorial Workflow for This Guide

This directory adapts the academic-writing workflow in
[guide section 16](../docs/16_Academic_Writing_with_LLM_Agents.md) to this documentation
repository. It supports reviewable edits to the numbered guide sections; it is
not a manuscript workspace and does not require paper-specific files such as a
bibliography or claims register.

Before an agent edits a section, it should read the repository's
[`AGENTS.md`](../AGENTS.md), these [writing rules](writing-rules.md), the
complete target section, and nearby sections covering the same topic. For a
writing or style edit, it should also read the [style profile](style.md). It
should inspect existing terminology, commands, and links before adding a
competing explanation.

Use [sources.md](sources.md) for references that support material changes to
externally verifiable claims. The record identifies sources that have been
checked for this guide; it does not make an uninspected source applicable to a
new claim. When a suitable authoritative source is unavailable, add a
descriptive HTML TODO comment instead of inventing a reference.

Use the [documentation-review runbook](workflows/documentation-review.md) for a
read-only review. Reviewers report located findings and do not edit files. One
editor applies selected revisions and performs the checks required by
`AGENTS.md`.
