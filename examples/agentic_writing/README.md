# Agentic Academic Writing Example

This example separates agent rules, a paper voice profile, and review skills.
It contains no paper sources; the project layout is illustrative.

## Shared Files

- [AGENTS.md](AGENTS.md): concise evidence and review rules.
- [manuscript/style.md](manuscript/style.md): preserved voice profile; adapt to the author and venue.
- [manuscript/project-context.md](manuscript/project-context.md): illustrative paper layout and validation notes.
- [Review coordination](manuscript/workflows/review.md).
- [Logic review](manuscript/workflows/logic-review.md).
- [Proof review](manuscript/workflows/proof-review.md).
- [Mathematics review](manuscript/workflows/math-review.md).
- [Algorithm review](manuscript/workflows/algorithm-review.md).
- [Evidence review](manuscript/workflows/evidence-review.md).
- [Writing review](manuscript/workflows/writing-review.md).
- [Author style](manuscript/workflows/author-style.md).
- [Codex wrappers](.agents/skills/) and [Claude Code wrappers](.claude/skills/).

## Adoption

1. Merge the entry point into the target project's existing `AGENTS.md`, or
   `CLAUDE.md` for Claude Code. Keep existing project instructions.
2. Copy `manuscript/` into the target project, or update links and wrapper paths
   to the locations the project actually uses. The wrappers resolve runbooks as
   `manuscript/workflows/<name>.md` from the project root. That path is this
   example's choice; a LaTeX-only project with sources at the root may prefer a
   shorter directory such as `editorial/`. Edit the wrapper paths to match.
3. Replace paper-specific notes and verify the cited style samples. They are
   absent here. Agree on author preferences and venue requirements.
4. Copy your agent's wrappers into its corresponding project skill directory.
   Wrappers assume shared files at the target project root.
5. In a fresh session, ask the agent to identify applicable rules, missing
   evidence, target draft, and review scope before proceeding.

Invoke `$logic-review` in Codex or `/logic-review` in Claude Code with the
section and draft version. Other names are `proof-review`, `math-review`,
`algorithm-review`, `evidence-review`, `writing-review`, and `author-style`.
Specify review or edit for style work; review is the default.

Run roles sequentially or in separate sessions. This example does not configure
automatic agent spawning. See [Guide section 16](../../docs/16_Academic_Writing_with_LLM_Agents.md).
