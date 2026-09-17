# Source Record for Guide Revisions

Use this file as a compact record when a change adds or materially revises an
externally verifiable claim. It is intentionally empty until a source is checked
for a specific revision. Existing section links are not automatically verified
by this record.

For each source, add an entry using this template:

```markdown
## <short source title>

- Guide location: <guide path and section>.
- Claim supported: <bounded statement>.
- Source: [<descriptive title>](<authoritative URL>).
- Source location: <section, heading, table, or other precise location>.
- Checked: <YYYY-MM-DD>.
- Limits or follow-up: <conditions, version scope, or TODO>.
```

Do not record inaccessible pages, search snippets, or remembered references as
verified. If the source cannot be checked, leave a descriptive HTML TODO comment
in the affected section instead.

## CheckIfExist reference verification

- Guide location: `docs/16_Academic_Writing_with_LLM_Agents.md`, "Validate a
  Candidate Reference Before Adding It" and "Further Reading."
- Claim supported: CheckIfExist compares candidate references with records from
  multiple scholarly indexes and is useful for locating and checking candidate
  bibliographic metadata.
- Source: [CheckIfExist: Detecting Citation Hallucinations in the Era of
  AI-Generated Content](https://arxiv.org/abs/2602.15871).
- Source location: Abstract.
- Checked: 2026-09-17.
- Limits or follow-up: The work is an arXiv preprint. Its tool output is treated
  as a lead for human review, not as evidence that a source supports a claim.

## Crossref REST API metadata retrieval

- Guide location: `docs/16_Academic_Writing_with_LLM_Agents.md`, "Validate a
  Candidate Reference Before Adding It" and "Further Reading."
- Claim supported: Crossref provides a REST API for retrieving deposited
  scholarly metadata.
- Source: [Crossref REST API documentation](https://www.crossref.org/documentation/retrieve-metadata/rest-api/).
- Source location: "REST API" overview and "Endpoints."
- Checked: 2026-09-17.
- Limits or follow-up: Crossref metadata is a bibliographic record, not proof
  that the cited work supports a manuscript claim.

## Codex subagents and custom agents

- Guide location: `docs/15_Agentic_Workflow.md`, "Configure Codex Subagents by
  Role."
- Claim supported: Local Codex supports project- and user-scoped custom-agent
  TOML files, global subagent defaults, per-role model and reasoning settings,
  inheritance and override behavior, and role-specific sandbox settings.
- Source: [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents).
- Source location: "Choosing models and reasoning," "Approvals and sandbox
  controls," "Custom agents," "Global settings," and "Custom agent file
  schema."
- Checked: 2026-09-15.
- Limits or follow-up: Model availability, supported reasoning levels, and
  configuration fields may change; recheck the page before adapting the
  example.

## Claude Code subagents and model configuration

- Guide location: `docs/15_Agentic_Workflow.md`, "Claude Code Subagent Roles."
- Claim supported: Claude Code stores subagent roles as Markdown files with YAML frontmatter in `.claude/agents/` (project) and `~/.claude/agents/` (user); `name` and `description` are required; `model` accepts aliases, full model IDs, or `inherit`; `effort` accepts `low` through `max`; `tools` is an allowlist and `disallowedTools` a denylist; `CLAUDE_CODE_SUBAGENT_MODEL` sets the default subagent model and resolves after a per-invocation model and the role's `model` field but before the main conversation's model; the `opusplan` alias uses Opus in plan mode and Sonnet for execution; subagents start without the parent conversation's history or previously read files.
- Source: [Claude Code subagents](https://code.claude.com/docs/en/sub-agents) and [Claude Code model configuration](https://code.claude.com/docs/en/model-config).
- Source location: "File Locations," "File Format," "Frontmatter Fields," "Model Field Values," "Tool Access Control," and "Context Isolation" on the subagents page; "Available Model Aliases" and "Subagent Model Configuration" on the model-configuration page.
- Checked: 2026-09-15.
- Limits or follow-up: Model aliases, frontmatter fields, and environment-variable names change between releases; recheck both pages before adapting the example.
