# Agentic Research Workflow: Knowledge, Rules, and State

The goal of this workflow is to keep a human in the loop while allowing
**large language model (LLM) agents** to make useful progress across sessions.
Humans and LLM agents share persistent project memory through version-controlled
documentation: humans record decisions, constraints, and corrections, while LLM
agents read and update the same record as work progresses. The human remains
responsible for approving scientific decisions, consequential actions, and final
results.

LLM agents are most useful in a research project when they are provided with
three kinds of information for every query, either directly in the query or
through the project context loaded by the LLM agent:

1. **Knowledge:** facts about the project and its scientific context.
2. **Rules:** instructions for how work must be performed and which actions are
   prohibited.
3. **State:** what has already happened, what is currently in progress, and what
   remains to be verified.

These concerns need different storage. Putting everything into a tool-specific
instruction file creates a long prompt that becomes difficult to maintain,
while keeping everything in chat makes knowledge disappear between sessions.
This chapter defines an agent-independent workflow for deciding where each kind
of information belongs.

The durable knowledge and state of a research project should remain
**tool-neutral**. Store architecture, decisions, findings, plans, and task state
in ordinary files under `docs/` so that collaborators, Claude Code, GitHub
Copilot, Codex, and future tools can use the same record. Tool-specific files
should tell an LLM agent how to find and apply that shared record; they should
not become the only place where the record exists.

Shared persistent memory also reduces the need to reconstruct project context
in every prompt. This can reduce token usage and leave more of the context window
available for the current task. Concise, relevant, and current documentation can
also improve agent reliability by giving the model stable facts and an explicit
work state. These benefits are not automatic: duplicated, stale, or excessively
long documents consume context and can reduce performance.

The workflow applies to Claude Code, Codex, GitHub Copilot, and other LLM
agents. The accompanying
[Claude example](../examples/ai_coding_examples/claude/) provides one concrete
implementation from a Python mass-spectrometry project. Its project names,
paths, environments, and scientific heuristics are not portable defaults.

LLM-agent products change frequently. Verify tool-specific feature details in the
official documentation for the agent being configured.

## Table of Contents

1. [Core Setup](#core-setup)
2. [Keep Project Knowledge](#keep-project-knowledge)
3. [Define and Enforce Rules](#define-and-enforce-rules)
4. [Keep Work State](#keep-work-state)
5. [Use a Repeatable Session Workflow](#use-a-repeatable-session-workflow)
6. [Package Recurring Research Workflows](#package-recurring-research-workflows)
7. [Apply the Pattern Across Agents](#apply-the-pattern-across-agents)
8. [Audit the System](#audit-the-system)
9. [Further Reading](#further-reading)

## Core Setup

Use three layers:

| Layer                        | Purpose                                                   | Examples                                           |
| ---------------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| **Project record**     | Shared knowledge and durable state                        | `docs/`, Git, experiment records                 |
| **LLM-agent guidance** | How an LLM agent should find information and perform work | `AGENTS.md`, `CLAUDE.md`, scoped rules, skills |
| **Local context**      | Temporary information for the current LLM agent or task   | Chat history, local memory, working notes          |

The project record is the source of truth. LLM-agent guidance should point to that
record rather than duplicate it. Local context can help an agent continue, but
it is not a durable or shared record.

This creates a human-in-the-loop cycle: the human defines goals and reviews
consequential decisions; the agent reads the shared record, performs bounded
work, and writes back verified state; the human then reviews the result and
decides what becomes accepted project knowledge.

A simple test is whether another LLM agent can continue by reading the repository.
If not, the necessary knowledge or state should be written to `docs/` or another
documented project store.

The rest of this chapter provides Claude Code and Codex examples where their
implementations differ. The workflow itself applies to any LLM agent.

## Keep Project Knowledge

Project knowledge includes architecture, terminology, data contracts,
experimental assumptions, and commands that are already known to work. Divide
it by how broadly and frequently LLM agents need it.

### Use docs as the shared knowledge layer

Detailed knowledge should live in ordinary project documents, close to the work
it explains. Typical examples include:

- `docs/architecture.md` for components, entry points, and data flow;
- `docs/data.md` for schemas, provenance, units, and split definitions;
- `docs/experiments.md` for metrics, baselines, and evaluation protocols;
- `docs/decisions/` for decisions and their alternatives;
- `docs/findings/` for measurements and supported conclusions; and
- `README.md` for setup and navigation.

These files are the shared interface between humans and LLM agents. They do
not depend on one vendor's memory format, installation, or session history.
Update them when the underlying fact changes, and link to the canonical document
instead of copying the same explanation into `CLAUDE.md`, `AGENTS.md`, Copilot
instructions, and several agent memories.

Tool-specific entry points should point to the same documents:

```text
docs/architecture.md              <- shared project knowledge
docs/experiments.md               <- shared scientific protocol
docs/project-state.md             <- shared work state
CLAUDE.md                         <- Claude Code routing and behavior
AGENTS.md                         <- Codex routing and cross-agent instructions
.github/copilot-instructions.md   <- Copilot-specific routing and behavior
```

This arrangement allows each tool to keep concise operating rules while using
the same definitions, decisions, and current state.

The example's
[architecture.md](../examples/ai_coding_examples/claude/architecture.md)
illustrates an architecture inventory. In a real project, verify it against the
repository and update it when files move. A stale inventory is worse than a
shorter document that identifies only stable boundaries and entry points.

### Use a thin LLM-agent-specific entry point

Each LLM-agent tool may have a preferred instruction file. Use that file as a
thin entry point containing information the LLM agent needs in nearly every
session. For Claude Code this is `CLAUDE.md` or `.claude/CLAUDE.md`; Codex uses
`AGENTS.md`; GitHub Copilot can use `.github/copilot-instructions.md`. Other
LLM-agent tools may use another format.

| Tool           | Repository entry point                 | More specific guidance                              |
| -------------- | -------------------------------------- | --------------------------------------------------- |
| Claude Code    | `CLAUDE.md` or `.claude/CLAUDE.md` | Nested`CLAUDE.md` files and `.claude/rules/`    |
| Codex          | `AGENTS.md`                          | Nested`AGENTS.md` or `AGENTS.override.md` files |
| GitHub Copilot | `.github/copilot-instructions.md`    | `.github/instructions/*.instructions.md`          |

Claude Code and Codex both combine instructions according to their own discovery
rules. Do not assume that nesting, precedence, or import syntax is identical.
Verify the behavior in the tool's current documentation.

The entry point should contain:

- the project's purpose and main scientific task;
- the canonical documentation paths;
- the supported environment and routine validation commands;
- important architectural boundaries;
- the location and policy for data, configurations, and results; and
- a small set of universal workflow rules.

Prefer links or imports to shared documentation over copied content. A
tool-specific entry should say where authoritative knowledge lives and when the
LLM agent must read it. If a fact would also help another LLM agent or a human
collaborator, put the fact in `docs/` first.

For example:

```markdown
# Project context

This project trains graph neural networks from versioned molecular datasets.

## Canonical knowledge

- Architecture and entry points: [docs/architecture.md](docs/architecture.md)
- Dataset schema and provenance: [docs/data.md](docs/data.md)
- Evaluation protocol: [docs/experiments.md](docs/experiments.md)
- Active work and plans: [docs/project-state.md](docs/project-state.md)

## Environment and validation

- Run Python through `conda run -n research-gpu`.
- Focused tests: `conda run -n research-gpu pytest <test-file> -q`
- Full tests: `conda run -n research-gpu pytest tests -q`
- Lint: `conda run -n research-gpu ruff check src tests`

## Workflow

- Read the relevant canonical document before editing.
- State assumptions and distinguish evidence from hypotheses.
- Make only changes required by the task.
- Do not commit, submit cluster jobs, or modify datasets unless requested.
```

The included [CLAUDE.md](../examples/ai_coding_examples/claude/CLAUDE.md)
provides a good behavioral core that can also be adapted to `AGENTS.md`: think
before acting, prefer simplicity, make surgical changes, and define verifiable
success criteria. Add repository facts to that core, but keep long procedures
elsewhere.

Claude Code loads `CLAUDE.md`, while Codex constructs an instruction chain from
applicable `AGENTS.md` files before it starts work. In both cases, concise and
specific instructions are easier to maintain than a long handbook. See the
[Claude memory guide](https://code.claude.com/docs/en/memory) and
[Codex `AGENTS.md` guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

### Use scoped instructions for specialized knowledge

Use the agent's scoped-instruction mechanism for language-, directory-, or
task-specific knowledge. In Claude Code, place topic files under
`.claude/rules/`. A rule without path frontmatter loads in every session; a
path-scoped rule loads when Claude works with matching files.

```markdown
---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Python rules

- Follow the import aliases configured in `pyproject.toml`.
- Add type annotations to public interfaces.
- Run Ruff on every changed Python file.
```

The example already separates:

- [testing](../examples/ai_coding_examples/claude/rules/testing.md);
- [formatting](../examples/ai_coding_examples/claude/rules/formatting.md);
- [imports](../examples/ai_coding_examples/claude/rules/import-conventions.md); and
- [environment selection](../examples/ai_coding_examples/claude/rules/environment.md).

In a working repository, these belong under `.claude/rules/` and should be
scoped where appropriate.

For Codex, place shared guidance in the repository-root `AGENTS.md` and add a
nested `AGENTS.md` only when a directory needs more specific instructions. Use
`AGENTS.override.md` when the local guidance should replace the regular file at
that directory level. For example:

```text
project-root/
├── AGENTS.md
├── src/
│   └── AGENTS.md
└── experiments/
    └── AGENTS.override.md
```

Codex reads from the project root toward the current working directory, so the
more specific file appears later in its instruction chain. Codex `.rules` files
serve a different purpose: they control which commands may run outside the
sandbox and should not be used as substitutes for project knowledge.

### Use agent-local memory as a cache, not a record

Some agents provide local memory across sessions. Claude Code auto-memory and
Codex local memories can retain useful context from earlier work, including
debugging hints and workflow habits. These memories may be machine-local, differ
between collaborators, and are not reproducible project records.

Use auto-memory for facts such as:

- a command that is repeatedly useful on one workstation;
- a local debugging observation that has not yet been confirmed; or
- a personal preference about how results are displayed.

Promote a memory entry into tool-neutral repository documentation when it
becomes team knowledge, an architectural decision, a required workaround, task
state, or evidence used to interpret a result. This makes it available to other
LLM agents rather than only to one tool on one machine. In Claude Code, inspect
loaded project context and auto-memory through `/memory`. In Codex clients that
support local memories, use `/memories` to control and review their use.

Do not instruct a shared workflow to write to a specific user's absolute memory
path. Agent runtimes own their local memory formats and locations, which are not
portable across users, machines, or products. Both Anthropic and OpenAI advise
keeping required team guidance in checked-in instructions or documentation
rather than relying on local memory alone.

## Define and Enforce Rules

Rules answer how an LLM agent should work. Separate guidance that requires judgment
from controls that must apply regardless of the model's interpretation.

### Behavioral rules belong in instructions

Use the tool-specific instruction entry point and scoped instruction files for
rules such as:

- explain assumptions before implementing ambiguous scientific behavior;
- preserve existing interfaces unless a migration is requested;
- avoid unrelated formatting and refactoring;
- write a regression test before fixing a reproducible bug;
- distinguish measured results from interpretations; and
- report which validation commands were actually run.

The example's
[git workflow rule](../examples/ai_coding_examples/claude/rules/git-workflow.md)
uses this approach to keep commits under human control. This is particularly
useful when Git revisions are used as experiment provenance.

### Enforced rules belong in settings or hooks

An instruction such as "never read `.env`" is a request to the model. If the
action must be blocked, use the LLM-agent runtime, operating system, sandbox, or
CI to enforce it.

- In Claude Code, configure allow, ask, and deny permissions in
  `.claude/settings.json`, and review the result with `/permissions`.
- In Codex, use its sandbox and approval settings. Experimental Codex `.rules`
  files can control which matching commands may run outside the sandbox.
- For every tool, retain operating-system permissions, secret management, and
  CI protection as independent security boundaries.

Keep allow rules narrow because they remove per-use prompts or sandbox
restrictions. Verify current behavior in the
[Claude Code permission documentation](https://code.claude.com/docs/en/permissions)
or [Codex rules documentation](https://learn.chatgpt.com/docs/agent-configuration/rules).

Use a hook or deterministic automation when an action must run at a particular
lifecycle event. Examples include validating a configuration after it changes,
running a fast formatter after an edit, or loading a current task summary when a
session starts. Hooks
are deterministic triggers, while skills and instruction files require model
interpretation. Keep hooks fast, idempotent, and safe with untrusted input.
Where an LLM-agent tool does not provide a suitable hook, enforce the check in
the repository's scripts, pre-commit configuration, or CI. See the
[Claude Code hooks guide](https://code.claude.com/docs/en/hooks-guide) for one
implementation.

### Keep rules consistent

Conflicting rules make agent behavior unpredictable. Maintain one canonical
rule for each concern:

| Concern                        | Preferred source          |
| ------------------------------ | ------------------------- |
| Universal project workflow     | Tool-specific entry point |
| Python or directory convention | Path-scoped rule          |
| Hard tool or file restriction  | Permission setting        |
| Deterministic lifecycle action | Hook                      |
| Multi-step task procedure      | Skill                     |

When a rule changes, search all instruction files and remove obsolete copies.
Rules should describe the target repository's actual configuration. For example,
do not state that Black, isort, and Ruff are all required unless their settings
and validation commands agree in the repository.

## Keep Work State

State records where the work stands. Research projects need more than chat
history because code, data, experiments, and interpretations evolve at different
rates.

### Distinguish four kinds of state

1. **Repository state** is the current Git revision, branch, working-tree diff,
   and staged changes.
2. **Task state** records the objective, the plan, completed work, open
   questions, and next checks.
3. **Experiment state** records configurations, input identity, environment,
   execution status, logs, checkpoints, metrics, and failures.
4. **Conversation state** is the agent session transcript and loaded context.

Repository, task, and experiment state are ground truth that every agent should
read fresh rather than recall from private memory. Store durable records in
tool-neutral formats and documented locations so another LLM agent can take over.
Only committed, versioned parts are durable, while an uncommitted diff or a
running experiment is current but still provisional. A resumed conversation is
convenient, but it can contain outdated assumptions and does not substitute for
inspecting the current files.

### Record task state in the repository

For work that spans sessions, maintain a short version-controlled state file,
issue, or plan. A useful format is:

```markdown
# Dataset validation state

> **Status:** ACTIVE
> **Updated:** 2026-09-03

## Objective

Reject duplicate sample IDs before dataset construction without changing the
manifest schema.

## Established facts

- Validation begins in `src/project/dataset.py`.
- Existing callers expect one `ValueError` for an invalid manifest.

## Plan

- Report all duplicate IDs in one error.
- Do not read sample contents during manifest validation.

## Completed

- Added a failing regression test for duplicate IDs.

## Current state

- Implementation is written but the full test suite has not run.

## Next checks

1. Run `pytest tests/test_dataset.py -q`.
2. Run `pytest tests -q`.
3. Review the diff for changes to dataset splitting.

## Open questions

- Should IDs be compared before or after Unicode normalization?
```

Write facts as facts and unresolved ideas as questions or hypotheses. Update the
file at meaningful checkpoints, not after every small tool call. Close the state
record with the final validation result and link to the resulting commit,
issue, experiment, or finding.

Do not write this state only to Claude auto-memory. A tool-neutral state document
lets a new Claude session, another LLM agent, or a human collaborator resume
from the same evidence.

### Record experiment state separately

An experiment record should contain enough information to reproduce or explain
the run:

- code revision and dirty-working-tree status;
- environment or lock-file identity;
- dataset version, manifest, and provenance;
- complete configuration and command-line arguments;
- random seeds and determinism settings;
- hardware, accelerator, and scheduler allocation;
- start time, completion status, and exit reason;
- log, checkpoint, and result locations; and
- metrics with their definitions and aggregation method.

Do not ask an LLM agent to infer missing provenance after the run. Capture it
when the experiment starts. An agent hook may load current task context, while
experiment wrappers should record execution metadata independently of any LLM
agent.

### Use agent sessions for continuity, not truth

LLM-agent transcripts can help continue an interrupted task, but their storage
and availability depend on the tool. For example, Claude Code provides
`claude --continue` and `claude --resume`; Codex clients may provide conversation
history and local memories. Use the selected tool's supported continuation
mechanism when the conversation remains useful, but begin resumed work by
checking:

```text
1. Read the current task-state document.
2. Inspect git status and the relevant diff.
3. Compare the repository with assumptions in the task state.
4. Report any mismatch before making changes.
```

This prevents an old transcript from overriding changes made by a collaborator
or another LLM agent. Claude's
[session guide](https://code.claude.com/docs/en/sessions) and Codex's
[memory guide](https://learn.chatgpt.com/docs/customization/memories) describe
their respective continuity mechanisms.

For concurrent tasks, use separate Git worktrees so sessions do not modify the
same working tree. Do not use parallel sessions merely to increase activity;
split work only when file ownership and integration boundaries are clear.

## Use a Repeatable Session Workflow

The following lifecycle keeps knowledge, rules, and state synchronized.

### 1. Orient

The LLM agent should read the relevant instructions and durable state before
planning:

```text
Read the repository's agent instructions and docs/project-state.md.
Inspect git status and the existing implementation. Summarize the current state,
identify conflicts with the state document, and do not edit files yet.
```

The user verifies that the LLM agent found the correct environment, files, and
task.

### 2. Define the outcome

Convert the request into observable success criteria:

```text
Objective: reject duplicate sample IDs before dataset construction.

Constraints:
- Preserve the public Dataset constructor and manifest schema.
- Do not change dataset splitting.
- Report all duplicates in one error.

Verification:
- A regression test fails before the fix and passes afterward.
- Focused and complete test suites pass.
- The final diff contains no unrelated changes.
```

Ambiguous scientific choices remain open questions until a person or repository
source resolves them.

### 3. Plan

For a multi-file or scientifically consequential task, enter Plan mode or ask
for a read-only plan. The plan should name files, risks, expected state changes,
and checks. Store the accepted plan when the work will span sessions or involve
other collaborators.

### 4. Implement a bounded increment

Make the smallest change that produces a testable result. After each meaningful
increment, inspect the diff and run the narrowest relevant check. Do not combine
a scientific change, dependency upgrade, and broad refactor into one increment.

### 5. Validate

Validate software behavior and scientific meaning separately:

- tests, linting, types, error paths, and compatibility;
- data provenance, leakage, units, metrics, baselines, and interpretation.

The LLM agent must report observed command results rather than claiming a
command was run. Generated scientific explanations and chemical assignments remain
hypotheses until supported by repository evidence or an authoritative source.

### 6. Update durable state

Before ending the session, update the relevant task or experiment record with:

- what changed;
- the plan followed and why;
- validation that passed or failed;
- unresolved questions;
- working-tree or artifact locations; and
- the next concrete action.

Do not store ephemeral narration or the entire chat transcript. Preserve only
information another person or fresh session needs to continue correctly.

### 7. Hand off

A useful handoff is short and evidence-based:

```text
Changed:
- Added duplicate-ID collection in src/project/dataset.py.
- Added three regression cases in tests/test_dataset.py.

Verified:
- Focused tests: 3 passed.
- Full tests: 214 passed.
- Ruff: passed on changed files.

Not verified:
- Unicode normalization behavior remains an open question.

State:
- Changes remain uncommitted.
- docs/project-state.md records the open question for the next session.
```

## Package Recurring Research Workflows

Use a reusable, version-controlled workflow when a task repeats and needs
detailed knowledge or a procedure. Prefer the cross-agent `SKILL.md` convention
when the participating agents support it; otherwise keep a tool-neutral runbook
and add a thin tool-specific wrapper. This keeps the scientific procedure
available to humans and agents that do not load the same skill directory.

| Agent       | Project skill location             | Explicit invocation                 |
| ----------- | ---------------------------------- | ----------------------------------- |
| Claude Code | `.claude/skills/<name>/SKILL.md` | `/<name>`                         |
| Codex       | `.agents/skills/<name>/SKILL.md` | `$<name>` or the `/skills` menu |

Both implementations use a `SKILL.md` file and can keep supporting scripts,
references, and assets beside it. Discovery, invocation, metadata, and
permission behavior remain tool-specific, so consult the current documentation
before sharing one package unchanged between agents.

A research skill should specify:

- when it applies and required inputs;
- authoritative project files and data sources;
- preconditions and environment checks;
- ordered analysis steps;
- evidence required for each classification or conclusion;
- safe tool permissions;
- output schema and destination;
- state that must be updated; and
- validation and stopping conditions.

For example:

```markdown
---
name: analyze-subclass
description: Analyze fragmentation-DAG misses for one chemical subclass.
---

# Analyze one chemical subclass

1. Validate the subclass argument against the analysis table.
2. Load inputs through the repository's read-only analysis helper.
3. Separate absent fragments from underpredicted fragments.
4. Check assignments against measured exact masses.
5. Draft a finding that separates evidence, interpretation, and uncertainty.
6. Update the task-state file only after validation succeeds.
```

Keep permissions and invocation controls in the agent-specific wrapper or
configuration. For example, Claude Code supports skill metadata such as
`disable-model-invocation` and `allowed-tools`; Codex applies its sandbox,
approval policy, and command rules independently of the skill instructions.
See [Claude Code skills](https://code.claude.com/docs/en/slash-commands) and
[Codex skills](https://learn.chatgpt.com/docs/build-skills).

The example uses legacy single-file commands under
[`commands/`](../examples/ai_coding_examples/claude/commands/). These still
work when placed under `.claude/commands/`, but new multi-step workflows are
better represented as skills with supporting scripts and reference files. A
Codex version should use `.agents/skills/` and refer to the same canonical
runbook and repository helpers.

## Apply the Pattern Across Agents

The included Claude example contains the right categories, but its durable
knowledge should be moved into shared locations. A repository that supports
both Claude Code and Codex can use this structure:

```text
project-root/
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   ├── architecture.md
│   ├── project-state.md
│   ├── workflows/
│   │   ├── analyze-subclass.md
│   │   └── add-smarts-rule.md
│   └── findings/
├── scripts/
│   └── research-analysis/
├── .claude/
    ├── settings.json
    ├── rules/
    │   ├── code-style.md
    │   ├── environment.md
    │   ├── formatting.md
    │   ├── git-workflow.md
    │   ├── import-conventions.md
    │   └── testing.md
    └── skills/
        ├── analyze-subclass/
        │   └── SKILL.md
        ├── analyze-subclass-mh/
        │   └── SKILL.md
        └── add-smarts-rule/
            └── SKILL.md
└── .agents/
    └── skills/
        ├── analyze-subclass/
        │   └── SKILL.md
        ├── analyze-subclass-mh/
        │   └── SKILL.md
        └── add-smarts-rule/
            └── SKILL.md
```

Apply these changes when turning the example into a real configuration:

1. Put shared behavior and links to canonical architecture, data, experiment,
   workflow, and state documents in the root `AGENTS.md`.
2. Keep `CLAUDE.md` as a concise Claude Code entry point to the same shared
   documents. Put Claude-specific rules under `.claude/rules/`, with path scopes
   where appropriate.
3. Keep architecture as human-readable repository documentation rather than
   duplicating the full tree in several prompts.
4. Convert the subclass commands into Claude Code and Codex skill wrappers that
   refer to a shared workflow and distinguish measurements from chemical
   interpretation.
5. Split the 509-line SMARTS command into a concise workflow, reference pages,
   and shared executable helpers with tests.
6. Remove `/home/feiw/.../memory` paths. Durable findings belong in the
   repository; agent-local memory is optional and managed by each tool.
7. Replace `FRAGNNET-GPU`, package paths, dataset paths, and test names only with
   values verified in the target repository.
8. Verify scientific constants and heuristics before treating them as rules.
9. Add narrow permissions for secrets, Git mutations, dependency installation,
   data modification, and SLURM submission.
10. Add a task-state template and require each long-running workflow to update
    it at meaningful checkpoints.
11. Keep scientific knowledge, decisions, findings, and task state outside
    `.claude/`, `.agents/`, and other tool-owned directories so humans and all
    LLM agents can discover and use them.

## Audit the System

Audit the same layers regardless of which LLM agent is used:

| Layer        | Question to answer                                                                    |
| ------------ | ------------------------------------------------------------------------------------- |
| Instructions | Which repository and user instruction files were loaded, in what order?               |
| Knowledge    | Does every required fact resolve to a current, shared source?                         |
| State        | Can a new human or agent identify the current goal, evidence, and next action?        |
| Skills       | Which reusable workflows are discoverable, and are their inputs and outputs explicit? |
| Permissions  | Which actions are allowed, prompted, sandboxed, or forbidden?                         |
| Automation   | Which hooks, scripts, and CI checks can change or validate work?                      |
| Tools        | Which external services, environments, and data stores are available?                 |

Use the tool's own inspection features for the implementation details:

| Concern                       | Claude Code                    | Codex                                                                                              |
| ----------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------- |
| Instructions and local memory | Inspect with`/memory`        | Check the applicable`AGENTS.md` chain; inspect local memories with `/memories` where supported |
| Skills                        | Inspect with`/skills`        | Inspect with`/skills` or explicitly invoke `$<skill-name>`                                     |
| Permissions                   | Inspect with`/permissions`   | Review sandbox and approval settings, plus any applicable`.rules` files                          |
| Configuration                 | Use`/doctor` and `/status` | Review the active Codex client configuration and repository instructions                           |

Audit the workflow periodically:

- remove stale or duplicated knowledge;
- promote useful local memory into version-controlled documentation;
- close or update abandoned task-state records;
- verify that documented commands still run;
- test permission rules and hooks in a safe environment;
- review skills for excessive permissions and stale paths; and
- confirm that experiment records still identify their code, data, and
  environment.

For tool-specific diagnostics, see the
[Claude Code configuration debugging guide](https://code.claude.com/docs/en/debug-your-config)
and the [Codex `AGENTS.md` guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

## Further Reading

### Codex

- [Codex `AGENTS.md`](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Codex skills](https://learn.chatgpt.com/docs/build-skills)
- [Codex memories](https://learn.chatgpt.com/docs/customization/memories)
- [Codex command rules](https://learn.chatgpt.com/docs/agent-configuration/rules)

### Claude Code

- [How Claude remembers a project](https://code.claude.com/docs/en/memory)
- [Extend Claude Code](https://code.claude.com/docs/en/features-overview)
- [Explore the `.claude` directory](https://code.claude.com/docs/en/claude-directory)
- [Configure permissions](https://code.claude.com/docs/en/permissions)
- [Automate workflows with hooks](https://code.claude.com/docs/en/hooks-guide)
- [Extend Claude with skills](https://code.claude.com/docs/en/slash-commands)
- [Manage Claude Code sessions](https://code.claude.com/docs/en/sessions)
- [Debug Claude Code configuration](https://code.claude.com/docs/en/debug-your-config)
