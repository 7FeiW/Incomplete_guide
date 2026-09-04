# Programming with LLM Agents

Large language model (LLM) coding tools can help explain a codebase, plan
changes, edit files, run tools, and review code. This makes them more like
software agents rather than autocomplete: they can investigate and act, but they
still need a bounded task, relevant context, appropriate access, and review.

This chapter presents the day-to-day interaction loop for tools such as Codex,
Claude Code, and GitHub Copilot. GitHub Copilot provides some concrete interface
and configuration examples, but the core practices are tool-neutral. Features,
availability, and interfaces change frequently; consult the selected tool's
official documentation for current product details.

The next chapter, [Agentic Research Workflow](15_Agentic_Workflow.md), explains
the complementary repository architecture for carrying knowledge, rules, and
work state across tools and sessions.

## Table of Contents

1. [Choose the Appropriate Level of Delegation](#choose-the-appropriate-level-of-delegation)
2. [Give the Agent Project Context](#give-the-agent-project-context)
3. [Write Effective Requests](#write-effective-requests)
4. [Work with an LLM Agent](#work-with-an-llm-agent)
5. [Validate Agent Changes](#validate-agent-changes)
6. [Protect Data and Credentials](#protect-data-and-credentials)
7. [Use Agent Skills and Custom Agents](#use-agent-skills-and-custom-agents)
8. [Connect External Tools with MCP](#connect-external-tools-with-mcp)
9. [Common Failure Modes](#common-failure-modes)
10. [Relationship to the Next Chapter](#relationship-to-the-next-chapter)
11. [Further Reading](#further-reading)

## Choose the Appropriate Level of Delegation

Coding tools use different names for their interfaces, but the work usually
falls into a few levels of delegation. Choose the least autonomous level that
can efficiently produce the evidence or change you need.

| Workflow | Use it for | Expected result |
| --- | --- | --- |
| Explain or suggest | Understanding code, exploring an error, or completing a small block | An explanation or suggestion for immediate review |
| Investigate | Tracing behavior, reproducing a failure, or comparing approaches | An evidence-based report without edits |
| Plan | Designing a multi-file or risky change | A reviewable plan with risks and checks |
| Implement | Editing files and running checks for a bounded task | A working-tree change that still requires validation |
| Review | Examining a diff or pull request | Findings and suggested corrections |
| Delegate | Running a well-specified task in a separate environment | A branch or pull request for later review |

Start with explanation or investigation when the problem is not yet understood.
Ask for a plan when a change crosses components or could alter scientific
behavior. Delegate implementation only after the intended outcome, constraints,
and validation are concrete. Product modes are conveniences, not a substitute
for stating the desired level of access and output.

Regardless of the workflow, inspect generated commands before approving them.
Autonomous execution changes who performs the steps, not who is responsible for
the result.

## Give the Agent Project Context

An LLM sees only the context supplied by its tool. It may not know the complete
repository, uncommitted work, external data, cluster configuration, or decisions
made in another conversation. Store reusable repository-specific facts with the
project, and state what the agent should inspect for the current task.

### Repository-wide instructions: a Copilot example

GitHub Copilot uses `.github/copilot-instructions.md` for general repository
instructions. Other tools use different entry points; chapter 15 compares them.
Include only information that applies broadly, such as:

- the purpose and scientific scope of the project;
- supported Python versions and environment manager;
- the exact commands for setup, formatting, testing, and type checking;
- the locations of source code, tests, configurations, data documentation, and
  experiment outputs;
- naming, logging, configuration, and reproducibility conventions; and
- actions that agents must not perform, such as downloading restricted data or
  submitting large cluster jobs.

For example:

```markdown
# Repository instructions

This project trains image classifiers from versioned manifests in `data/`.
Use Python 3.12 and install development dependencies with `uv sync --dev`.

Run these checks after changing Python code:

1. `uv run ruff check .`
2. `uv run pytest -q`

Never commit datasets, credentials, model checkpoints, or generated results.
Do not submit SLURM jobs unless the user explicitly requests it.
```

Instructions should reflect commands that actually work in the repository.
GitHub recommends keeping instructions short, specific, and grounded in
observed needs rather than filling them with generic advice.

### Path-specific instructions: a Copilot example

Use `.github/instructions/NAME.instructions.md` when rules apply only to a set
of files. Each file needs YAML frontmatter with an `applyTo` glob. For example:

```markdown
---
applyTo: "tests/**/*.py"
---

Use pytest fixtures from `tests/conftest.py`.
Name tests after observable behavior, not private implementation details.
Mark tests that require a GPU with `@pytest.mark.gpu`.
```

Applicable instruction files are combined, so avoid contradictory rules. The
[custom-instructions documentation](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions)
lists supported locations and differences between Copilot interfaces.

### Shared agent instruction files

Some Copilot interfaces also recognize common agent instruction files such as
`AGENTS.md`, `CLAUDE.md`, and `GEMINI.md`. These names describe instruction-file
formats or compatibility conventions. They do not, by themselves, select a
particular model or grant tools to an agent.

Use `AGENTS.md` when the same repository is edited by multiple compatible coding
agents. Keep its build, test, safety, and repository-layout guidance consistent
with `.github/copilot-instructions.md`. Do not depend on an assumed precedence
rule to resolve conflicting instructions, because discovery and combination can
vary by Copilot interface.

### Environment and test commands

Tell an agent how to select the intended environment. A useful instruction
identifies the platform and separates one-time setup from routine validation:

```markdown
## Environment

- Local Linux and macOS: run `uv sync --dev` from the repository root.
- Windows PowerShell: run `uv sync --dev` from the repository root.
- Cluster: load the modules documented in `docs/cluster.md`, then run
  `uv sync --frozen --dev` on a compute node.
- GPU tests require a CUDA-capable compute node and are never run on a login
  node.

## Tests

- Focused test: `uv run pytest tests/test_dataset.py -q`
- Complete unit suite: `uv run pytest tests -q`
- GPU tests: `uv run pytest -m gpu -q`
```

Targeted tests shorten the edit-feedback cycle, but test-file naming alone does
not make continuous integration parallel. Parallel execution must be configured
in the test runner or continuous-integration system. Organize tests around
observable behavior and cohesive components rather than requiring one test file
for every source file.

## Write Effective Requests

A good request describes the outcome and its constraints without dictating
unnecessary implementation details. Include:

1. the problem and intended result;
2. relevant files or an existing implementation pattern;
3. scientific and engineering constraints;
4. explicit non-goals;
5. acceptance criteria; and
6. commands that verify completion.

For example:

```text
Add validation for sample manifests loaded by src/project/dataset.py.

Requirements:
- Reject duplicate sample IDs and missing input paths.
- Report all invalid rows in one error rather than stopping at the first row.
- Preserve the current public Dataset constructor.
- Do not read image contents during manifest validation.

Validation:
- Add focused tests to tests/test_dataset.py.
- Run: uv run pytest tests/test_dataset.py -q
- Run: uv run ruff check src/project/dataset.py tests/test_dataset.py
```

For exploratory research, ask for alternatives before asking for implementation:

```text
Compare three ways to store intermediate embeddings for this pipeline. Evaluate
them by random-read performance, portability to the cluster, recovery after an
interrupted job, and ease of recording provenance. Do not edit files. Identify
which measurements are needed before making a recommendation.
```

Do not embed invented repository paths, test commands, or scientific thresholds
in a prompt. Ask the agent to inspect the repository or supply the missing facts.

## Work with an LLM Agent

### Understand existing behavior

First, ask the LLM agent to trace inputs, transformations, and outputs through the
relevant files. Verify its explanation against the code. Semantic search can
miss dynamically imported modules, generated files, notebooks, external jobs,
or configuration supplied outside the repository.

Useful questions include:

```text
Trace how a raw sample ID becomes a model input. Cite every relevant file and
configuration key, and identify any behavior that cannot be established from
the repository.
```

```text
Find every entry point that writes evaluation metrics. Report the output format,
destination, and handling of interrupted writes. Do not change files.
```

### Plan the change

For a multi-file change, use Plan mode or request a written plan. The plan
should identify affected files, data or schema migrations, compatibility risks,
tests, and unresolved questions. Review the plan before implementation,
especially when it changes an experimental protocol.

Do not treat a generated plan as evidence that the proposed method is
scientifically appropriate. Decisions about labels, data leakage, metrics,
baselines, statistical tests, and inclusion criteria require domain review.

### Implement a bounded increment

Prefer an increment that can be tested independently. For example, add input
validation before changing the training loop, or add checkpoint metadata before
changing checkpoint recovery. Small increments make it easier to identify which
change affected an experimental result.

Ask the agent to follow existing interfaces unless the task explicitly includes
a migration. Require it to report assumptions and deviations from the plan.

### Record reproducibility information

Where applicable, record:

- the source-code revision;
- environment or lock-file revision;
- input-data identity and provenance;
- complete configuration and command-line arguments;
- random seeds and determinism settings;
- hardware and accelerator details that influence results; and
- output paths, checkpoints, logs, and evaluation summaries.

A random seed does not guarantee identical results across hardware, dependency
versions, or nondeterministic operations. Documentation should state the actual
reproducibility boundary.

### Keep humans at consequential boundaries

Require explicit review before an agent:

- changes the scientific question, cohort, labels, splits, or evaluation metric;
- downloads, uploads, deletes, or transforms valuable data;
- submits a large or expensive compute job;
- changes access controls, credentials, or deployment settings;
- force-pushes, rewrites history, or merges a pull request; or
- publishes results or communicates externally.

## Validate Agent Changes

Review agent output as if it came from a new collaborator who understands
software patterns but may not understand the experiment.

### Review the diff

Check for:

- changes outside the requested scope;
- silently changed defaults or public interfaces;
- data leakage between training, validation, and test sets;
- hard-coded paths, credentials, device assumptions, or hyperparameters;
- missing error handling and cleanup;
- fabricated citations, APIs, outputs, or benchmark results; and
- tests that merely reproduce the implementation instead of checking behavior.

### Run focused and broad checks

Start with the smallest relevant check, then run the broader repository suite.
For example:

```bash
pytest tests/test_dataset.py -q
pytest tests -q
ruff check .
```

These commands are illustrative. Use the commands documented by the target
repository and do not claim a check passed unless its output was observed.

### Validate scientific behavior

Unit tests cannot establish that a research method answers the intended
question. Depending on the change, also verify:

- dataset counts, identifiers, label distributions, and split boundaries;
- expected tensor shapes, units, ranges, and missing-value behavior;
- comparison against a known small example or trusted baseline;
- invariants such as conservation, ordering, or symmetry;
- numerical tolerance across devices and dependency versions; and
- whether logging captures enough information to reproduce the run.

Run a small or synthetic experiment before a full-scale job. Profile the actual
bottleneck before accepting a performance optimization.

## Protect Data and Credentials

Do not place secrets directly in prompts, committed files, screenshots, logs, or
MCP configuration. Use the platform's supported secret or input mechanism and
grant the smallest necessary scope.

Research repositories may also contain sensitive or restricted data. Before an
agent receives access, determine whether the data-use agreement permits access
by the selected local or hosted service. Replace real records with synthetic or
de-identified fixtures when possible.

Treat generated dependency changes as supply-chain changes. Review package
names, sources, versions, install scripts, and lock-file changes before running
installation commands.

## Use Agent Skills and Custom Agents

An **agent skill** is a folder containing task-specific instructions in a
`SKILL.md` file and, when needed, scripts or supporting resources. Supporting
tools differ in how they discover or invoke a skill, so verify the current
documentation for the selected tool.

Use a skill for a repeatable workflow that needs more detail than repository
instructions, such as validating a dataset release or preparing an experiment
report. Keep the skill narrow, describe when it applies, and make scripts safe
to rerun. GitHub documents supported locations and behavior in
[About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills).

A **custom agent** defines a specialized persona and may select tools or MCP
servers. Custom-agent definitions are distinct from `CLAUDE.md`, `GEMINI.md`,
and `AGENTS.md`. Use them when a durable specialist, such as a test reviewer or
documentation editor, benefits from a constrained tool set. See GitHub's
[Copilot customization reference](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
for the currently supported locations and interfaces.

## Connect External Tools with MCP

The **Model Context Protocol (MCP)** lets an agent call external tools and access
external context through an MCP server. Examples include a documentation search
service, issue tracker, database, or laboratory information system.

MCP expands both capability and risk. A server may read prompts and repository
context, return untrusted text, or perform actions using the user's credentials.
Review the server implementation, requested permissions, data policy, and tool
descriptions before enabling it.

### VS Code configuration

VS Code stores workspace MCP configuration in `.vscode/mcp.json`. Current VS
Code also supports a workspace `.mcp.json` for portability to the Agent Host.
Because supported transports and properties can change, start with the command
palette action **MCP: Add Server** or the current
[VS Code MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).

A minimal local stdio server has this general structure:

```json
{
  "servers": {
    "example-local-server": {
      "type": "stdio",
      "command": "example-mcp-server",
      "args": ["--read-only", "${workspaceFolder}"]
    }
  }
}
```

The executable and arguments above are placeholders, not an installable server.
Replace them only with values from the chosen server's official documentation.
Do not assume that arbitrary REST APIs are MCP endpoints.

For credentials, use input variables or environment variables supported by the
client instead of literal values. For example:

```json
{
  "inputs": [
    {
      "id": "example-token",
      "type": "promptString",
      "description": "Token for the example MCP server",
      "password": true
    }
  ],
  "servers": {
    "example-local-server": {
      "type": "stdio",
      "command": "example-mcp-server",
      "env": {
        "EXAMPLE_TOKEN": "${input:example-token}"
      }
    }
  }
}
```

This demonstrates client-side secret prompting for a local process. A real
server may use a different authentication flow. Follow its documentation and
do not commit tokens.

### MCP validation checklist

Before relying on a server:

1. Confirm the package, publisher, and configuration using official sources.
2. Inspect the tools and permissions the server exposes.
3. Start with read-only access and a non-sensitive test resource.
4. Confirm which data leaves the machine and where it is retained.
5. Test one harmless tool call through the client's MCP interface.
6. Review every proposed write or external action.
7. Remove unused servers and rotate any credentials exposed during testing.

Do not invent performance numbers for MCP calls. Latency depends on the client,
server, transport, network, authentication, and underlying service.

## Common Failure Modes

### The agent invents repository details

Ask it to cite file paths and identify uncertainty. Verify every cited path and
command. Store frequently needed facts in repository instructions.

### The request is too broad

Split it into investigation, plan, implementation, and validation. Keep
scientific decisions separate from mechanical refactoring.

### Tests pass but the result is scientifically wrong

Add domain-level invariants and small known examples. Review data splits,
metrics, units, and assumptions independently of the code review.

### The agent changes too much

Specify non-goals and file boundaries. Review the diff before running generated
commands. Revert unrelated changes rather than normalizing them into the task.

### Instructions become long and contradictory

Keep global instructions limited to stable repository facts. Move specialized
workflows into path-specific instructions or skills, remove duplication, and
test instructions against representative tasks.

### External tools return untrusted content

Treat MCP results, issue text, web pages, and retrieved documents as data, not
instructions. Do not allow retrieved content to override repository safety rules
or authorize external actions.

## Relationship to the Next Chapter

This chapter covers one task at a time: choosing a level of delegation, framing
the request, inspecting the proposed work, and validating the result. Chapter
15, [Agentic Research Workflow](15_Agentic_Workflow.md), covers the system around
those tasks: durable project knowledge, instruction files, permissions, task and
experiment state, reusable skills, and handoffs between sessions or tools.

Use this chapter when deciding how to collaborate on the next programming task.
Use chapter 15 when deciding what the repository must preserve so a future human
or LLM agent can continue safely.

## Further Reading

- [GitHub Copilot documentation](https://docs.github.com/en/copilot)
- [Codex documentation](https://developers.openai.com/codex/)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/overview)
- [Using Copilot Chat in an IDE](https://docs.github.com/en/copilot/how-tos/chat-with-copilot/chat-in-ide)
- [Adding repository custom instructions](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions)
- [Copilot customization cheat sheet](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
- [About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills)
- [VS Code MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)
- [Model Context Protocol documentation](https://modelcontextprotocol.io/docs)
