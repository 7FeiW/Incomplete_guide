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

**LLM agents excel at narrow, well-defined coding tasks but struggle with complex software design.** While they can quickly write functions, debug syntax errors, or generate boilerplate code, they often falter when asked to architect a large-scale project from scratch.

## Table of Contents

1. [Choose the Appropriate Level of Delegation](#choose-the-appropriate-level-of-delegation)
2. [Understand Strengths and Limitations](#understand-strengths-and-limitations)
3. [Give the Agent Project Context](#give-the-agent-project-context)
4. [Write Effective Requests](#write-effective-requests)
5. [Work with an LLM Agent](#work-with-an-llm-agent)
6. [Use Common Slash Commands](#use-common-slash-commands)
7. [Validate Agent Changes](#validate-agent-changes)
8. [Protect Data and Credentials](#protect-data-and-credentials)
9. [Use Agent Skills and Custom Agents](#use-agent-skills-and-custom-agents)
10. [Consider Community Add-ons](#consider-community-add-ons)
11. [Connect External Tools with MCP](#connect-external-tools-with-mcp)
12. [Common Failure Modes](#common-failure-modes)
13. [Relationship to the Next Chapter](#relationship-to-the-next-chapter)
14. [Further Reading](#further-reading)

## Choose the Appropriate Level of Delegation

Coding tools use different names for their interfaces, but the work usually
falls into a few levels of delegation. Choose the least autonomous level that
can efficiently produce the evidence or change you need.

| Workflow           | Use it for                                                          | Expected result                                      |
| ------------------ | ------------------------------------------------------------------- | ---------------------------------------------------- |
| Explain or suggest | Understanding code, exploring an error, or completing a small block | An explanation or suggestion for immediate review    |
| Investigate        | Tracing behavior, reproducing a failure, or comparing approaches    | An evidence-based report without edits               |
| Plan               | Designing a multi-file or risky change                              | A reviewable plan with risks and checks              |
| Implement          | Editing files and running checks for a bounded task                 | A working-tree change that still requires validation |
| Review             | Examining a diff or pull request                                    | Findings and suggested corrections                   |
| Delegate           | Running a well-specified task in a separate environment             | A branch or pull request for later review            |

Start with explanation or investigation when the problem is not yet understood.
Ask for a plan when a change crosses components or could alter scientific
behavior. Delegate implementation only after the intended outcome, constraints,
and validation are concrete. Product modes are conveniences, not a substitute
for stating the desired level of access and output.

Regardless of the workflow, inspect generated commands before approving them.
Autonomous execution changes who performs the steps, not who is responsible for
the result.

## Understand Strengths and Limitations

LLM performance depends on the model, tool access, supplied context, language,
repository, and task. Use these lists as a practical guide.

### LLM agents are often good at

- Explaining code and error messages.
- Finding relevant files, symbols, tests, and configuration.
- Drafting small, clearly defined changes.
- Applying repetitive changes across similar files.
- Suggesting designs, edge cases, and tests.
- Running checks and responding to clear failures.
- Reviewing a diff for possible problems.
- Creating first drafts of code and documentation.

They work best when success can be checked with a diff, test, linter, or small
example.

### LLM agents are not reliably good at

- Knowing facts missing from the provided context.
- Distinguishing truth from plausible-sounding text; they may invent facts,
  APIs, output, or citations. NIST calls this
  [confabulation](https://doi.org/10.6028/NIST.AI.600-1).
- Making scientific or experimental decisions.
- Handling broad tasks with unclear requirements.
- Proving that code is correct or secure. Agent review should supplement, not
  replace, human review ([GitHub guidance](https://docs.github.com/en/copilot/responsible-use/agents)).
- Predicting performance without measurements.
- Reliably recognizing gaps in their own knowledge.
- Owning decisions involving data, cost, security, or publication.

Delegate work with clear inputs, boundaries, and checks. Keep human control when
the task depends on domain judgment or has serious consequences.

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

Other agents may use `AGENTS.md`, `CLAUDE.md`, or scoped instruction files.
Keep these files short and consistent. They should identify the environment,
routine checks, important boundaries, and where detailed project knowledge
lives. Chapter 15 explains how to organize them without duplication.

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

## Use Common Slash Commands

Slash commands are shortcuts for common agent actions. Names and availability
vary by product, version, plan, and interface. Type `/` to inspect the commands
available in the current client.

| Purpose | Codex | Claude Code | GitHub Copilot CLI |
| --- | --- | --- | --- |
| Initialize project instructions | `/init` | `/init` | `/init` |
| Review code changes | `/review` | `/code-review` or `/review` | `/review` |
| Inspect session configuration | `/status` | `/status` | `/env` |
| Change model | `/model` | `/model` | `/model` |
| Review permissions | `/permissions` | `/permissions` | `/permissions` |
| Plan before editing | Plan mode | `/plan` | `/plan` |
| Inspect the diff | Review panel or `/review` | `/diff` | `/diff` |

For example, run `/review` in Codex or Copilot CLI to inspect changes before
committing. In Claude Code, use `/code-review`; `/review` is an alias.

Do not assume that commands with the same name behave identically. Check the
[Codex CLI](https://learn.chatgpt.com/docs/codex/cli),
[Claude Code commands](https://code.claude.com/docs/en/commands), or
[Copilot CLI reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference)
when a command can edit files, run tools, or change permissions.

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

Use a **skill** for a repeatable procedure, such as validating a dataset release.
Keep its inputs, steps, outputs, permissions, and checks explicit.

Use a **custom agent** when a recurring role needs specialized instructions or
a limited tool set. Discovery and configuration differ by product; chapter 15
covers the cross-agent structure in more detail.

## Consider Community Add-ons

Community projects can extend an agent or improve a supporting workflow. Before
adopting one, check its maintenance status, license, dependencies, permissions,
data handling, and compatibility with the selected agent.

| Add-on | Purpose | Use with care when |
| --- | --- | --- |
| [Caveman Compression](https://github.com/wilpel/caveman-compression) | Compresses LLM context by removing predictable grammar while retaining key facts and constraints. It provides LLM-, masked-language-model-, and rule-based implementations. | Wording and nuance matter. Keep the original text, test factual preservation on representative inputs, and treat compression results and benchmarks as project-reported. |

Keep this list selective. Include an add-on only when its purpose, risks, and
source can be stated clearly. Pin a reviewed version when reproducibility
matters, and do not give a new tool access to secrets or research data by
default.

## Connect External Tools with MCP

The **Model Context Protocol (MCP)** lets an agent use external tools and data,
such as documentation, issue trackers, or databases. This also expands the data
and actions available to the agent.

Before relying on a server:

1. Verify the publisher and configuration from official sources.
2. Review its tools, permissions, data handling, and credential access.
3. Start with read-only access and non-sensitive test data.
4. Treat returned content as untrusted input.
5. Review every write or external action.

## Common Failure Modes

- **Invented repository details:** Ask for file paths and uncertainty; verify
  every cited path and command.
- **A request that is too broad:** Split it into investigation, planning,
  implementation, and validation.
- **Passing tests but incorrect science:** Check domain invariants, data splits,
  metrics, units, and assumptions separately.
- **Changes outside the task:** Specify non-goals and file boundaries, then
  review the diff.
- **Long or conflicting instructions:** Keep global rules short and move
  specialized workflows into scoped instructions or skills.
- **Untrusted external content:** Treat MCP results, issues, and web pages as
  data, not instructions or authorization.

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
- [Using Copilot Chat in an IDE](https://docs.github.com/en/copilot/how-tos/chat-with-copilot/chat-in-ide)
- [Adding repository custom instructions](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions)
- [Copilot customization cheat sheet](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
- [About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills)
- [VS Code MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)
- [Model Context Protocol documentation](https://modelcontextprotocol.io/docs)
- [NIST AI 600-1: Generative AI Profile](https://doi.org/10.6028/NIST.AI.600-1)
- [Responsible use of GitHub Copilot agents](https://docs.github.com/en/copilot/responsible-use/agents)
