# Programming with LLM Agents

Large language model (LLM) coding tools can explain a codebase, plan a change,
edit files, run commands, and review the result. That makes them more capable
than autocomplete, but it does not make them independent developers. They still
need a clear task, the right context, sensible access, and human review.

This chapter covers the day-to-day working loop for tools such as Codex, Claude
Code, and GitHub Copilot. Some examples use GitHub Copilot, but the underlying
practices apply to other agents too. Product details change quickly, so check the
official documentation when an interface or feature matters.

This chapter follows the sample project used in chapter 15: an image-classifier
project whose dataset constructor needs to reject duplicate sample IDs. The
examples take this bounded task from delegation through validation. The
next chapter, [Agentic Research Workflow](15_Agentic_Workflow.md), explains the
complementary repository architecture for carrying knowledge, rules, and plans across tools and sessions.

Agents are easiest to trust on focused tasks with results you can check. Let
them investigate and implement bounded changes; do not hand them responsibility
for the project's architecture or scientific decisions.

## Table of Contents

1. [Delegation](#delegation)
2. [Preparation](#preparation)
3. [Work Cycle](#work-cycle)
4. [Tool Controls](#tool-controls)
5. [Review and Validation](#review-and-validation)
6. [Safety Boundaries](#safety-boundaries)
7. [Workflow Extensions](#workflow-extensions)
8. [Common Failure Modes](#common-failure-modes)
9. [Chapter 15](#chapter-15)
10. [Further Reading](#further-reading)

## Delegation

Start by asking how uncertain the task is, what could go wrong, and how you will
check the result. Those answers determine how much work to hand over.

### Delegation Levels

Products use different names for their modes, but most work falls into the same
few levels. Choose the lowest level of autonomy that can produce the evidence or
change you need without unnecessary overhead.

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

Whatever mode you use, read generated commands before approving them. The agent
may perform the steps, but responsibility for the result still rests with you.

### Strengths and Limits

What an agent can do well depends on the model, its tools, the context it can
see, and the codebase itself. The following lists are a starting point, not a
guarantee.

#### Strengths

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

#### Limits

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

## Preparation

A useful request combines two kinds of context: repository guidance that applies
again and again, and details about the task in front of you.

### Sample Project

Assume an existing Python project with `src/project/dataset.py` and
`tests/test_dataset.py`. Its `Dataset` constructor consumes manifest rows with
`sample_id`, `path`, and `split` fields. The project uses uv with pytest and Ruff
declared as development dependencies, and its tests can import the project code.
These are illustrative assumptions; inspect your repository before using the
prompts. Run commands from the project root in its configured environment.
See [chapter 15's setup walkthrough](15_Agentic_Workflow.md#step-by-step-setup)
for the layout, environment setup, and shared instruction files.

The accompanying [shared documentation examples](../examples/ai_coding_examples/docs/)
live outside the agent-specific folders. In the sample project, use `docs/rules/`
for constraints, `docs/knowledge/` for explanations and procedures,
`docs/plans/` for plans with their status and progress, and `docs/findings/`
for supported conclusions. The included architecture and sample plan use this
image-classifier project. The supplementary rules come from a mass-spectrometry
project; adapt their content before using them here.

**Duplicate sample IDs** means that two or more manifest rows have the same
`sample_id`. For example, rows pairing `s001` with `images/a.png` and `s001` with
`images/b.png` have a duplicate ID even though their image paths differ.

This sample project requires one unique ID per manifest row. Repeated
measurements can legitimately share a subject or sample identifier in other
research projects; define which identifier must be unique before adding this
validation.

The task is to reject duplicate IDs before dataset construction. For this
example, the project owner has chosen exact string comparison, with no case or
Unicode normalization. These synthetic cases define the expected behavior:

| Sample IDs in manifest order | Expected result |
| --- | --- |
| `s1`, `s2` | Accept; preserve row order and split assignments. |
| `s1`, `s1` | Raise one `ValueError` identifying `s1`. |
| `s2`, `s1`, `s2`, `s1`, `s1` | Raise one `ValueError` identifying both `s1` and `s2`, each once. |
| `s1`, `S1` | Treat as distinct IDs under the agreed comparison rule. |

Use synthetic rows and test fixtures for paths; no research images are needed.
Missing-path checks are outside this task, and existing path-validation behavior
must be preserved. The table specifies acceptance criteria, not observed results.

### Project Context

An agent only sees what its tool makes available. It may miss uncommitted work,
external data, cluster settings, or decisions from another conversation. Put
reusable facts in the repository, then tell the agent what to inspect for this
task.

#### Copilot Example

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

This project trains image classifiers from sample manifests.
Use the Python environment specified by the project and documented in README.md.
Install development dependencies with `uv sync --dev`.

Read the relevant shared rules in docs/rules/ before editing. When present,
read the current task's plan in docs/plans/, including its status and evidence.
Use docs/knowledge/ for procedures and docs/findings/ for supported conclusions.

Run these checks after changing Python code:

1. `uv run pytest tests/test_dataset.py -q` for manifest changes.
2. `uv run pytest tests -q` for the full suite.
3. `uv run ruff check src tests`.

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

Keep shared rules in `docs/rules/` and have each agent's entry point direct it to
the relevant files. The [shared rules examples](../examples/ai_coding_examples/docs/rules/)
illustrate this content; tool-specific instruction files provide the entry points.

### Task Requests

A good request says what success looks like and where the boundaries are. It
usually includes:

1. the problem and intended result;
2. relevant files or an existing implementation pattern;
3. scientific and engineering constraints;
4. explicit non-goals;
5. acceptance criteria; and
6. commands that verify completion.

For example:

```text
Reject duplicate sample IDs in manifests loaded by src/project/dataset.py.

Requirements:
- Compare IDs as exact strings without case or Unicode normalization.
- Raise one ValueError listing every duplicated ID once.
- Preserve the public Dataset constructor, manifest schema, and path checks.
- Preserve row order and existing split assignments for valid manifests.
- Do not read image contents during manifest validation.
- Do not modify research datasets or start training.

Validation:
- Add focused tests to tests/test_dataset.py.
- Run: uv run pytest tests/test_dataset.py -q
- Run: uv run pytest tests -q
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

## Work Cycle

Work from investigation to planning and then implementation. If the code
contradicts the plan, stop and revise the plan; do not build more work on top of
a doubtful assumption.

### Inspect Existing Behavior

First, ask the agent to trace inputs, transformations, and outputs through the
relevant files. Check its explanation against the code. Semantic search can
miss dynamically imported modules, generated files, notebooks, external jobs,
or configuration supplied outside the repository.

Start the sample task with this prompt:

```text
Read the repository instructions and inspect git status, src/project/dataset.py,
and tests/test_dataset.py. Trace where manifest validation happens relative to
dataset construction and image loading. Identify current duplicate-ID behavior
and existing tests. Cite files and flag unknowns. Do not edit files yet.
```

Check that the response identifies the actual constructor and validation path.
If the code differs from the sample assumptions, revise the task before proceeding.
For other investigations, use a similarly bounded prompt:

```text
Find every entry point that writes evaluation metrics. Report the output format,
destination, and handling of interrupted writes. Do not change files.
```

### Plan

For a multi-file change, use Plan mode or request a written plan. The plan
should identify affected files, data or schema migrations, compatibility risks,
tests, and unresolved questions. Review the plan before implementation,
especially when it changes an experimental protocol.

For the duplicate-ID task, send:

```text
Plan the duplicate-ID change described above. Name the files to edit, where the
check belongs, and the regression cases. Include unique IDs, one repeated ID,
multiple distinct repeated IDs, and case-sensitive IDs. Explain how tests will
check that validation does not read images or alter valid rows and splits.
List the validation commands and any unresolved choices. Do not edit files yet.
```

Review whether the proposed tests check the agreed behavior and whether the plan
preserves existing path checks. Resolve new identity or schema questions before
accepting it. The plan should produce a small change to the dataset module and
its tests; additional files need a task-related reason.

Do not treat a generated plan as evidence that the proposed method is
scientifically appropriate. Decisions about labels, data leakage, metrics,
baselines, statistical tests, and inclusion criteria require domain review.

### Implement

Prefer an increment that can be tested independently. For example, add input
validation before changing the training loop, or add checkpoint metadata before
changing checkpoint recovery. Small increments make it easier to identify which
change affected an experimental result.

Ask the agent to follow existing interfaces unless the task explicitly includes
a migration. Require it to report assumptions and deviations from the plan.

After accepting the sample task's plan, send:

```text
Implement the accepted plan. First add a regression test and run it against the
current implementation. Confirm that it fails because duplicate IDs are accepted,
not because of an import or fixture error. Then add the validation and run the
focused tests, full test suite, and Ruff check from the task request.
Preserve unrelated changes and leave the result uncommitted. Report actual
command results, deviations from the plan, and anything not verified.
```

If inspection shows duplicates are already rejected, reassess the reported
problem instead of manufacturing a failing test. If a check cannot run, have the
agent report the blocker rather than treating the change as verified.

### Record Reproducibility

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

## Tool Controls

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

## Review and Validation

Review the result as you would work from a new collaborator: assume useful
programming knowledge, but do not assume a full understanding of the experiment.

### Diff Review

Check for:

- changes outside the requested scope;
- silently changed defaults or public interfaces;
- data leakage between training, validation, and test sets;
- hard-coded paths, credentials, device assumptions, or hyperparameters;
- missing error handling and cleanup;
- fabricated citations, APIs, outputs, or benchmark results; and
- tests that merely reproduce the implementation instead of checking behavior.

### Run Checks

Start with the smallest relevant check, then run the broader repository suite.
For example:

```bash
uv run pytest tests/test_dataset.py -q
uv run pytest tests -q
uv run ruff check src/project/dataset.py tests/test_dataset.py
```

These commands are illustrative. Use the commands documented by the target
repository and do not claim a check passed unless its output was observed.

For the sample task, request a final review:

```text
Review the duplicate-ID diff against the accepted plan and sample cases.
Check exact string comparison, one error containing all duplicated IDs once,
unchanged path checks, preserved row order and splits, and no image reads during
validation. Report findings with file references. Summarize the regression
failure before the fix, checks after the fix, and any checks not run. Do not edit.
```

Compare the report with the diff and captured command output. Accept the change
only when the required behavior and checks are accounted for; keep any remaining
failure or uncertainty explicit. There is no predetermined passing test count.

### Scientific Validation

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

## Safety Boundaries

### Human Review

Require explicit review before an agent:

- changes the scientific question, cohort, labels, splits, or evaluation metric;
- downloads, uploads, deletes, or transforms valuable data;
- submits a large or expensive compute job;
- changes access controls, credentials, or deployment settings;
- force-pushes, rewrites history, or merges a pull request; or
- publishes results or communicates externally.

### Sensitive Access

Do not place secrets directly in prompts, committed files, screenshots, logs, or
MCP configuration. Use the platform's supported secret or input mechanism and
grant the smallest necessary scope.

Research repositories may also contain sensitive or restricted data. Before an
agent receives access, determine whether the data-use agreement permits access
by the selected local or hosted service. Replace real records with synthetic or
de-identified fixtures when possible.

Treat generated dependency changes as supply-chain changes. Review package
names, sources, versions, install scripts, and lock-file changes before running
installation commands. Access to a tool does not imply permission to use it for
a consequential action.

## Workflow Extensions

Skills, custom agents, community add-ons, and external integrations can save
time on repeated work. Each one also brings more instructions, code,
permissions, or data access into the project, so review it before relying on it.

### Skills and Custom Agents

Use a **skill** for a repeatable procedure, such as validating a dataset release.
Keep its inputs, steps, outputs, permissions, and checks explicit.

Use a **custom agent** when a recurring role needs specialized instructions or
a limited tool set. Discovery and configuration differ by product; chapter 15
covers the cross-agent structure in more detail.

### Community Add-ons

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

### MCP Connections

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

## Chapter 15

This chapter covers one task at a time: choosing a level of delegation, framing
the request, inspecting the proposed work, and validating the result. Chapter
15, [Agentic Research Workflow](15_Agentic_Workflow.md), covers the system around
those tasks: durable project knowledge, instruction files, permissions, plans and
experiment records, reusable skills, and handoffs between sessions or tools.

Continue with the same duplicate-ID task in
[chapter 15's setup walkthrough](15_Agentic_Workflow.md#step-by-step-setup).
It shows where to save the agreed constraints, plan, observed checks, and next
action so a fresh session can recover the work described here.

Come back to this chapter when you are working through a specific programming
task. Use chapter 15 when you are deciding what the repository must preserve so
someone else—or another agent session—can pick up the work safely.

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
