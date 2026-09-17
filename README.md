# FW's Incomplete Guide to Python Research Codebase

## What is this

This is repository of documents for Python-based computational research. This is a incomplete guide (and very working in progress) for python research codebase from research scientist and programer's perspective.

## Adopt This Guide with Codex or Claude Code

Do not copy this guide into a project unchanged. Its recommendations are a
starting point; an agent should first determine which ones fit the project's
language, research workflow, existing conventions, and operational constraints.

Give Codex or Claude Code access to this repository and the target project.
Ask it to read this README and the relevant sections in `docs/`, inspect the
target project's structure, instructions, environment, tests, and current
workflow, and report:

- which guidance is already in use;
- which recommendations fit and why;
- which recommendations need adaptation or do not fit, with the project-specific
  constraints; and
- a small, ordered adoption plan, including proposed files, changes, and checks.

Require the agent to ask for your approval before it edits the target project.
Review the assessment and choose the parts to adopt; then ask the agent to make
only those approved changes and show the resulting diff and validation results.

For example:

```text
Read FW's Incomplete Guide to Python Research Codebase and inspect this project from <url>
Determine which guidance is appropriate for this project's current language,
research workflow, environment, tests, and operational constraints. Do not edit
anything yet. Report what is already adopted, what to adopt, what to adapt or
skip, and a small ordered plan with proposed files and validation checks. Ask me
which parts I want to adopt before making changes.
```

See [Programming with LLM Agents](docs/14_Programming_with_LLM_Agents.md) for
task delegation and review, and [Agentic Research Workflow](docs/15_Agentic_Workflow.md)
for project instructions, rules, and plans.

## Docs

Below are the documents in the `docs/` folder (click to open):

- [Coding Convention](docs/01_Coding_Convention.md)
- [Project Structure](docs/02_Project_Structure.md)
- [Version Control and Git](docs/03_Version_Contorl_and_Git.md)
- [PIP and Conda / Python Env](docs/04_Python_Env.md)
- [Configurations](docs/05_Config.md)
- [Logging](docs/06_Logging.md)
- [Cross Platform](docs/07_Cross_Platform.md)
- [Remote Machine / Deploy](docs/08_Remote_Machine.md)
- [Do or Not](docs/09_Things_Should_Consdier.md)
- [Profiling and Speed](docs/10_Profiing_and_Speed.md)
- [Testing](docs/11_Testing.md)
- [Go With Large Scale](docs/12_Go_With_Large_Scale.md)
- [Using Apptainer on Compute Canada](docs/13_Apptainer_Compute_Canada.md)
- [Programming with LLM Agents](docs/14_Programming_with_LLM_Agents.md)
- [Agentic Research Workflow: Knowledge, Rules, and Plans](docs/15_Agentic_Workflow.md)
- [Academic Writing with Codex and Claude](docs/16_Academic_Writing_with_LLM_Agents.md)

## Editorial Workflow

This guide uses a lightweight, repository-specific writing workflow for agent
assistance. See the [editorial instructions](editorial/README.md) for the
shared rules, source record, and read-only review procedure.
