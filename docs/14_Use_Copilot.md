# Using GitHub Copilot for Planning & Agentic Development

This document covers practical workflows for using **GitHub Copilot** as an AI agent for software development. It focuses on planning strategies, agentic workflows, and how to guide agents to understand your codebase and complete complex tasks autonomously.

---

## Important: AI Agents Are Tools, Not Replacements

**AI agents are not magic.** While GitHub Copilot and other AI coding assistants are powerful tools, they are **force multipliers**, not replacements for software engineering expertise.

**Why understanding software systems matters more than ever:**

1. **AI agents follow your instructions** - Poor understanding leads to poor instructions, which leads to poor code. You need to know what to ask for and how to evaluate the results.

2. **You must verify agent outputs** - AI-generated code can contain subtle bugs, security vulnerabilities, or architectural flaws that only domain expertise can catch. Blind trust in AI outputs is dangerous.

3. **Complex systems require human judgment** - AI agents excel at implementing well-defined tasks, but system architecture, design trade-offs, and long-term maintainability decisions still require human expertise.

4. **Debugging AI-generated code requires deeper knowledge** - When agent-generated code fails, you need strong fundamentals to diagnose whether the issue is in your requirements, the AI's interpretation, or the generated implementation.

5. **AI agents augment, not replace** - Think of Copilot as a senior developer who can write code incredibly fast but needs your guidance on what to build and why. You remain the architect, code reviewer, and decision-maker.

**What AI agents supercharge:**
- Writing boilerplate and repetitive code
- Implementing well-specified features quickly
- Generating tests and documentation
- Refactoring code following established patterns
- Exploring multiple implementation approaches rapidly

**What still requires human expertise:**
- System architecture and design decisions
- Understanding business requirements and edge cases
- Security threat modeling and vulnerability assessment
- Performance optimization trade-offs
- Code review and quality assurance
- Debugging complex issues
- Making technical trade-off decisions

**The bottom line**: AI agents make excellent software engineers even more productive. They do not make inexperienced developers into experts. If anything, using AI agents effectively requires **stronger fundamentals** because you need to rapidly evaluate, debug, and integrate AI-generated code into production systems.

---

## Table of Contents

### Part 1: Understanding Copilot
1. [What is Copilot?](#what-is-copilot)
2. [Copilot Models & Modes](#copilot-models--modes)
3. [Choosing the Right Mode](#choosing-the-right-mode)

### Part 2: Using Copilot for Agents
4. [Why Use Copilot Agents?](#why-use-copilot-agents)
5. [Plan Organization Strategy](#plan-organization-strategy)
6. [Repository Custom Instructions for Agents](#repository-custom-instructions-for-agents)
7. [Planning & Executing Agent Workflows](#planning--executing-agent-workflows)

### Part 3: Best Practices & Patterns
8. [Practical Guide to Agent Development](#practical-guide-to-agent-development)
   - 8.1 [Data Pipeline Workflows](#81-data-pipeline-workflows)
   - 8.2 [Model Development Workflows](#82-model-development-workflows)
   - 8.3 [Training Loop & Distributed Training](#83-training-loop--distributed-training)
   - 8.4 [Experiment Tracking](#84-experiment-tracking)
   - 8.5 [Hyperparameter Tuning](#85-hyperparameter-tuning)
   - 8.6 [Model Evaluation & Testing](#86-model-evaluation--testing)
   - 8.7 [Molecular Data Processing](#87-molecular-data-processing)
   - 8.8 [Molecular Dynamics & Simulations](#88-molecular-dynamics--simulations)
   - 8.9 [Omics Data Analysis](#89-omics-data-analysis)
   - 8.10 [Structure Prediction & Modeling](#810-structure-prediction--modeling)
   - 8.11 [Cheminformatics & QSAR Modeling](#811-cheminformatics--qsar-modeling)
   - 8.12 [Structural Bioinformatics Tools](#812-structural-bioinformatics-tools)

### Part 4: Quality & Safety
9. [Security & Code Quality Considerations](#security--code-quality-considerations)
10. [Anti-patterns & Pitfalls](#anti-patterns--pitfalls)
11. [Integration with Development Workflow](#integration-with-development-workflow)

### Part 5: Reference
12. [When to Use, When Not to Use](#when-to-use-when-not-to-use)
13. [Troubleshooting & FAQ](#troubleshooting--faq)
14. [Quick Command Reference](#quick-command-reference)
15. [References & Further Reading](#references--further-reading)

### Part 6: MCP Integration
16. [What is MCP?](#what-is-mcp)
17. [Setting Up MCP Servers](#setting-up-mcp-servers)
18. [Using MCP During ML Development](#using-mcp-during-ml-development)
19. [MCP Troubleshooting](#mcp-troubleshooting)
20. [MCP Server Resources](#mcp-server-resources)
21. [Additional MCP Resources](#additional-mcp-resources)

---

# Part 1: Understanding Copilot

## What is Copilot?

### GitHub Copilot & Agentic Mode

- **What it is**: AI-powered development agent that can understand codebases, plan changes, and implement features autonomously.
- **Modes**: 
  - **Interactive mode** (Copilot Chat) — human-guided, exploratory
  - **Agentic mode** (Copilot Coding Agent) — autonomous planning and execution, creates pull requests
- **Primary use**: understanding complex requirements, planning multi-file changes, generating pull requests, code review
- **Model**: based on OpenAI's GPT-4 and vision capabilities
- **Access**: 
  - Copilot Chat: $10-$20/month
  - Copilot Coding Agent: available on GitHub.com (beta/preview)

---

## Copilot Models & Modes

GitHub Copilot offers different models optimized for different tasks:

### 1. Ask Mode (Copilot Chat)

**Use case**: Explore, understand, prototype, debug, explain

**Characteristics**:
- Interactive and conversational
- You ask questions, Copilot responds
- Can reference code, files, or entire repositories
- Good for learning and iteration
- No autonomous action

**Examples**:
```
"How do I implement pagination in Django?"
"What's wrong with this code? [paste error]"
"Explain this algorithm to me"
"Refactor this function to use async/await"
```

**Workflow**:
1. Ask a question
2. Review response
3. Ask follow-up or request changes
4. Copy/paste result into your code

**Pros**: Safe, exploratory, good for learning  
**Cons**: Manual work to apply changes, slow for large tasks

---

### 2. Plan Agent Mode (Copilot Coding Agent)

**Use case**: Autonomous planning and implementation of complete features

**Characteristics**:
- Analyzes codebase to understand architecture
- Plans multi-file changes before implementing
- Runs tests and validates work
- Creates pull requests automatically
- Goal-oriented (complete feature, not just answer questions)

**Examples**:
```
"Add distributed training support using PyTorch DDP for multi-GPU experiments"
"Implement automated hyperparameter tuning using Optuna with experiment tracking"
"Build data validation pipeline for preprocessing datasets with quality metrics and logging"
```

**Workflow**:
1. Write clear requirements with acceptance criteria
2. Agent analyzes your codebase and custom instructions
3. Agent creates a detailed plan (visible in PR description)
4. Agent implements changes across all necessary files
5. Agent runs tests and validation
6. Agent creates PR with summary
7. You review, request changes if needed, or merge

**Pros**: Autonomous, handles complex tasks, creates complete PRs  
**Cons**: Requires good custom instructions, needs review before merge

---

### 3. Code Review Mode (Copilot)

**Use case**: Review pull requests and suggest improvements

**Characteristics**:
- Analyzes PR diffs
- Suggests improvements (performance, security, style)
- Can explain changes
- Applies repo custom instructions

**Examples**:
```
"Review this PR for security issues"
"Are there any performance problems here?"
"Does this follow our coding conventions?"
```

**Workflow**:
1. Request Copilot code review on a PR (GitHub UI)
2. Copilot analyzes changes in context of your codebase
3. Copilot leaves suggestions as comments
4. You address suggestions and update PR

**Pros**: Catches issues early, applies conventions  
**Cons**: Not replacement for human review

---

### 4. Inline Edit Mode (VS Code / IDEs)

**Use case**: Quick edits, completions, refactoring in your editor

**Characteristics**:
- Works as you code
- Suggests completions and edits
- No PR or autonomous planning

**Shortcuts**:
- `Ctrl+I` (VS Code) — inline edit
- `Alt+\` — accept suggestion
- `Esc` — reject

**Examples**: Generating getters/setters, writing repetitive boilerplate, adding docstrings

**Workflow**:
1. Write function signature
2. Copilot suggests body
3. Accept or modify

**Pros**: Fast for boilerplate  
**Cons**: Limited context, no planning

---

## Choosing the Right Mode

| Task | Ask Mode | Plan Agent | Code Review | Inline |
|------|----------|-----------|------------|--------|
| Learn how to do something | ✓✓ | — | — | — |
| Implement a small feature | ✓ | ✓✓ | — | — |
| Implement a large feature | — | ✓✓ | — | — |
| Refactor multiple files | — | ✓✓ | — | — |
| Quick function generation | ✓ | — | — | ✓✓ |
| Review code | — | — | ✓✓ | — |
| Debug/explain code | ✓✓ | — | — | — |
| Generate tests | ✓ | ✓ | — | ✓ |

---

---

# Part 2: Using Copilot for Agents

## Why Use Copilot Agents?

1. **Planning**: AI reasons through requirements and breaks tasks into steps
2. **Multi-file understanding**: agents analyze dependencies across files
3. **Autonomous execution**: creates pull requests with complete, tested implementations
4. **Codebase context**: agents learn your architecture and conventions
5. **Reduced human iteration**: better pre-planning = fewer review cycles

**Key difference from Chat**: Agents are goal-oriented, not just responsive. They plan first, then execute.

---

## Plan Organization Strategy

Plans guide Copilot agents and maintain project continuity. Use two types depending on scope and lifecycle:

### Persistent Plans (Long-Lived Features)

Use `docs/feature-name-plan.md` for plans that represent **persistent project specifications** — features that span multiple sprints, ongoing architectural decisions, or cross-team initiatives.

**Characteristics**:
- Versioned in Git alongside codebase
- Referenced in `.github/copilot-instructions.md`
- Updated as requirements evolve
- Remain as documentation after implementation
- Shared reference for team and future developers

**Example**: `docs/experiment-tracking-plan.md`
```markdown
# Experiment Tracking & Reproducibility Plan

## Goals
- Centralize experiment logging with MLflow or WandB
- Capture hyperparameters, metrics, and model artifacts
- Enable reproducible model checkpoints with seed tracking
- Support cross-team experiment comparison

## Implementation Phases
1. Phase 1: MLflow integration for local experiments (week 1-2)
2. Phase 2: Distributed logging for multi-GPU training (week 3-4)
3. Phase 3: Web UI dashboard and artifact versioning (week 5)

## Reference in copilot-instructions.md
See docs/experiment-tracking-plan.md for logging requirements and integration phases.
```

### Temporary Plans (One-Off Refactors)

Use inline plans **next to the relevant module** for one-off refactors or sprint-specific tasks — save them in default location or as `.plan` files in the module directory, then **delete after merge**.

**Characteristics**:
- Temporary; not versioned long-term
- Scoped to a single refactor or sprint task
- Deleted once PR merges
- Not shared in `copilot-instructions.md`
- Local guidance for quick agent tasks

**Example**: `src/models/.refactor-cache-layer.plan`
```markdown
# Temp Plan: Add Caching Layer to Model Loading

## Scope
Refactor model loading to use Redis cache for performance.

## Changes
1. Update src/models/loader.py to check Redis first
2. Add Redis config to src/config.py
3. Update tests in tests/test_models.py
4. Update requirements.txt with redis

## Delete after merge to main.
```

**Why delete after merge?**
- Keeps repo clean of stale plans
- Prevents confusion (old plans become outdated)
- Temporary guidance shouldn't be permanent

**Summary**: Use persistent plans for features you'll reference repeatedly; use temporary plans for focused refactors, then clean them up.

---

## Repository Custom Instructions for Agents

GitHub Copilot supports three types of repository custom instructions. Understanding these allows you to provide context at different levels of granularity:

### 1. Repository-Wide Custom Instructions

**File**: `.github/copilot-instructions.md`

**Scope**: Applies to **all requests** made in the context of the repository

**Use for**:
- Project overview and architecture
- Core technologies and dependencies
- Build, test, and deployment commands
- Coding standards and conventions
- Project structure and file organization
- Testing expectations
- Security and compliance requirements
- **References to persistent plans** (link to `docs/feature-name-plan.md` files)

**When to use**: This is the **primary** custom instructions file. Every repository should have one. It provides the baseline context that Copilot uses for all interactions.

### 2. Path-Specific Custom Instructions

**File**: `.github/instructions/<NAME>.instructions.md`

**Scope**: Applies to requests made in the context of **files matching a specified path pattern**

**Use for**:
- Module-specific conventions (e.g., API routes vs data processing scripts)
- Domain-specific patterns (e.g., ML model files vs preprocessing scripts)
- Specialized testing requirements for specific components
- Different coding styles for different parts of the codebase

**When to use**: When different parts of your codebase have distinct conventions or requirements. For example:
- `ml-models.instructions.md` — instructions for PyTorch model files
- `preprocessing.instructions.md` — instructions for data processing scripts
- `notebooks.instructions.md` — instructions for Jupyter notebooks

**How they combine**: If both repository-wide and path-specific instructions exist, Copilot uses **both**. Path-specific instructions supplement (not replace) repository-wide instructions.

**Example structure**:
```
.github/
├── copilot-instructions.md                    # Repository-wide
└── instructions/
    ├── ml-models.instructions.md              # For src/my_package/models/
    ├── preprocessing.instructions.md          # For preprocess_scripts/
    └── notebooks.instructions.md              # For notebooks/
```

### 3. Agent Instructions (AGENTS.md)

**File**: `AGENTS.md` (can be placed anywhere in the repository)

**Scope**: Used by **AI agents** for autonomous task execution

**Use for**:
- Agent personas and specialized roles (architect, reviewer, researcher)
- Task-specific workflows and procedures
- Agent decision-making guidelines
- Integration patterns with external tools
- Multi-agent coordination

**When to use**: For advanced agentic workflows where you want fine-grained control over how AI agents behave in different contexts.

**How precedence works**: The **nearest** `AGENTS.md` file in the directory tree takes precedence. This allows hierarchical agent instructions.

**Reference**: [OpenAI Agents Documentation](https://github.com/openai/agents.md)

### Choosing the Right Type

| Scenario | Instruction Type | File Location |
|----------|------------------|---------------|
| Project-wide coding standards | Repository-wide | `.github/copilot-instructions.md` |
| Link to persistent feature plans | Repository-wide | `.github/copilot-instructions.md` |
| ML model-specific patterns | Path-specific | `.github/instructions/ml-models.instructions.md` |
| Data preprocessing conventions | Path-specific | `.github/instructions/preprocessing.instructions.md` |
| Agent task planning workflow | Agent instructions | `AGENTS.md` or `docs/AGENTS.md` |
| General project architecture | Repository-wide | `.github/copilot-instructions.md` |
| Module-specific security rules | Path-specific | `.github/instructions/security.instructions.md` |

### Best Practices for Multiple Instruction Files

1. **Start simple**: Begin with only `.github/copilot-instructions.md`. Add path-specific instructions only when needed.
2. **Avoid duplication**: Don't repeat content across files. Path-specific instructions should **extend**, not duplicate, repository-wide instructions.
3. **Be explicit about scope**: In path-specific instructions, clearly state which files/directories they apply to.
4. **Keep instructions focused**: Each file should have a clear purpose and scope.
5. **Document your structure**: In your repository-wide instructions, mention that path-specific instructions exist and explain their purpose.
6. **Reference persistent plans**: Link to `docs/feature-name-plan.md` files in your repository-wide instructions so agents can find detailed specifications.

---

## Planning & Executing Agent Workflows

### How Agents Plan & Execute

Agents follow this internal process:

1. **Understand the requirement** — parse your request and acceptance criteria
2. **Explore codebase** — read relevant files, understand architecture and conventions
3. **Plan the solution** — break into logical steps, identify all files to modify
4. **Implement** — write code following discovered patterns
5. **Test** — run tests, validate changes
6. **Deliver** — code is ready for integration (as branch or PR)

Your job is to **provide clear inputs** at step 1, and **custom instructions** (in `.github/copilot-instructions.md`) to accelerate steps 2-3.

### Structuring Requests for Agents

#### ❌ Vague Request
```
"Add user authentication to the API"
```

#### ✓ Clear Request with Context
```
Add JWT-based user authentication to the API with the following requirements:

1. Create a new /auth/login endpoint that accepts username/password
2. Validate credentials against the users table
3. Return a JWT token with 24-hour expiration
4. Protect existing /api/users endpoints with JWT middleware
5. Add unit tests for auth logic

Architecture notes: 
- We use Flask with SQLAlchemy
- Tokens should be stored in Redis for token blacklisting
- Follow the pattern in routes/admin.py for middleware
```

### Key Elements of a Good Agent Request

1. **Acceptance criteria** — what success looks like
2. **References** — point to similar patterns in codebase ("like admin.py")
3. **Constraints** — tech stack, patterns, non-goals
4. **File scope** — which areas should change
5. **Testing** — what tests should pass

### Step-by-Step Workflow

#### Step 0: Set Up Custom Instructions (One-Time Setup)

Ensure `.github/copilot-instructions.md` exists and is comprehensive. Agents use this as their knowledge base. This is a **one-time setup** for your repository—once created, agents will reference it for all future requests.

See [Repository Custom Instructions for Agents](#repository-custom-instructions-for-agents) section below for detailed guidance on what to include.

#### Step 1: Open Copilot Agent

Navigate to [github.com/copilot/agents](https://github.com/copilot/agents) to access Copilot Coding Agent.

#### Step 2: Write a Clear Request

```
Task: Add distributed training support for multi-GPU experiments

Requirements:
1. Create src/my_package/distributed.py with DDP initialization and cleanup
2. Wrap model training loop to support torch.distributed across GPUs
3. Save checkpoints only on rank-0 to avoid write conflicts
4. Add learning rate scheduling that accounts for batch size scaling
5. Log training metrics to MLflow for experiment tracking
6. Add unit tests that validate gradient synchronization

Success criteria:
- Training script runs on single GPU and multi-GPU without code changes
- Training time scales near-linearly with GPU count
- Checkpoints save correctly without corruption
- All tests pass with 2+ GPU simulation

Reference pattern: See distributed setup in scripts/fine_tuning.py and model initialization patterns in src/my_package/models.py
```

#### Step 3: Agent Implements & Tests

- Agent explores your codebase using custom instructions
- Agent creates a branch and implements changes
- Agent runs tests and validates work
- Agent is ready for review and integration

Agents rely heavily on custom instructions to work efficiently. Without them, agents waste time exploring your codebase.

### What to Include in `.github/copilot-instructions.md`

**1. High-Level Overview** (agents need context fast)
```markdown
## Project Overview

This is a machine learning research project for model training and inference pipelines.
- **Language**: Python 3.11
- **Core libraries**: PyTorch, NumPy, scikit-learn
- **Data processing**: Pandas, Polars
- **Testing**: pytest with fixtures
- **Execution**: SLURM on HPC clusters or local GPU training
- **Key patterns**: Modular src/ package, numbered preprocessing scripts, experiment config tracking
```

**2. Plans & Specifications** (link persistent feature plans)
```markdown
## Feature Plans & Specifications

Ongoing and cross-cutting features are documented in docs/:
- `docs/experiment-tracking-plan.md` — MLflow integration and distributed logging phases
- `docs/data-validation-plan.md` — Data quality checks and preprocessing pipeline schema
- `docs/distributed-training-plan.md` — Multi-GPU and SLURM cluster training setup

When implementing these features, reference the corresponding plan file for requirements and implementation phases.
```

**3. Exact Build & Test Commands** (agents must validate their work)
```markdown
## Setup & Validation

### Install
1. `python -m venv venv`
2. `source venv/bin/activate`  (Unix) or `venv\Scripts\activate.ps1` (Windows)
3. `pip install -r requirements.txt`
4. `pip install -r requirements-dev.txt` (for testing/development)

### Test
- **Run all tests**: `pytest tests/ -v`
- **Run specific test file**: `pytest tests/test_models.py -v`
- **Run with coverage**: `pytest --cov=src tests/`

### Data Processing
- `python preprocess_scripts/01_extract_data.py --config configs/preprocess.yaml`
- `python preprocess_scripts/02_create_dataset.py --config configs/dataset.yaml`

### Training
- **Local GPU**: `python scripts/fine_tuning.py --config configs/model_config.yaml`
- **SLURM cluster**: `sbatch slurm_scripts/train_job.sh`

### Inference
- `python scripts/inference.py --model weights/model.pt --input data/test.csv`
```

**3. Project Layout** (agents must know where to put files)
```markdown
## Project Structure
src/
  └── my_package/         # Main Python package
      ├── __init__.py
      ├── models.py       # Model definitions (PyTorch modules)
      ├── utils.py        # Shared utilities (preprocessing, metrics)
      ├── config.py       # Config loading and validation
      └── data.py         # Dataset classes

tests/
  ├── conftest.py         # pytest fixtures (sample data, mock models)
  ├── test_models.py      # Model unit tests
  ├── test_utils.py
  └── fixtures/           # Test data and pre-trained models
      ├── sample_data.csv
      └── test_weights.pt

preprocess_scripts/
  ├── 01_extract_data.py         # Download/extract raw data
  └── 02_create_dataset.py       # Clean and prepare dataset

scripts/
  ├── fine_tuning.py     # Model training script
  └── inference.py       # Inference/prediction script

notebooks/
  ├── data_exploration.ipynb
  └── results_analysis.ipynb

configs/
  ├── model_config.yaml   # Model hyperparameters
  ├── dataset.yaml        # Dataset configuration
  └── preprocess.yaml     # Preprocessing settings

slurm_scripts/
  ├── train_job.sh        # SLURM job submission script
  └── inference_job.sh

setup_scripts/
  ├── env_local.sh        # Local environment setup
  └── env_hpc_cpu.sh      # HPC environment setup

.github/
  └── copilot-instructions.md
```

**4. Coding Conventions** (agents must follow your style)
```markdown
## Coding Standards

- **Type hints**: All functions must have type hints where applicable
- **Naming**: Snake case for functions/vars, PascalCase for classes
- **Device handling**: Always check for CUDA availability; support CPU fallback
- **Random seeds**: Set seeds (torch, numpy, random) for reproducibility
- **Logging**: Use Python logging module, not print statements
- **Tests**: Write unit tests for models and utilities; use pytest fixtures
- **Configs**: Use YAML for all experiment configs; never hardcode hyperparameters
```

**5. Critical Constraints** (prevents agents from making breaking changes)
```markdown
## Important Constraints

⚠️ **Never modify**:
- Model input/output shapes without updating all downstream code
- The config schema without deprecation warnings
- The preprocessing steps (breaks reproducibility of past experiments)

✓ **Always do**:
- Save model checkpoints with seed and config info
- Add tests for new model architectures
- Update configs/ when adding new hyperparameters
- Include docstrings explaining mathematical operations
- Run full test suite and validate on validation set before submitting PR
```

### Advanced: Nested & Model-Specific Instructions

For larger projects, complement `.github/copilot-instructions.md` with specialized instruction files:

#### Nested Instructions (Folder-Specific)

**Feature**: Supports nested `copilot-instructions.md` files for folder-specific guidance (experimental in VS Code via `chat.useNestedAgentsMdFiles`).

**Example structure**:
```
.github/
  └── copilot-instructions.md        # Project-wide instructions

src/
  └── copilot-instructions.md        # Specific to src/ package
      └── models/
          └── copilot-instructions.md # Specific to model implementations

tests/
  └── copilot-instructions.md        # Testing-specific conventions

scripts/
  └── copilot-instructions.md        # Script and automation patterns
```

**Use cases**:
- **Model implementations** (`src/models/copilot-instructions.md`): Define PyTorch module patterns, layer types, validation requirements
- **Testing** (`tests/copilot-instructions.md`): Specify pytest fixtures, mocking strategies, edge cases to always test
- **Scripts** (`scripts/copilot-instructions.md`): Document CLI argument patterns, error handling, SLURM submission conventions
- **Data pipelines** (`preprocess_scripts/copilot-instructions.md`): Specify data format validation, logging levels, output schemas

#### Model-Specific Instructions

**Feature**: Define agent personas and tool access using model-specific files like `CLAUDE.md` or `GEMINI.md`.

**Example structure**:
```
.github/
  ├── copilot-instructions.md    # Universal guidelines
  ├── CLAUDE.md                  # Claude-specific (agentic reasoning, planning)
  ├── GEMINI.md                  # Gemini-specific (multimodal, vision tasks)
  └── GPT4.md                    # GPT-4 specific (code complexity, edge cases)
```

**CLAUDE.md example**:
```markdown
# Instructions for Claude Models

Claude excels at reasoning through complex refactoring and architecture decisions.

## Agent Personas for Claude
- **Architect**: Design multi-file refactorings with full dependency understanding
- **Reviewer**: Critique code for logical correctness and edge cases
- **Researcher**: Propose novel algorithmic improvements with mathematical rigor

## Tool Boundaries
- ✓ Full access to plan and execute multi-step features
- ✓ Can propose significant architectural changes with reasoning
- ⚠️ Always request safety review for model inference changes

## Specialized Tasks
- Large refactors: Use Claude for design planning phase
- Code review: Claude is excellent at catching logical flaws
- Documentation: Claude writes clear mathematical explanations
```

**GEMINI.md example**:
```markdown
# Instructions for Gemini Models

Gemini excels at visual analysis and data-heavy tasks.

## Tool Boundaries
- ✓ Full access to analyze dataset structure and statistics
- ✓ Can review notebook visualizations and plots
- ⚠️ Defer complex numerical algorithm design to other models

## Specialized Tasks
- Data exploration: Use Gemini for EDA and visualization generation
- Notebook analysis: Gemini understands notebook structure well
- Performance profiling: Gemini can identify bottleneck patterns
```

**Why use model-specific files?**
- **Optimize for strengths**: Route tasks to models that excel at them
- **Define boundaries**: Set clear expectations for each model's access level
- **Agent personas**: Give agents specialized roles (architect, reviewer, researcher)
- **Safety & trust**: Declare which models handle sensitive code (auth, ML pipeline validation)

---

# Part 3: Best Practices & Patterns

## Practical Guide to Agent Development

### Tips for Effective Agent Requests

#### Tip 1: Leverage Reference Patterns

Guide agents to existing similar code:

```
❌ Vague:
"Add a new API endpoint"

✓ Specific:
"Add a new model class for image preprocessing similar to the existing 
ImageDataset in src/my_package/data.py. Follow the same pattern including 
docstrings, type hints, and error handling."
```

#### Tip 2: Break Large Tasks Into Steps

Agents work better with decomposed tasks:

```
❌ Too big:
"Refactor the payment system to support Stripe and PayPal"

✓ Better:
"Add a new preprocessing pipeline for image augmentation:
1. Create new models class models/augmentation.py with RandomCrop, RandomFlip
2. Integrate augmentation into dataset loading in src/my_package/data.py
3. Add augmentation config parameters to configs/dataset.yaml
4. Add unit tests in tests/test_augmentation.py
Success: agents should apply augmentations during training without breaking existing code"
```

#### Tip 3: Specify Test Expectations

Agents test their work; tell them what to verify:

```
Verification:
- Test that model loads correctly from checkpoint with all hyperparameters
- Test that preprocessing pipeline handles missing values (NaN) correctly
- Test that inference script outputs predictions in expected format
- Load test: 1000 training batches on GPU should complete without memory errors
```

#### Tip 4: Point Out Edge Cases

```
Edge cases to handle:
- Empty or zero values in training data should be skipped or imputed
- Model should handle variable input sequence lengths
- Inference script should gracefully fail if GPU memory is exhausted
- Preprocessing should support both CSV and Parquet file formats
```

### Best Practices & Do's/Don'ts

#### ✓ Do

- **Be explicit about requirements** — agents are literal; they follow specs precisely
- **Provide multiple reference patterns** — agents learn from examples
- **Include error scenarios** — tell agents what to do when things fail
- **Test incrementally** — ask agent to implement and test one piece at a time
- **Review generated code** — understand what the agent built
- **Update instructions** — as codebase evolves, update custom instructions

#### ❌ Don't

- **Don't be vague** — "improve performance" is useless; specify the bottleneck
- **Don't mix concerns** — one task per agent request
- **Don't skip custom instructions** — agents without context make mistakes
- **Don't merge without review** — agents make errors; always review
- **Don't assume agent knows undocumented patterns** — if it's not in instructions or code, it won't be followed
- **Don't ignore test failures** — agent might have broken something subtle

### Practical Development Patterns

#### Pattern 1: Comment-Driven Development

```python
# Parse JSON input and validate against schema
def validate_request(data, schema):
    # Copilot often generates correct implementation from intent
    pass
```

**Tip**: Be specific about the "what" (intent) and let AI fill in the "how".

#### Pattern 2: Iterative Refinement

1. Generate initial code
2. Ask for improvements: "add error handling", "optimize for performance"
3. Repeat until satisfied

#### Pattern 3: Context Injection

If agent misses your coding style:

```
# Our codebase uses dataclasses, dependency injection, and async/await
# Add a new User model following this pattern
```

#### Pattern 4: Testing-First with AI

1. **Write test with AI help** (test describes intent)
2. **Generate implementation** to pass tests
3. **Refactor** with AI suggestions

---

### ML/AI Research Workflows

This section provides practical workflows for using Copilot agents in machine learning and AI research projects. All examples focus on PyTorch, TensorFlow, and common ML libraries (NumPy, pandas, scikit-learn).

#### 8.1 Data Pipeline Workflows

Copilot excels at generating data preprocessing and validation pipelines. The key is to provide clear specifications about data format, validation rules, and quality metrics.

##### Example: Data Validation Pipeline

```
Task: Create a data validation pipeline for preprocessing image datasets

Requirements:
1. Create preprocessing/validate_dataset.py with the following validators:
   - Check image dimensions are within acceptable range (min 224x224, max 4096x4096)
   - Verify file formats (accept .jpg, .png, .webp only)
   - Detect corrupted images using PIL image loading
   - Check for class imbalance in labels (warn if any class < 5% of dataset)
   - Log data quality metrics (total images, invalid images, size distribution)

2. Validation should support both local paths and cloud storage (S3 URIs)

3. Output validation report as JSON with structure:
   {
     "total_images": int,
     "valid_images": int,
     "invalid_images": [{"path": str, "reason": str}],
     "class_distribution": {class_name: count},
     "quality_warnings": [str]
   }

4. Add pytest tests in tests/test_validation.py for edge cases:
   - Empty dataset directory
   - Mixed valid/invalid images
   - Extreme class imbalance

Reference pattern: Follow the structure in preprocessing/download_dataset.py for
cloud storage handling and logging configuration.

Success criteria:
- Script handles 10,000+ images without memory overflow
- Detects all common corruption types (truncated, wrong format, unreadable)
- Reports complete in < 1 minute for 1000 images on CPU
```

##### Example: Data Augmentation Pipeline

```
Task: Add image augmentation pipeline for training data

Requirements:
1. Create src/my_package/augmentation.py with PyTorch-based augmentations:
   - RandomResizedCrop (scale 0.8-1.0)
   - RandomHorizontalFlip (p=0.5)
   - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
   - RandomRotation (±15 degrees)
   - Normalize using ImageNet stats

2. Integrate into existing Dataset class (src/my_package/data.py):
   - Add augmentation parameter to __init__
   - Apply only during training, not validation
   - Support disabling individual augmentations via config

3. Add augmentation settings to configs/dataset.yaml

4. Add tests that verify:
   - Augmentation changes image tensors deterministically with fixed seed
   - Validation data remains unaugmented
   - All augmentations preserve image shape

Reference: Use torchvision.transforms and follow the pattern in src/my_package/data.py

Success: Training with augmentation should reduce overfitting (validation accuracy improves)
```

**Pattern for reproducible preprocessing**:
- Always set random seeds for augmentation
- Log augmentation parameters to experiment tracking
- Version preprocessing code alongside model checkpoints
- Validate that preprocessing doesn't introduce label leakage

---

#### 8.2 Model Development Workflows

Use Copilot to generate model architectures following your existing codebase patterns.

##### Example: Custom CNN Architecture

```
Task: Implement a custom ResNet-style CNN for image classification

Requirements:
1. Create src/my_package/models/resnet_custom.py with:
   - ResidualBlock class with skip connections
   - CustomResNet class accepting num_classes parameter
   - Architecture: 4 residual blocks with [64, 128, 256, 512] channels
   - Global average pooling before final linear layer
   - Support for pretrained backbone (load ImageNet weights for first 3 blocks)

2. Follow existing patterns:
   - Use same initialization as models/base_model.py
   - Include forward pass type hints
   - Add docstrings explaining architecture choices

3. Add model registration to models/__init__.py

4. Create tests/test_resnet_custom.py:
   - Test forward pass shape (batch_size=16, num_classes=10)
   - Test gradient flow through skip connections
   - Test pretrained weight loading

Architecture notes:
- Each ResidualBlock: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> Add -> ReLU
- Use stride=2 in first conv of each block for downsampling
- Final output shape: (batch_size, num_classes)

Success: Model trains without NaN losses and achieves > 80% validation accuracy on CIFAR-10
```

##### Example: Transformer Model Implementation

```
Task: Add a Vision Transformer (ViT) model for image classification

Requirements:
1. Create src/my_package/models/vit.py with:
   - PatchEmbedding layer (16x16 patches)
   - TransformerEncoder with multi-head self-attention
   - Classification head with LayerNorm
   - Positional encoding (learnable or sinusoidal)

2. Configuration:
   - Support variable image sizes (default 224x224)
   - Configurable: num_layers, hidden_dim, num_heads, mlp_ratio
   - Add all hyperparameters to configs/model_config.yaml

3. Use PyTorch's nn.TransformerEncoder as base, customize attention masks if needed

4. Tests:
   - Forward pass with different image sizes
   - Attention weights have correct shape
   - Positional encoding broadcasts correctly

Reference: Follow the modular structure in models/base_model.py

Success: Model architecture matches ViT-Base specifications (12 layers, 768 hidden dim, 12 heads)
```

**Tips for model generation**:
- Reference existing model files in your request ("like models/base_model.py")
- Specify exact layer names and dimensions to avoid ambiguity
- Request shape validation tests to catch dimension mismatches early
- Ask for gradient flow tests to detect vanishing/exploding gradients

---

#### 8.3 Training Loop & Distributed Training

Copilot can generate complete training loops with distributed training support.

##### Example: DDP Multi-GPU Training

```
Task: Add distributed data parallel (DDP) training support for multi-GPU experiments

Requirements:
1. Create src/my_package/distributed.py with:
   - setup_distributed(): Initialize process group (NCCL backend)
   - cleanup_distributed(): Destroy process group
   - get_rank(): Return current process rank
   - get_world_size(): Return total number of processes
   - is_main_process(): Check if rank == 0

2. Update scripts/train.py to support DDP:
   - Wrap model with torch.nn.parallel.DistributedDataParallel
   - Use DistributedSampler for DataLoader
   - Scale learning rate by world_size (lr = base_lr * world_size)
   - Save checkpoints only on rank 0 to avoid write conflicts
   - Aggregate metrics across all processes before logging
   - Support both single-GPU and multi-GPU modes (auto-detect)

3. Add DDP configuration to configs/training.yaml:
   - distributed: bool (enable/disable)
   - backend: str (nccl, gloo, mpi)
   - init_method: str (env:// or tcp://)

4. Add gradient synchronization verification:
   - Assert gradients are synchronized across GPUs
   - Log gradient norms per rank for debugging

5. Tests (tests/test_distributed.py):
   - Mock multi-GPU with 2 processes using torch.multiprocessing
   - Verify model parameters identical after 1 training step
   - Verify checkpoint saving only on rank 0

Launch command:
- Single GPU: python scripts/train.py --config configs/training.yaml
- Multi GPU: torchrun --nproc_per_node=4 scripts/train.py --config configs/training.yaml

Reference: See distributed patterns in existing scripts/fine_tuning.py

Success criteria:
- Training time scales linearly with GPU count (2x GPUs → ~2x faster)
- Validation metrics identical between single-GPU and multi-GPU runs (same seed)
- No deadlocks or NCCL timeout errors during training
```

##### Example: Mixed Precision Training (AMP)

```
Task: Add automatic mixed precision (AMP) training support

Requirements:
1. Update scripts/train.py to use torch.cuda.amp:
   - Wrap forward pass with autocast()
   - Use GradScaler for gradient scaling
   - Add amp_enabled flag to configs/training.yaml

2. Ensure loss scaling handles overflow/underflow gracefully

3. Benchmark: Compare training speed and memory usage with/without AMP

Success: AMP training achieves ~1.5x speedup with < 0.5% accuracy difference
```

**Distributed training best practices**:
- Always set `torch.cuda.set_device(local_rank)` before model creation
- Use `torch.distributed.barrier()` to synchronize processes before checkpointing
- Log only on rank 0 to avoid duplicate log entries
- Test with `torchrun` locally before scaling to multi-node clusters

---

#### 8.4 Experiment Tracking

Generate experiment tracking integrations for reproducibility.

##### Example: MLflow Integration

```
Task: Integrate MLflow for experiment tracking

Requirements:
1. Create src/my_package/tracking.py with:
   - init_tracking(experiment_name, run_name): Start MLflow run
   - log_params(config_dict): Log all hyperparameters from config
   - log_metrics(metrics_dict, step): Log training/validation metrics
   - log_model(model, artifact_path): Save model as MLflow artifact
   - log_config(config_path): Log config file as artifact

2. Update scripts/train.py:
   - Initialize tracking at start of training
   - Log hyperparameters (lr, batch_size, num_epochs, model architecture)
   - Log metrics every epoch (train_loss, val_loss, val_accuracy)
   - Log final model checkpoint and training curves
   - Support MLflow tracking URI from environment variable MLFLOW_TRACKING_URI

3. Add MLflow server setup instructions to docs/experiment_tracking.md

4. Add to requirements.txt: mlflow>=2.0.0

Success: All experiments logged to MLflow with complete reproducibility information
```

##### Alternative: Weights & Biases Integration

```
Task: Add Weights & Biases (wandb) integration for experiment tracking

Requirements:
- Similar structure to MLflow example above
- Use wandb.init(), wandb.log(), wandb.save()
- Log system metrics (GPU usage, memory) automatically
- Create visualization dashboard for metric comparison

Reference: https://docs.wandb.ai/guides/integrations/pytorch
```

**Experiment tracking tips**:
- Log random seeds for full reproducibility
- Save git commit hash with each experiment
- Log data preprocessing parameters (augmentation settings, normalization stats)
- Track model architecture as JSON for easy comparison

---

#### 8.5 Hyperparameter Tuning

Generate automated hyperparameter search scripts.

##### Example: Optuna Hyperparameter Search

```
Task: Add automated hyperparameter tuning using Optuna

Requirements:
1. Create scripts/tune_hyperparameters.py with:
   - objective(trial): Define search space and return validation metric
   - Search space: learning_rate (1e-5 to 1e-1, log scale)
                   batch_size (16, 32, 64, 128)
                   num_layers (2, 4, 6, 8)
                   hidden_dim (128, 256, 512, 1024)
   - Run Optuna study with TPE sampler
   - Save best hyperparameters to configs/best_params.yaml
   - Generate optimization history plot

2. Integration:
   - Reuse training loop from scripts/train.py
   - Use early stopping (patience=3 epochs) for failed trials
   - Log all trials to MLflow for comparison

3. Add to requirements.txt: optuna>=3.0.0, optuna-dashboard>=0.9.0

Launch: python scripts/tune_hyperparameters.py --n-trials 50 --study-name my_study

Success: Automatically find hyperparameters achieving > 85% validation accuracy
```

##### Alternative: Ray Tune for Distributed Tuning

```
Brief: Add Ray Tune for distributed hyperparameter search across multiple GPUs/nodes
- Similar to Optuna but with distributed trial execution
- Support async hyperband scheduling
```

---

#### 8.6 Model Evaluation & Testing

Generate comprehensive evaluation and testing scripts.

##### Example: Model Testing Suite

```
Task: Create comprehensive model evaluation suite

Requirements:
1. Create tests/test_model_evaluation.py with:
   - test_forward_pass_shapes: Verify output shapes for various input sizes
   - test_gradient_flow: Ensure gradients propagate to all parameters
   - test_deterministic_output: Same input + seed → same output
   - test_batch_independence: Predictions independent of batch composition
   - test_device_consistency: CPU and GPU produce same results (tolerance 1e-5)

2. Create scripts/evaluate.py with:
   - Load trained model from checkpoint
   - Compute metrics: accuracy, precision, recall, F1, confusion matrix
   - Generate ROC curve and AUC for binary/multiclass
   - Compute statistical significance (bootstrap confidence intervals)
   - Save evaluation report as JSON and plots

3. Statistical tests:
   - McNemar's test for comparing two models
   - Paired t-test for metric differences across folds
   - Calibration plots (predicted probabilities vs actual)

Success: Complete evaluation report generated in < 5 minutes for 10K test samples
```

**Testing best practices for ML models**:
- Always test with fixed random seeds for reproducibility
- Test edge cases (empty batches, single samples, very large batches)
- Verify model works on both CPU and GPU
- Test checkpoint save/load cycle (saved model produces same results)
- Use property-based testing (Hypothesis library) for data transformations

---

### Computational Biology & Chemistry Workflows

This section provides workflows specifically for computational biology, chemistry, and molecular modeling research. Examples focus on BioPython, RDKit, MDAnalysis, PyMOL, and scientific computing libraries.

#### 8.7 Molecular Data Processing

Copilot can help generate pipelines for molecular structure processing, sequence analysis, and chemical data validation.

##### Example: Protein Sequence Analysis Pipeline

```
Task: Create a protein sequence analysis pipeline with quality control

Requirements:
1. Create bioprocessing/sequence_analysis.py with:
   - Load FASTA files using BioPython
   - Validate sequences (check for non-standard amino acids)
   - Compute basic properties:
     - Molecular weight
     - Isoelectric point (pI)
     - Hydrophobicity (Kyte-Doolittle scale)
     - Secondary structure prediction (simple method or API call)
   - Identify conserved domains using NCBI CDD or Pfam API
   - Generate sequence alignment with ClustalW or MUSCLE

2. Quality control:
   - Flag sequences shorter than 50 amino acids
   - Detect duplicate sequences
   - Check for low-complexity regions
   - Log quality metrics (total sequences, avg length, composition)

3. Output:
   - CSV with sequence properties
   - Multiple sequence alignment in FASTA format
   - Quality report as JSON

4. Tests (tests/test_sequence_analysis.py):
   - Handle malformed FASTA files
   - Verify property calculations against known proteins
   - Test with edge cases (very short, very long sequences)

Reference: Use BioPython's SeqIO and ProtParam modules

Success: Pipeline processes 1000 sequences in < 2 minutes with accurate properties
```

##### Example: Chemical Structure Validation with RDKit

```
Task: Build a chemical structure validation pipeline for small molecule datasets

Requirements:
1. Create chemistry/validate_molecules.py with:
   - Parse SMILES strings using RDKit
   - Validate chemical structures:
     - Check for valid valence
     - Detect unusual substructures (reactive groups, PAINS filters)
     - Calculate molecular descriptors (MW, LogP, TPSA, num_rotatable_bonds)
   - Standardize molecules (neutralize charges, remove salts, canonicalize tautomers)
   - Detect duplicates using InChI keys

2. Generate quality report:
   - Number of valid/invalid SMILES
   - Distribution of molecular properties
   - Flagged structures with reasons
   - Duplicate clusters

3. Export:
   - Cleaned SDF file with standardized molecules
   - CSV with molecular descriptors
   - JSON validation report

4. Tests:
   - Known invalid SMILES (malformed, impossible structures)
   - Tautomer standardization (ensure consistency)
   - Duplicate detection across different SMILES representations

Reference: Use RDKit's Chem, Descriptors, and MolStandardize modules

Success: Validate 10,000 molecules in < 5 minutes, catch all invalid structures
```

---

#### 8.8 Molecular Dynamics & Simulations

Generate code for setting up MD simulations, trajectory analysis, and visualization.

##### Example: MD Trajectory Analysis with MDAnalysis

```
Task: Create a molecular dynamics trajectory analysis pipeline

Requirements:
1. Create md_analysis/trajectory_analyzer.py with:
   - Load MD trajectories using MDAnalysis (GROMACS, AMBER, NAMD formats)
   - Compute trajectory metrics:
     - RMSD (backbone and all-atom)
     - RMSF per residue
     - Radius of gyration over time
     - Hydrogen bond analysis
     - Distance between specified atoms/residues
   - Identify stable conformations (clustering by RMSD)

2. Visualization:
   - Plot time series (RMSD, Rg, distances)
   - Generate heatmaps (RMSF by residue, contact maps)
   - Export key frames for PyMOL visualization

3. Integration:
   - Accept trajectory path and topology file from config
   - Support multiple trajectory formats (XTC, DCD, TRR)
   - Parallel processing for large trajectories (use Dask or multiprocessing)

4. Tests:
   - Verify RMSD calculation against reference
   - Test with different trajectory formats
   - Handle incomplete trajectories gracefully

Reference: Use MDAnalysis.analysis modules for RMSD, RMSF, hydrogen bonds

Success: Analyze 10 ns trajectory (100K frames) in < 10 minutes on single CPU
```

##### Example: Molecular Docking Workflow with AutoDock

```
Task: Automate molecular docking workflow for virtual screening

Requirements:
1. Create docking/autodock_pipeline.py with:
   - Prepare protein receptor (add hydrogens, assign charges)
   - Prepare ligands from SMILES or SDF (generate 3D coords, assign charges)
   - Define docking box (center coordinates and size)
   - Run AutoDock Vina for each ligand
   - Parse docking results (binding affinity, poses)
   - Rank ligands by predicted affinity

2. Output:
   - CSV with ligand ID, SMILES, binding affinity, best pose
   - SDF files with docked poses
   - Top 10 hits for further analysis

3. Parallel execution:
   - Use multiprocessing to dock ligands in parallel
   - Support SLURM for HPC cluster submission

4. Tests:
   - Verify docking reproduces known protein-ligand complex
   - Test with different box sizes
   - Handle docking failures gracefully

Reference: Use RDKit for ligand prep, OpenBabel for format conversion, subprocess for Vina

Success: Screen 1000 ligands against one receptor in < 2 hours (16 CPUs)
```

---

#### 8.9 Omics Data Analysis

Generate workflows for genomics, proteomics, and transcriptomics analysis.

##### Example: RNA-seq Differential Expression Analysis

```
Task: Create RNA-seq differential expression analysis pipeline

Requirements:
1. Create rnaseq/diff_expression.py with:
   - Load count matrix (genes x samples) from CSV or HDF5
   - Normalize counts (TPM, RPKM, or DESeq2 normalization)
   - Perform differential expression analysis:
     - Use scipy.stats for t-tests or edgeR/DESeq2 via rpy2
     - Calculate log2 fold change and adjusted p-values (FDR)
   - Filter significant genes (padj < 0.05, |log2FC| > 1)

2. Visualization:
   - Volcano plot (log2FC vs -log10(padj))
   - MA plot (mean expression vs log2FC)
   - Heatmap of top differentially expressed genes
   - PCA plot of samples (colored by condition)

3. Gene set enrichment:
   - Query enriched pathways using Enrichr API or local gene sets
   - Generate enrichment report (pathway, p-value, genes)

4. Export:
   - CSV with all genes and statistics
   - CSV with significant genes only
   - PDF with all plots
   - JSON enrichment results

Reference: Use pandas, scipy, seaborn/matplotlib, and statsmodels

Success: Analyze 20K genes x 50 samples in < 5 minutes, generate publication-ready plots
```

##### Example: Mass Spectrometry Proteomics Pipeline

```
Task: Process and analyze mass spectrometry proteomics data

Requirements:
1. Create proteomics/ms_analysis.py with:
   - Load peptide identification results (MaxQuant, Mascot CSV)
   - Filter by confidence (FDR < 0.01, peptide score > threshold)
   - Protein inference (group peptides to proteins)
   - Quantification:
     - Label-free quantification (LFQ intensities)
     - Normalize across samples (median normalization)
     - Impute missing values (KNN or min-value imputation)
   - Differential abundance analysis (t-test, limma-like)

2. Quality control:
   - Coefficient of variation (CV) for replicates
   - Missing value heatmap
   - Sample correlation matrix
   - PCA for batch effect detection

3. Visualization:
   - Protein abundance heatmap (hierarchical clustering)
   - Volcano plot for differential proteins
   - Venn diagrams for protein overlap between conditions

4. Tests:
   - Verify normalization reduces technical variation
   - Test imputation methods on simulated missing data
   - Validate statistical tests against known standards

Reference: Use pandas, scipy, scikit-learn for imputation, seaborn for viz

Success: Process 5000 proteins x 30 samples in < 3 minutes with full QC report
```

---

#### 8.10 Structure Prediction & Modeling

Generate code for protein structure prediction, homology modeling, and structure analysis.

##### Example: AlphaFold Integration Workflow

```
Task: Automate AlphaFold2 structure prediction and analysis

Requirements:
1. Create structure_prediction/alphafold_runner.py with:
   - Prepare input FASTA sequences
   - Generate MSA (multiple sequence alignment) using MMseqs2 or HHblits
   - Run AlphaFold2 prediction (via LocalColabFold or API)
   - Parse prediction results:
     - Extract predicted structures (PDB format)
     - Parse pLDDT (confidence scores per residue)
     - Extract PAE (predicted aligned error) matrices

2. Analysis:
   - Identify high-confidence regions (pLDDT > 70)
   - Detect disordered regions (pLDDT < 50)
   - Compare multiple models (if generated)
   - Superimpose with known structure (if available) and calculate RMSD

3. Visualization scripts:
   - Color PDB by pLDDT (PyMOL script)
   - Plot pLDDT along sequence
   - Heatmap of PAE matrix

4. Integration:
   - Support batch prediction for multiple sequences
   - Save results in organized directory structure
   - Log prediction metadata (sequence length, runtime, confidence)

Reference: Use BioPython for PDB parsing, numpy for PAE analysis

Success: Predict structure for 100 sequences (avg 300 aa) in < 24 hours with full QC
```

##### Example: Protein-Protein Docking Analysis

```
Task: Analyze protein-protein docking results from HADDOCK or ClusPro

Requirements:
1. Create docking/protein_docking_analysis.py with:
   - Load docking poses (PDB files)
   - Calculate interface metrics:
     - Interface area (buried surface area)
     - Number of interface residues
     - Hydrogen bonds at interface
     - Salt bridges
   - Energy analysis (if scores available)
   - Cluster poses by interface RMSD

2. Visualization:
   - Interface residue mapping on sequence
   - Contact map (residue-residue distances < 5 Å)
   - PyMOL scripts to highlight interface

3. Export:
   - CSV with pose ID, score, interface area, num contacts
   - Top 5 poses as separate PDB files
   - Interface residue list

Reference: Use BioPython, ProDy, or MDAnalysis for structure analysis

Success: Analyze 100 docking poses in < 5 minutes, identify key interface residues
```

---

#### 8.11 Cheminformatics & QSAR Modeling

Generate machine learning models for molecular property prediction and drug design.

##### Example: QSAR Model for Molecular Property Prediction

```
Task: Build QSAR (Quantitative Structure-Activity Relationship) model for bioactivity prediction

Requirements:
1. Create qsar/property_predictor.py with:
   - Load molecular dataset (SMILES + activity values)
   - Calculate molecular descriptors:
     - Morgan fingerprints (RDKit)
     - 2D descriptors (MW, LogP, TPSA, num_rings, etc.)
     - 3D descriptors (if 3D structures available)
   - Feature selection (remove low-variance, high-correlation features)
   - Train ML models:
     - Random Forest regression
     - XGBoost regression
     - Neural network (sklearn MLPRegressor or PyTorch)
   - Cross-validation (5-fold or leave-one-out)
   - Evaluate with R², RMSE, MAE

2. Model interpretation:
   - Feature importance (for tree models)
   - SHAP values for model explainability
   - Identify key structural features correlated with activity

3. Prediction pipeline:
   - Load trained model from checkpoint
   - Predict on new SMILES
   - Provide confidence intervals

4. Tests:
   - Verify descriptor calculation reproducibility
   - Test model save/load cycle
   - Validate predictions on known molecules

Reference: Use RDKit for descriptors, scikit-learn for ML, SHAP for interpretation

Success: Achieve R² > 0.7 on test set for bioactivity prediction
```

---

#### 8.12 Structural Bioinformatics Tools

Generate utilities for protein structure analysis, visualization, and quality assessment.

##### Example: Protein Structure Quality Assessment

```
Task: Assess quality of protein structures (experimental or predicted)

Requirements:
1. Create structure_qc/quality_checker.py with:
   - Load PDB structure using BioPython
   - Geometry validation:
     - Ramachandran plot (phi-psi angles)
     - Bond length and angle deviations
     - Clash detection (steric clashes)
     - Rotamer outliers
   - Stereochemistry checks (using MolProbity-style analysis)
   - Calculate quality scores:
     - Overall quality Z-score
     - Per-residue quality scores

2. Comparison with experimental data (if available):
   - Calculate R-factor and R-free (for X-ray structures)
   - Compare B-factors distribution
   - Identify poorly resolved regions

3. Output:
   - Quality report as JSON
   - Ramachandran plot (matplotlib)
   - List of residues with issues
   - PyMOL script to highlight problem regions

4. Tests:
   - Known high-quality structure (should pass checks)
   - Known low-quality structure (should flag issues)
   - Synthetic bad geometry (should detect all issues)

Reference: Use BioPython, ProDy, or pymol for structure analysis

Success: Accurately identify quality issues matching MolProbity validation
```

---

### Best Practices for Computational Bio/Chem with Copilot

**When requesting workflows**:
- Specify exact software versions (RDKit 2023.9.1, BioPython 1.81) as APIs change
- Reference established protocols (e.g., "follow GROMACS tutorial workflow")
- Include typical dataset sizes for performance requirements
- Request citation information for methods used

**Domain-specific patterns**:
- Always validate chemical structures before analysis (use RDKit sanitization)
- Set random seeds for reproducible simulations and splits
- Log all parameters (simulation time, force field, cutoffs) for reproducibility
- Include unit tests with known reference molecules/structures
- Request proper error handling for file format variations

**Integration with research workflows**:
- Generate scripts that output standardized formats (PDB, SDF, FASTA)
- Request logging compatible with lab notebooks (Jupyter, LabArchives)
- Include data provenance tracking (input sources, processing steps)
- Support both local execution and HPC cluster submission (SLURM, PBS)

**Common tools to request integration with**:
- Structure: PyMOL, Chimera/ChimeraX, VMD
- Docking: AutoDock Vina, DOCK, Glide, HADDOCK
- MD: GROMACS, AMBER, NAMD, OpenMM
- Cheminformatics: RDKit, OpenBabel, CDK
- Biology: BioPython, Biopandas, ProDy
- Analysis: pandas, numpy, scipy, scikit-learn

---

# Part 4: Quality & Safety

## Security & Code Quality Considerations

### Risks with AI-Generated Code

- **License compliance**: ensure generated code doesn't violate open-source licenses
- **Security vulnerabilities**: AI may suggest insecure patterns (SQL injection, weak crypto)
- **Outdated libraries**: suggestions might use deprecated APIs
- **Logic errors**: AI-generated code can contain subtle bugs

### Mitigation Strategies

1. **Always review** generated code before merging
2. **Run security linters**: use tools like Bandit (Python), Snyk, or similar
3. **Test thoroughly**: unit tests, integration tests, security tests
4. **Use IDE inspections**: let your IDE flag warnings
5. **Understand the code**: don't accept code you can't explain
6. **Check dependencies**: verify suggested libraries are maintained and safe

### Code Review Practices

- **Treat AI-generated code as a draft**, not final
- **Ask questions**: if logic is unclear, ask Copilot to explain it
- **Compare alternatives**: regenerate suggestions and pick the best
- **Document reasoning**: add comments explaining why you kept/rejected AI suggestions

---

## Anti-patterns & Pitfalls

❌ **Don't**: Accept first suggestion without review  
✓ **Do**: Review, test, understand before merging

❌ **Don't**: Use AI for security-critical code without heavy scrutiny  
✓ **Do**: Have security experts review authentication, crypto, access control

❌ **Don't**: Assume AI understands your entire codebase  
✓ **Do**: Provide context files, coding standards, architecture docs

❌ **Don't**: Rely on AI for complex business logic without tests  
✓ **Do**: Write unit tests that define expected behavior

❌ **Don't**: Copy-paste generated code across projects  
✓ **Do**: Adapt suggestions to your project's conventions and security posture

---

## Integration with Development Workflow

### Pull Request Workflows (Team Environments)

For collaborative codebases with pull request reviews:

#### Step 5: Review & Iterate

- Review agent-generated PR (understand the logic)
- Request changes if needed: "add rate limiting", "improve error messages"
- Agent implements changes, pushes updates, and updates PR

#### Step 6: Merge

Once satisfied with the implementation and tests pass, merge the PR to main branch.

### Solo & Research Workflows (No PR Review)

For individual researchers or solo development:

Instead of creating a PR, agent can:
- Commit changes directly to a branch
- Generate a summary of changes (similar to PR description)
- Provide before/after comparison for your review
- You validate locally and merge when ready

**Approach**:
```
After step 3 (Clear Request), agent implements directly:
1. Agent creates a feature branch
2. Agent implements and tests changes
3. Agent generates change summary with diffs
4. You review summary locally
5. You merge branch to main
```

This skips the PR infrastructure overhead while maintaining code review discipline.

### In a Team

1. **Establish guidelines**: team consensus on AI usage (what, where, how to review)
2. **Code review process**: peer reviews must include AI-generated code examination
3. **Security review**: have one person/role verify AI suggestions for security
4. **Documentation**: note where AI was used (commit messages, comments)
5. **Training**: ensure team understands AI limitations

### In CI/CD

- **Pre-commit hooks**: lint/test AI-generated code before commit
- **Automated testing**: all AI suggestions must pass test suite
- **Type checking**: use mypy, TypeScript strict mode, etc., to catch errors
- **Security scanning**: run SAST tools to flag potential vulnerabilities

### Documentation & Knowledge Sharing

- Use AI to draft internal docs, architecture guides, README sections
- Have team review and verify accuracy
- Store AI-generated templates for common tasks

---

# Part 5: Reference

## When to Use, When Not to Use

### ✓ Good Use Cases

- Boilerplate code (getters, setters, data classes)
- Test scaffolds (setup/teardown, test cases)
- Documentation (docstrings, API docs, README sections)
- Repetitive patterns (logging, error handling)
- Code exploration and learning
- Refactoring suggestions
- Bug-fix brainstorming

### ✗ Poor Use Cases

- Security-critical code (authentication, encryption, access control)
- Business logic that defines revenue/legal requirements
- Real-time systems where correctness is critical
- Code with strict performance constraints
- First-time designs without existing patterns in codebase

---

## Troubleshooting & FAQ

**Q: Copilot's suggestions are poor quality or irrelevant.**  
A: Provide more context (comments, imports, function signatures). Use Copilot Chat for complex requests instead of inline suggestions.

**Q: How do I know if AI-generated code is secure?**  
A: Run security linters, have security review, test edge cases. Never assume AI-generated security code is correct.

**Q: Can I use AI suggestions for production code?**  
A: Yes, but only after thorough review, testing, and security validation. Same standards as human code.

**Q: My AI tool doesn't understand my codebase style.**  
A: Provide explicit context: comment with coding conventions, include style examples, or create a "coding standards" file you reference in prompts.

**Q: Does using AI tools count as plagiarism?**  
A: No, if you review, understand, and test the code. The AI output is a tool, like a code generator or framework. However, always attribute AI usage in team discussions and code reviews.

---

## References & Further Reading

- **GitHub Copilot**:
  - Official docs: https://docs.github.com/en/copilot
  - Best practices: https://github.blog/2023-06-20-how-to-build-with-github-copilot/
  - VS Code integration: https://marketplace.visualstudio.com/items?itemName=GitHub.copilot

- **AI Code Security & Quality**:
  - OWASP secure coding: https://owasp.org/
  - CWE (Common Weakness Enumeration): https://cwe.mitre.org/
  - Snyk security scanning: https://snyk.io/

- **Workflow & Team Practices**:
  - GitHub flow: https://guides.github.com/introduction/flow/
  - Code review best practices: https://google.github.io/eng-practices/review/reviewer/

---

## Quick Command Reference

### VS Code + Copilot

| Action | Shortcut |
|--------|----------|
| Accept suggestion | Tab |
| Reject suggestion | Esc |
| Open Copilot Chat | Ctrl+Shift+I |
| Inline suggestion | Alt+\ |
| Trigger completion | Ctrl+Space |

### Tips for Effective Prompts

```
Good: "write a function to validate email addresses using regex"
Better: "write a function that validates email format (local@domain.ext) and returns True/False"
Best: "validate email using regex; handle common formats like name+tag@domain.co.uk"
```

---

# Part 6: MCP Integration

## What is MCP?

**MCP (Model Context Protocol)** is an open protocol that enables AI assistants like GitHub Copilot and Claude to connect to external data sources and tools during conversations. Think of MCP servers as "plugins" that extend Copilot's capabilities beyond just your local codebase.

### Why MCP Matters for ML Research

In ML research workflows, you often need to:
- **Query research literature** (papers, documentation, API references)
- **Access external datasets** (cloud storage, databases, data lakes)
- **Fetch experiment results** (metrics from MLflow, W&B, TensorBoard)
- **Search code repositories** (GitHub, GitLab for reference implementations)
- **Query APIs** (model serving endpoints, data validation services)

Without MCP, Copilot is limited to your local code and its training data. With MCP servers, Copilot can fetch real-time information during code generation, making suggestions more accurate and context-aware.

### How MCP Extends Copilot

**Example workflow without MCP**:
```
You: "Implement a ResNet model"
Copilot: [Generates generic ResNet from training data, may be outdated]
```

**Example workflow with MCP (Perplexity server)**:
```
You: "Implement a ResNet model using the latest PyTorch 2.0 features"
Copilot: [Uses MCP to query Perplexity for PyTorch 2.0 ResNet docs]
Copilot: [Generates ResNet with torch.compile() and new APIs]
```

MCP servers can provide:
- Up-to-date library documentation
- Research paper implementations
- Database schema information
- Real-time metrics and logs
- File system access to large datasets

---

## Setting Up MCP Servers

MCP servers are configured in VS Code using a `.vscode/mcp.json` file in your workspace. This file defines which servers to connect to and how to authenticate.

### Basic MCP Configuration Structure

Create `.vscode/mcp.json` in your project root:

```json
{
  "servers": {
    "server-name": {
      "type": "http",
      "url": "https://api.example.com/mcp/v1",
      "env": {
        "API_KEY": "${input:api-key-id}"
      }
    }
  },
  "inputs": [
    {
      "id": "api-key-id",
      "description": "API key for the service",
      "type": "promptString",
      "password": true
    }
  ]
}
```

**Key components**:
- `servers`: Dictionary of MCP server configurations
- `type`: Connection type (`http` for remote servers, `stdio` for local processes)
- `url`: API endpoint for the MCP server
- `env`: Environment variables (like API keys) passed to the server
- `inputs`: User prompts for sensitive values (API keys, tokens)

---

### Example 1: Perplexity MCP Server (Research Literature)

**Use case**: Query research papers, documentation, and technical articles during development.

This example is from the repository at [`examples/vibe_coding_examples/research_repo/.vscode/mcp.json`](../examples/vibe_coding_examples/research_repo/.vscode/mcp.json).

**Configuration**:
```json
{
  "servers": {
    "perplexity-research": {
      "type": "http",
      "url": "https://api.perplexity.ai/mcp/v1",
      "env": {
        "PERPLEXITY_API_KEY": "${input:perplexity-api-key}"
      }
    }
  },
  "inputs": [
    {
      "id": "perplexity-api-key",
      "description": "Perplexity API key",
      "type": "promptString",
      "password": true
    }
  ]
}
```

**How to get Perplexity API key**:
1. Sign up at https://www.perplexity.ai/
2. Navigate to Settings → API
3. Generate a new API key
4. Copy the key (you'll be prompted to enter it when VS Code loads)

**Usage during development**:
```
You: "@perplexity-research What are the best practices for implementing
     PyTorch DDP training with gradient accumulation?"

Copilot: [Queries Perplexity for latest DDP documentation and best practices]
Copilot: [Generates code based on current PyTorch recommendations]
```

**When to use**:
- Implementing new features using latest library versions
- Finding optimal hyperparameters from recent papers
- Checking current best practices for ML patterns
- Debugging with up-to-date Stack Overflow discussions

---

### Example 2: GitHub MCP Server (Code Search)

**Use case**: Search other repositories for reference implementations during development.

**Configuration**:
```json
{
  "servers": {
    "github-search": {
      "type": "http",
      "url": "https://api.github.com/mcp/v1",
      "env": {
        "GITHUB_TOKEN": "${input:github-token}"
      }
    }
  },
  "inputs": [
    {
      "id": "github-token",
      "description": "GitHub Personal Access Token",
      "type": "promptString",
      "password": true
    }
  ]
}
```

**How to get GitHub token**:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with `repo` scope
3. Copy the token

**Usage**:
```
You: "@github-search Find PyTorch implementations of Vision Transformer
     with multi-GPU training support"

Copilot: [Searches GitHub for relevant repositories]
Copilot: [Shows top implementations with stars and recent activity]
You: "Use the pattern from repository X for my implementation"
Copilot: [Generates code following that repository's structure]
```

---

### Example 3: Filesystem MCP Server (Large Dataset Access)

**Use case**: Access large datasets and experiment logs without loading them into VS Code.

**Configuration**:
```json
{
  "servers": {
    "data-filesystem": {
      "type": "stdio",
      "command": "mcp-server-filesystem",
      "args": ["/path/to/data", "/path/to/experiments"],
      "env": {}
    }
  }
}
```

**Installation**:
```bash
npm install -g @modelcontextprotocol/server-filesystem
```

**Usage**:
```
You: "@data-filesystem What's the schema of train.csv in /path/to/data?"

Copilot: [Reads first few rows and infers schema]
Copilot: Columns: id (int), image_path (str), label (int), split (str)

You: "Generate a PyTorch Dataset class for this CSV"
Copilot: [Generates Dataset with correct column names and types]
```

**When to use**:
- Working with datasets too large to open in editor
- Reading experiment logs and metrics from training runs
- Accessing cloud-mounted storage (S3, GCS, Azure Blob)

---

### Example 4: Database MCP Server (Experiment Metadata)

**Use case**: Query experiment tracking databases (MLflow, W&B) for metrics and configurations.

**Configuration** (SQLite example):
```json
{
  "servers": {
    "mlflow-db": {
      "type": "stdio",
      "command": "mcp-server-sqlite",
      "args": ["--db-path", "/path/to/mlflow.db"],
      "env": {}
    }
  }
}
```

**Usage**:
```
You: "@mlflow-db What were the hyperparameters for the best run
     with val_accuracy > 0.9?"

Copilot: [Queries MLflow database]
Copilot: Best run ID: a1b2c3, lr=0.001, batch_size=64, num_layers=12

You: "Use those hyperparameters in my training script"
Copilot: [Updates configs/training.yaml with those values]
```

---

### Environment Variable Management

**Security best practices**:

1. **Never commit API keys** to `.vscode/mcp.json`
   - Use `${input:key-id}` pattern to prompt at runtime
   - Or use `${env:ENV_VAR_NAME}` to read from environment

2. **Use VS Code secret storage** for sensitive tokens:
   ```json
   {
     "env": {
       "API_KEY": "${secret:my-api-key}"
     }
   }
   ```

3. **For team repositories**, use environment variables:
   ```json
   {
     "env": {
       "PERPLEXITY_API_KEY": "${env:PERPLEXITY_API_KEY}"
     }
   }
   ```

   Then set in your shell:
   ```bash
   export PERPLEXITY_API_KEY="your-key-here"
   ```

4. **Add `.vscode/mcp.json` to `.gitignore`** if it contains sensitive URLs or proxy configurations

---

### Multiple MCP Servers Configuration

You can configure multiple servers simultaneously:

```json
{
  "servers": {
    "perplexity-research": {
      "type": "http",
      "url": "https://api.perplexity.ai/mcp/v1",
      "env": {
        "PERPLEXITY_API_KEY": "${input:perplexity-api-key}"
      }
    },
    "github-search": {
      "type": "http",
      "url": "https://api.github.com/mcp/v1",
      "env": {
        "GITHUB_TOKEN": "${env:GITHUB_TOKEN}"
      }
    },
    "local-data": {
      "type": "stdio",
      "command": "mcp-server-filesystem",
      "args": ["/mnt/datasets", "/mnt/experiments"],
      "env": {}
    }
  },
  "inputs": [
    {
      "id": "perplexity-api-key",
      "description": "Perplexity API key for research queries",
      "type": "promptString",
      "password": true
    }
  ]
}
```

---

## Using MCP During ML Development

### Workflow 1: Finding Optimal Implementation Patterns

**Scenario**: You need to implement a new feature but aren't sure of the best approach.

```
You: "@perplexity-research What's the most efficient way to implement
     mixed precision training with PyTorch AMP for large transformer models?"

Copilot: [Queries Perplexity for latest research and documentation]
Copilot: "Based on current best practices, here's an implementation using
         torch.cuda.amp.autocast() with gradient scaling..."

[Copilot generates code with proper AMP context managers and scaler]
```

### Workflow 2: Debugging with Context

**Scenario**: You encounter an error and need to understand recent solutions.

```
You: "@perplexity-research I'm getting 'NCCL timeout' errors during DDP
     training with PyTorch 2.0. What are common causes and fixes?"

Copilot: [Searches recent discussions and GitHub issues]
Copilot: "Common causes include: 1) Mismatched NCCL versions,
         2) Firewall blocking ports, 3) Uneven batch sizes across GPUs..."

[Copilot suggests diagnostic code and configuration changes]
```

### Workflow 3: Dataset Schema Discovery

**Scenario**: Working with an unfamiliar dataset structure.

```
You: "@local-data Show me the structure of files in /mnt/datasets/imagenet"

Copilot: [Lists directory structure]
Copilot:
  train/
    n01440764/ (tench)
    n01443537/ (goldfish)
    ...
  val/
    ...

You: "Generate a PyTorch ImageFolder dataset loader for this structure"
Copilot: [Generates dataset class with correct paths and transforms]
```

---

## MCP Troubleshooting

### Common Connection Errors

#### Error: "Failed to connect to MCP server"

**Causes**:
- Incorrect URL or server offline
- Network/firewall blocking connection
- Missing authentication credentials

**Solutions**:
1. Verify URL is correct and server is running
2. Test connection: `curl -H "Authorization: Bearer $API_KEY" <url>`
3. Check VS Code Output panel → "MCP" for detailed logs
4. Verify API key is set correctly (check for trailing spaces)

---

#### Error: "Authentication failed"

**Causes**:
- Invalid or expired API key
- API key not properly passed to server
- Wrong environment variable name

**Solutions**:
1. Regenerate API key from provider dashboard
2. Verify `${input:id}` matches `inputs` array in config
3. Check environment variable is set: `echo $VAR_NAME`
4. Restart VS Code after updating API keys

---

#### Error: "Rate limit exceeded"

**Causes**:
- Too many requests to MCP server
- API quota exhausted

**Solutions**:
1. Check API usage dashboard (Perplexity, GitHub, etc.)
2. Reduce request frequency
3. Upgrade API plan if needed
4. Cache responses locally for repeated queries

---

#### Error: "MCP server not found in chat"

**Causes**:
- MCP server not registered properly
- VS Code hasn't loaded `.vscode/mcp.json`
- Syntax error in JSON config

**Solutions**:
1. Reload VS Code window (Cmd/Ctrl+Shift+P → "Reload Window")
2. Validate JSON syntax: `jq . .vscode/mcp.json`
3. Check VS Code settings: "Copilot: Enable MCP" is turned on
4. Verify server name matches when using `@server-name` in chat

---

### Performance Considerations

**MCP queries add latency** to Copilot responses:

- **Local MCP servers** (filesystem, sqlite): < 100ms overhead
- **Remote HTTP servers** (Perplexity, GitHub): 500ms - 2s overhead
- **Multiple MCP queries**: Cumulative latency

**Best practices**:
- Use MCP only when external context is needed
- Cache responses for repeated queries
- Prefer local MCP servers for frequently accessed data
- Batch multiple questions in one query when possible

---

### Debugging MCP Issues

**Enable verbose logging**:

1. Open VS Code settings (Cmd/Ctrl+,)
2. Search for "Copilot MCP"
3. Enable "Copilot: MCP Verbose Logging"
4. Check Output panel → "MCP" for detailed connection logs

**Test MCP server independently**:

For HTTP servers:
```bash
curl -X POST https://api.example.com/mcp/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

For stdio servers:
```bash
echo '{"method": "ping"}' | mcp-server-filesystem /path
```

**Common JSON syntax errors**:
- Trailing commas (not allowed in JSON)
- Unescaped quotes in strings
- Missing closing braces
- Comments (not allowed in standard JSON, use JSONC)

---

## MCP Server Resources

### Official MCP Servers

- **Filesystem**: `@modelcontextprotocol/server-filesystem` - Local file access
- **GitHub**: Built-in GitHub integration for code search
- **Perplexity**: Research and documentation queries
- **Brave Search**: Web search capabilities
- **SQLite**: Local database queries

### Community MCP Servers

- **Postgres MCP**: PostgreSQL database access
- **AWS S3 MCP**: S3 bucket file access
- **Slack MCP**: Slack workspace search
- **Notion MCP**: Notion database queries

### Building Custom MCP Servers

For advanced use cases (proprietary data sources, internal APIs), you can build custom MCP servers:

- **MCP SDK**: https://github.com/modelcontextprotocol/typescript-sdk
- **Python MCP**: https://github.com/modelcontextprotocol/python-sdk
- **Specification**: https://spec.modelcontextprotocol.io/

**Use cases for custom servers**:
- Internal company knowledge bases
- Proprietary experiment tracking systems
- Custom data validation APIs
- Legacy database integrations

---

## Additional MCP Resources

- **Microsoft Announcement**: https://developer.microsoft.com/blog/announcing-awesome-copilot-mcp-server
- **Awesome Copilot MCP**: https://github.com/github/awesome-copilot/tree/main?tab=readme-ov-file
- **MCP Documentation**: https://modelcontextprotocol.io/docs
