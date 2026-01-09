# Using GitHub Copilot for Planning & Agentic Development

This document covers practical workflows for using **GitHub Copilot** as an AI agent for software development. It focuses on planning strategies, agentic workflows, and how to guide agents to understand your codebase and complete complex tasks autonomously.

---

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
"Add JWT authentication to the API with email verification"
"Refactor the payment system to support multiple currencies"
"Add caching layer using Redis to improve performance"
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

## Why Use Copilot Agents?

1. **Planning**: AI reasons through requirements and breaks tasks into steps
2. **Multi-file understanding**: agents analyze dependencies across files
3. **Autonomous execution**: creates pull requests with complete, tested implementations
4. **Codebase context**: agents learn your architecture and conventions
5. **Reduced human iteration**: better pre-planning = fewer review cycles

**Key difference from Chat**: Agents are goal-oriented, not just responsive. They plan first, then execute.

---

## Planning with Copilot Agents

### Agent Planning Workflow

Agents follow this internal process:

1. **Understand the requirement** — parse user request and acceptance criteria
2. **Explore codebase** — read relevant files, understand architecture and conventions
3. **Plan the solution** — break into logical steps, identify all files to modify
4. **Implement** — write code following discovered patterns
5. **Test** — run tests, validate changes
6. **Create PR** — submit with description and change summary

Your job is to **provide good inputs** at step 1, and **custom instructions** to accelerate steps 2-3.

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

---

## Repository Custom Instructions for Agents

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

**2. Exact Build & Test Commands** (agents must validate their work)
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

```
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

---

## Agentic Workflow: Step-by-Step

### 1. Set Up Custom Instructions (Foundation)

Ensure `.github/copilot-instructions.md` exists and is comprehensive. Agents use this as their knowledge base.

### 2. Open Copilot Agent

Navigate to [github.com/copilot/agents](https://github.com/copilot/agents) to access Copilot Coding Agent.

### 3. Write a Clear Request

```
Task: Add a data validation module for training datasets

Requirements:
1. Create src/my_package/validation.py with functions to check data quality
2. Validate that features have no NaN values (impute or drop if found)
3. Check that labels are within expected ranges
4. Log validation results with sample statistics
5. Raise ValueError if validation fails with descriptive message
6. Add comprehensive unit tests

Success criteria:
- Training script validates dataset before training starts
- Validation output includes row counts, missing value rates, label distribution
- All tests pass locally

Reference pattern: See how preprocessing is done in preprocess_scripts/02_create_dataset.py
```

### 4. Agent Creates a Draft PR

- Agent explores your codebase
- Agent creates a branch and implements changes
- Agent runs tests and validates
- Agent opens a PR with a detailed description

### 5. Review & Iterate

- Review the PR (understand the logic)
- Request changes if needed: "add rate limiting to /auth/verify-email"
- Agent implements, pushes updates, and updates PR

### 6. Merge

Once satisfied, merge the PR.

---

## Practical Tips for Agent Success

### Tip 1: Leverage Reference Patterns

Guide agents to existing similar code:

```
❌ Vague:
"Add a new API endpoint"

✓ Specific:
"Add a new model class for image preprocessing similar to the existing 
ImageDataset in src/my_package/data.py. Follow the same pattern including 
docstrings, type hints, and error handling."
```

### Tip 2: Break Large Tasks Into Steps

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

### Tip 3: Specify Test Expectations

Agents test their work; tell them what to verify:

```
Verification:
- Test that model loads correctly from checkpoint with all hyperparameters
- Test that preprocessing pipeline handles missing values (NaN) correctly
- Test that inference script outputs predictions in expected format
- Load test: 1000 training batches on GPU should complete without memory errors
```

### Tip 4: Point Out Edge Cases

```
Edge cases to handle:
- Empty or zero values in training data should be skipped or imputed
- Model should handle variable input sequence lengths
- Inference script should gracefully fail if GPU memory is exhausted
- Preprocessing should support both CSV and Parquet file formats
```

---

## Best Practices for Agentic Development

### ✓ Do

- **Be explicit about requirements** — agents are literal; they follow specs precisely
- **Provide multiple reference patterns** — agents learn from examples
- **Include error scenarios** — tell agents what to do when things fail
- **Test incrementally** — ask agent to implement and test one piece at a time
- **Review generated code** — understand what the agent built
- **Update instructions** — as codebase evolves, update custom instructions

### ❌ Don't

- **Don't be vague** — "improve performance" is useless; specify the bottleneck
- **Don't mix concerns** — one task per agent request
- **Don't skip custom instructions** — agents without context make mistakes
- **Don't merge without review** — agents make errors; always review
- **Don't assume agent knows undocumented patterns** — if it's not in instructions or code, it won't be followed
- **Don't ignore test failures** — agent might have broken something subtle


---

## Practical Tips & Patterns

### Pattern 1: Comment-Driven Development

```python
# Parse JSON input and validate against schema
def validate_request(data, schema):
    # Copilot often generates correct implementation from intent
    pass
```

**Tip**: Be specific about the "what" (intent) and let AI fill in the "how".

### Pattern 2: Iterative Refinement

1. Generate initial code (Copilot or Antigravity)
2. Ask for improvements: "add error handling", "optimize for performance"
3. Repeat until satisfied

### Pattern 3: Context Injection

If AI suggestions miss your coding style:

```
# Our codebase uses dataclasses, dependency injection, and async/await
# Please follow this pattern for the following function:
def process_data(data):
    pass
```

### Pattern 4: Testing-First with AI

1. **Write test with AI help** (test describes intent)
2. **Generate implementation** to pass tests
3. **Refactor** with AI suggestions---

## Practical Tips & Patterns

### Pattern 1: Comment-Driven Development

```python
# Parse JSON input and validate against schema
def validate_request(data, schema):
    # Copilot often generates correct implementation from intent
    pass
```

**Tip**: Be specific about the "what" (intent) and let AI fill in the "how".

### Pattern 2: Iterative Refinement

1. Generate initial code
2. Ask for improvements: "add error handling", "optimize for performance"
3. Repeat until satisfied

### Pattern 3: Context Injection

If agent misses your coding style:

```
# Our codebase uses dataclasses, dependency injection, and async/await
# Add a new User model following this pattern
```

### Pattern 4: Testing-First with AI

1. **Write test with AI help** (test describes intent)
2. **Generate implementation** to pass tests
3. **Refactor** with AI suggestions

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

## MCP Server and more advanced 
extra: https://developer.microsoft.com/blog/announcing-awesome-copilot-mcp-server
extra: https://github.com/github/awesome-copilot/tree/main?tab=readme-ov-file
