# Using GitHub Copilot for Planning & Agentic Development

This document covers practical workflows for using **GitHub Copilot** as an AI agent for software development. It focuses on planning strategies, agentic workflows, and how to guide agents to understand your codebase and complete complex tasks autonomously.

---

## What is Copilot?

### GitHub Copilot & Agentic Mode

- **What it is**: AI-powered development agent that can understand codebases, plan changes, and implement features autonomously.
- **Primary use**: understanding complex requirements, planning multi-file changes, generating pull requests, code review
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

## Planning with Copilot

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

This is a Python REST API for ecommerce order management.
- **Language**: Python 3.11
- **Framework**: Flask 2.x with SQLAlchemy ORM
- **Database**: PostgreSQL 15
- **Testing**: pytest with factories
- **Key patterns**: Blueprint routing, dependency injection with Flask extensions
```

**2. Exact Build & Test Commands** (agents must validate their work)
```markdown
## Setup & Validation

### Install
1. `python -m venv venv`
2. `source venv/bin/activate`  (Unix) or `venv\Scripts\activate.ps1` (Windows)
3. `pip install -r requirements.txt`
4. `pip install -r requirements-dev.txt`

### Test
- **Run all tests**: `pytest tests/ -v`
- **Run specific test file**: `pytest tests/test_auth.py -v`
- **Run with coverage**: `pytest --cov=src tests/`

### Lint
- `flake8 src/ tests/`
- `mypy src/ --strict`

### Start Dev Server
- `python -m src.app`
- Server runs on http://localhost:5000
```

**3. Project Layout** (agents must know where to put files)
```markdown
## Project Structure

```
src/
  ├── app.py              # Flask app factory
  ├── models/             # SQLAlchemy models
  │   ├── user.py
  │   ├── order.py
  │   └── __init__.py
  ├── routes/             # Flask Blueprints
  │   ├── auth.py         # Authentication endpoints
  │   ├── orders.py       # Order CRUD
  │   └── __init__.py
  ├── schemas/            # Pydantic validation
  │   ├── auth.py
  │   └── order.py
  └── utils/              # Shared helpers
      ├── jwt.py          # JWT token utilities
      ├── decorators.py   # Auth decorators
      └── db.py           # DB helpers

tests/
  ├── conftest.py         # pytest fixtures (database, factories)
  ├── test_auth.py
  ├── test_orders.py
  └── factories/          # Factory Boy fixtures
      ├── user_factory.py
      └── order_factory.py

.github/
  └── copilot-instructions.md
```

**4. Coding Conventions** (agents must follow your style)
```markdown
## Coding Standards

- **Type hints**: All functions must have type hints (strict mypy)
- **Naming**: Snake case for functions/vars, PascalCase for classes
- **Error handling**: Use custom exceptions from `src.exceptions`
- **Database**: Always use ORM; never raw SQL
- **Tests**: 1 test per behavior; use factories for setup
- **Async**: Never used; code is synchronous throughout
```

**5. Critical Constraints** (prevents agents from making breaking changes)
```markdown
## Important Constraints

⚠️ **Never modify**:
- The database schema without running `alembic revision`
- The response format of existing endpoints (backward compatibility)
- The JWT token structure (mobile apps depend on it)

✓ **Always do**:
- Add migrations when touching models
- Add tests for new endpoints
- Update API documentation in docs/api.md
- Run full test suite before submitting PR
```

---

## Agentic Workflow: Step-by-Step

### 1. Set Up Custom Instructions (Foundation)

Ensure `.github/copilot-instructions.md` exists and is comprehensive. Agents use this as their knowledge base.

### 2. Open Copilot Agent

Navigate to [github.com/copilot/agents](https://github.com/copilot/agents) to access Copilot Coding Agent.

### 3. Write a Clear Request

```
Task: Add email verification for new user signups

Requirements:
1. Generate a 6-digit OTP when user signs up
2. Send OTP via email using SendGrid
3. Create a /auth/verify-email endpoint that validates OTP
4. Mark user.email_verified = true after successful verification
5. Expire OTP after 15 minutes or 3 failed attempts
6. Add comprehensive unit tests

Success criteria:
- New user cannot call authenticated endpoints until email verified
- Resend OTP endpoint exists at /auth/resend-otp
- All tests pass locally

Reference pattern: See how password reset works in routes/auth.py
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
"Add a /users/{id}/preferences endpoint similar to /users/{id}/settings 
but with different fields. Follow the same pattern used in routes/settings.py"
```

### Tip 2: Break Large Tasks Into Steps

Agents work better with decomposed tasks:

```
❌ Too big:
"Refactor the payment system to support Stripe and PayPal"

✓ Better:
"Add PayPal support to the payment system:
1. Create new models/paypal_transaction.py
2. Add PayPal credential validation in services/paypal_service.py
3. Update /payments/create to accept paypal_method parameter
4. Add tests in tests/test_paypal_integration.py
Success: agent can process PayPal webhooks and store transactions"
```

### Tip 3: Specify Test Expectations

Agents test their work; tell them what to verify:

```
Verification:
- Test that existing users cannot be created twice with same email
- Test that JWT middleware rejects invalid tokens
- Test that admin endpoints return 403 for non-admin users
- Load test: 1000 concurrent requests to /health should not crash
```

### Tip 4: Point Out Edge Cases

```
Edge cases to handle:
- Empty strings should be treated as null
- Concurrent requests to /order/checkout should not double-charge
- If Redis is down, fall back to in-memory cache
- Unicode usernames should be supported
```

---