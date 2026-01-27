# GitHub Copilot Instructions

## Project Overview

This repository contains a Python 3.11 web API for ecommerce order management.
- Framework: FastAPI
- Database: PostgreSQL via SQLAlchemy
- Testing: pytest with factory-based fixtures
- Async: All request handlers are async

Copilot should prioritize readability, testability, and explicit error handling.

## How to Run & Test

- Install: `pip install -r requirements.txt`
- Run app: `uvicorn app.main:app --reload`
- Run all tests: `pytest -q`
- Run type checks: `mypy app`
- Run lint: `ruff check app tests`

Always ensure tests and type checks pass before considering a change complete.

## Project Structure

- `app/main.py`: FastAPI app and startup configuration
- `app/api/`: Route handlers (organized by domain)
- `app/models/`: SQLAlchemy models
- `app/schemas/`: Pydantic models (request/response)
- `app/services/`: Business logic and integrations
- `tests/`: Unit and integration tests

When adding new features:
- Put HTTP handlers in `app/api/<domain>.py`
- Put business logic in `app/services/`
- Put persistence logic in `app/models/` or dedicated repositories

## Coding Standards

- Use type hints everywhere; code must pass `mypy` in strict mode.
- Use snake_case for functions/variables, PascalCase for classes.
- Prefer dependency injection (FastAPI `Depends`) over global state.
- Do not use raw SQL; always go through SQLAlchemy models/sessions.
- Write small, focused functions; avoid deeply nested logic.
- For errors, raise domain-specific exceptions and map them to HTTP errors in one place.

## Testing Expectations

- Every new endpoint or non-trivial behavior requires tests in `tests/`.
- Use factories/fixtures instead of hand-rolled test data.
- Write at least:
  - Happy-path tests.
  - Validation / error-path tests.
  - Edge-case tests when relevant (empty values, limits, concurrency).

## Important Constraints

- Do not change existing public API paths or response shapes without explicit instructions.
- Do not change database schemas without adding Alembic migrations in `migrations/`.
- Do not introduce new external dependencies unless explicitly requested.
- Authentication and authorization logic must not be weakened or bypassed.

## How Copilot Should Work

- Before editing, inspect surrounding code to match existing patterns.
- Prefer extending existing patterns over inventing new ones.
- When adding code:
  - Update or add tests.
  - Keep log messages structured and minimal (no sensitive data).
- If multiple approaches are possible, choose the simplest one that is consistent with existing code.
