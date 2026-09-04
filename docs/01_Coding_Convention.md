# Coding Conventions

Consistent names and documentation make research code easier to review, reuse,
and maintain. Use the conventions below as defaults, then document any
project-specific exceptions.

## Naming Python Code

[PEP 8](https://peps.python.org/pep-0008/) is the Python style guide. Follow its
naming guidance unless the project has a documented reason to differ.

Function and variable names should use **snake_case** and describe their purpose.

- Separate lowercase words with underscores: use `do_something()`, not
  `dosomething()`, and `training_data`, not `trainingdata`.
- Prefer descriptive names over unclear abbreviations: use `training_split`
  instead of `tsplit`.
- Avoid ambiguous names such as `val`, `data`, or `process` when a more specific
  name is available.
- Name functions for what they do: use `calculate_tax()` instead of `tax()`.

Class names should use **PascalCase**.

- Use `MyClass`, not `my_class` or `myClass`.
- Preserve familiar capitalized acronyms where they improve clarity, such as
  `CUDAError`.

Constants should use uppercase words separated by underscores, such as
`MAX_RETRIES`.

Modules and packages should use short, lowercase names, such as `utils` or
`analysis`.

Use leading or trailing underscores with care:

- A single leading underscore, as in `_variable`, conventionally marks a name
  for internal use.
- Two leading underscores, as in `__variable`, trigger name mangling in a class.
- Names with two leading and trailing underscores, such as `__init__` and
  `__str__`, are reserved for special methods defined by Python.

Here is a table of examples:

| Entity | Convention | Example |
| --- | --- | --- |
| Variable | `snake_case` | `user_id` |
| Function | `snake_case` | `get_user_data` |
| Class | PascalCase | `UserProfile` |
| Constant | `UPPER_CASE_WITH_UNDERSCORES` | `MAX_RETRIES` |
| Module | Lowercase; underscores optional | `data_utils.py` |
| Package | Lowercase | `datapackage` |
| Exception | PascalCase ending in `Error` | `InputValidationError` |

## Documentation and Type Hints

[PEP 257](https://peps.python.org/pep-0257/) defines conventions for Python
docstrings. Follow its guidance unless the project documents a different style.

<!-- TODO: Add practical docstring and type-hint examples. -->
