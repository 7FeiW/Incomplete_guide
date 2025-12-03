# Coding Convention

PEP8 is the *official* naming standard for Python code (<https://peps.python.org/pep-0008/>). Follow the guideline, but tweak it for your specific needs.

Function and variable names should follow **snake_case** and clearly describe what the function does
   * Use lowercase with words separated by underscores: use `do_something()`, not `dosomething()`, use `training_data` not `trainingdata`
   * Use longer, more descriptive names over abbreviations, e.g., `get_data()` instead of `data()` or `gd()`, `training_split()` instead of `tsplit`
   * The name should NOT be ambiguous: do not use `val()`, `data()`, `process()`, `data`
   * Function names should describe what the function does: use `calculate_tax()` instead of `tax()`

Class names should follow **PascalCase**
   * Use `MyClass` not `my_class` or `myClass`
   * If class names contain an acronym, use capitalized letters for acronyms, e.g., `CUDAError`

Constants
   * Use all uppercase letters and separate words with underscores, e.g., `MAX_RETRIES`

Modules and Packages
   * Use short, all-lowercase names, e.g., `util` or `math`

Special Names - These are something you should be aware of at least; use leading or trailing `_` with caution
   * Names with a single leading underscore `_variable` indicate "internal use" or "protected" members. If you don't know what private methods are, check out [this guide](https://www.datacamp.com/tutorial/python-private-methods-explained)
   * Names with double leading underscores `__variable` trigger name mangling in classes - Python will rename this to `Class__variable`
   * Names with double leading and trailing underscores `__init__`, `__str__` are reserved for special methods

Here is a table of examples:

|Entity	|Convention	|Example|
| ------- |  ------- |  ------- |
|Variable	|snake_case	|user_id|
|Function	|snake_case	|get_user_data|
|Class	|CapWords (PascalCase)	|UserProfile|
|Constant	|UPPER_CASE_WITH_UNDERSCORES	|MAX_RETRIES|
|Module	|lowercase (underscores optional)	|data_utils.py|
|Package	|lowercase (no underscores)	|datapackage|
|Exception	|CapWords + 'Error'	|InputValidationError|

#### A crash course on documentation and type hint

PEP257 (<https://peps.python.org/pep-0257/>) is the *official* standard for Python documentation. Follow the guideline, but tweak it for your specific needs
   * TODO: add something here
