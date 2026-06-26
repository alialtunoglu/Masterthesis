# Clean Code Rules

Write clean, readable, testable Python code.

General rules:

1. Use clear and descriptive names.
2. Avoid large functions. Prefer small functions with one responsibility.
3. Avoid duplicated code. Extract reusable logic into utilities.
4. Use type hints for function arguments and return values.
5. Use pathlib.Path instead of raw string path manipulation.
6. Prefer explicit parameters over hidden global state.
7. Avoid magic numbers. Put constants at the top of the file or in config files.
8. Use docstrings for public functions and classes.
9. Add comments only when they explain why something is done, not obvious what.
10. Fail fast with clear error messages.

Python style:

1. Follow PEP8.
2. Use snake_case for functions and variables.
3. Use PascalCase for classes.
4. Use UPPER_CASE for constants.
5. Keep imports ordered:
   - standard library
   - third-party packages
   - local project imports

Error handling:

1. Validate paths before using them.
2. If a file is missing, raise FileNotFoundError with a helpful message.
3. If a dataset format is invalid, raise ValueError with a clear explanation.
4. Do not silently ignore errors.

Do not write notebook-style messy code inside project scripts.
