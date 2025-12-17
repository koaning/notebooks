# Agent Instructions

## Python Version

Prefer Python 3.12 for now. Some packages (like `av` for manim) have compatibility issues with Python 3.14. Use `requires-python = ">=3.12,<3.14"` in script headers.

## Marimo Notebook Validation

After every edit to a `.py` file that is a Marimo notebook, run:

```bash
uvx marimo check <filename>.py
```

This validates that the notebook has no critical issues such as:
- Variables defined in multiple cells (use `_` prefix for private/temporary variables)
- Missing imports
- Invalid cell dependencies

Only proceed once `marimo check` passes with no errors.
