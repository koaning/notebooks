# Marimo Notebooks

## Code style

- Don't alias imports with underscore prefixes in marimo cells (e.g. `import warnings as _warnings` or `from sklearn.linear_model import LogisticRegression as _LR`). Instead, move shared imports to a common imports cell and pass them as cell dependencies.
- Don't use underscore prefixes on variables inside functions — they're already function-scoped and can't leak into marimo's cell graph. Only use underscores on cell-level temporaries that the linter requires.
