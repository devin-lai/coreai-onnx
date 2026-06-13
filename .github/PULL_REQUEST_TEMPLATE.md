## Summary

- <!-- Describe the change. -->

## Validation

- [ ] `ruff check .`
- [ ] `python -m compileall src tests`
- [ ] `pytest -m "not apple and not integration"`
- [ ] `python -m build && twine check dist/*`
- [ ] Local Apple Core AI validation completed when conversion/runtime behavior changes

## Checklist

- [ ] Public API, CLI JSON envelope, and exit-code behavior are unchanged or documented.
- [ ] Tests cover new behavior or changed lowerings.
- [ ] Documentation is updated when user-facing behavior changes.
