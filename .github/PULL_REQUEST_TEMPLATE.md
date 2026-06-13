## Summary

- <!-- Describe the change. -->

## Validation

- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy --ignore-missing-imports src/coreai_onnx`
- [ ] `pytest -m "not slow"`
- [ ] `pytest -m "not slow" --cov=coreai_onnx --cov-report=term-missing`
- [ ] `python -m build --sdist --wheel && twine check dist/*`

## Checklist

- [ ] Public API, CLI JSON envelope, and exit-code behavior are unchanged or documented.
- [ ] Tests cover new behavior or changed lowerings.
- [ ] Documentation is updated when user-facing behavior changes.
