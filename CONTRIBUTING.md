# Contributing to coreai-onnx

## Dev setup

```bash
python -m pip install -e ".[test,dev]"
pre-commit install
```

## Running tests

```bash
pytest -m "not slow"       # all fast tests
pytest -m ops              # operator parity tests only
pytest -n auto -m ir       # MLIR-output tests, safe to parallelize
pytest -m slow             # slow / large-model tests
pytest -m e2e              # end-to-end model tests
pytest -m "not slow" --cov=coreai_onnx --cov-report=term-missing
```

Keep Core AI runtime-executing tests serial. The native runtime uses
process-external specialization/cache state and can fail spuriously under
multi-process xdist loads; plain `pytest` is the source of truth.

**Markers:**

| Marker | Meaning |
|--------|---------|
| `ops`  | Core operator parity tests — compare ONNX runtime vs. Core AI output |
| `ir`   | MLIR-output tests; do not require the Core AI runtime |
| `slow` | Long-running tests; skipped by default in CI on PRs |
| `e2e`  | End-to-end model conversion and execution tests |

**Runtime requirement:** op parity and e2e tests require the Core AI runtime, which
is only available on **macOS 27+**. Tests that need the runtime are guarded by the
`requires_coreai_runtime` marker/fixture in `tests/helpers.py`. They will be skipped
automatically on unsupported platforms.

## Adding an op lowering

1. **Write a failing parity test** in the appropriate `tests/test_ops_*.py`
   file using the `single_op_model` helper and `assert_parity` from
   `tests/helpers.py`. Use `zlib.crc32`-seeded inputs for reproducibility:

   ```python
   import zlib

   import numpy as np
   import pytest

   from .helpers import assert_parity, requires_coreai_runtime, single_op_model

   pytestmark = [pytest.mark.ops, requires_coreai_runtime]


   async def test_my_op_basic():
       rng = np.random.default_rng(zlib.crc32(b"my_op_basic"))
       x = rng.random((2, 3)).astype(np.float32)
       model = single_op_model("MyOp", {"x": x})
       await assert_parity(model, {"x": x})
   ```

2. **Port or adapt the lowering** from `coreai-torch`'s `_aten_to_core.py`
   patterns where applicable. Lowerings live under `src/coreai_onnx/_lowerings/`.
   Each lowering is a function

   ```python
   (values_map: dict[str, Value], node: onnx.NodeProto, loc: Location) -> Value | list[Value]
   ```

   that reads its inputs from `values_map`, emits Core AI ops, and returns the
   result `Value` (or a `list[Value]`, one entry per non-empty output slot, for
   multi-output nodes).

3. **Register the lowering** in the appropriate
   `src/coreai_onnx/_lowerings/_*.py` module's `REGISTRY` dict:

   ```python
   REGISTRY["MyOp"] = lower_my_op
   ```

4. **Unsupported attribute combinations** must raise `ValueError` with a clear
   message. Add a corresponding test:

   ```python
   def test_my_op_unsupported_attr():
       with pytest.raises(ConversionError, match="MyOp: attr X is not supported"):
           convert(single_op_model("MyOp", {"x": x}, attrs={"X": 99}))
   ```

5. **Before opening a PR**, run:

   ```bash
   ruff check . && ruff format --check .
   mypy --ignore-missing-imports src/coreai_onnx
   pytest -m "not slow"
   pytest -m "not slow" --cov=coreai_onnx --cov-report=term-missing
   python -m build --sdist --wheel && twine check dist/*
   ```

### Error handling in lowerings

Raise plain `ValueError` with a precise message for anything the lowering
cannot handle (unsupported attribute combination, non-static dim, ...).
The converter wraps it into `ConversionError` with the node name and op
key attached — lowerings never raise `ConversionError` themselves. This
split is deliberate; don't "fix" it.

## PR checklist

- [ ] Failing test added before the fix / feature
- [ ] Lowering registered in `REGISTRY`
- [ ] Unsupported combos raise `ValueError` / `ConversionError` with clear message
- [ ] `ruff check . && ruff format --check .` passes
- [ ] `mypy --ignore-missing-imports src/coreai_onnx` passes
- [ ] `pytest -m "not slow"` passes
- [ ] `pytest -m "not slow" --cov=coreai_onnx --cov-report=term-missing` passes
- [ ] `python -m build --sdist --wheel && twine check dist/*` passes

## Known runtime quirks

These are confirmed bugs / limitations in the Core AI runtime as of the initial
release. Work around them in tests with `pytest.mark.xfail` or `pytest.mark.skip`
and leave a comment referencing this list.

- **Bool graph inputs crash logical primitives.** Passing a boolean tensor as a
  top-level graph input (rather than casting inside the graph) crashes certain
  logical ops. Workaround: cast to int32 at the graph boundary.

- **f16 graph-input crash.** Float16 tensors as graph inputs trigger a runtime
  crash. Cast to float32 at the boundary and cast back as needed.

- **Multi-output `If` execution hang.** An `If` node whose branches return more
  than one output value hangs indefinitely at runtime. Single-output `If` works.

- **Xcode Performance runner static-concat abort.** The Xcode model performance
  service can abort inside MPSGraph's GPU `ConcatOpHandler` for common static
  channel-concat image models. Keep static ONNX `Concat` lowered through the
  pad+add/pad+where workaround unless the OS compiler bug is confirmed fixed.

- **Reduce ops ignore `keepdims=0`.** Reduce operators (ReduceSum, ReduceMean,
  etc.) always keep the reduced dimension regardless of the `keepdims` attribute.
  Squeeze the output manually as a workaround.

- **`floor_divide` truncates on integers.** Integer floor-division truncates
  toward zero (C semantics) rather than flooring toward negative infinity
  (Python / NumPy semantics). For negative operands the results will differ.
