# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import inspect
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import onnx
from coreai.authoring import AIProgram
from coreai.authoring import Context as _CoreAIAuthoringContext
from onnx import numpy_helper

from ._fusion import fuse_activations, fuse_attention
from ._ir import (
    InsertionPoint,
    Location,
    Module,
    OpResultList,
    Type,
    Value,
    program_from_module,
)
from ._ir import coreai_dialect as coreai
from ._lowerings import _onnx_to_core_resolver
from ._passes import preprocess
from ._type_mapping import narrow_array, tensor_type_from_value_info
from ._utils import iter_graph_nodes, op_key
from .errors import ConversionError, UnsupportedOpError

_RESERVED_DOMAINS = ("ai.onnx", "coreai")


def _warn_if_float16(graph: onnx.GraphProto, entrypoint_name: str) -> None:
    """Warn on float16 inputs/initializers: the Core AI runtime's load support
    for f16 programs is partial (default specialization options can fail with
    'Program load failure'), a known Core AI limitation."""
    f16 = [
        vi.name
        for vi in graph.input
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    ]
    f16 += [
        init.name
        for init in graph.initializer
        if init.data_type == onnx.TensorProto.FLOAT16
    ]
    if f16:
        warnings.warn(
            f"graph '{entrypoint_name}' has float16 inputs/initializers "
            f"({', '.join(f16[:5])}): Core AI runtime support for float16 is "
            "partial and the saved .aimodel may fail to load with default "
            "specialization options (known Core AI limitation)",
            stacklevel=3,
        )


class Context(_CoreAIAuthoringContext):
    """Authoring context that also pushes a default Location (mirrors coreai-torch)."""

    def __init__(self) -> None:
        super().__init__()
        self._location: Location = Location.unknown(self._mlir_context)

    def __enter__(self) -> Self:
        super().__enter__()
        self._location.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._location.__exit__(exc_type, exc_value, traceback)
        super().__exit__(exc_type, exc_value, traceback)


@dataclass
class _StagedEntry:
    model: onnx.ModelProto
    input_names: Sequence[str] | None
    output_names: Sequence[str] | None
    entrypoint_name: str


class OnnxConverter:
    def __init__(self) -> None:
        self.context = Context()

        # user defined ONNX op lowerings (reusable across conversions)
        self._user_defined_lowering: dict[str, Callable[..., Any]] = {}

        # cache: lowering fn -> whether it takes the converter as a 4th arg
        self._lowering_wants_converter: dict[Callable[..., Any], bool] = {}

        # staged models awaiting conversion
        self._staged: list[_StagedEntry] = []
        self._conversion_started = False

    def add_onnx_model(
        self,
        model: onnx.ModelProto | str | Path,
        *,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        entrypoint_name: str = "main",
    ) -> None:
        """Stage an ONNX model (proto or path) for conversion."""
        if isinstance(model, str | Path):
            model = onnx.load(str(model))
        else:
            # The preprocessing/fusion passes mutate the proto in place; the
            # caller's model must stay usable (e.g. for verify() against
            # onnxruntime afterwards).
            copied = onnx.ModelProto()
            copied.CopyFrom(model)
            model = copied
        if any(e.entrypoint_name == entrypoint_name for e in self._staged):
            raise ValueError(f"entrypoint '{entrypoint_name}' is already staged")
        model = preprocess(model)
        model = fuse_attention(model)
        model = fuse_activations(model)
        _warn_if_float16(model.graph, entrypoint_name)
        self._staged.append(
            _StagedEntry(model, input_names, output_names, entrypoint_name)
        )

    def register_onnx_lowering(
        self, qualified_name: str, *, allow_override: bool = False
    ) -> Callable:
        """Register a custom ONNX node lowering with this converter.

        Used as a decorator. The decorated function receives
        ``(values_map, node, loc)`` and must return a Core AI ``Value`` or a
        list of ``Value``s. If it declares a 4th positional parameter that is
        literally named ``converter``, this ``OnnxConverter`` is passed there
        (control-flow lowerings use it to lower subgraphs recursively); a 4th
        parameter with any other name is never bound by the converter.
        ``qualified_name`` is a bare op type for the default domain (e.g.
        ``"Det"``) or ``"domain::OpType"`` otherwise. The ``ai.onnx`` and
        ``coreai`` domains are reserved. Set ``allow_override=True`` to
        replace an existing lowering.
        """
        if "::" in qualified_name:
            domain = qualified_name.split("::", 1)[0]
            if domain in _RESERVED_DOMAINS:
                raise ValueError(
                    f"Cannot register lowering for '{qualified_name}': "
                    f"domain '{domain}' is reserved."
                )

        if not allow_override and (
            qualified_name in _onnx_to_core_resolver
            or qualified_name in self._user_defined_lowering
        ):
            raise ValueError(
                f"a lowering for '{qualified_name}' already exists; "
                "pass allow_override=True to replace it"
            )

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._user_defined_lowering[qualified_name] = fn
            return fn

        return decorator

    def _check_coverage(self, graph: onnx.GraphProto) -> None:
        """Aggregate every op in ``graph`` (including subgraphs) with no lowering.

        Unlike ``_coverage.analyze``, this also honors instance-registered
        custom lowerings, so a model is gated against what *this* converter can
        actually lower.
        """
        missing: dict[str, list[str]] = {}
        for i, node in enumerate(iter_graph_nodes(graph)):
            key = op_key(node)
            if (
                key not in _onnx_to_core_resolver
                and key not in self._user_defined_lowering
            ):
                missing.setdefault(key, []).append(node.name or f"node_{i}")
        if missing:
            raise UnsupportedOpError(missing)

    def to_coreai(self) -> AIProgram:
        """Convert all staged ONNX models to a Core AI AIProgram."""
        if not self._staged:
            raise RuntimeError("No models to convert. Call add_onnx_model() first.")
        if self._conversion_started:
            raise RuntimeError(
                "OnnxConverter.to_coreai() can only be called once; "
                "create a new OnnxConverter for another conversion."
            )

        # Coverage-gate every staged graph before emitting anything.
        for entry in self._staged:
            self._check_coverage(entry.model.graph)
        self._conversion_started = True

        with self.context:
            module: Module = Module.create()
            with InsertionPoint(module.body):
                for entry in self._staged:
                    self._emit_graph(entry)
        return program_from_module(module)

    def _emit_graph(self, entry: _StagedEntry) -> None:
        graph = entry.model.graph
        loc = Location.current if Location.current else Location.unknown()

        initializer_names = {init.name for init in graph.initializer}
        graph_inputs = [vi for vi in graph.input if vi.name not in initializer_names]

        if entry.input_names is not None:
            if len(entry.input_names) != len(graph_inputs):
                raise ValueError(
                    f"input_names has {len(entry.input_names)} name(s) but the "
                    f"graph has {len(graph_inputs)} input(s)"
                )
            input_names = list(entry.input_names)
        else:
            input_names = [vi.name for vi in graph_inputs]

        if entry.output_names is not None:
            if len(entry.output_names) != len(graph.output):
                raise ValueError(
                    f"output_names has {len(entry.output_names)} name(s) but the "
                    f"graph has {len(graph.output)} output(s)"
                )
            output_names = list(entry.output_names)
        else:
            output_names = [o.name for o in graph.output]

        input_types: list[Type] = []
        for vi in graph_inputs:
            try:
                input_types.append(tensor_type_from_value_info(vi))
            except ValueError as exc:
                raise ConversionError(vi.name, "(graph input)", exc) from exc

        graph_op: coreai.GraphOp = coreai.GraphOp(
            name=entry.entrypoint_name,
            input_types=input_types,
            result_types=[],
            input_names=input_names,
            loc=loc,
        )

        values_map: dict[str, Value] = {}
        for i, vi in enumerate(graph_inputs):
            values_map[vi.name] = graph_op.arguments[i]

        with graph_op.block:
            self._lower_graph_body(graph, values_map, loc)

        # Validate graph outputs before building the OrderedDict.
        seen_resolved: set[str] = set()
        for out, resolved_name in zip(graph.output, output_names, strict=True):
            if out.name not in values_map:
                cause = ValueError(
                    f"graph output '{out.name}' (resolved as '{resolved_name}') "
                    "was not produced by any node"
                )
                raise ConversionError(
                    entry.entrypoint_name,
                    "(graph output)",
                    cause,
                ) from cause
            if resolved_name in seen_resolved:
                raise ValueError(
                    f"duplicate resolved output name '{resolved_name}' in graph "
                    f"'{entry.entrypoint_name}'"
                )
            seen_resolved.add(resolved_name)

        graph_op.set_outputs_spec_from_dict(
            OrderedDict(
                (resolved_name, values_map[out.name])
                for out, resolved_name in zip(graph.output, output_names, strict=True)
            )
        )

    def _lower_graph_body(
        self, graph: onnx.GraphProto, values_map: dict[str, Value], loc: Location
    ) -> None:
        """Emit initializers and lower all nodes of an ONNX (sub)graph into the current block."""
        for init in graph.initializer:
            try:
                arr = narrow_array(
                    numpy_helper.to_array(init),
                    context=f"initializer '{init.name}'",
                )
            except OverflowError as exc:
                raise ConversionError(init.name, "(initializer)", exc) from exc
            values_map[init.name] = coreai.constant(arr, loc=loc)
        for node in graph.node:
            self._lower_node(values_map, node, loc)

    def _wants_converter(self, fn: Callable[..., Any]) -> bool:
        """True if *fn*'s 4th positional parameter is literally named ``converter``.

        Most lowerings are pure ``(values_map, node, loc)`` functions; control
        flow lowerings declare a 4th parameter named ``converter`` so they can
        lower branch subgraphs recursively. The name is the contract: a 4th
        parameter with any other name (e.g. a defaulted tuning knob on a user
        lowering) is never silently bound to the converter. Cached per callable.
        """
        cached = self._lowering_wants_converter.get(fn)
        if cached is None:
            try:
                params = inspect.signature(fn).parameters.values()
            except (TypeError, ValueError):
                cached = False
            else:
                positional = [
                    p
                    for p in params
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                cached = len(positional) >= 4 and positional[3].name == "converter"
            self._lowering_wants_converter[fn] = cached
        return cached

    def _lower_node(
        self, values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> None:
        key = op_key(node)
        fn = self._user_defined_lowering.get(key)
        if fn is None:
            fn = _onnx_to_core_resolver[key]

        try:
            if self._wants_converter(fn):
                results = fn(values_map, node, loc, self)
            else:
                results = fn(values_map, node, loc)
        except Exception as exc:
            raise ConversionError(node.name or node.op_type, key, exc) from exc

        if isinstance(results, Value):
            results = [results]
        elif isinstance(results, OpResultList):
            # The upstream stub omits OpResultList's Sequence protocol
            # (__iter__/__len__); the runtime binding is iterable.
            results = list(cast(Iterable[Value], results))

        required = [o for o in node.output if o]
        if len(results) != len(required):
            cause = ValueError(
                f"lowering returned {len(results)} result(s) but node has "
                f"{len(required)} outputs (op {key})"
            )
            raise ConversionError(
                node.name or node.op_type,
                key,
                cause,
            ) from cause

        # Bind the i-th result to the i-th requested (non-empty) output. The
        # contract is "one result per non-empty output, in order" (every
        # built-in multi-output lowering follows it); zipping against the raw
        # node.output instead would skip a result on each "" placeholder and
        # mis-bind any output that follows an omitted optional one.
        values_map.update(zip(required, results, strict=True))
