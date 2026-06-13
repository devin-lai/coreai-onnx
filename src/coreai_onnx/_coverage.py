# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model coverage analysis - powers `coreai-onnx inspect` and the docs table."""

from dataclasses import dataclass, field

import onnx

from ._lowerings import _onnx_to_core_resolver
from ._utils import iter_graph_nodes, op_key


def supported_ops() -> set[str]:
    """Return the set of all supported ONNX op keys."""
    return set(_onnx_to_core_resolver)


@dataclass
class CoverageReport:
    total_nodes: int = 0
    op_histogram: dict[str, int] = field(default_factory=dict)
    unsupported: dict[str, int] = field(default_factory=dict)

    @property
    def convertible(self) -> bool:
        return not self.unsupported


def _walk_graph(graph: onnx.GraphProto, report: CoverageReport) -> None:
    """Count nodes in *graph* and any nested sub-graphs against the built-ins."""
    resolver = _onnx_to_core_resolver
    for node in iter_graph_nodes(graph):
        report.total_nodes += 1
        key = op_key(node)
        report.op_histogram[key] = report.op_histogram.get(key, 0) + 1
        if key not in resolver:
            report.unsupported[key] = report.unsupported.get(key, 0) + 1


def analyze(model: onnx.ModelProto) -> CoverageReport:
    """Walk *model*'s graph (recursively into sub-graphs) and return a report.

    Note: custom op lowerings registered on an ``OnnxConverter`` instance via
    ``register_onnx_lowering`` are not visible here — ``analyze`` reports
    coverage against the built-in resolver only.
    """
    report = CoverageReport()
    _walk_graph(model.graph, report)
    return report


def coverage_markdown() -> str:
    """Return a Markdown table of all supported lowering keys, sorted by name."""
    lines = [
        "| Op key | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {op} | supported |" for op in sorted(supported_ops()))
    return "\n".join(lines)
