# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Automatic, known-safe conversion repairs.

``apply_repairs`` runs every applicable strategy (see ``_strategies``) over a
copy of the model and reports what it changed. Repairs are semantics-preserving
by construction; callers re-verify parity against ONNX Runtime before trusting
a repaired model, so the engine never produces a silently-wrong result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import onnx

from ._strategies import STRATEGIES, RepairStrategy

__all__ = ["STRATEGIES", "RepairRecord", "RepairStrategy", "apply_repairs"]


@dataclass(frozen=True)
class RepairRecord:
    """One applied repair: which strategy ran and what it touched."""

    name: str
    summary: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "details": self.details}


def apply_repairs(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, list[RepairRecord]]:
    """Apply every applicable known-safe repair to a copy of *model*.

    Returns the repaired model and one ``RepairRecord`` per applied strategy
    (empty if nothing applied). The input proto is never mutated.
    """
    repaired = onnx.ModelProto()
    repaired.CopyFrom(model)
    records: list[RepairRecord] = []
    for strategy in STRATEGIES:
        triggered = strategy.detect(repaired)
        if triggered:
            strategy.apply(repaired)
            records.append(
                RepairRecord(
                    strategy.name, strategy.summary, {"inputs": sorted(triggered)}
                )
            )
    return repaired, records
