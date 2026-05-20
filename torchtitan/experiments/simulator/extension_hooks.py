# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any


def collect_extension_metadata(trainer: Any, capture: Any) -> dict[str, Any]:
    """
    Optional extension hook for simulator side-loads.

    Trainers can implement ``collect_simulation_metadata(capture)`` and return
    JSON-serializable metadata. The root simulator deliberately uses duck typing
    here so experiment packages such as torchtitan-npu can add annotations
    without creating an import dependency from core TorchTitan simulator code.
    """
    collector = getattr(trainer, "collect_simulation_metadata", None)
    if not callable(collector):
        return {}
    metadata = collector(capture)
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(
            "collect_simulation_metadata(capture) must return a dict or None, "
            f"got {type(metadata).__name__}"
        )
    return metadata


def postprocess_extension_result(result: Any, trainer: Any, sim_opts: Any) -> Any:
    """
    Optional extension hook for mutating a SimulationResult before export.

    ``postprocess_simulation_result(result, sim_opts)`` may mutate ``result`` in
    place and return ``None`` or return the result object to export.
    """
    postprocess = getattr(trainer, "postprocess_simulation_result", None)
    if not callable(postprocess):
        return result
    processed = postprocess(result, sim_opts)
    return result if processed is None else processed
