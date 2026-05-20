# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchDispatchMode-based interceptor that records every tensor operation
dispatched through PyTorch's dispatcher.

Usage::

    recorder = OpRecorder()
    with capture_ops(recorder, phase="forward"):
        output = model(inputs)

After the context exits, ``recorder.nodes`` contains an :class:`OpNode` for
every operator that was dispatched.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from .nodes import OpNode, TensorMeta

# ---------------------------------------------------------------------------
# Op categorisation
# ---------------------------------------------------------------------------

# Prefixes / substrings that identify collective communication ops
_COMM_COLLECTIVE_MARKERS = (
    "_c10d_functional",
    "c10d_functional",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast_",
    "wait_tensor",
    "barrier",
)

# Substrings that identify point-to-point comm ops
_COMM_P2P_MARKERS = ("_send", "_recv", ".send", ".recv")

# Substrings that identify data-movement ops (device copy, dtype convert)
_DATA_MOVE_MARKERS = ("_to_copy", "copy_", ".to.")

# Allocation ops
_MEMORY_MARKERS = (
    "aten.empty",
    "aten.zeros",
    "aten.ones",
    "aten.full",
    "aten.arange",
    "aten.rand",
)

# Ops that are too trivial/low-signal to record individually
_SKIP_SUFFIXES = (
    "aten.detach.default",
    "aten.detach_.default",
    "aten.alias.default",
    "aten.t.default",
    "aten.as_strided.default",
    "aten._unsafe_view.default",
    "aten.view.default",
    "aten.lift_fresh_copy.default",
    "aten.lift.default",
)

# Map from substring → canonical comm_op name
_COMM_OP_MAP: list[tuple[str, str]] = [
    ("reduce_scatter", "reduce_scatter"),
    ("all_gather", "all_gather"),
    ("all_reduce", "all_reduce"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("wait_tensor", "wait"),
    ("barrier", "barrier"),
    ("_send", "send"),
    ("_recv", "recv"),
]


def _categorize_op(func_name: str) -> tuple[str, str | None]:
    """Return ``(op_type, comm_op)`` for a given function name string."""
    if any(m in func_name for m in _COMM_P2P_MARKERS):
        for substr, canonical in _COMM_OP_MAP:
            if substr in func_name:
                return "comm_p2p", canonical
        return "comm_p2p", "p2p_unknown"

    if any(m in func_name for m in _COMM_COLLECTIVE_MARKERS):
        for substr, canonical in _COMM_OP_MAP:
            if substr in func_name:
                return "comm_collective", canonical
        return "comm_collective", "collective_unknown"

    if any(m in func_name for m in _DATA_MOVE_MARKERS):
        return "data_move", None

    if any(func_name.startswith(m) or m in func_name for m in _MEMORY_MARKERS):
        return "memory", None

    return "compute", None


def _is_trivial(func_name: str) -> bool:
    return any(func_name.endswith(skip) for skip in _SKIP_SUFFIXES)


def _collect_tensor_metas(args: Any, kwargs: Any) -> list[TensorMeta]:
    flat, _ = pytree.tree_flatten((args, kwargs))
    metas: list[TensorMeta] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            try:
                metas.append(TensorMeta.from_tensor(item))
            except Exception:
                pass
    return metas


def _collect_input_tensors(args: Any, kwargs: Any) -> list[torch.Tensor]:
    flat, _ = pytree.tree_flatten((args, kwargs))
    return [item for item in flat if isinstance(item, torch.Tensor)]


def _collect_output_tensors(output: Any) -> list[torch.Tensor]:
    flat, _ = pytree.tree_flatten(output)
    return [item for item in flat if isinstance(item, torch.Tensor)]


def _collect_output_metas(output: Any) -> list[TensorMeta]:
    flat, _ = pytree.tree_flatten(output)
    metas: list[TensorMeta] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            try:
                metas.append(TensorMeta.from_tensor(item))
            except Exception:
                pass
    return metas


# ---------------------------------------------------------------------------
# Recorder (thread-safe)
# ---------------------------------------------------------------------------


class OpRecorder:
    """Thread-safe container that accumulates :class:`OpNode` entries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self.nodes: list[OpNode] = []
        self.edges: list[tuple[str, str, str]] = []
        self._tensor_producer: dict[int, str] = {}
        # Mutable phase/context fields; callers update these between sections
        self.current_phase: str = "forward"
        self.current_pp_stage: int | None = None
        self.current_microbatch: int | None = None

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"op_{self._counter:07d}"

    def record(
        self,
        func: Any,
        input_metas: list[TensorMeta],
        output_metas: list[TensorMeta],
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        attrs: dict[str, Any] | None = None,
    ) -> OpNode:
        func_name = str(func)
        op_type, comm_op = _categorize_op(func_name)

        node = OpNode(
            node_id=self._next_id(),
            op_name=func_name,
            op_type=op_type,
            phase=self.current_phase,
            inputs=input_metas,
            outputs=output_metas,
            attrs=attrs or {},
            pp_stage=self.current_pp_stage,
            microbatch_idx=self.current_microbatch,
            comm_op=comm_op,
        )
        with self._lock:
            input_producers = set()
            for t in input_tensors:
                producer = self._tensor_producer.get(id(t))
                if producer is not None:
                    input_producers.add(producer)
            for producer in sorted(input_producers):
                self.edges.append((producer, node.node_id, "data"))

            self.nodes.append(node)
            for t in output_tensors:
                self._tensor_producer[id(t)] = node.node_id
        return node

    def get_producer(self, tensor: torch.Tensor | None) -> str | None:
        if tensor is None:
            return None
        with self._lock:
            return self._tensor_producer.get(id(tensor))

    def set_producer(self, tensor: torch.Tensor | None, node_id: str) -> None:
        if tensor is None:
            return
        with self._lock:
            self._tensor_producer[id(tensor)] = node_id


# ---------------------------------------------------------------------------
# TorchDispatchMode
# ---------------------------------------------------------------------------


class OpCaptureMode(TorchDispatchMode):
    """
    Intercepts every tensor operation and records it in the given
    :class:`OpRecorder`.

    This works in both eager and FakeTensor modes.
    """

    def __init__(self, recorder: OpRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        func_name = str(func)

        # Execute first so output shapes are available
        result = func(*args, **kwargs)

        if _is_trivial(func_name):
            return result

        input_metas = _collect_tensor_metas(args, kwargs)
        output_metas = _collect_output_metas(result)
        input_tensors = _collect_input_tensors(args, kwargs)
        output_tensors = _collect_output_tensors(result)

        # Collect non-tensor scalar attrs for context
        attrs: dict[str, Any] = {}
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float, bool, str)):
                attrs[f"arg_{i}"] = arg

        self.recorder.record(
            func,
            input_metas,
            output_metas,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
        )
        return result


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------

_RECORDER_STACK: list[OpRecorder] = []


def get_current_recorder() -> OpRecorder | None:
    """Return the innermost active OpRecorder, or None."""
    return _RECORDER_STACK[-1] if _RECORDER_STACK else None


@contextmanager
def capture_ops(
    recorder: OpRecorder,
    phase: str = "forward",
) -> Generator[OpRecorder, None, None]:
    """
    Context manager that activates :class:`OpCaptureMode` and sets the
    initial phase on the recorder.

    Args:
        recorder: Target recorder to write into.
        phase: Initial phase annotation (``"forward"``, ``"backward"``,
            ``"optimizer"``).

    Yields:
        The same ``recorder`` instance for convenience.
    """
    recorder.current_phase = phase
    _RECORDER_STACK.append(recorder)
    with OpCaptureMode(recorder):
        yield recorder
    _RECORDER_STACK.pop()
