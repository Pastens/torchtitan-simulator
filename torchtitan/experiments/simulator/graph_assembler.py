# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph assembler: builds :class:`ComputeGraph` instances from captured data.

Two construction paths:

1. **FX path** (``GraphAssembler.from_fx``): Converts an ``fx.GraphModule``
   (produced by ``make_fx``) into a ``ComputeGraph``.  Delegates to
   :func:`fx_graph_to_compute_graph`.

2. **Runtime path** (``GraphAssembler.from_runtime``): Converts the list of
   :class:`OpNode` objects collected by :class:`OpRecorder` (dispatch
   interception) into a ``ComputeGraph``.  Edges are sequential
   (each op depends on the immediately preceding op in the same phase).

3. **Comm merge** (``GraphAssembler.merge_comm_events``): Inserts ``OpNode``
   entries for every collective / P2P captured by :class:`CommRecorder` so
   that the graph contains communication nodes even when they are not
   visible in the dispatch trace (functional collectives bypass it).
"""

from __future__ import annotations

from typing import Any

import torch.fx as fx

from .fx_capture import fx_graph_to_compute_graph
from .nodes import ComputeGraph, DataEdge, OpNode, TensorMeta


class GraphAssembler:
    """Utility class with static factory methods for :class:`ComputeGraph`."""

    # ------------------------------------------------------------------
    # FX path
    # ------------------------------------------------------------------

    @staticmethod
    def from_fx(
        gm: fx.GraphModule,
        phase: str = "forward",
        metadata: dict[str, Any] | None = None,
    ) -> ComputeGraph:
        """
        Build a :class:`ComputeGraph` from an ``fx.GraphModule``.

        Trivial ops (detach, alias, view, …) are filtered out.  Edges reflect
        true data-flow dependencies taken from FX node arguments.

        Args:
            gm: An FX graph module produced by ``make_fx`` or
                ``torch.export.export``.
            phase: Phase label to stamp on every node (``"forward"``,
                ``"backward"``, ``"joint"``).
            metadata: Optional extra metadata dict.

        Returns:
            A populated :class:`ComputeGraph`.
        """
        return fx_graph_to_compute_graph(gm, phase=phase, metadata=metadata)

    # ------------------------------------------------------------------
    # Runtime path
    # ------------------------------------------------------------------

    @staticmethod
    def from_runtime(
        nodes: list[OpNode],
        edges: list[tuple[str, str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComputeGraph:
        """
        Build a :class:`ComputeGraph` from a list of :class:`OpNode` objects
        captured by :class:`OpRecorder`.

        Within the same ``(phase, pp_stage, microbatch_idx)`` group, edges are
        sequential (each op depends on the previous one).  Cross-group edges
        are not inferred — those come from the schedule assembler instead.

        Args:
            nodes: Ordered list of :class:`OpNode` objects as produced by
                :class:`dispatch_interceptor.OpRecorder`.
            metadata: Optional extra metadata dict.

        Returns:
            A populated :class:`ComputeGraph`.
        """
        graph = ComputeGraph(metadata=metadata or {})
        for n in nodes:
            graph.add_node(n)

        if edges is not None:
            for src, dst, edge_type in edges:
                graph.add_edge(DataEdge(src_node_id=src, dst_node_id=dst, edge_type=edge_type))
        else:
            # Backward-compatible fallback: sequential edges
            GroupKey = tuple  # (phase, pp_stage, microbatch_idx)
            last_in_group: dict[GroupKey, str] = {}

            for n in nodes:
                key = GroupKey((n.phase, n.pp_stage, n.microbatch_idx))
                prev_id = last_in_group.get(key)
                if prev_id is not None:
                    graph.add_edge(
                        DataEdge(
                            src_node_id=prev_id,
                            dst_node_id=n.node_id,
                            edge_type="sequential",
                        )
                    )
                last_in_group[key] = n.node_id

        return graph

    # ------------------------------------------------------------------
    # Comm event merge
    # ------------------------------------------------------------------

    @staticmethod
    def merge_comm_events(
        graph: ComputeGraph,
        comm_events: list[dict[str, Any]],
        phase_override: str | None = None,
    ) -> ComputeGraph:
        """
        Insert :class:`OpNode` entries for distributed comm events into an
        existing graph.

        Comm nodes are labelled with ``op_type="comm_collective"`` or
        ``op_type="comm_p2p"`` and annotated with metadata from the event
        dict (``group``, ``tag``, ``src_rank``, ``dst_rank``, etc.).

        If a matching compute node with the same ``(phase, pp_stage,
        microbatch_idx)`` already exists, a sequential edge is added from the
        last node in that group to the comm node.

        Args:
            graph: The graph to mutate in-place.
            comm_events: List of comm event dicts from
                :class:`comm_interceptor.CommRecorder`.
            phase_override: If given, stamp all inserted nodes with this phase.

        Returns:
            The same graph (mutated).
        """
        last_by_phase: dict[str, str] = {}

        # Build reverse index: find last node id per phase from existing graph
        for n in graph.nodes.values():
            key = n.phase or "unknown"
            last_by_phase[key] = n.node_id

        for ev in comm_events:
            node_id = ev.get("event_id", f"comm_{len(graph.nodes)+1:07d}")
            op_name = ev.get("op", "collective_unknown")
            phase = phase_override or ev.get("phase", "unknown")

            # Build output TensorMeta from event shape info if available
            output_metas: list[TensorMeta] = []
            for shape_entry in ev.get("tensor_shapes", []):
                if shape_entry is not None:
                    output_metas.append(
                        TensorMeta(
                            shape=tuple(shape_entry.get("shape", [])),
                            dtype=shape_entry.get("dtype", "unknown"),
                            device=shape_entry.get("device", "cpu"),
                        )
                    )

            op_type = ev.get("op_type", "comm_collective")
            node = OpNode(
                node_id=node_id,
                op_name=op_name,
                op_type=op_type,
                phase=phase,
                inputs=[],
                outputs=output_metas,
                comm_op=op_name,
                pp_stage=ev.get("pp_stage"),
                microbatch_idx=ev.get("microbatch"),
                attrs={
                    "group": str(ev.get("group", "")),
                    "tag": str(ev.get("tag", "")),
                    "src_rank": ev.get("src_rank"),
                    "dst_rank": ev.get("dst_rank"),
                    "rank": ev.get("rank"),
                },
            )
            graph.add_node(node)

            for src_id in ev.get("source_node_ids", []):
                if src_id in graph.nodes:
                    graph.add_edge(
                        DataEdge(
                            src_node_id=src_id,
                            dst_node_id=node_id,
                            edge_type="data",
                        )
                    )

            prev_id = last_by_phase.get(phase)
            if prev_id is not None and prev_id != node_id:
                graph.add_edge(
                    DataEdge(
                        src_node_id=prev_id,
                        dst_node_id=node_id,
                        edge_type="control",
                    )
                )
            last_by_phase[phase] = node_id

        return graph
