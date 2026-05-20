# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export utilities: write :class:`SimulationResult` / :class:`ComputeGraph` to
multiple output formats.

Supported formats
-----------------
* **JSON** — full structured dump, loadable back into Python dicts.
* **DOT** — Graphviz dot format with colour-coded nodes by op type.
* **Chrome Trace** — ``chrome://tracing`` compatible JSON for timeline views.
* **Text summary** — human-readable console output with statistics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .nodes import ComputeGraph, OpNode, SimulationResult, TrainingSchedule

# ---------------------------------------------------------------------------
# Colour scheme for DOT export (by op_type)
# ---------------------------------------------------------------------------

_DOT_COLORS: dict[str, str] = {
    "compute": "#AED6F1",          # light blue
    "comm_collective": "#F9E79F",  # yellow
    "comm_p2p": "#FAD7A0",         # orange
    "data_move": "#A9DFBF",        # light green
    "memory": "#D7BDE2",           # light purple
    "unknown": "#D5D8DC",          # grey
}


def _node_color(op_type: str) -> str:
    return _DOT_COLORS.get(op_type, _DOT_COLORS["unknown"])


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(result: SimulationResult, path: str | os.PathLike) -> None:
    """
    Serialize a :class:`SimulationResult` to a JSON file.

    The output is pretty-printed with ``indent=2`` for readability.

    Args:
        result: The simulation result to serialize.
        path: Output file path (will be created / overwritten).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# DOT export
# ---------------------------------------------------------------------------


def _graph_to_dot(
    graph: ComputeGraph,
    title: str = "ComputeGraph",
    include_shapes: bool = True,
) -> str:
    """Render a :class:`ComputeGraph` as a Graphviz DOT string."""
    lines: list[str] = [
        f'digraph "{title}" {{',
        "  rankdir=TB;",
        '  node [shape=box fontname="Helvetica" fontsize=9];',
    ]

    for node in graph.nodes.values():
        color = _node_color(node.op_type)
        label_parts = [node.op_name]
        if include_shapes and node.outputs:
            shape_strs = [str(o.shape) for o in node.outputs[:2]]
            label_parts.append("out: " + ", ".join(shape_strs))
        if node.comm_op:
            label_parts.append(f"[{node.comm_op}]")
        label = "\\n".join(label_parts)
        node_id_safe = node.node_id.replace("-", "_")
        lines.append(
            f'  {node_id_safe} [label="{label}" fillcolor="{color}" style=filled'
            f' tooltip="{node.op_type}"];'
        )

    for edge in graph.edges:
        src = edge.src_node_id.replace("-", "_")
        dst = edge.dst_node_id.replace("-", "_")
        style = "dashed" if edge.edge_type in ("comm_dep", "sequential") else "solid"
        lines.append(f'  {src} -> {dst} [style={style}];')

    lines.append("}")
    return "\n".join(lines)


def export_dot(
    graph: ComputeGraph,
    path: str | os.PathLike,
    title: str = "ComputeGraph",
    include_shapes: bool = True,
) -> None:
    """
    Write a :class:`ComputeGraph` as a Graphviz DOT file.

    Nodes are colour-coded by op type:
    - Blue: compute
    - Yellow: collective comms
    - Orange: P2P comms
    - Green: data movement
    - Purple: memory alloc
    - Grey: unknown

    Args:
        graph: The graph to export.
        path: Output ``.dot`` file path.
        title: Graph title embedded in the DOT file.
        include_shapes: Whether to annotate nodes with output tensor shapes.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dot = _graph_to_dot(graph, title=title, include_shapes=include_shapes)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot)


# ---------------------------------------------------------------------------
# Chrome trace export
# ---------------------------------------------------------------------------


def _op_to_chrome_event(
    node: OpNode,
    pid: int = 0,
    tid: int = 0,
    ts_us: float = 0.0,
    dur_us: float = 1.0,
) -> dict[str, Any]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": ts_us,
        "dur": dur_us,
        "name": node.op_name,
        "cat": node.op_type,
        "args": {
            "node_id": node.node_id,
            "phase": node.phase,
            "pp_stage": node.pp_stage,
            "microbatch": node.microbatch_idx,
            "outputs": [str(o.shape) for o in node.outputs],
            "comm_op": node.comm_op,
        },
    }


def export_chrome_trace(
    result: SimulationResult,
    path: str | os.PathLike,
    us_per_op: float = 1.0,
) -> None:
    """
    Write a ``chrome://tracing``-compatible JSON trace file.

    Each op becomes a duration event (``"ph": "X"``).  Events are laid out
    sequentially per phase on separate *threads* (tid).  The timeline is
    purely logical (not wall-clock time): each event is assigned a
    *us_per_op* microsecond slot.

    Args:
        result: The simulation result to render.
        path: Output JSON file path.
        us_per_op: Duration in microseconds to assign each op slot.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    phase_order = ["forward", "backward", "optimizer", "unknown", "joint"]
    phase_tid: dict[str, int] = {}
    tid_counter = [0]

    def _get_tid(phase: str) -> int:
        if phase not in phase_tid:
            phase_tid[phase] = tid_counter[0]
            tid_counter[0] += 1
        return phase_tid[phase]

    phase_ts: dict[str, float] = {}

    events: list[dict[str, Any]] = []
    for node in result.compute_graph.nodes.values():
        phase = node.phase or "unknown"
        tid = _get_tid(phase)
        ts = phase_ts.get(phase, 0.0)
        events.append(_op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts, dur_us=us_per_op))
        phase_ts[phase] = ts + us_per_op

    # Add FSDP events as a separate process
    for ev in result.fsdp_events:
        phase = ev.get("phase", "unknown")
        tid = _get_tid(f"fsdp_{phase}")
        ts = phase_ts.get(f"fsdp_{phase}", 0.0)
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": _get_tid(f"fsdp_{phase}"),
                "ts": ts,
                "dur": us_per_op,
                "name": ev.get("event_type", "fsdp_event"),
                "cat": "fsdp",
                "args": ev,
            }
        )
        phase_ts[f"fsdp_{phase}"] = ts + us_per_op

    # Metadata events
    for phase, tid in phase_tid.items():
        events.append(
            {
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": phase},
            }
        )

    trace = {"traceEvents": events, "displayTimeUnit": "us"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Text summary export
# ---------------------------------------------------------------------------


def export_text_summary(result: SimulationResult) -> str:
    """
    Return a human-readable text summary of a :class:`SimulationResult`.

    Prints statistics about the compute graph, communication ops, FSDP
    lifecycle events, and schedule events.

    Args:
        result: The simulation result to summarise.

    Returns:
        A multi-line string.
    """
    lines: list[str] = []
    sep = "=" * 72

    def section(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    graph = result.compute_graph

    section("Compute Graph Summary")
    lines.append(f"  Total ops : {len(graph.nodes)}")
    lines.append(f"  Total edges: {len(graph.edges)}")

    # Count by type
    type_counts: dict[str, int] = {}
    for n in graph.nodes.values():
        type_counts[n.op_type] = type_counts.get(n.op_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    # Count by phase
    phase_counts: dict[str, int] = {}
    for n in graph.nodes.values():
        p = n.phase or "unknown"
        phase_counts[p] = phase_counts.get(p, 0) + 1
    lines.append("")
    lines.append("  By phase:")
    for p, c in sorted(phase_counts.items()):
        lines.append(f"    {p:<22}: {c}")

    section("Communication Events")
    lines.append(f"  Total comm events: {len(result.comm_events)}")
    op_counts: dict[str, int] = {}
    for ev in result.comm_events:
        op = ev.get("op", "unknown")
        op_counts[op] = op_counts.get(op, 0) + 1
    for op, c in sorted(op_counts.items()):
        lines.append(f"    {op:<22}: {c}")

    section("FSDP Events")
    lines.append(f"  Total FSDP events: {len(result.fsdp_events)}")
    ev_type_counts: dict[str, int] = {}
    for ev in result.fsdp_events:
        t = ev.get("event_type", "unknown")
        ev_type_counts[t] = ev_type_counts.get(t, 0) + 1
    for t, c in sorted(ev_type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    section("PP Events")
    lines.append(f"  Total PP events: {len(result.pp_events)}")
    pp_type_counts: dict[str, int] = {}
    for ev in result.pp_events:
        t = ev.get("event_type", ev.get("action_type", "unknown"))
        pp_type_counts[t] = pp_type_counts.get(t, 0) + 1
    for t, c in sorted(pp_type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    if result.schedule:
        section("Training Schedule")
        sched = result.schedule
        lines.append(f"  Total schedule events: {len(sched.events)}")
        lines.append(f"  Total schedule deps  : {len(sched.deps)}")
        if sched.metadata:
            for k, v in sched.metadata.items():
                lines.append(f"    {k}: {v}")

    section("Metadata")
    for k, v in result.metadata.items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)
