"""Serialize state graphs and deltas into structured text for the proposer.

Converts the typed internal representations into the text format
the proposer model consumes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import (
    Delta,
    GridContext,
    ObjectNode,
    RelationEdge,
    StateGraph,
)


def serialize_object(obj: ObjectNode) -> str:
    x, y, w, h = obj.bbox
    sym = ", ".join(s.name for s in sorted(obj.symmetry, key=lambda s: s.name))
    return (
        f"  {{id:{obj.id}, color:{obj.color}, shape:{obj.shape.name}, "
        f"bbox:({x},{y},{w},{h}), size:{obj.size}"
        + (f", symmetry:{{{sym}}}" if sym else "")
        + "}"
    )


def serialize_relation(rel: RelationEdge) -> str:
    spatial = ", ".join(s.name for s in sorted(rel.spatial, key=lambda s: s.name))
    topo = ", ".join(s.name for s in sorted(rel.topo, key=lambda s: s.name))
    parts = [f"src:{rel.src}", f"dst:{rel.dst}"]
    if spatial:
        parts.append(f"spatial:{{{spatial}}}")
    if topo:
        parts.append(f"topo:{{{topo}}}")
    return "  {" + ", ".join(parts) + "}"


def serialize_context(ctx: GridContext) -> str:
    r, c = ctx.dims
    palette = ", ".join(str(c) for c in sorted(ctx.palette))
    parts = [
        f"dims:({r},{c})",
        f"bg_color:{ctx.bg_color}",
        f"obj_count:{ctx.obj_count}",
        f"palette:{{{palette}}}",
    ]
    if ctx.is_tiled:
        tr, tc = ctx.is_tiled
        parts.append(f"tiled:({tr},{tc})")
    if ctx.symmetry:
        sym = ", ".join(s.name for s in sorted(ctx.symmetry, key=lambda s: s.name))
        parts.append(f"symmetry:{{{sym}}}")
    return "  {" + ", ".join(parts) + "}"


def serialize_state_graph(sg: StateGraph) -> str:
    lines = ["[STATE_GRAPH]"]
    lines.append("  objects: [")
    for obj in sg.objects:
        lines.append(serialize_object(obj) + ",")
    lines.append("  ]")
    lines.append("  relations: [")
    for rel in sg.relations:
        lines.append(serialize_relation(rel) + ",")
    lines.append("  ]")
    lines.append("  context: " + serialize_context(sg.context))
    return "\n".join(lines)


def serialize_delta(delta: Delta, demo_idx: int) -> str:
    parts = [f"  demo_{demo_idx}: {{"]
    added_ids = [str(o.id) for o in delta.added]
    parts.append(f"    added:[{', '.join(added_ids)}]")
    parts.append(f"    removed:[{', '.join(str(r) for r in delta.removed)}]")
    mods = []
    for obj_id, field, old, new in delta.modified:
        mods.append(f"{{id:{obj_id}, field:{field}, {old}→{new}}}")
    parts.append(f"    modified:[{', '.join(mods)}]")
    if delta.dims_changed:
        old_d, new_d = delta.dims_changed
        parts.append(f"    dims_changed:{old_d}→{new_d}")
    parts.append("  }")
    return "\n".join(parts)


def serialize_library_index(core_ops: list[str], library_ops: list[dict[str, Any]]) -> str:
    lines = ["[LIBRARY]"]
    lines.append(f"  core: [{', '.join(core_ops)}]")
    if library_ops:
        learned = []
        for op in library_ops:
            params = ", ".join(f"{n}:{t}" for n, t in op["params"])
            learned.append(f"{op['name']}({params})")
        lines.append(f"  learned: [{', '.join(learned)}]")
    return "\n".join(lines)


def serialize_prior_attempt(
    attempt_num: int = 0,
    program_text: str = "",
    failed_demo: int | None = None,
    error_type: str | None = None,
    diff: dict[str, Any] | None = None,
    step_trace: list[dict[str, Any]] | None = None,
    error_detail: str | None = None,
    **_kwargs: Any,
) -> str:
    lines = [f"  attempt_{attempt_num}:"]
    lines.append(f'    program: "{program_text}"')
    if failed_demo is not None:
        lines.append(f"    failed_demo: {failed_demo}")
    if error_type:
        lines.append(f"    error_type: {error_type}")
    if error_detail:
        lines.append(f"    error_detail: {error_detail}")
    if diff:
        lines.append("    diff: {")
        for k, v in diff.items():
            lines.append(f"      {k}: {v}")
        lines.append("    }")
    if step_trace:
        lines.append("    step_trace: [")
        for entry in step_trace:
            status = "ok" if entry.get("ok") else f'SUSPECT: "{entry.get("suspect", "")}"'
            lines.append(f'      {{step: "{entry["step_name"]}", value: "{entry["value"]}", {status}}}')
        lines.append("    ]")
    return "\n".join(lines)


def build_proposer_input(
    state_graphs: list[StateGraph],
    deltas: list[Delta],
    core_ops: list[str],
    library_ops: list[dict[str, Any]],
    prior_attempts: list[dict[str, Any]] | None = None,
) -> str:
    """Build the full structured input for the proposer model."""
    sections = []

    # State graph (from first demo input, representative)
    if state_graphs:
        sections.append(serialize_state_graph(state_graphs[0]))

    # Deltas
    if deltas:
        delta_lines = ["[DELTAS]"]
        for i, d in enumerate(deltas):
            delta_lines.append(serialize_delta(d, i))
        sections.append("\n".join(delta_lines))

    # Library
    sections.append(serialize_library_index(core_ops, library_ops))

    # Prior attempts (rounds 2+)
    if prior_attempts:
        attempt_lines = ["[PRIOR_ATTEMPTS]"]
        for att in prior_attempts:
            attempt_lines.append(serialize_prior_attempt(**att))
        sections.append("\n".join(attempt_lines))

    return "\n\n".join(sections)
