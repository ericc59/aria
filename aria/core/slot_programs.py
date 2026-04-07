"""Slot-conditioned symbolic program templates.

Programs with open structural slots filled from entity graph search.
Templates are fixed; slots are filled from graph queries and verified exactly.

Stage 1 templates:
  1. relocate_marker_to_gap — move markers into host gaps
  2. recolor_by_paired_color — recolor objects using paired object's color
  3. fill_gaps_with_marker_color — fill host gaps using paired marker color
  4. erase_by_role — erase objects matching a role
  5. swap_paired_colors — swap colors between paired objects
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aria.core.entity_graph import (
    DemoGraphSet,
    EntityGraph,
    build_demo_graphs,
    query_marker_host_pairs,
    query_objects_by_role,
)
from aria.types import (
    Bind,
    Call,
    DemoPair,
    Grid,
    Literal,
    Program,
    Ref,
    Type,
)
from aria.verify.verifier import verify


# ---------------------------------------------------------------------------
# Slot-conditioned program result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotProgramResult:
    """Result of slot-conditioned program search."""
    solved: bool
    winning_program: Program | None = None
    template_name: str = ""
    slot_values: dict = None  # type: ignore[assignment]
    candidates_tested: int = 0


# ---------------------------------------------------------------------------
# Template 1: relocate_marker_to_gap
# ---------------------------------------------------------------------------


def _try_relocate_marker_to_gap(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Try relocating markers into host gaps via match_and_place."""
    candidates: list[tuple[Program, str, dict]] = []

    # Check structural preconditions
    if not demo_graphs.n_markers_consistent:
        return candidates
    if not all(len(g.gaps) > 0 for g in demo_graphs.graphs):
        return candidates

    # Try all match_rule × align combinations
    from aria.runtime.ops.relate_paint import ALL_MATCH_RULES, ALL_ALIGNS

    for match_rule in ALL_MATCH_RULES:
        for align in ALL_ALIGNS:
            prog = Program(
                steps=(
                    Bind(
                        name="v0",
                        typ=Type.GRID,
                        expr=Call(
                            op="relocate_objects",
                            args=(
                                Ref(name="input"),
                                Literal(value=int(match_rule), typ=Type.INT),
                                Literal(value=int(align), typ=Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )
            candidates.append((
                prog,
                "relocate_marker_to_gap",
                {"match_rule": match_rule, "align": align},
            ))

    return candidates


# ---------------------------------------------------------------------------
# Template 2: recolor_by_paired_color
# ---------------------------------------------------------------------------


def _try_recolor_by_paired_color(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Recolor objects using the color of a paired neighbor.

    For each demo, check if the output recolors some objects to match
    the color of a related object. Then search for the pairing rule.
    """
    candidates: list[tuple[Program, str, dict]] = []

    # Check if any demos have same-shape or same-size pairs
    if not demo_graphs.n_hosts_consistent:
        return candidates

    # Try conditional_fill with colors from related objects
    # This is a proxy for recolor-by-pair: for each pair of colors
    # where one becomes the other in the output
    for d_idx, demo in enumerate(demos):
        if demo.input.shape != demo.output.shape:
            return []
        diff_mask = demo.input != demo.output
        if not np.any(diff_mask):
            continue

        # Collect (from_color, to_color) changes
        from_colors = demo.input[diff_mask]
        to_colors = demo.output[diff_mask]
        color_changes: set[tuple[int, int]] = set()
        for fc, tc in zip(from_colors.ravel(), to_colors.ravel()):
            color_changes.add((int(fc), int(tc)))

        for fc, tc in color_changes:
            prog = Program(
                steps=(
                    Bind(
                        name="v0",
                        typ=Type.GRID,
                        expr=Call(
                            op="conditional_fill",
                            args=(
                                Ref(name="input"),
                                Literal(value=fc, typ=Type.INT),
                                Literal(value=tc, typ=Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )
            candidates.append((
                prog,
                "recolor_by_paired_color",
                {"from_color": fc, "to_color": tc},
            ))
        break  # only need one demo for candidates

    return candidates


# ---------------------------------------------------------------------------
# Template 3: fill_gaps_with_marker_color
# ---------------------------------------------------------------------------


def _try_fill_gaps_with_marker_color(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Fill enclosed regions using marker color."""
    candidates: list[tuple[Program, str, dict]] = []

    if not all(len(g.gaps) > 0 for g in demo_graphs.graphs):
        return candidates
    if not all(any(r.role == "marker" for r in g.roles) for g in demo_graphs.graphs):
        return candidates

    # Infer fill color from markers in each demo
    for graph, demo in zip(demo_graphs.graphs, demos):
        markers = query_objects_by_role(graph, "marker")
        if not markers:
            continue
        for marker in markers:
            fill_color = marker.color
            # Try fill_enclosed with this color
            prog = Program(
                steps=(
                    Bind(
                        name="v0",
                        typ=Type.GRID,
                        expr=Call(
                            op="fill_enclosed",
                            args=(
                                Ref(name="input"),
                                Literal(value=fill_color, typ=Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )
            candidates.append((
                prog,
                "fill_gaps_with_marker_color",
                {"fill_color": fill_color},
            ))
        break  # candidates from first demo

    # Also try fill_enclosed_regions and fill_enclosed_regions_auto
    candidates.append((
        Program(
            steps=(Bind(name="v0", typ=Type.GRID,
                        expr=Call(op="fill_enclosed_regions_auto",
                                  args=(Ref(name="input"),))),),
            output="v0",
        ),
        "fill_gaps_auto",
        {},
    ))

    return candidates


# ---------------------------------------------------------------------------
# Template 4: erase_by_role
# ---------------------------------------------------------------------------


def _try_erase_by_role(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Erase objects matching a role (markers, hosts, specific colors)."""
    candidates: list[tuple[Program, str, dict]] = []

    if any(d.input.shape != d.output.shape for d in demos):
        return candidates

    # Check which colors are erased (present in input, absent in output)
    for demo in demos:
        in_colors = set(int(v) for v in np.unique(demo.input))
        out_colors = set(int(v) for v in np.unique(demo.output))
        erased = in_colors - out_colors
        for c in erased:
            prog = Program(
                steps=(
                    Bind(name="v0", typ=Type.GRID,
                         expr=Call(op="erase_color",
                                   args=(Ref(name="input"),
                                         Literal(value=c, typ=Type.INT)))),
                ),
                output="v0",
            )
            candidates.append((prog, "erase_by_role", {"erase_color": c}))
        break

    # Try size-based erasure
    for threshold in (1, 2, 3, 5):
        for op_name in ("erase_by_max_size", "keep_by_min_size"):
            prog = Program(
                steps=(
                    Bind(name="v0", typ=Type.GRID,
                         expr=Call(op=op_name,
                                   args=(Ref(name="input"),
                                         Literal(value=threshold, typ=Type.INT)))),
                ),
                output="v0",
            )
            candidates.append((prog, "erase_by_role", {"op": op_name, "threshold": threshold}))

    return candidates


# ---------------------------------------------------------------------------
# Template 5: select_relate_paint (existing relational op)
# ---------------------------------------------------------------------------


def _try_select_relate_paint(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Try the existing select_relate_paint op."""
    candidates: list[tuple[Program, str, dict]] = []

    if any(d.input.shape != d.output.shape for d in demos):
        return candidates

    prog = Program(
        steps=(
            Bind(name="v0", typ=Type.GRID,
                 expr=Call(op="select_relate_paint",
                           args=(Ref(name="input"),))),
        ),
        output="v0",
    )
    candidates.append((prog, "select_relate_paint", {}))
    return candidates


# ---------------------------------------------------------------------------
# Template 6: graph-guided depth-2 compositions
# ---------------------------------------------------------------------------


def _try_graph_guided_compose(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Graph-guided depth-2 compositions.

    Uses entity graph structure to constrain which depth-2 programs to try:
    - If markers + hosts: try extract_markers → correction
    - If frames + content: try crop_frame → correction
    - If color changes are role-based: try role-based recolor chains
    """
    candidates: list[tuple[Program, str, dict]] = []

    has_markers = all(any(r.role == "marker" for r in g.roles)
                      for g in demo_graphs.graphs)
    has_hosts = all(any(r.role == "host" for r in g.roles)
                    for g in demo_graphs.graphs)
    has_gaps = all(len(g.gaps) > 0 for g in demo_graphs.graphs)

    # Analyze color changes from demo 0
    demo0 = demos[0]
    if demo0.input.shape != demo0.output.shape:
        return candidates

    diff_mask = demo0.input != demo0.output
    if not np.any(diff_mask):
        return candidates

    from_vals = demo0.input[diff_mask]
    to_vals = demo0.output[diff_mask]
    bg = int(np.bincount(demo0.input.ravel()).argmax())

    # Collect unique color changes
    changes: set[tuple[int, int]] = set()
    for f, t in zip(from_vals.ravel(), to_vals.ravel()):
        changes.add((int(f), int(t)))

    # If markers exist and changes involve bg→color: marker-seeded fill
    if has_markers and any(f == bg for f, t in changes):
        fill_colors = set(t for f, t in changes if f == bg)
        for fc in fill_colors:
            # extract_markers → fill_enclosed
            step1 = Bind(name="v0", typ=Type.GRID,
                         expr=Call(op="extract_markers",
                                   args=(Ref(name="input"),)))
            step2 = Bind(name="v1", typ=Type.GRID,
                         expr=Call(op="fill_enclosed",
                                   args=(Ref(name="v0"),
                                         Literal(value=fc, typ=Type.INT))))
            candidates.append((
                Program(steps=(step1, step2), output="v1"),
                "extract_markers+fill_enclosed",
                {"fill_color": fc},
            ))

    # If hosts + gaps: try relocate then erase
    if has_markers and has_hosts:
        from aria.runtime.ops.relate_paint import ALL_MATCH_RULES, ALL_ALIGNS
        # relocate → erase_color(marker_color)
        marker_colors = set()
        for g in demo_graphs.graphs:
            for r in g.roles:
                if r.role == "marker":
                    marker_colors.add(r.color)

        for mr in ALL_MATCH_RULES:
            for al in ALL_ALIGNS:
                step1 = Bind(name="v0", typ=Type.GRID,
                             expr=Call(op="relocate_objects",
                                       args=(Ref(name="input"),
                                             Literal(value=int(mr), typ=Type.INT),
                                             Literal(value=int(al), typ=Type.INT))))
                # After relocation, try erasing marker colors
                for mc in marker_colors:
                    step2 = Bind(name="v1", typ=Type.GRID,
                                 expr=Call(op="erase_color",
                                           args=(Ref(name="v0"),
                                                 Literal(value=mc, typ=Type.INT))))
                    candidates.append((
                        Program(steps=(step1, step2), output="v1"),
                        "relocate+erase_marker",
                        {"match_rule": mr, "align": al, "erase_color": mc},
                    ))

    # Symmetry-based: complete_symmetry → conditional_fill for corrections
    for sym_op in ("complete_symmetry_h", "complete_symmetry_v"):
        step1 = Bind(name="v0", typ=Type.GRID,
                     expr=Call(op=sym_op, args=(Ref(name="input"),)))
        for f, t in changes:
            step2 = Bind(name="v1", typ=Type.GRID,
                         expr=Call(op="conditional_fill",
                                   args=(Ref(name="v0"),
                                         Literal(value=f, typ=Type.INT),
                                         Literal(value=t, typ=Type.INT))))
            candidates.append((
                Program(steps=(step1, step2), output="v1"),
                f"{sym_op}+conditional_fill",
                {"from": f, "to": t},
            ))

    # Erase → fill: erase a color then fill enclosed
    for f, t in changes:
        if f != bg:
            step1 = Bind(name="v0", typ=Type.GRID,
                         expr=Call(op="erase_color",
                                   args=(Ref(name="input"),
                                         Literal(value=f, typ=Type.INT))))
            for fc in set(t2 for _, t2 in changes):
                step2 = Bind(name="v1", typ=Type.GRID,
                             expr=Call(op="fill_enclosed",
                                       args=(Ref(name="v0"),
                                             Literal(value=fc, typ=Type.INT))))
                candidates.append((
                    Program(steps=(step1, step2), output="v1"),
                    "erase+fill_enclosed",
                    {"erase": f, "fill": fc},
                ))

    return candidates


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------


def _try_directional_fill(
    demos: tuple[DemoPair, ...],
    demo_graphs: DemoGraphSet,
) -> list[tuple[Program, str, dict]]:
    """Try directional fill_between_v / fill_between_h with all color combos."""
    candidates: list[tuple[Program, str, dict]] = []

    if any(d.input.shape != d.output.shape for d in demos):
        return candidates

    for op_name in ("fill_between_v", "fill_between_h"):
        for bc in range(10):
            for fc in range(10):
                prog = Program(
                    steps=(
                        Bind(name="v0", typ=Type.GRID,
                             expr=Call(op=op_name,
                                       args=(Ref(name="input"),
                                             Literal(value=bc, typ=Type.INT),
                                             Literal(value=fc, typ=Type.INT)))),
                    ),
                    output="v0",
                )
                candidates.append((prog, f"{op_name}", {"boundary": bc, "fill": fc}))

    return candidates


_TEMPLATE_FNS = [
    _try_relocate_marker_to_gap,
    _try_recolor_by_paired_color,
    _try_fill_gaps_with_marker_color,
    _try_erase_by_role,
    _try_select_relate_paint,
    _try_directional_fill,
    _try_graph_guided_compose,
]


def search_slot_programs(
    demos: tuple[DemoPair, ...],
    *,
    max_candidates: int = 2000,
) -> SlotProgramResult:
    """Search for slot-conditioned programs using entity graph guidance.

    1. Build entity graphs for all demos
    2. Generate candidates from each template
    3. Verify each candidate on all demos
    4. Return first verified program
    """
    demo_graphs = build_demo_graphs(demos)

    total_tested = 0
    seen: set[str] = set()

    for template_fn in _TEMPLATE_FNS:
        candidates = template_fn(demos, demo_graphs)

        for prog, template_name, slots in candidates:
            if total_tested >= max_candidates:
                return SlotProgramResult(
                    solved=False,
                    candidates_tested=total_tested,
                )

            # Deduplicate by program text
            from aria.runtime.program import program_to_text
            key = program_to_text(prog)
            if key in seen:
                continue
            seen.add(key)

            total_tested += 1
            vr = verify(prog, demos)
            if vr.passed:
                return SlotProgramResult(
                    solved=True,
                    winning_program=prog,
                    template_name=template_name,
                    slot_values=slots,
                    candidates_tested=total_tested,
                )

    return SlotProgramResult(
        solved=False,
        candidates_tested=total_tested,
    )
