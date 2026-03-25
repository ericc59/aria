"""End-to-end integration test for the ARIA pipeline.

Tests the full loop: extract -> propose -> verify -> solve
using a simple ARC-like task (recolor all objects to a single color).
"""

import numpy as np

from aria.types import (
    DemoPair, Task, Program, Bind, Call, Ref, Literal, Type,
    grid_from_list, grid_eq,
)
from aria.graph.extract import extract, extract_with_delta
from aria.graph.signatures import compute_task_signatures
from aria.proposer.parser import parse_program
from aria.verify.verifier import verify
from aria.verify.mode import detect_mode, VerifyMode
from aria.library.store import Library
from aria.proposer.models import MockProposer
from aria.proposer.harness import propose_and_verify


# ---- State graph extraction ----

def test_extract_simple_grid():
    grid = grid_from_list([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    sg = extract(grid)
    assert sg.context.bg_color == 0
    assert sg.context.dims == (5, 5)
    assert len(sg.objects) >= 2  # at least the two colored objects


def test_extract_partition_scene_and_roles():
    grid = grid_from_list([
        [1, 1, 1, 1, 1],
        [1, 2, 1, 3, 1],
        [1, 1, 1, 1, 1],
        [1, 4, 1, 5, 1],
        [1, 1, 1, 1, 1],
    ])
    sg = extract(grid)

    assert sg.partition is not None
    assert sg.partition.separator_color == 1
    assert sg.partition.n_rows == 2
    assert sg.partition.n_cols == 2
    assert any(binding.role.name == "SEPARATOR" for binding in sg.roles)


def test_extract_legend_scene():
    grid = grid_from_list([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [9, 9, 9, 9],
        [0, 0, 5, 0],
        [0, 0, 0, 6],
    ])
    sg = extract(grid)

    assert sg.legend is not None
    assert len(sg.legend.entries) == 2
    assert any(binding.role.name == "LEGEND" for binding in sg.roles)


def test_extract_sparse_points_do_not_fake_partition_or_legend():
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    ])

    sg = extract(grid)

    assert sg.partition is None
    assert sg.legend is None


def test_extract_aligned_sparse_markers_do_not_fake_partition():
    grid = grid_from_list([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    sg = extract(grid)

    assert sg.partition is None


def test_extract_with_delta():
    in_grid = grid_from_list([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    out_grid = grid_from_list([
        [0, 2, 0],
        [0, 2, 0],
        [0, 0, 0],
    ])
    sg_in, sg_out, delta = extract_with_delta(in_grid, out_grid)
    assert sg_in.context.dims == (3, 3)
    assert sg_out.context.dims == (3, 3)
    # Delta should show color modification
    assert len(delta.modified) > 0 or len(delta.added) > 0 or len(delta.removed) > 0


def test_compute_task_signatures_uses_partition_and_legend():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [9, 9, 9, 9],
                [0, 0, 5, 0],
                [0, 0, 0, 6],
            ]),
            output=grid_from_list([
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [9, 9, 9, 9],
                [0, 0, 5, 0],
                [0, 0, 0, 6],
            ]),
        ),
    )

    signatures = compute_task_signatures(demos)
    assert "role:has_legend" in signatures
    assert "legend:present" in signatures


def test_compute_task_signatures_require_cross_demo_support_for_scene_tags():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [9, 9, 9, 9],
                [0, 0, 5, 0],
                [0, 0, 0, 6],
            ]),
            output=grid_from_list([
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [9, 9, 9, 9],
                [0, 0, 5, 0],
                [0, 0, 0, 6],
            ]),
        ),
        DemoPair(
            input=grid_from_list([
                [0, 0, 5, 0],
                [0, 0, 0, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            output=grid_from_list([
                [0, 0, 5, 0],
                [0, 0, 0, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        ),
    )

    signatures = compute_task_signatures(demos)

    assert "legend:present" not in signatures
    assert "role:has_legend" not in signatures
    assert "role:has_marker" in signatures


def test_compute_task_signatures_captures_specific_size_rules():
    demos = (
        DemoPair(
            input=grid_from_list([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            output=grid_from_list([[0] * 6 for _ in range(6)]),
        ),
        DemoPair(
            input=grid_from_list([[1] * 4 for _ in range(4)]),
            output=grid_from_list([[0] * 8 for _ in range(8)]),
        ),
    )

    signatures = compute_task_signatures(demos)
    assert "size:multiplicative" in signatures
    assert "size:rows_x2" in signatures
    assert "size:cols_x2" in signatures
    assert "size:scale_2x" in signatures


# ---- Parser ----

def test_parse_simple_program():
    text = """\
bind objects = find_objects(input)
bind n = count(objects)
bind out_dims = dims_make(n, 1)
bind canvas = new_grid(out_dims, 0)
yield canvas
"""
    prog = parse_program(text)
    assert len(prog.steps) == 4
    assert prog.output == "canvas"


def test_parse_with_enum_literals():
    text = """\
bind objects = find_objects(input)
bind sorted = sort_by(size, descending, objects)
yield sorted
"""
    prog = parse_program(text)
    assert len(prog.steps) == 2
    assert prog.output == "sorted"


# ---- Mode detection ----

def test_mode_stateless():
    prog = parse_program("""\
bind objects = find_objects(input)
bind result = recolor(3, objects)
yield result
""")
    assert detect_mode(prog) == VerifyMode.STATELESS


def test_mode_loo():
    prog = parse_program("""\
bind mapping = infer_map(ctx, color, color)
bind result = apply_color_map(mapping, input)
yield result
""")
    assert detect_mode(prog) == VerifyMode.LEAVE_ONE_OUT


def test_mode_sequential():
    prog = parse_program("""\
bind rule = infer_step(ctx)
bind result = repeat_apply(4, rule, input)
yield result
""")
    assert detect_mode(prog) == VerifyMode.SEQUENTIAL


# ---- Verification (stateless, with a hand-built program) ----

def test_verify_identity_program():
    """A program that just yields the input should pass if input == output."""
    prog = Program(
        steps=(),
        output="input",
    )
    demo = DemoPair(
        input=grid_from_list([[1, 2], [3, 4]]),
        output=grid_from_list([[1, 2], [3, 4]]),
    )
    result = verify(prog, (demo,))
    assert result.passed


def test_verify_identity_program_fails():
    """Identity program fails when input != output."""
    prog = Program(
        steps=(),
        output="input",
    )
    demo = DemoPair(
        input=grid_from_list([[1, 2], [3, 4]]),
        output=grid_from_list([[5, 6], [7, 8]]),
    )
    result = verify(prog, (demo,))
    assert not result.passed
    assert result.failed_demo == 0


# ---- Library ----

def test_library_basic():
    lib = Library()
    assert len(lib.all_entries()) == 0
    assert lib.names() == []


# ---- Full pipeline smoke test ----

def test_full_pipeline_smoke():
    """Smoke test: extract graphs, parse program, verify."""
    in_grid = grid_from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    out_grid = grid_from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])

    # Extract
    sg_in, sg_out, delta = extract_with_delta(in_grid, out_grid)
    assert sg_in is not None
    assert sg_out is not None

    # For this identity task, the trivial program works
    prog = Program(steps=(), output="input")
    demo = DemoPair(input=in_grid, output=out_grid)

    # Verify
    result = verify(prog, (demo,))
    assert result.passed


def test_proposer_harness_rejects_static_type_errors():
    in_grid = grid_from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    out_grid = in_grid.copy()
    sg_in, _sg_out, delta = extract_with_delta(in_grid, out_grid)
    demos = (DemoPair(input=in_grid, output=out_grid),)

    model = MockProposer(programs=[
        """bind bad = count(input)
yield bad"""
    ])

    result = propose_and_verify(
        model=model,
        demos=demos,
        state_graphs=[sg_in],
        deltas=[delta],
        library=Library(),
        max_rounds=1,
        k=1,
        task_id="static-type-error",
    )

    assert not result.solved
    attempt = result.all_attempts[0][0]
    assert attempt.execution_error is not None
    assert "Type check failed" in attempt.execution_error
