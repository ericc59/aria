"""Cross-demo context operations.

These ops read from the TaskContext to support multi-example reasoning.
They analyze demo pairs to discover consistent patterns (color maps,
dimension rules, progression sequences, predicate selection).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from aria.types import DemoPair, Grid, ObjectNode, TaskContext, Type
from aria.graph.extract import extract
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _demo_count(ctx: TaskContext) -> int:
    """Return the number of demo pairs."""
    return len(ctx.demos)


def _demo_at(ctx: TaskContext, idx: int) -> tuple[Grid, Grid]:
    """Return (input_grid, output_grid) for the demo at index `idx`."""
    if idx < 0 or idx >= len(ctx.demos):
        raise IndexError(f"demo_at: index {idx} out of range for {len(ctx.demos)} demos")
    pair = ctx.demos[idx]
    return (pair.input.copy(), pair.output.copy())


def _infer_map(
    ctx: TaskContext, prop_in: int, prop_out: int
) -> dict[int, int]:
    """Infer a color mapping by analyzing all demo pairs pixel-by-pixel.

    For each demo, compares input and output grids at every cell where the
    color changes. Builds per-demo mappings, then intersects them to find
    the mapping that holds across all demos.

    Parameters
    ----------
    ctx : TaskContext
        The task context containing demo pairs.
    prop_in : int
        Property channel for input (unused in color mode, reserved for future
        property-based mapping). Pass 0 for color-to-color mapping.
    prop_out : int
        Property channel for output. Pass 0 for color-to-color mapping.

    Returns
    -------
    dict[int, int]
        A mapping from input color to output color that is consistent
        across all demos.
    """
    if not ctx.demos:
        return {}

    # Collect per-demo mappings. Each demo produces a dict of observed
    # color->color transitions. We track only cells where both grids
    # have valid positions (same shape region).
    per_demo_maps: list[dict[int, Counter]] = []

    for demo in ctx.demos:
        inp, out = demo.input, demo.output
        # Only compare the overlapping region
        min_rows = min(inp.shape[0], out.shape[0])
        min_cols = min(inp.shape[1], out.shape[1])
        inp_region = inp[:min_rows, :min_cols]
        out_region = out[:min_rows, :min_cols]

        color_votes: dict[int, Counter] = {}
        for r in range(min_rows):
            for c in range(min_cols):
                ci = int(inp_region[r, c])
                co = int(out_region[r, c])
                if ci not in color_votes:
                    color_votes[ci] = Counter()
                color_votes[ci][co] += 1

        per_demo_maps.append(color_votes)

    # For each input color, determine the dominant output color per demo,
    # then check if that mapping is consistent across all demos.
    all_input_colors: set[int] = set()
    for dm in per_demo_maps:
        all_input_colors.update(dm.keys())

    result: dict[int, int] = {}
    for ci in sorted(all_input_colors):
        # For each demo that saw this input color, find the dominant output
        dominant_per_demo: list[int] = []
        for dm in per_demo_maps:
            if ci in dm:
                dominant_per_demo.append(dm[ci].most_common(1)[0][0])

        if not dominant_per_demo:
            continue

        # Check consistency: all demos agree on the output color
        if len(set(dominant_per_demo)) == 1:
            result[ci] = dominant_per_demo[0]
        else:
            # Majority vote across demos as fallback
            vote = Counter(dominant_per_demo)
            winner, count = vote.most_common(1)[0]
            # Only include if a strong majority (> half the demos)
            if count > len(dominant_per_demo) / 2:
                result[ci] = winner

    return result


def _infer_step(ctx: TaskContext) -> Callable[[Grid], Grid]:
    """Infer a transformation step from ordered demo pairs.

    Detects progression rules by comparing consecutive demos:
    - Translation: an object moves by a constant offset each step.
    - Growth: the grid grows by a constant amount each step.
    - Iteration: a consistent pixel-level diff is applied repeatedly.

    Falls back to a color-mapping transform if no spatial progression is
    found but a consistent color map exists.

    Returns a callable Grid -> Grid that applies one step.
    """
    if len(ctx.demos) < 2:
        # With fewer than 2 demos, try color mapping as a single-step transform
        return _step_from_color_map(ctx)

    # Analyze consecutive demo pairs for spatial progression
    translation = _detect_translation(ctx.demos)
    if translation is not None and translation != (0, 0):
        dr, dc = translation
        return _make_translation_step(dr, dc)

    growth = _detect_growth(ctx.demos)
    if growth is not None:
        grow_r, grow_c = growth
        return _make_growth_step(grow_r, grow_c)

    # Try pixel-level diff (iteration pattern)
    diff_step = _detect_pixel_diff(ctx.demos)
    if diff_step is not None:
        return diff_step

    # Fallback: color mapping as a transform
    return _step_from_color_map(ctx)


def _detect_translation(demos: tuple[DemoPair, ...]) -> tuple[int, int] | None:
    """Check if objects translate by a constant offset between consecutive demos.

    Compares the output grids of consecutive demos. Finds non-background
    pixels and checks if their center of mass shifts by a constant
    integer vector. Returns None if the offset is not near-integer or
    is (0, 0).
    """
    if len(demos) < 2:
        return None

    offsets: list[tuple[int, int]] = []
    for i in range(len(demos) - 1):
        out_a = demos[i].output
        out_b = demos[i + 1].output

        if out_a.shape != out_b.shape:
            return None  # Shape changes rule out simple translation

        # Compute center of mass of non-zero pixels
        com_a = _center_of_mass(out_a)
        com_b = _center_of_mass(out_b)

        if com_a is None or com_b is None:
            return None

        raw_dr = com_b[0] - com_a[0]
        raw_dc = com_b[1] - com_a[1]

        # Translation must be close to integer offsets
        dr = int(round(raw_dr))
        dc = int(round(raw_dc))
        if abs(raw_dr - dr) > 0.1 or abs(raw_dc - dc) > 0.1:
            return None

        offsets.append((dr, dc))

    if not offsets:
        return None

    # All offsets must be identical and non-zero
    if offsets[0] == (0, 0):
        return None
    if all(o == offsets[0] for o in offsets):
        return offsets[0]
    return None


def _center_of_mass(grid: Grid) -> tuple[float, float] | None:
    """Compute center of mass of non-zero pixels in a grid."""
    nonzero = np.argwhere(grid > 0)
    if len(nonzero) == 0:
        return None
    return (float(np.mean(nonzero[:, 0])), float(np.mean(nonzero[:, 1])))


def _make_translation_step(dr: int, dc: int) -> Callable[[Grid], Grid]:
    """Return a callable that shifts non-background pixels by (dr, dc)."""
    def translate(grid: Grid) -> Grid:
        rows, cols = grid.shape
        result = np.zeros_like(grid)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = grid[r, c]
        return result
    return translate


def _detect_growth(demos: tuple[DemoPair, ...]) -> tuple[int, int] | None:
    """Check if output grids grow by a constant amount between demos."""
    if len(demos) < 2:
        return None

    deltas: list[tuple[int, int]] = []
    for i in range(len(demos) - 1):
        out_a = demos[i].output
        out_b = demos[i + 1].output
        d_r = int(out_b.shape[0]) - int(out_a.shape[0])
        d_c = int(out_b.shape[1]) - int(out_a.shape[1])
        deltas.append((d_r, d_c))

    if not deltas:
        return None
    if all(d == deltas[0] for d in deltas) and deltas[0] != (0, 0):
        return deltas[0]
    return None


def _make_growth_step(grow_r: int, grow_c: int) -> Callable[[Grid], Grid]:
    """Return a callable that grows the grid by (grow_r, grow_c)."""
    def grow(grid: Grid) -> Grid:
        old_r, old_c = grid.shape
        new_r = max(1, old_r + grow_r)
        new_c = max(1, old_c + grow_c)
        result = np.zeros((new_r, new_c), dtype=np.uint8)
        # Copy old content into top-left
        copy_r = min(old_r, new_r)
        copy_c = min(old_c, new_c)
        result[:copy_r, :copy_c] = grid[:copy_r, :copy_c]
        return result
    return grow


def _detect_pixel_diff(demos: tuple[DemoPair, ...]) -> Callable[[Grid], Grid] | None:
    """Detect a consistent pixel-level transformation across demos.

    For each demo, computes the diff between input and output (where they
    have the same shape). If the diff pattern is consistent across all
    demos, returns a callable that applies that diff.
    """
    if not demos:
        return None

    # All demos must have same-shape input/output for this approach
    for demo in demos:
        if demo.input.shape != demo.output.shape:
            return None

    # Compute the color transformation at each cell for the first demo
    ref_inp = demos[0].input
    ref_out = demos[0].output
    rows, cols = ref_inp.shape

    # Build a position-independent color transform map from first demo
    color_transform: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            ci = int(ref_inp[r, c])
            co = int(ref_out[r, c])
            if ci in color_transform:
                if color_transform[ci] != co:
                    # Inconsistent within a single demo - not a simple color map
                    return None
            else:
                color_transform[ci] = co

    # Verify this color transform works for all other demos
    for demo in demos[1:]:
        inp, out = demo.input, demo.output
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ci = int(inp[r, c])
                co = int(out[r, c])
                expected = color_transform.get(ci)
                if expected is not None and expected != co:
                    return None
                color_transform.setdefault(ci, co)

    def apply_color_transform(grid: Grid) -> Grid:
        result = grid.copy()
        for ci, co in color_transform.items():
            result[grid == ci] = co
        return result

    return apply_color_transform


def _step_from_color_map(ctx: TaskContext) -> Callable[[Grid], Grid]:
    """Build a step function from an inferred color map, falling back to identity."""
    cmap = _infer_map(ctx, 0, 0)
    if cmap:
        def apply_map(grid: Grid) -> Grid:
            result = grid.copy()
            for ci, co in cmap.items():
                result[grid == ci] = co
            return result
        return apply_map

    # True fallback: identity
    def identity(grid: Grid) -> Grid:
        return grid.copy()
    return identity


def _infer_iteration(ctx: TaskContext, grid: Grid) -> int:
    """Infer how many iterations of a step rule to apply.

    Strategies (in order of priority):
    1. If a growth progression is detected, the iteration count is
       demo_count + 1 (predict the next in sequence).
    2. If grid dimensions encode the iteration count (e.g., N rows -> N
       iterations), derive from the test input.
    3. Default: demo_count + 1.
    """
    n_demos = len(ctx.demos)
    if n_demos == 0:
        return 1

    # Strategy 1: check if demos show a progression where demo i corresponds
    # to i+1 iterations (or some linear function of the index)
    if n_demos >= 2:
        # Check for dimension-based progression
        growth = _detect_growth(ctx.demos)
        if growth is not None:
            # Next step in the sequence
            return n_demos + 1

    # Strategy 2: check if a grid property encodes iteration count
    if n_demos >= 2:
        # Check if output size / input size gives a consistent ratio per demo
        ratios: list[float] = []
        for i, demo in enumerate(ctx.demos):
            in_size = demo.input.shape[0] * demo.input.shape[1]
            out_size = demo.output.shape[0] * demo.output.shape[1]
            if in_size > 0:
                ratios.append(out_size / in_size)

        # If ratios form a linear sequence (1, 2, 3, ...) suggesting
        # iteration count = demo index + 1
        if len(ratios) >= 2:
            int_ratios = [round(r) for r in ratios]
            if all(int_ratios[i] == i + 1 for i in range(len(int_ratios))):
                return n_demos + 1

    # Default: next in sequence
    return n_demos + 1


def _predict_dims(ctx: TaskContext, grid: Grid) -> tuple[int, int]:
    """Predict output dimensions from demo patterns.

    Analyzes how output dimensions relate to input dimensions across demos:
    - Fixed output size: all demos produce the same output shape.
    - Multiplicative: output = k * input for some constant k.
    - Additive: output = input + c for some constant c.
    - Same as input: output shape matches input shape.

    Falls back to the input grid's own dimensions when no pattern is found.
    """
    in_rows, in_cols = int(grid.shape[0]), int(grid.shape[1])

    if not ctx.demos:
        return (in_rows, in_cols)

    # Gather (in_rows, in_cols, out_rows, out_cols) for each demo
    dims: list[tuple[int, int, int, int]] = []
    for demo in ctx.demos:
        ir = int(demo.input.shape[0])
        ic = int(demo.input.shape[1])
        outr = int(demo.output.shape[0])
        outc = int(demo.output.shape[1])
        dims.append((ir, ic, outr, outc))

    # Strategy 1: same as input (all demos have output == input dims)
    if all(ir == outr and ic == outc for ir, ic, outr, outc in dims):
        return (in_rows, in_cols)

    # Strategy 2: multiplicative relationship (output = k * input)
    #
    # In leave-one-out verification, `predict_dims` often runs with a single
    # remaining demo. A naive "fixed output size" check would always match
    # trivially in that setting and mask simple scaling rules like 2x/3x.
    # Prefer exact integer scaling before falling back to fixed-size output.
    row_ratios: list[float] = []
    col_ratios: list[float] = []
    for ir, ic, outr, outc in dims:
        if ir > 0:
            row_ratios.append(outr / ir)
        if ic > 0:
            col_ratios.append(outc / ic)

    if row_ratios and col_ratios:
        if _all_close(row_ratios) and _all_close(col_ratios):
            kr = row_ratios[0]
            kc = col_ratios[0]
            if len(dims) >= 2 or (_is_integer_scale(kr) and _is_integer_scale(kc)):
                pred_r = round(in_rows * kr)
                pred_c = round(in_cols * kc)
                return (max(1, pred_r), max(1, pred_c))

    # Strategy 3: additive relationship (output = input + c)
    row_diffs = [outr - ir for ir, _, outr, _ in dims]
    col_diffs = [outc - ic for _, ic, _, outc in dims]

    if len(dims) >= 2 and _all_equal(row_diffs) and _all_equal(col_diffs):
        pred_r = in_rows + row_diffs[0]
        pred_c = in_cols + col_diffs[0]
        return (max(1, pred_r), max(1, pred_c))

    # Strategy 4: fixed output size (all demos have the same output dims)
    out_dims_set = {(outr, outc) for _, _, outr, outc in dims}
    if len(out_dims_set) == 1:
        return out_dims_set.pop()

    # Fallback: use the input dimensions
    return (in_rows, in_cols)


def _all_close(values: list[float], tol: float = 1e-9) -> bool:
    """Check if all values in a list are approximately equal."""
    if not values:
        return True
    ref = values[0]
    return all(abs(v - ref) < tol for v in values)


def _all_equal(values: list[int]) -> bool:
    """Check if all integer values are identical."""
    if not values:
        return True
    return all(v == values[0] for v in values)


def _is_integer_scale(value: float, tol: float = 1e-9) -> bool:
    """Return True when a scale factor is a positive integer."""
    if value <= 0:
        return False
    return abs(value - round(value)) < tol


def _disambiguate(
    ctx: TaskContext, predicates: list[Any]
) -> Callable[[Any], bool]:
    """Choose the predicate most consistent across all demos.

    For each candidate predicate (Object -> Bool), tests it against the
    objects extracted from every demo's input and output grids. Scores
    each predicate by how consistently it separates "kept/modified"
    objects from "removed" objects across demos.

    If no demo pairs are available or no predicates are provided, returns
    a predicate that always returns True.

    Parameters
    ----------
    ctx : TaskContext
        The task context with demo pairs.
    predicates : list[Any]
        A list of callables, each with signature (ObjectNode) -> bool.

    Returns
    -------
    Callable[[Any], bool]
        The predicate with the highest consistency score.
    """
    if not predicates:
        def always_true(x: Any) -> bool:
            return True
        return always_true

    if not ctx.demos:
        return predicates[0]

    # For each demo, extract input objects and compute which ones survive
    # in the output (via delta). Then score each predicate on how well
    # it predicts "survives" vs "removed".
    demo_data: list[tuple[list[ObjectNode], set[int]]] = []
    for demo in ctx.demos:
        try:
            sg_in = extract(demo.input)
            sg_out = extract(demo.output)
        except Exception:
            continue

        # Determine which input objects "survive" in the output by
        # checking pixel overlap with output objects
        survived_ids: set[int] = set()
        out_pixels: set[tuple[int, int]] = set()
        for obj in sg_out.objects:
            bx, by, bw, bh = obj.bbox
            for r in range(bh):
                for c in range(bw):
                    if obj.mask[r, c]:
                        out_pixels.add((by + r, bx + c))

        for obj in sg_in.objects:
            bx, by, bw, bh = obj.bbox
            overlap = 0
            total = 0
            for r in range(bh):
                for c in range(bw):
                    if obj.mask[r, c]:
                        total += 1
                        if (by + r, bx + c) in out_pixels:
                            overlap += 1
            if total > 0 and overlap > total * 0.3:
                survived_ids.add(obj.id)

        demo_data.append((list(sg_in.objects), survived_ids))

    if not demo_data:
        return predicates[0]

    # Score each predicate
    best_pred = predicates[0]
    best_score = -1

    for pred in predicates:
        score = 0
        for objects, survived_ids in demo_data:
            for obj in objects:
                try:
                    pred_result = pred(obj)
                except Exception:
                    continue
                actually_survived = obj.id in survived_ids
                if pred_result == actually_survived:
                    score += 1
        if score > best_score:
            best_score = score
            best_pred = pred

    return best_pred


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "demo_count",
    OpSignature(params=(("ctx", Type.TASK_CTX),), return_type=Type.INT),
    _demo_count,
)

register(
    "demo_at",
    OpSignature(
        params=(("ctx", Type.TASK_CTX), ("idx", Type.INT)),
        return_type=Type.PAIR,
    ),
    _demo_at,
)

register(
    "infer_step",
    OpSignature(params=(("ctx", Type.TASK_CTX),), return_type=Type.GRID_TRANSFORM),
    _infer_step,
)

register(
    "infer_map",
    OpSignature(
        params=(
            ("ctx", Type.TASK_CTX),
            ("prop_in", Type.INT),
            ("prop_out", Type.INT),
        ),
        return_type=Type.COLOR_MAP,
    ),
    _infer_map,
)

register(
    "disambiguate",
    OpSignature(
        params=(("ctx", Type.TASK_CTX), ("predicates", Type.OBJECT_LIST)),
        return_type=Type.PREDICATE,
    ),
    _disambiguate,
)

register(
    "infer_iteration",
    OpSignature(
        params=(("ctx", Type.TASK_CTX), ("grid", Type.GRID)),
        return_type=Type.INT,
    ),
    _infer_iteration,
)

register(
    "predict_dims",
    OpSignature(
        params=(("ctx", Type.TASK_CTX), ("grid", Type.GRID)),
        return_type=Type.DIMS,
    ),
    _predict_dims,
)
