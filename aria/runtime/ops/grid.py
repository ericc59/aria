"""Grid construction and composition operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from aria.types import Color, Grid, ObjectNode, Type, make_grid
from aria.runtime.ops import OpSignature, register


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _new_grid(dims: tuple[int, int], color: int) -> Grid:
    """Create a new grid of the given dimensions filled with a single color."""
    rows, cols = dims
    return make_grid(rows, cols, fill=color)


def _from_object(obj: ObjectNode) -> Grid:
    """Convert an ObjectNode into a Grid.

    True cells in the mask become the object's color; False cells become 0
    (background).
    """
    h, w = obj.mask.shape
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[obj.mask] = obj.color
    return grid


def _crop(grid: Grid, region: tuple[int, int, int, int]) -> Grid:
    """Crop a rectangular region from a grid.

    region is (x, y, w, h) where x=col, y=row.
    """
    x, y, w, h = region
    return grid[y : y + h, x : x + w].copy()


def _place_at(obj: ObjectNode, pos: tuple[int, int], grid: Grid) -> Grid:
    """Place an object at (col, row) position on a grid, returning a NEW grid.

    Only cells where the mask is True are written.
    """
    result = grid.copy()
    col, row = pos
    mask = obj.mask
    mh, mw = mask.shape
    gr, gc = result.shape

    # Clip to grid bounds.
    r_start = max(0, row)
    r_end = min(gr, row + mh)
    c_start = max(0, col)
    c_end = min(gc, col + mw)

    mr_start = r_start - row
    mr_end = r_end - row
    mc_start = c_start - col
    mc_end = c_end - col

    sub_mask = mask[mr_start:mr_end, mc_start:mc_end]
    result[r_start:r_end, c_start:c_end][sub_mask] = obj.color
    return result


def _fill_region(region: tuple[int, int, int, int], color: int, grid: Grid) -> Grid:
    """Fill a rectangular region with a color, returning a NEW grid.

    region is (x, y, w, h).
    """
    result = grid.copy()
    x, y, w, h = region
    rows, cols = result.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(cols, x + w)
    y1 = min(rows, y + h)
    if x0 < x1 and y0 < y1:
        result[y0:y1, x0:x1] = color
    return result


def _square_region(obj: ObjectNode, radius: int) -> tuple[int, int, int, int]:
    """Return a square region centered on an object's center point."""
    if radius < 0:
        raise ValueError(f"square_region: radius must be >= 0, got {radius}")
    center_x = obj.bbox[0] + (obj.bbox[2] // 2)
    center_y = obj.bbox[1] + (obj.bbox[3] // 2)
    return (center_x - radius, center_y - radius, 2 * radius + 1, 2 * radius + 1)


def _box_region(obj: ObjectNode, half_w: int, half_h: int) -> tuple[int, int, int, int]:
    """Return a rectangular region centered on an object's center point."""
    if half_w < 0 or half_h < 0:
        raise ValueError(
            f"box_region: half sizes must be >= 0, got ({half_w}, {half_h})"
        )
    center_x = obj.bbox[0] + (obj.bbox[2] // 2)
    center_y = obj.bbox[1] + (obj.bbox[3] // 2)
    return (center_x - half_w, center_y - half_h, 2 * half_w + 1, 2 * half_h + 1)


def _embed(small: Grid, large: Grid, pos: tuple[int, int]) -> Grid:
    """Embed a smaller grid into a larger grid at (col, row) position.

    Non-zero cells in `small` overwrite cells in `large`.
    Returns a NEW grid.
    """
    result = large.copy()
    col, row = pos
    sh, sw = small.shape
    lr, lc = result.shape

    r_start = max(0, row)
    r_end = min(lr, row + sh)
    c_start = max(0, col)
    c_end = min(lc, col + sw)

    sr_start = r_start - row
    sr_end = r_end - row
    sc_start = c_start - col
    sc_end = c_end - col

    sub_small = small[sr_start:sr_end, sc_start:sc_end]
    nonzero = sub_small != 0
    result[r_start:r_end, c_start:c_end][nonzero] = sub_small[nonzero]
    return result


def _stack_h(a, b) -> Grid:
    """Horizontally stack two grids (side by side). Heights must match."""
    from aria.runtime.ops.compat import coerce_to_grid
    if not isinstance(a, np.ndarray):
        a = coerce_to_grid(a) or a
    if not isinstance(b, np.ndarray):
        b = coerce_to_grid(b) or b
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    if a.shape[0] != b.shape[0]:
        h = max(a.shape[0], b.shape[0])
        if a.shape[0] < h:
            a = np.pad(a, ((0, h - a.shape[0]), (0, 0)), constant_values=0).astype(np.uint8)
        if b.shape[0] < h:
            b = np.pad(b, ((0, h - b.shape[0]), (0, 0)), constant_values=0).astype(np.uint8)
    return np.hstack([a, b]).astype(np.uint8)


def _stack_v(a, b) -> Grid:
    """Vertically stack two grids (top and bottom). Widths must match."""
    from aria.runtime.ops.compat import coerce_to_grid
    if not isinstance(a, np.ndarray):
        a = coerce_to_grid(a) or a
    if not isinstance(b, np.ndarray):
        b = coerce_to_grid(b) or b
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    if a.shape[1] != b.shape[1]:
        # Pad the narrower one with zeros
        w = max(a.shape[1], b.shape[1])
        if a.shape[1] < w:
            a = np.pad(a, ((0, 0), (0, w - a.shape[1])), constant_values=0).astype(np.uint8)
        if b.shape[1] < w:
            b = np.pad(b, ((0, 0), (0, w - b.shape[1])), constant_values=0).astype(np.uint8)
    return np.vstack([a, b]).astype(np.uint8)


def _overlay(top, bottom) -> Grid:
    """Overlay `top` onto `bottom`. Non-zero cells in `top` overwrite.

    Accepts Grid or ObjectSet/ObjectNode (auto-coerced to Grid).
    If shapes differ, `top` is placed at (0,0) of `bottom`.
    """
    from aria.runtime.ops.compat import coerce_to_grid

    if not isinstance(top, np.ndarray):
        coerced = coerce_to_grid(top)
        if coerced is not None:
            top = coerced
        else:
            raise ValueError(f"overlay: cannot coerce top to Grid, got {type(top).__name__}")
    if not isinstance(bottom, np.ndarray):
        coerced = coerce_to_grid(bottom)
        if coerced is not None:
            bottom = coerced
        else:
            raise ValueError(f"overlay: cannot coerce bottom to Grid, got {type(bottom).__name__}")

    if top.shape != bottom.shape:
        # Embed top into bottom at (0,0) instead of crashing
        result = bottom.copy()
        th, tw = top.shape
        bh, bw = bottom.shape
        h = min(th, bh)
        w = min(tw, bw)
        sub = top[:h, :w]
        nonzero = sub != 0
        result[:h, :w][nonzero] = sub[nonzero]
        return result

    # Same shape — standard overlay
    result = bottom.copy()
    nonzero = top != 0
    result[nonzero] = top[nonzero]
    return result


def _apply_color_map(mapping: dict[int, int], grid: Grid) -> Grid:
    """Remap colors in a grid according to a mapping dict. Returns NEW grid."""
    result = grid.copy()
    for old_color, new_color in mapping.items():
        result[grid == old_color] = new_color
    return result


def _fill_cells(grid: Grid, values: list) -> Grid:
    """Fill grid cells with values.

    If values is a flat list of ints, fills in row-major order.
    If values is a list of (row, col, color) tuples, fills at those positions.
    Returns NEW grid.
    """
    result = grid.copy()
    if not values:
        return result

    if isinstance(values[0], (tuple, list)) and len(values[0]) == 3:
        for row, col, color in values:
            if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
                result[row, col] = int(color)
    else:
        rows, cols = result.shape
        for i, color in enumerate(values):
            if i >= rows * cols:
                break
            r, c = divmod(i, cols)
            result[r, c] = int(color)
    return result


# ---------------------------------------------------------------------------
# Grid-level transforms (the model keeps wanting these)
# ---------------------------------------------------------------------------


def _rotate_grid(degrees: int, grid: Grid) -> Grid:
    """Rotate a grid by 90, 180, or 270 degrees clockwise."""
    k = {90: 3, 180: 2, 270: 1}.get(degrees)  # np.rot90 is counter-clockwise
    if k is None:
        raise ValueError(f"rotate_grid: degrees must be 90, 180, or 270, got {degrees}")
    return np.rot90(grid, k=k).astype(np.uint8).copy()


def _reflect_grid(axis: int, grid: Grid) -> Grid:
    """Reflect a grid. axis: 0=HORIZONTAL (flip rows), 1=VERTICAL (flip cols),
    2=DIAG_MAIN (transpose), 3=DIAG_ANTI."""
    from aria.types import Axis
    if isinstance(axis, Axis):
        axis = {Axis.HORIZONTAL: 0, Axis.VERTICAL: 1,
                Axis.DIAG_MAIN: 2, Axis.DIAG_ANTI: 3}[axis]
    if axis == 0:
        return np.flipud(grid).astype(np.uint8).copy()
    if axis == 1:
        return np.fliplr(grid).astype(np.uint8).copy()
    if axis == 2:
        return grid.T.astype(np.uint8).copy()
    if axis == 3:
        return np.flip(grid.T).astype(np.uint8).copy()
    raise ValueError(f"reflect_grid: unknown axis {axis}")


def _transpose_grid(grid: Grid) -> Grid:
    """Transpose a grid (swap rows and columns)."""
    return grid.T.astype(np.uint8).copy()


def _tile_grid(grid: Grid, rows: int, cols: int) -> Grid:
    """Tile a grid into a rows x cols arrangement."""
    return np.tile(grid, (rows, cols)).astype(np.uint8)


def _upscale_grid(grid: Grid, factor: int) -> Grid:
    """Scale up a grid by repeating each pixel factor x factor times."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1).astype(np.uint8)


def _fill_enclosed(grid: Grid, fill_color: int) -> Grid:
    """Fill all enclosed (non-reachable-from-border) background regions.

    Flood-fills from all border cells of the background color. Whatever
    remains as background after that is enclosed and gets filled.
    """
    from collections import deque

    result = grid.copy()
    rows, cols = result.shape
    if rows == 0 or cols == 0:
        return result

    # Detect background as most frequent color
    unique, counts = np.unique(grid, return_counts=True)
    bg = int(unique[np.argmax(counts)])

    # BFS from all border cells that are bg color — mark as "exterior"
    visited = np.zeros_like(grid, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Any bg cell NOT visited is enclosed
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == bg and not visited[r, c]:
                result[r, c] = fill_color

    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(
    "rotate_grid",
    OpSignature(params=(("degrees", Type.INT), ("grid", Type.GRID)), return_type=Type.GRID),
    _rotate_grid,
)

register(
    "reflect_grid",
    OpSignature(params=(("axis", Type.AXIS), ("grid", Type.GRID)), return_type=Type.GRID),
    _reflect_grid,
)

register(
    "transpose_grid",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _transpose_grid,
)

register(
    "tile_grid",
    OpSignature(params=(("grid", Type.GRID), ("rows", Type.INT), ("cols", Type.INT)), return_type=Type.GRID),
    _tile_grid,
)

register(
    "upscale_grid",
    OpSignature(params=(("grid", Type.GRID), ("factor", Type.INT)), return_type=Type.GRID),
    _upscale_grid,
)

register(
    "fill_enclosed",
    OpSignature(params=(("grid", Type.GRID), ("fill_color", Type.COLOR)), return_type=Type.GRID),
    _fill_enclosed,
)

register(
    "new_grid",
    OpSignature(
        params=(("dims", Type.DIMS), ("color", Type.COLOR)),
        return_type=Type.GRID,
    ),
    _new_grid,
)

register(
    "from_object",
    OpSignature(params=(("obj", Type.OBJECT),), return_type=Type.GRID),
    _from_object,
)

register(
    "crop",
    OpSignature(
        params=(("grid", Type.GRID), ("region", Type.REGION)),
        return_type=Type.GRID,
    ),
    _crop,
)

register(
    "place_at",
    OpSignature(
        params=(("obj", Type.OBJECT), ("pos", Type.DIMS), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _place_at,
)

register(
    "fill_region",
    OpSignature(
        params=(("region", Type.REGION), ("color", Type.COLOR), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _fill_region,
)

register(
    "square_region",
    OpSignature(
        params=(("obj", Type.OBJECT), ("radius", Type.INT)),
        return_type=Type.REGION,
    ),
    _square_region,
)

register(
    "box_region",
    OpSignature(
        params=(("obj", Type.OBJECT), ("half_w", Type.INT), ("half_h", Type.INT)),
        return_type=Type.REGION,
    ),
    _box_region,
)

register(
    "embed",
    OpSignature(
        params=(("small", Type.GRID), ("large", Type.GRID), ("pos", Type.DIMS)),
        return_type=Type.GRID,
    ),
    _embed,
)

register(
    "stack_h",
    OpSignature(
        params=(("a", Type.GRID), ("b", Type.GRID)),
        return_type=Type.GRID,
    ),
    _stack_h,
)

register(
    "stack_v",
    OpSignature(
        params=(("a", Type.GRID), ("b", Type.GRID)),
        return_type=Type.GRID,
    ),
    _stack_v,
)

register(
    "overlay",
    OpSignature(
        params=(("top", Type.GRID), ("bottom", Type.GRID)),
        return_type=Type.GRID,
    ),
    _overlay,
)

register(
    "apply_color_map",
    OpSignature(
        params=(("mapping", Type.COLOR_MAP), ("grid", Type.GRID)),
        return_type=Type.GRID,
    ),
    _apply_color_map,
)

register(
    "fill_cells",
    OpSignature(
        params=(("grid", Type.GRID), ("values", Type.INT_LIST)),
        return_type=Type.GRID,
    ),
    _fill_cells,
)


# ---------------------------------------------------------------------------
# Grid shifting
# ---------------------------------------------------------------------------


def _shift_grid(dr: int, dc: int, fill: int, grid: Grid) -> Grid:
    """Shift all grid content by (dr, dc) rows/cols, filling vacated cells.

    Positive dr = shift down, positive dc = shift right.
    Content that shifts out of bounds is clipped (not wrapped).
    """
    rows, cols = grid.shape
    result = np.full_like(grid, fill)
    # Compute source and destination slices
    src_r0 = max(0, -dr)
    src_r1 = min(rows, rows - dr)
    dst_r0 = max(0, dr)
    dst_r1 = min(rows, rows + dr)
    src_c0 = max(0, -dc)
    src_c1 = min(cols, cols - dc)
    dst_c0 = max(0, dc)
    dst_c1 = min(cols, cols + dc)
    if src_r0 < src_r1 and src_c0 < src_c1:
        result[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return result


register(
    "shift_grid",
    OpSignature(
        params=(
            ("dr", Type.INT),
            ("dc", Type.INT),
            ("fill", Type.COLOR),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _shift_grid,
)


# ---------------------------------------------------------------------------
# Object reassembly
# ---------------------------------------------------------------------------


def _paint_objects(objects, grid: Grid) -> Grid:
    """Paint each object from the set onto a copy of grid at its bbox position.

    This is the generic reassembly step: after extracting and transforming
    objects, paint them back onto the grid. Non-zero pixels in the object
    overwrite the grid.
    """
    from aria.runtime.ops.compat import coerce_to_grid

    # Handle ObjectNode, set, list
    if isinstance(objects, ObjectNode):
        objects = {objects}
    elif isinstance(objects, (list, tuple)):
        objects = set(objects)
    elif not isinstance(objects, (set, frozenset)):
        coerced = coerce_to_grid(objects)
        if coerced is not None:
            # If it's already a grid, overlay
            result = grid.copy()
            mask = coerced != 0
            if coerced.shape == grid.shape:
                result[mask] = coerced[mask]
            return result
        return grid.copy()

    result = grid.copy()
    rows, cols = grid.shape
    for obj in objects:
        x, y, w, h = obj.bbox
        mask = obj.mask
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    gr = y + r
                    gc = x + c
                    if 0 <= gr < rows and 0 <= gc < cols:
                        result[gr, gc] = obj.color
    return result


register(
    "paint_objects",
    OpSignature(
        params=(
            ("objects", Type.OBJECT_SET),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _paint_objects,
)


# ---------------------------------------------------------------------------
# Pattern stamping: apply spatial patterns around matching objects
# ---------------------------------------------------------------------------


def _stamp_around(
    pred,
    radius: int,
    fill_color: int,
    grid: Grid,
    *,
    mode: str = "chebyshev",
) -> Grid:
    """For each object matching pred, fill nearby bg pixels with fill_color.

    mode controls which offsets are filled:
    - "chebyshev": all pixels within Chebyshev distance (full square minus center)
    - "cardinal": only pixels at cardinal offsets (Manhattan distance == radius, on axes)
    - "diagonal": only pixels at diagonal offsets
    """
    from aria.runtime.ops.selection import _find_objects

    result = grid.copy()
    rows, cols = grid.shape
    objects = _find_objects(grid)
    matching = [obj for obj in objects if pred(obj)]

    for obj in matching:
        cx = obj.bbox[0] + obj.bbox[2] // 2  # center col
        cy = obj.bbox[1] + obj.bbox[3] // 2  # center row

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = cy + dr, cx + dc
                if not (0 <= r < rows and 0 <= c < cols):
                    continue

                if mode == "chebyshev":
                    if max(abs(dr), abs(dc)) <= radius:
                        if int(result[r, c]) == 0 or int(grid[r, c]) == 0:
                            result[r, c] = fill_color
                elif mode == "cardinal":
                    if (dr == 0) != (dc == 0):  # exactly one is zero
                        if abs(dr) + abs(dc) <= radius:
                            if int(result[r, c]) == 0 or int(grid[r, c]) == 0:
                                result[r, c] = fill_color
                elif mode == "diagonal":
                    if abs(dr) == abs(dc) and abs(dr) <= radius:
                        if int(result[r, c]) == 0 or int(grid[r, c]) == 0:
                            result[r, c] = fill_color

    return result


def _fill_around(pred, radius: int, fill_color: int, grid: Grid) -> Grid:
    """Fill all background pixels within Chebyshev distance of matching objects."""
    return _stamp_around(pred, radius, fill_color, grid, mode="chebyshev")


def _fill_cardinal(pred, radius: int, fill_color: int, grid: Grid) -> Grid:
    """Fill background pixels at cardinal offsets from matching objects."""
    return _stamp_around(pred, radius, fill_color, grid, mode="cardinal")


def _fill_diagonal(pred, radius: int, fill_color: int, grid: Grid) -> Grid:
    """Fill background pixels at diagonal offsets from matching objects."""
    return _stamp_around(pred, radius, fill_color, grid, mode="diagonal")


register(
    "fill_around",
    OpSignature(
        params=(
            ("pred", Type.PREDICATE),
            ("radius", Type.INT),
            ("fill_color", Type.COLOR),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _fill_around,
)

register(
    "fill_cardinal",
    OpSignature(
        params=(
            ("pred", Type.PREDICATE),
            ("radius", Type.INT),
            ("fill_color", Type.COLOR),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _fill_cardinal,
)

register(
    "fill_diagonal",
    OpSignature(
        params=(
            ("pred", Type.PREDICATE),
            ("radius", Type.INT),
            ("fill_color", Type.COLOR),
            ("grid", Type.GRID),
        ),
        return_type=Type.GRID,
    ),
    _fill_diagonal,
)


# ---------------------------------------------------------------------------
# Periodic repair: detect and fix periodic pattern violations in framed regions
# ---------------------------------------------------------------------------


def _repair_periodic(grid: Grid, axis: int, period: int) -> Grid:
    """Compatibility wrapper: repair periodic violations in framed regions.

    Delegates to the primitive pipeline:
    1. peel_frame → strip outermost frame(s)
    2. partition_grid → find sub-cells via separators
    3. repair_grid_lines → infer motif + fix mismatches per cell

    Args:
        grid: input grid
        axis: 0 for row-periodic, 1 for col-periodic
        period: hint period length

    Returns:
        repaired grid (copy; input is not modified)
    """
    result = grid.copy()
    rows, cols = result.shape

    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])

    # Use the recursive handler which composes: peel → partition → repair
    _repair_regions_recursive(result, bg, axis, period, depth=3)

    return result


def _repair_regions_recursive(
    grid: Grid, bg: int, axis: int, period: int, depth: int,
) -> None:
    """Find framed regions and repair periodic violations in-place.

    Key: only repair at the deepest framed level. If a sub-frame is found
    inside, the outer frame's interior is NOT directly repaired (the sub-frame
    handler takes care of the content).
    """
    if depth <= 0:
        return

    rows, cols = grid.shape
    if rows < 1 or cols < 3:
        return

    # Small grids (< 3 rows): no frame possible, repair content directly
    if rows < 3:
        _repair_interior_periodic(grid, axis, period)
        return

    # Check if this grid has a frame
    frame_color = _detect_frame_color(grid)
    if frame_color is not None:
        interior = grid[1:rows - 1, 1:cols - 1]
        if interior.size > 0:
            # Check if interior itself has a frame (nested)
            inner_frame = _detect_frame_color(interior)
            if inner_frame is not None and depth > 1:
                # Recurse deeper — don't repair this level's interior
                int_vals, int_counts = np.unique(interior, return_counts=True)
                int_bg = int(int_vals[int(np.argmax(int_counts))])
                _repair_regions_recursive(interior, int_bg, axis, period, depth - 1)
            else:
                # Deepest framed level — check for separator sub-regions
                had_separators = _repair_separator_regions(interior, bg, axis, period, depth - 1)
                if not had_separators:
                    # No separators found — repair content directly
                    _repair_interior_periodic(interior, axis, period)
        return  # frame path handled

    # No frame: look for separator-bounded sub-regions
    _repair_separator_regions(grid, bg, axis, period, depth)


def _detect_frame_color(grid: Grid) -> int | None:
    """Check if the grid border is a uniform single color."""
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return None

    border = set()
    for c in range(cols):
        border.add(int(grid[0, c]))
        border.add(int(grid[rows - 1, c]))
    for r in range(rows):
        border.add(int(grid[r, 0]))
        border.add(int(grid[r, cols - 1]))

    return border.pop() if len(border) == 1 else None


def _repair_interior_periodic(interior: Grid, axis: int, period: int) -> None:
    """Repair periodic violations in an interior grid (in-place).

    Delegates to _repair_grid_periodic_lines and copies results back.
    """
    repaired = _repair_grid_periodic_lines(interior, axis, period)
    interior[:] = repaired


def _repair_line_periodic(line: np.ndarray, hint_period: int) -> None:
    """Infer motif and fix violations in-place. Thin wrapper over primitives."""
    motif_info = _infer_line_motif(line, hint_period)
    if motif_info is None:
        return
    pattern, period, violations, lo, hi = motif_info
    for i in violations:
        line[lo + i] = pattern[i % period]


def _repair_separator_regions(
    grid: Grid, bg: int, axis: int, period: int, depth: int,
) -> bool:
    """Find sub-regions bounded by separator rows AND cols, repair each.

    Partitions the grid into a 2D grid of sub-cells using the finest
    row and column separators, then recurses into each cell independently.
    Returns True if any separator-bounded sub-regions were found.
    """
    rows, cols = grid.shape

    # Find row separators: full rows of a single color
    row_seps = _find_best_separators_axis(grid, "row")
    # Find column separators: full columns of a single color
    col_seps = _find_best_separators_axis(grid, "col")

    if not row_seps and not col_seps:
        return False

    # Build row intervals
    if row_seps:
        row_intervals = []
        for i in range(len(row_seps) - 1):
            r0 = row_seps[i] + 1
            r1 = row_seps[i + 1]
            if r1 > r0:
                row_intervals.append((r0, r1))
    else:
        row_intervals = [(0, rows)]

    # Build column intervals
    if col_seps:
        col_intervals = []
        for i in range(len(col_seps) - 1):
            c0 = col_seps[i] + 1
            c1 = col_seps[i + 1]
            if c1 > c0:
                col_intervals.append((c0, c1))
    else:
        col_intervals = [(0, cols)]

    found = False
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            sub = grid[r0:r1, c0:c1]
            if sub.size > 0:
                found = True
                sub_vals, sub_counts = np.unique(sub, return_counts=True)
                sub_bg = int(sub_vals[int(np.argmax(sub_counts))])
                _repair_regions_recursive(sub, sub_bg, axis, period, depth - 1)

    return found


def _find_best_separators_axis(grid: Grid, axis: str) -> list[int]:
    """Find the finest set of full-line separators on the given axis.

    axis="row": find rows where all values are the same color (needs ≥2 cols).
    axis="col": find columns where all values are the same color (needs ≥2 rows).
    Returns the separator indices for the color that creates the most sub-regions.
    """
    rows, cols = grid.shape
    separators: dict[int, list[int]] = {}

    if axis == "row":
        if cols < 2:
            return []  # need at least 2 cols for a meaningful row separator
        for r in range(rows):
            vals = set(int(grid[r, c]) for c in range(cols))
            if len(vals) == 1:
                separators.setdefault(vals.pop(), []).append(r)
    else:
        if rows < 2:
            return []  # need at least 2 rows for a meaningful col separator
        for c in range(cols):
            vals = set(int(grid[r, c]) for r in range(rows))
            if len(vals) == 1:
                separators.setdefault(vals.pop(), []).append(c)

    # Pick the color whose separators create the best partition.
    # "Best" = most sub-regions of size ≥ 2 (ignore 1-wide gaps from
    # interleaved separator colors).
    best_color = None
    best_score = 0
    for color, seps in separators.items():
        if len(seps) < 2:
            continue
        # Count sub-regions of size ≥ 2
        n_good = sum(
            1 for i in range(len(seps) - 1)
            if seps[i + 1] - seps[i] > 2  # at least 2 content lines between separators
        )
        if n_good > best_score:
            best_score = n_good
            best_color = color
        elif n_good == best_score and n_good > 0:
            # Tie-break: fewer total separators (coarser partition)
            if len(seps) < len(separators.get(best_color, [])):
                best_color = color

    if best_color is None or best_score < 1:
        return []
    return separators[best_color]


# Compatibility wrapper — delegates to the primitive pipeline
register(
    "repair_periodic",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("axis", Type.INT),
            ("period", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _repair_periodic,
)


# ---------------------------------------------------------------------------
# Explicit reusable primitives for the periodic-repair pipeline
# ---------------------------------------------------------------------------


def _peel_frame(grid: Grid) -> Grid:
    """Strip the outermost uniform-color frame and return the interior.

    If the grid has no uniform border, returns the grid unchanged.
    """
    fc = _detect_frame_color(grid)
    if fc is None:
        return grid
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return grid
    return grid[1:rows - 1, 1:cols - 1].copy()


register(
    "peel_frame",
    OpSignature(params=(("grid", Type.GRID),), return_type=Type.GRID),
    _peel_frame,
)


def _partition_grid(grid: Grid) -> list:
    """Partition a grid into sub-cells using row and column separators.

    Returns a list of (row_start, col_start, sub_grid) tuples.
    If no separators found, returns [(0, 0, grid)].
    """
    rows, cols = grid.shape
    row_seps = _find_best_separators_axis(grid, "row")
    col_seps = _find_best_separators_axis(grid, "col")

    if not row_seps and not col_seps:
        return [(0, 0, grid)]

    row_intervals = []
    if row_seps:
        for i in range(len(row_seps) - 1):
            r0, r1 = row_seps[i] + 1, row_seps[i + 1]
            if r1 > r0:
                row_intervals.append((r0, r1))
    else:
        row_intervals = [(0, rows)]

    col_intervals = []
    if col_seps:
        for i in range(len(col_seps) - 1):
            c0, c1 = col_seps[i] + 1, col_seps[i + 1]
            if c1 > c0:
                col_intervals.append((c0, c1))
    else:
        col_intervals = [(0, cols)]

    cells = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            sub = grid[r0:r1, c0:c1]
            if sub.size > 0:
                cells.append((r0, c0, sub))
    return cells


def _infer_line_motif(line: np.ndarray, hint_period: int) -> tuple | None:
    """Infer the majority-vote periodic motif of a 1D line.

    Returns (pattern, period, violations, lo, hi) or None if no
    repairable period is found.
    - pattern: the repeating unit (list of ints)
    - period: the period length
    - violations: list of content-relative indices that deviate
    - lo, hi: content region bounds (excluding border)
    """
    from collections import Counter

    n = len(line)
    if n < 3:
        return None

    # Detect border
    edge_val = int(line[0])
    start_border = 0
    while start_border < n and int(line[start_border]) == edge_val:
        start_border += 1
    if start_border >= n:
        return None

    end_val = int(line[n - 1])
    end_border = 0
    while end_border < n and int(line[n - 1 - end_border]) == end_val:
        end_border += 1

    lo = max(start_border, 1)
    hi = min(n - end_border, n - 1)
    if hi - lo < 3:
        return None

    content = line[lo:hi]
    cn = len(content)
    candidates = [hint_period] + [p for p in range(2, cn // 2 + 1) if p != hint_period]

    best = None
    for period in candidates:
        if period < 2 or period > cn // 2:
            continue
        pattern = []
        for phase in range(period):
            vals = [int(content[i]) for i in range(phase, cn, period)]
            counts = Counter(vals)
            pattern.append(counts.most_common(1)[0][0])

        violations = [i for i in range(cn) if int(content[i]) != pattern[i % period]]
        if not violations or len(violations) > cn // 3:
            continue
        if best is None or len(violations) < len(best[2]):
            best = (tuple(pattern), period, violations, lo, hi)

    return best


def _repair_line_from_motif(
    line: np.ndarray, motif: tuple, period: int,
    violations: list, lo: int,
) -> np.ndarray:
    """Apply a known motif to repair violations in a line. Returns copy."""
    result = line.copy()
    for i in violations:
        result[lo + i] = motif[i % period]
    return result


def _repair_grid_periodic_lines(grid: Grid, axis: int, hint_period: int) -> Grid:
    """Infer motif + repair mismatches per line on the given axis.

    Tries the specified axis first. If no repairs made, tries the other.
    This is the primitive that replaces the old _repair_interior_periodic.
    """
    result = grid.copy()
    rows, cols = result.shape

    def _try_axis(ax: int) -> int:
        repaired = 0
        if ax == 0:
            for r in range(rows):
                motif_info = _infer_line_motif(result[r], hint_period)
                if motif_info is not None:
                    pattern, period, violations, lo, hi = motif_info
                    for i in violations:
                        result[r, lo + i] = pattern[i % period]
                    repaired += 1
        else:
            for c in range(cols):
                col = result[:, c].copy()
                motif_info = _infer_line_motif(col, hint_period)
                if motif_info is not None:
                    pattern, period, violations, lo, hi = motif_info
                    for i in violations:
                        result[lo + i, c] = pattern[i % period]
                    repaired += 1
        return repaired

    n = _try_axis(axis)
    if n == 0:
        _try_axis(1 - axis)

    return result


register(
    "repair_grid_lines",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("axis", Type.INT),
            ("hint_period", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _repair_grid_periodic_lines,
)


def _repair_framed_lines(grid: Grid, axis: int, hint_period: int) -> Grid:
    """Peel frames, partition into cells, repair lines in each cell.

    Composed from explicit primitives:
    1. peel_frame → strip outermost frame(s) to reach content
    2. partition_grid → find sub-cells via row/column separators
    3. repair_grid_lines → infer motif + fix mismatches per cell

    The frame structure is preserved; only content within the deepest
    framed regions is modified.
    """
    result = grid.copy()
    _repair_framed_cells_recursive(result, axis, hint_period, depth=3)
    return result


def _repair_framed_cells_recursive(
    grid: Grid, axis: int, hint_period: int, depth: int,
) -> None:
    """Recursive peel→partition→repair using explicit primitives (in-place)."""
    if depth <= 0:
        return

    rows, cols = grid.shape
    if rows < 1 or cols < 3:
        return

    if rows < 3:
        # Too small for frame — repair content directly
        repaired = _repair_grid_periodic_lines(grid, axis, hint_period)
        grid[:] = repaired
        return

    # Step 1: detect and peel frame
    frame_color = _detect_frame_color(grid)
    if frame_color is not None:
        interior = grid[1:rows - 1, 1:cols - 1]
        if interior.size > 0:
            inner_frame = _detect_frame_color(interior)
            if inner_frame is not None and depth > 1:
                # Nested frame — recurse into interior
                _repair_framed_cells_recursive(interior, axis, hint_period, depth - 1)
            else:
                # Step 2: partition interior into cells
                cells = _partition_grid(interior)
                if len(cells) > 1:
                    # Step 3: repair each cell
                    for r0, c0, cell in cells:
                        repaired = _repair_grid_periodic_lines(cell, axis, hint_period)
                        cell[:] = repaired
                else:
                    # No partition — repair interior directly
                    repaired = _repair_grid_periodic_lines(interior, axis, hint_period)
                    interior[:] = repaired
        return

    # No frame: try partition at this level
    cells = _partition_grid(grid)
    if len(cells) > 1:
        for r0, c0, cell in cells:
            _repair_framed_cells_recursive(cell, axis, hint_period, depth - 1)


register(
    "repair_framed_lines",
    OpSignature(
        params=(
            ("grid", Type.GRID),
            ("axis", Type.INT),
            ("hint_period", Type.INT),
        ),
        return_type=Type.GRID,
    ),
    _repair_framed_lines,
)


# ---------------------------------------------------------------------------
# 2D motif primitives: infer tile → detect mismatch → repair
# ---------------------------------------------------------------------------


def _infer_2d_motif(
    cell: Grid,
    min_tile_h: int = 2,
    max_tile_h: int = 8,
) -> tuple[Grid, int, list[tuple[int, int]]] | None:
    """Infer a 2D tiling motif from a cell by majority vote.

    Tries vertical tile heights from min_tile_h to max_tile_h.
    For each height, stacks all non-overlapping tiles and takes
    the majority value at each (row, col) position.

    Returns (motif, tile_height, violations) or None if:
    - no tile height produces violations (cell is already perfect)
    - violations exceed 1/3 of tile area × tile count (not a real tile)

    violations is a list of (absolute_row, col) positions within the cell
    that differ from the inferred motif.
    """
    from collections import Counter

    rows, cols = cell.shape
    best_motif = None
    best_th = 0
    best_violations: list[tuple[int, int]] = []
    best_count = rows * cols  # worse than anything

    for th in range(min_tile_h, min(max_tile_h + 1, rows)):
        # Allow trimming up to th-1 trailing rows for clean divisibility
        usable_rows = (rows // th) * th
        n_tiles = usable_rows // th
        if n_tiles < 2:
            continue

        # Majority-vote motif over usable rows only
        motif = np.zeros((th, cols), dtype=np.uint8)
        agreement = np.zeros((th, cols), dtype=int)
        for r in range(th):
            for c in range(cols):
                vals = [int(cell[i * th + r, c]) for i in range(n_tiles)]
                counts = Counter(vals)
                majority_val, majority_count = counts.most_common(1)[0]
                motif[r, c] = majority_val
                agreement[r, c] = majority_count

        # Violations: only at positions where ALL other tiles unanimously agree
        # (agreement == n_tiles means the violating tile is the only outlier)
        violations = []
        for i in range(n_tiles):
            for r in range(th):
                for c in range(cols):
                    if int(cell[i * th + r, c]) != int(motif[r, c]):
                        # This tile disagrees. Check: do all OTHER tiles agree?
                        # agreement[r,c] counts total agreeing tiles.
                        # If agreement == n_tiles - 1, this IS the only outlier.
                        if agreement[r, c] == n_tiles - 1:
                            violations.append((i * th + r, c))

        if not violations:
            continue
        if len(violations) > (n_tiles * th * cols) // 3:
            continue

        if len(violations) < best_count:
            best_count = len(violations)
            best_th = th
            best_motif = motif
            best_violations = violations

    if best_motif is None:
        return None

    return best_motif, best_th, best_violations


def _repair_2d_cell(
    cell: Grid, motif: Grid, tile_h: int,
    violations: list[tuple[int, int]] | None = None,
) -> Grid:
    """Repair a cell by replacing only violation positions with motif values.

    If violations is provided, only those positions are repaired.
    Otherwise, all positions differing from the motif are repaired.
    """
    result = cell.copy()
    if violations is not None:
        for r, c in violations:
            result[r, c] = motif[r % tile_h, c]
    else:
        n_tiles = result.shape[0] // tile_h
        for i in range(n_tiles):
            for r in range(tile_h):
                for c in range(result.shape[1]):
                    result[i * tile_h + r, c] = motif[r, c]
    return result


def _repair_framed_2d_motif(grid: Grid) -> Grid:
    """2D motif repair — for cells with genuine 2D tiling structure.

    Only repairs pixels where the majority vote is unanimous except for
    the violating tile position (i.e., N-1 out of N tiles agree).
    """
    result = grid.copy()
    _repair_framed_2d_recursive(result, depth=3)
    return result


def _repair_framed_2d_recursive(grid: Grid, depth: int) -> None:
    """Recursive peel→partition→2D-motif-repair (in-place)."""
    if depth <= 0:
        return

    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return

    # Step 1: detect and peel frame
    frame_color = _detect_frame_color(grid)
    if frame_color is not None and rows >= 3 and cols >= 3:
        interior = grid[1:rows - 1, 1:cols - 1]
        if interior.size > 0:
            inner_frame = _detect_frame_color(interior)
            if inner_frame is not None and depth > 1:
                _repair_framed_2d_recursive(interior, depth - 1)
            else:
                # Step 2: partition interior
                cells = _partition_grid(interior)
                if len(cells) > 1:
                    for _, _, cell in cells:
                        _repair_cell_2d_or_1d(cell)
                else:
                    _repair_cell_2d_or_1d(interior)
        return

    # No frame: partition and recurse
    cells = _partition_grid(grid)
    if len(cells) > 1:
        for _, _, cell in cells:
            _repair_framed_2d_recursive(cell, depth - 1)


def _repair_cell_2d_or_1d(cell: Grid) -> None:
    """Try 2D motif repair first; fall back to 1D line repair.

    Strips uniform border rows/cols before motif inference. Handles
    cells that include separator/frame borders on some sides.
    """
    rows, cols = cell.shape

    # Try peeling full frame first
    fc = _detect_frame_color(cell)
    if fc is not None and rows >= 3 and cols >= 3:
        interior = cell[1:rows - 1, 1:cols - 1]
        _repair_cell_2d_or_1d(interior)
        return

    # Strip uniform top/bottom rows (partial frame from separators)
    r_lo, r_hi = 0, rows
    while r_lo < r_hi and len(set(int(cell[r_lo, c]) for c in range(cols))) == 1:
        r_lo += 1
    while r_hi > r_lo and len(set(int(cell[r_hi - 1, c]) for c in range(cols))) == 1:
        r_hi -= 1
    # Strip uniform left/right cols
    c_lo, c_hi = 0, cols
    while c_lo < c_hi and len(set(int(cell[r, c_lo]) for r in range(r_lo, r_hi))) == 1:
        c_lo += 1
    while c_hi > c_lo and len(set(int(cell[r, c_hi - 1]) for r in range(r_lo, r_hi))) == 1:
        c_hi -= 1

    if r_lo < r_hi and c_lo < c_hi and (r_lo > 0 or r_hi < rows or c_lo > 0 or c_hi < cols):
        content = cell[r_lo:r_hi, c_lo:c_hi]
        if content.size >= 4:
            _repair_cell_2d_or_1d(content)
            return

    # Try 2D motif repair — very conservative:
    # - Must have ≥4 tile repetitions
    # - Must have genuine 2D structure (≥2 distinct rows in motif)
    # - Violations must be very few (≤ 1 per tile on average)
    # - Each violation must have UNANIMOUS agreement from all other tiles
    rows_h, cols_w = cell.shape
    if rows_h >= 6 and cols_w >= 2:
        motif_info = _infer_2d_motif(cell)
        if motif_info is not None:
            motif, tile_h, violations = motif_info
            usable = (rows_h // tile_h) * tile_h
            n_tiles = usable // tile_h
            unique_rows = len(set(tuple(int(v) for v in motif[r]) for r in range(tile_h)))
            if (unique_rows >= 2
                    and n_tiles >= 4
                    and len(violations) >= 2
                    and len(violations) <= n_tiles):
                repaired = _repair_2d_cell(cell, motif, tile_h, violations)
                cell[:] = repaired


register(
    "repair_framed_2d_motif",
    OpSignature(
        params=(("grid", Type.GRID),),
        return_type=Type.GRID,
    ),
    _repair_framed_2d_motif,
)
