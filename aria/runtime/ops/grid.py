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
