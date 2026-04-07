"""Perception layer: report raw structural facts about a grid.

No role labels. No interpretation. Just facts.
Objects, their properties, pairwise relationships, group statistics.
Ordered by size (largest first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from scipy import ndimage

from aria.types import Grid


# ---------------------------------------------------------------------------
# Object facts
# ---------------------------------------------------------------------------

@dataclass
class ObjFact:
    """Raw facts about one object."""
    oid: int
    color: int
    row: int
    col: int
    height: int
    width: int
    size: int                     # pixel count
    mask: np.ndarray              # bool within bbox

    # Shape properties
    is_rectangular: bool          # mask fills entire bbox
    is_square: bool
    is_line: bool                 # height==1 or width==1
    aspect_ratio: float           # width / height

    # Uniqueness
    n_same_color: int             # how many objects share this color
    n_same_size: int              # how many objects share this pixel count
    n_same_shape: int             # how many objects have identical mask shape

    # Position relative to grid
    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool
    center_row: float
    center_col: float


# ---------------------------------------------------------------------------
# Pair facts
# ---------------------------------------------------------------------------

@dataclass
class PairFact:
    """Raw facts about a pair of objects."""
    oid_a: int
    oid_b: int
    # Spatial
    row_gap: int                  # b.row - a.row
    col_gap: int                  # b.col - a.col
    distance: float               # euclidean between centers
    same_row: bool
    same_col: bool
    aligned_h: bool               # top or bottom edges align
    aligned_v: bool               # left or right edges align
    adjacent: bool                # 4-connected touching
    a_contains_b: bool
    b_contains_a: bool
    # Color/shape
    same_color: bool
    same_shape: bool              # identical mask
    same_size: bool


# ---------------------------------------------------------------------------
# Grid facts
# ---------------------------------------------------------------------------

@dataclass
class SeparatorFact:
    """A row or column that spans the full grid width/height in one color."""
    axis: str                     # "row" or "col"
    index: int                    # which row/col
    color: int
    # Regions defined by this separator
    region_before: tuple[int, int, int, int]  # (r0, c0, r1, c1) above/left
    region_after: tuple[int, int, int, int]   # below/right


@dataclass
class RegionFact:
    """A rectangular region of the grid (defined by separators, borders, etc.)."""
    r0: int
    c0: int
    r1: int  # inclusive
    c1: int  # inclusive
    role: str                    # "data", "answer", "full"
    # Color stats within this region
    pixel_color_counts: dict[int, int]  # color -> pixel count (excluding bg)
    most_common_color: int       # most frequent non-bg color (-1 if empty)
    center_row: int
    center_col: int


@dataclass
class GridFacts:
    """All perceivable facts about one grid."""
    rows: int
    cols: int
    bg: int
    n_objects: int
    objects: list[ObjFact]        # sorted by size descending
    pairs: list[PairFact]
    # Group statistics
    color_counts: dict[int, int]  # color -> n objects of that color
    size_counts: dict[int, int]   # size -> n objects of that size
    unique_colors: int            # number of distinct object colors
    has_symmetric_h: bool         # grid has horizontal symmetry
    has_symmetric_v: bool         # grid has vertical symmetry
    # Separators and regions
    separators: list[SeparatorFact] = field(default_factory=list)
    regions: list[RegionFact] = field(default_factory=list)
    # Derived dimension candidates: named values that could be output dimensions
    # Each is (name, value) — the output size inducer checks which names
    # consistently match output dimensions across demos
    dim_candidates: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def perceive(grid: Grid) -> GridFacts:
    """Extract all structural facts from a grid."""
    bg = _detect_bg(grid)
    rows, cols = grid.shape
    raw_objs = _extract_objects(grid, bg)

    # Compute group statistics for uniqueness
    color_counts = Counter(o['color'] for o in raw_objs)
    size_counts = Counter(o['size'] for o in raw_objs)
    shape_groups = _group_by_shape(raw_objs)

    # Build ObjFacts sorted by size descending
    obj_facts = []
    for i, o in enumerate(sorted(raw_objs, key=lambda x: -x['size'])):
        mask = o['mask']
        h, w = mask.shape
        obj_facts.append(ObjFact(
            oid=i,
            color=o['color'],
            row=o['row'], col=o['col'],
            height=h, width=w,
            size=o['size'],
            mask=mask,
            is_rectangular=bool(np.all(mask)),
            is_square=(h == w),
            is_line=(h == 1 or w == 1),
            aspect_ratio=w / max(1, h),
            n_same_color=color_counts[o['color']],
            n_same_size=size_counts[o['size']],
            n_same_shape=shape_groups.get(_shape_key(mask), 1),
            touches_top=(o['row'] == 0),
            touches_bottom=(o['row'] + h >= rows),
            touches_left=(o['col'] == 0),
            touches_right=(o['col'] + w >= cols),
            center_row=o['row'] + h / 2,
            center_col=o['col'] + w / 2,
        ))

    # Build PairFacts (limit to avoid O(n^2) explosion)
    pairs = []
    n = len(obj_facts)
    for i in range(min(n, 30)):
        for j in range(i + 1, min(n, 30)):
            a, b = obj_facts[i], obj_facts[j]
            pairs.append(_make_pair(a, b, grid.shape))

    # Grid-level
    has_h_sym = bool(np.array_equal(grid, grid[:, ::-1]))
    has_v_sym = bool(np.array_equal(grid, grid[::-1, :]))

    # Separators and regions
    separators = _detect_separators(grid, bg)
    regions = _extract_regions(grid, bg, separators)

    return GridFacts(
        rows=rows, cols=cols, bg=bg,
        n_objects=len(obj_facts),
        objects=obj_facts,
        pairs=pairs,
        color_counts=dict(color_counts),
        size_counts=dict(size_counts),
        unique_colors=len(color_counts),
        has_symmetric_h=has_h_sym,
        has_symmetric_v=has_v_sym,
        separators=separators,
        regions=regions,
        dim_candidates=_compute_dim_candidates(
            rows, cols, bg, obj_facts, separators, regions, color_counts, grid),
    )


# ---------------------------------------------------------------------------
# Dimension candidates: all perception-derived values that could be output dims
# ---------------------------------------------------------------------------

def _compute_dim_candidates(rows, cols, bg, objects, separators, regions, color_counts, grid=None):
    """Compute all named values that could plausibly be an output dimension."""
    d = {}

    # Grid dimensions
    d['rows'] = rows
    d['cols'] = cols
    d['rows*2'] = rows * 2
    d['cols*2'] = cols * 2
    d['rows*3'] = rows * 3
    d['cols*3'] = cols * 3
    d['rows//2'] = rows // 2
    d['cols//2'] = cols // 2
    d['rows//3'] = rows // 3
    d['cols//3'] = cols // 3
    d['rows-1'] = rows - 1
    d['cols-1'] = cols - 1
    d['rows-2'] = max(0, rows - 2)
    d['cols-2'] = max(0, cols - 2)
    d['rows-3'] = max(0, rows - 3)
    d['cols-3'] = max(0, cols - 3)
    d['rows+1'] = rows + 1
    d['cols+1'] = cols + 1
    d['rows+2'] = rows + 2
    d['cols+2'] = cols + 2
    d['rows+3'] = rows + 3
    d['cols+3'] = cols + 3
    d['rows+4'] = rows + 4
    d['cols+4'] = cols + 4
    d['rows*rows'] = rows * rows
    d['cols*cols'] = cols * cols
    d['rows*cols'] = rows * cols
    if rows != cols:
        d['min_dim'] = min(rows, cols)
        d['max_dim'] = max(rows, cols)
        d['min_dim*min_dim'] = min(rows, cols) ** 2

    # Object counts
    n = len(objects)
    d['n_objects'] = n
    d['n_objects+1'] = n + 1
    d['n_colors'] = len(color_counts)

    if objects:
        sizes = [o.size for o in objects]
        heights = [o.height for o in objects]
        widths = [o.width for o in objects]

        d['max_obj_size'] = max(sizes)
        d['min_obj_size'] = min(sizes)
        d['max_obj_height'] = max(heights)
        d['max_obj_width'] = max(widths)
        d['min_obj_height'] = min(heights)
        d['min_obj_width'] = min(widths)
        d['sum_obj_heights'] = sum(heights)
        d['sum_obj_widths'] = sum(widths)

        # Largest object dimensions
        largest = objects[0]  # already sorted by size desc
        d['largest_height'] = largest.height
        d['largest_width'] = largest.width

        # Smallest object dimensions
        smallest = min(objects, key=lambda o: o.size)
        d['smallest_height'] = smallest.height
        d['smallest_width'] = smallest.width

        # Integer sqrt of max object size
        d['sqrt_max_obj'] = int(max(sizes) ** 0.5)

        # Number of singletons
        d['n_singletons'] = sum(1 for o in objects if o.size == 1)
        d['n_non_singletons'] = sum(1 for o in objects if o.size > 1)

        # Number of rectangular (solid) objects — filters out noise
        d['n_rectangles'] = sum(1 for o in objects if o.is_rectangular)
        d['n_large_rectangles'] = sum(1 for o in objects if o.is_rectangular and o.size >= 4)

        # Linear combinations of grid and object dims
        d['rows+cols'] = rows + cols
        d['rows+cols-1'] = rows + cols - 1
        d['rows-cols'] = abs(rows - cols)
        d['2*rows-1'] = 2 * rows - 1
        d['2*cols-1'] = 2 * cols - 1
        d['2*rows+1'] = 2 * rows + 1
        d['2*cols+1'] = 2 * cols + 1
        d['rows-2'] = max(0, rows - 2)
        d['cols-2'] = max(0, cols - 2)
        d['rows//4'] = rows // 4
        d['cols//4'] = cols // 4

        # Object-derived linear
        d['max_height+max_width'] = max(heights) + max(widths)
        d['largest_h+largest_w'] = largest.height + largest.width
        d['2*largest_height'] = 2 * largest.height
        d['2*largest_width'] = 2 * largest.width
        d['largest_height-1'] = max(0, largest.height - 1)
        d['largest_width-1'] = max(0, largest.width - 1)

        # Per-color properties
        for color, count in color_counts.items():
            d[f'n_color_{color}'] = count
            # Bounding box of ALL objects of this color combined
            color_objs = [o for o in objects if o.color == color]
            if color_objs:
                min_r = min(o.row for o in color_objs)
                max_r = max(o.row + o.height - 1 for o in color_objs)
                min_c = min(o.col for o in color_objs)
                max_c = max(o.col + o.width - 1 for o in color_objs)
                d[f'color_{color}_bbox_h'] = max_r - min_r + 1
                d[f'color_{color}_bbox_w'] = max_c - min_c + 1
                d[f'color_{color}_total_px'] = sum(o.size for o in color_objs)

        # Unique shapes count
        shape_set = set()
        for o in objects:
            shape_set.add((o.height, o.width, o.mask.tobytes()))
        d['n_unique_shapes'] = len(shape_set)

        # Max/min pixel count in any single color group
        if color_counts:
            d['max_color_count'] = max(color_counts.values())
            d['min_color_count'] = min(color_counts.values())

        # Total non-bg pixels and bg pixels (direct pixel count)
        if grid is not None:
            n_nonbg_pixels = int(np.sum(grid != bg))
        else:
            n_nonbg_pixels = sum(sizes)
        n_bg_pixels = rows * cols - n_nonbg_pixels
        d['total_nonbg_pixels'] = n_nonbg_pixels
        d['n_bg_pixels'] = n_bg_pixels
        d['sqrt_total_pixels'] = int(n_nonbg_pixels ** 0.5)
        d['sqrt_bg_pixels'] = int(n_bg_pixels ** 0.5)
        d['rows*n_bg_pixels'] = rows * n_bg_pixels
        d['cols*n_bg_pixels'] = cols * n_bg_pixels
        d['rows*total_nonbg'] = rows * n_nonbg_pixels
        d['cols*total_nonbg'] = cols * n_nonbg_pixels

            # Hole count per object (bg cells inside object bbox)
        if grid is not None:
            hole_counts = []
            for obj in objects:
                sub = grid[obj.row:obj.row+obj.height, obj.col:obj.col+obj.width]
                holes = int(np.sum(sub == bg))
                hole_counts.append(holes)
            if hole_counts:
                d['total_holes'] = sum(hole_counts)
                d['max_holes'] = max(hole_counts)
                d['min_holes'] = min(hole_counts)
                # Per-rank hole count
                for rank, h in enumerate(sorted(hole_counts, reverse=True)[:3]):
                    d[f'holes_rank{rank}'] = h

        # Unique-color object bbox (object whose color appears only once)
        # Use consistent name regardless of whether there's 1 or multiple
        unique_color_objs = [o for o in objects if o.n_same_color == 1]
        if unique_color_objs:
            uc = max(unique_color_objs, key=lambda o: o.size)
            d['unique_color_obj_height'] = uc.height
            d['unique_color_obj_width'] = uc.width
            d['unique_color_obj_size'] = uc.size
            # Keep the old names too for backward compat
            if len(unique_color_objs) == 1:
                d['only_unique_color_height'] = uc.height
                d['only_unique_color_width'] = uc.width
            else:
                d['largest_unique_color_height'] = uc.height
                d['largest_unique_color_width'] = uc.width

        # Second-largest object
        if len(objects) >= 2:
            by_size = sorted(objects, key=lambda o: -o.size)
            second = by_size[1]
            d['second_largest_height'] = second.height
            d['second_largest_width'] = second.width

        # Smallest non-singleton object
        non_singletons = [o for o in objects if o.size > 1]
        if non_singletons:
            smallest_ns = min(non_singletons, key=lambda o: o.size)
            d['smallest_nonsing_height'] = smallest_ns.height
            d['smallest_nonsing_width'] = smallest_ns.width

        # Object that doesn't touch border (if exactly one)
        interior = [o for o in objects if not (o.touches_top or o.touches_bottom or o.touches_left or o.touches_right)]
        if len(interior) == 1:
            d['interior_obj_height'] = interior[0].height
            d['interior_obj_width'] = interior[0].width
        elif interior:
            biggest_interior = max(interior, key=lambda o: o.size)
            d['biggest_interior_height'] = biggest_interior.height
            d['biggest_interior_width'] = biggest_interior.width

        # Most-common-color object bbox
        if color_counts:
            mc_color = max(color_counts, key=color_counts.get)
            mc_objs = [o for o in objects if o.color == mc_color]
            if mc_objs:
                mc_largest = max(mc_objs, key=lambda o: o.size)
                d['most_common_color_obj_height'] = mc_largest.height
                d['most_common_color_obj_width'] = mc_largest.width

    # Rank-based: dimensions of 2nd, 3rd largest objects
        by_size = sorted(objects, key=lambda o: -o.size)
        for rank in range(min(len(by_size), 5)):
            obj = by_size[rank]
            d[f'obj_rank{rank}_height'] = obj.height
            d[f'obj_rank{rank}_width'] = obj.width
            d[f'obj_rank{rank}_size'] = obj.size

        # Arithmetic: grid_dim ± n_objects
        d['rows-n_objects'] = max(0, rows - n)
        d['cols-n_objects'] = max(0, cols - n)
        d['rows+n_objects'] = rows + n
        d['cols+n_objects'] = cols + n
        if n > 0:
            d['rows//n_objects'] = rows // n
            d['cols//n_objects'] = cols // n
        d['2*rows-n_objects'] = max(0, 2 * rows - n)
        d['2*cols-n_objects'] = max(0, 2 * cols - n)

        # Multiplicative: grid_dim * object_fact
        for fact_name, fact_val in [
            ('n_objects', n),
            ('n_colors', len(color_counts)),
            ('n_singletons', sum(1 for o in objects if o.size == 1)),
            ('n_non_singletons', sum(1 for o in objects if o.size > 1)),
            ('n_unique_shapes', len(shape_set)),
            ('max_obj_size', max(sizes)),
            ('max_obj_height', max(heights)),
            ('max_obj_width', max(widths)),
        ]:
            if fact_val > 0:
                d[f'rows*{fact_name}'] = rows * fact_val
                d[f'cols*{fact_name}'] = cols * fact_val

        # Per-color counts (useful for tasks where a specific color's count determines size)
        for color, count in color_counts.items():
            if count > 0:
                d[f'rows*n_c{color}'] = rows * count
                d[f'cols*n_c{color}'] = cols * count

    # Separator-derived
    if separators:
        row_seps = sorted([s for s in separators if s.axis == "row"], key=lambda s: s.index)
        col_seps = sorted([s for s in separators if s.axis == "col"], key=lambda s: s.index)
        d['n_row_separators'] = len(row_seps)
        d['n_col_separators'] = len(col_seps)
        d['n_separators'] = len(separators)
        if row_seps:
            d['n_row_regions'] = len(row_seps) + 1
        if col_seps:
            d['n_col_regions'] = len(col_seps) + 1
        # Partition cell dimensions
        row_bounds = [0] + [s.index + 1 for s in row_seps] + [rows]
        col_bounds = [0] + [s.index + 1 for s in col_seps] + [cols]
        cell_heights = [row_bounds[i+1] - row_bounds[i] for i in range(len(row_bounds)-1) if row_bounds[i+1] > row_bounds[i]]
        cell_widths = [col_bounds[i+1] - col_bounds[i] for i in range(len(col_bounds)-1) if col_bounds[i+1] > col_bounds[i]]
        d['n_row_cells'] = len(cell_heights)
        d['n_col_cells'] = len(cell_widths)
        if cell_heights:
            d['first_cell_height'] = cell_heights[0]
            d['max_cell_height'] = max(cell_heights)
            d['min_cell_height'] = min(cell_heights)
        if cell_widths:
            d['first_cell_width'] = cell_widths[0]
            d['max_cell_width'] = max(cell_widths)
            d['min_cell_width'] = min(cell_widths)

    # Large-object grid layout: try different counts of "top N" objects
    # and record the grid dimensions for each.
    # Filter to rectangular objects first — noise (scattered, irregular pixels)
    # pollutes the top-N and breaks clustering.
    rect_objects = [o for o in objects if o.is_rectangular]
    layout_objects = rect_objects if len(rect_objects) >= 4 else objects
    if len(layout_objects) >= 4:
        def _verify_grid(centers, target_k):
            """Check if 'centers' form exactly target_k evenly-spaced groups.

            Uses range-based threshold: between-cluster gaps must be at least
            total_range / (target_k * 2). Each cluster must have n/target_k
            elements (±1).
            """
            n = len(centers)
            if n < target_k or target_k < 1:
                return False
            sorted_c = sorted(centers)
            total_range = sorted_c[-1] - sorted_c[0]
            if total_range == 0:
                return target_k == 1
            threshold = total_range / (target_k * 2)
            # Assign to clusters by scanning sorted values
            clusters = [[sorted_c[0]]]
            for i in range(1, n):
                if sorted_c[i] - sorted_c[i-1] >= threshold:
                    clusters.append([sorted_c[i]])
                else:
                    clusters[-1].append(sorted_c[i])
            if len(clusters) != target_k:
                return False
            # Verify roughly equal cluster sizes
            sizes = [len(c) for c in clusters]
            expected = n // target_k
            return all(expected - 1 <= s <= expected + 1 for s in sizes)

        best_grid = None
        best_n = 0
        for n_top in [4, 6, 8, 9, 10, 12, 15, 16, 20, 25]:
            if n_top > len(layout_objects):
                break
            top = layout_objects[:n_top]
            row_centers = [o.center_row for o in top]
            col_centers = [o.center_col for o in top]
            # Try all factorizations of n_top
            for nr in range(2, n_top + 1):
                if n_top % nr != 0:
                    continue
                nc = n_top // nr
                if nc < 1:
                    continue
                if _verify_grid(row_centers, nr) and _verify_grid(col_centers, nc):
                    if n_top > best_n:
                        best_grid = (nr, nc)
                        best_n = n_top

        if best_grid:
            nr, nc = best_grid
            d['large_obj_row_groups'] = nr
            d['large_obj_col_groups'] = nc
            d['n_large_objects'] = best_n

    # Topology-derived (Core Knowledge: connectivity, containment)
    if grid is not None:
        from collections import deque as _deque
        from scipy import ndimage as _ndi
        # Connected bg regions (flood fill from borders vs enclosed)
        _rows, _cols = grid.shape
        reachable = np.zeros((_rows, _cols), dtype=bool)
        q = _deque()
        for r in range(_rows):
            for c in range(_cols):
                if (r == 0 or r == _rows-1 or c == 0 or c == _cols-1) and grid[r, c] == bg:
                    reachable[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<_rows and 0<=nc<_cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        enclosed_bg = (grid == bg) & ~reachable
        n_enclosed = int(np.sum(enclosed_bg))
        d['n_enclosed_bg'] = n_enclosed

        # Number of separate enclosed regions
        if n_enclosed > 0:
            labeled_enc, n_enc_regions = _ndi.label(enclosed_bg, structure=np.ones((3,3)))
            d['n_enclosed_regions'] = n_enc_regions

        # Number of separate bg regions (connected components of bg)
        bg_mask = (grid == bg).astype(np.uint8)
        _, n_bg_components = _ndi.label(bg_mask, structure=np.ones((3,3)))
        d['n_bg_components'] = n_bg_components

        # Number of connected components of non-bg (8-connectivity)
        nonbg_mask = (grid != bg).astype(np.uint8)
        _, n_nonbg_components = _ndi.label(nonbg_mask, structure=np.ones((3,3)))
        d['n_nonbg_components'] = n_nonbg_components

    # Region-derived
    if regions:
        d['n_regions'] = len(regions)
        for i, r in enumerate(regions):
            d[f'region_{i}_height'] = r.r1 - r.r0 + 1
            d[f'region_{i}_width'] = r.c1 - r.c0 + 1

    # Filter: only positive values
    return {k: v for k, v in d.items() if isinstance(v, int) and v > 0}


# ---------------------------------------------------------------------------
# Separator detection
# ---------------------------------------------------------------------------

def _detect_separators(grid, bg):
    """Find rows/columns that are entirely one non-bg color."""
    rows, cols = grid.shape
    separators = []

    for r in range(rows):
        row = grid[r, :]
        unique = np.unique(row)
        if len(unique) == 1 and int(unique[0]) != bg:
            color = int(unique[0])
            before = (0, 0, r - 1, cols - 1) if r > 0 else None
            after = (r + 1, 0, rows - 1, cols - 1) if r < rows - 1 else None
            separators.append(SeparatorFact(
                axis="row", index=r, color=color,
                region_before=before if before else (0, 0, 0, 0),
                region_after=after if after else (0, 0, 0, 0),
            ))

    for c in range(cols):
        col = grid[:, c]
        unique = np.unique(col)
        if len(unique) == 1 and int(unique[0]) != bg:
            color = int(unique[0])
            before = (0, 0, rows - 1, c - 1) if c > 0 else None
            after = (0, c + 1, rows - 1, cols - 1) if c < cols - 1 else None
            separators.append(SeparatorFact(
                axis="col", index=c, color=color,
                region_before=before if before else (0, 0, 0, 0),
                region_after=after if after else (0, 0, 0, 0),
            ))

    return separators


def _extract_regions(grid, bg, separators):
    """Extract regions defined by separators."""
    rows, cols = grid.shape
    regions = []

    if not separators:
        # No separators — whole grid is one region
        cc = _color_counts_in_region(grid, bg, 0, 0, rows - 1, cols - 1)
        mc = max(cc, key=cc.get) if cc else -1
        regions.append(RegionFact(
            0, 0, rows - 1, cols - 1, "full", cc, mc,
            center_row=rows // 2, center_col=cols // 2,
        ))
        return regions

    # For row separators: extract regions above and below
    row_seps = sorted([s for s in separators if s.axis == "row"], key=lambda s: s.index)
    if row_seps:
        # Region above first separator
        first = row_seps[0]
        if first.index > 0:
            r0, c0, r1, c1 = 0, 0, first.index - 1, cols - 1
            cc = _color_counts_in_region(grid, bg, r0, c0, r1, c1)
            mc = max(cc, key=cc.get) if cc else -1
            regions.append(RegionFact(r0, c0, r1, c1, "data", cc, mc,
                                      center_row=(r0 + r1) // 2, center_col=(c0 + c1) // 2))

        # Regions between separators
        for i in range(len(row_seps) - 1):
            r0 = row_seps[i].index + 1
            r1 = row_seps[i + 1].index - 1
            if r0 <= r1:
                cc = _color_counts_in_region(grid, bg, r0, 0, r1, cols - 1)
                mc = max(cc, key=cc.get) if cc else -1
                regions.append(RegionFact(r0, 0, r1, cols - 1, "data", cc, mc,
                                          center_row=(r0 + r1) // 2, center_col=cols // 2))

        # Region below last separator
        last = row_seps[-1]
        if last.index < rows - 1:
            r0, c0, r1, c1 = last.index + 1, 0, rows - 1, cols - 1
            cc = _color_counts_in_region(grid, bg, r0, c0, r1, c1)
            mc = max(cc, key=cc.get) if cc else -1
            # If this region is all bg, it's the "answer" region
            role = "answer" if not cc else "data"
            regions.append(RegionFact(r0, c0, r1, c1, role, cc, mc,
                                      center_row=(r0 + r1) // 2, center_col=(c0 + c1) // 2))

    # Same for col separators
    col_seps = sorted([s for s in separators if s.axis == "col"], key=lambda s: s.index)
    if col_seps and not row_seps:
        first = col_seps[0]
        if first.index > 0:
            cc = _color_counts_in_region(grid, bg, 0, 0, rows - 1, first.index - 1)
            mc = max(cc, key=cc.get) if cc else -1
            regions.append(RegionFact(0, 0, rows - 1, first.index - 1, "data", cc, mc,
                                      center_row=rows // 2, center_col=(first.index - 1) // 2))
        last = col_seps[-1]
        if last.index < cols - 1:
            cc = _color_counts_in_region(grid, bg, 0, last.index + 1, rows - 1, cols - 1)
            mc = max(cc, key=cc.get) if cc else -1
            role = "answer" if not cc else "data"
            regions.append(RegionFact(0, last.index + 1, rows - 1, cols - 1, role, cc, mc,
                                      center_row=rows // 2, center_col=(last.index + 1 + cols - 1) // 2))

    return regions


def _color_counts_in_region(grid, bg, r0, c0, r1, c1):
    """Count non-bg pixel colors in a region."""
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    counts = {}
    for v in sub.flat:
        v = int(v)
        if v != bg:
            counts[v] = counts.get(v, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_bg(grid):
    """Detect background color.

    In ARC, 0 (black) is the bg in the vast majority of tasks.
    Use 0 if it's present. Only fall back to most-common if it
    clearly dominates (covers >30% of grid and is 1.5x the next color).
    If no color dominates, return -1 (no background — all colors are content).
    """
    vals, counts = np.unique(grid, return_counts=True)
    total = grid.size

    # Prefer 0 if it appears (even if not most common)
    if 0 in vals:
        return 0

    # Find most common color
    idx = int(np.argmax(counts))
    most_common = int(vals[idx])
    most_count = int(counts[idx])

    # Must clearly dominate: >30% of grid AND at least 1.5x the next color
    sorted_counts = sorted(counts, reverse=True)
    if most_count > total * 0.3:
        if len(sorted_counts) < 2 or sorted_counts[0] >= sorted_counts[1] * 1.5:
            return most_common

    # No dominant color — no background
    return -1


def _extract_objects(grid, bg):
    objects = []
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for color in range(10):
        if color == bg:
            continue
        binary = (grid == color).astype(np.uint8)
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary, structure=struct4)
        for lid in range(1, n + 1):
            ys, xs = np.where(labeled == lid)
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            mask = (labeled[r0:r1 + 1, c0:c1 + 1] == lid)
            objects.append({
                'color': color, 'row': r0, 'col': c0,
                'size': int(mask.sum()), 'mask': mask,
            })
    return objects


def _group_by_shape(objs):
    groups = Counter()
    for o in objs:
        key = _shape_key(o['mask'])
        groups[key] += 1
    return dict(groups)


def _shape_key(mask):
    return (mask.shape, mask.tobytes())


def _make_pair(a: ObjFact, b: ObjFact, grid_shape) -> PairFact:
    rows, cols = grid_shape
    # Containment
    a_contains = (a.row <= b.row and a.col <= b.col and
                  a.row + a.height >= b.row + b.height and
                  a.col + a.width >= b.col + b.width and a.size > b.size)
    b_contains = (b.row <= a.row and b.col <= a.col and
                  b.row + b.height >= a.row + a.height and
                  b.col + b.width >= a.col + a.width and b.size > a.size)

    # Adjacency (approximate: bbox-based)
    adj = False
    if (abs(a.row - (b.row + b.height)) <= 1 or abs(b.row - (a.row + a.height)) <= 1):
        if not (a.col + a.width <= b.col or b.col + b.width <= a.col):
            adj = True
    if (abs(a.col - (b.col + b.width)) <= 1 or abs(b.col - (a.col + a.width)) <= 1):
        if not (a.row + a.height <= b.row or b.row + b.height <= a.row):
            adj = True

    # Shape match
    same_shape = (a.height == b.height and a.width == b.width and
                  np.array_equal(a.mask, b.mask))

    dist = ((a.center_row - b.center_row) ** 2 + (a.center_col - b.center_col) ** 2) ** 0.5

    return PairFact(
        oid_a=a.oid, oid_b=b.oid,
        row_gap=b.row - a.row,
        col_gap=b.col - a.col,
        distance=dist,
        same_row=(a.row == b.row),
        same_col=(a.col == b.col),
        aligned_h=(a.row == b.row or a.row + a.height == b.row + b.height),
        aligned_v=(a.col == b.col or a.col + a.width == b.col + b.width),
        adjacent=adj,
        a_contains_b=a_contains,
        b_contains_a=b_contains,
        same_color=(a.color == b.color),
        same_shape=same_shape,
        same_size=(a.size == b.size),
    )
