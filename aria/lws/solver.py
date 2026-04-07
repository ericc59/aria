"""Layered Workspace Solver — vertical slice implementation.

Stages 0-9 in one module for the first slice.
Cross-demo accumulation is central: every stage refines hypotheses
across all demos before proceeding.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy import ndimage

from aria.decomposition import detect_bg, extract_objects, RawObject
from aria.lws.types import (
    Scaffold, Workspace, PreservedSupport, OutputUnit,
    Alignment, Selection, RewriteRule, UnifiedRule, SolveResult,
)
from aria.types import Grid, DemoPair


# ===================================================================
# MAIN ENTRY
# ===================================================================

def lws_solve(
    demos: tuple[DemoPair, ...],
    task_id: str = "",
) -> SolveResult:
    """Run the layered workspace solver on one task."""
    if not demos:
        return _empty(task_id)

    trace: dict[str, Any] = {}

    # --- Stage 0: Scaffold ---
    scaffolds = [_stage0_scaffold(d) for d in demos]
    same_shape = all(s.same_shape for s in scaffolds)
    if not same_shape:
        return _empty(task_id, details={"stage": "scaffold", "reason": "diff_shape_not_yet"})
    trace["stage0"] = "same_shape"

    # --- Stage 1: Workspace init ---
    workspaces = [_stage1_workspace(d, s) for d, s in zip(demos, scaffolds)]

    # --- Stage 2: Preservation (cross-demo) ---
    preservations = [_stage2_preservation(d) for d in demos]
    avg_preserved = np.mean([p.n_preserved / (p.n_preserved + p.n_residual)
                             for p in preservations])
    trace["stage2_avg_preserved"] = f"{avg_preserved:.1%}"

    if all(p.n_residual == 0 for p in preservations):
        return _empty(task_id, details={"stage": "preservation", "reason": "identity"})

    # --- Stage 3-7: Try each rewrite strategy ---
    # Strategy dispatch: try all strategies, take first that train-verifies
    strategies = [
        _strategy_periodic_repair,
        _strategy_symmetry_repair,
        _strategy_recolor_line_to_singleton,
        _strategy_fill_enclosed_by_legend,
        _strategy_gravity_fill,
    ]

    for strategy_fn in strategies:
        result = strategy_fn(demos, preservations, task_id, trace)
        if result is not None and result.train_verified:
            return result

    # No strategy worked — return best partial
    return _empty(task_id, details={"stage": "no_strategy_matched", **trace})


# ===================================================================
# STAGE 0: SCAFFOLD
# ===================================================================

def _stage0_scaffold(demo: DemoPair) -> Scaffold:
    bg = detect_bg(demo.input)
    return Scaffold(
        output_shape=demo.output.shape,
        same_shape=demo.input.shape == demo.output.shape,
        bg=bg,
    )


# ===================================================================
# STAGE 1: WORKSPACE
# ===================================================================

def _stage1_workspace(demo: DemoPair, scaffold: Scaffold) -> Workspace:
    bg = scaffold.bg
    rows, cols = scaffold.output_shape
    canvas = np.full((rows, cols), bg, dtype=np.uint8)
    palette = set(int(v) for v in np.unique(demo.input))
    return Workspace(canvas=canvas, bg=bg, palette=palette)


# ===================================================================
# STAGE 2: PRESERVATION
# ===================================================================

def _stage2_preservation(demo: DemoPair) -> PreservedSupport:
    preserved = demo.input == demo.output
    residual = ~preserved
    return PreservedSupport(
        preserved_mask=preserved,
        residual_mask=residual,
        n_preserved=int(np.sum(preserved)),
        n_residual=int(np.sum(residual)),
    )


# ===================================================================
# STRATEGY: PERIODIC REPAIR
# ===================================================================

def _strategy_periodic_repair(
    demos: tuple[DemoPair, ...],
    preservations: list[PreservedSupport],
    task_id: str,
    trace: dict,
) -> SolveResult | None:
    """Detect and repair periodic patterns in rows or columns.

    Cross-demo accumulation: all demos must be explainable as
    periodic repair on the same axis.
    """
    for axis in ["row", "col"]:
        all_ok = True
        for di, demo in enumerate(demos):
            bg = detect_bg(demo.input)
            # Try both paths, accept whichever works
            repaired = _apply_periodic_repair(demo.input, axis, bg)
            if not np.array_equal(repaired, demo.output):
                all_ok = False
                break

        if all_ok:
            rule = UnifiedRule(
                rule=RewriteRule("periodic_repair", {"axis": axis},
                                 f"periodic {axis} repair"),
                train_verified=True,
                train_diff=0,
            )
            trace["strategy"] = f"periodic_repair_{axis}"
            return SolveResult(
                task_id=task_id,
                train_verified=True,
                test_outputs=[],
                unified_rule=rule,
                train_diff=0,
                details={"strategy": f"periodic_repair_{axis}"},
                stage_trace=trace,
            )

    return None


def _apply_periodic_repair(grid: Grid, axis: str, bg: int) -> Grid:
    """Repair periodic anomalies, respecting framed/panel structure.

    Tries recursive frame detection first. If that produces no changes,
    tries panel-based decomposition. Returns whichever produces changes.
    """
    from aria.decomposition import detect_panels, detect_framed_regions

    # Path 1: recursive frame repair
    out_frame = grid.copy()
    _recursive_frame_repair(out_frame, 0, 0, grid.shape[0], grid.shape[1], axis, bg, depth=0)
    if not np.array_equal(out_frame, grid):
        return out_frame

    # Path 2: panel-based + recursive frame on each panel
    panels = detect_panels(grid, bg)
    if panels is not None and panels.n_panels >= 2:
        out_panel = grid.copy()
        for p in panels.panels:
            panel_bg = detect_bg(grid[p.row:p.row + p.height, p.col:p.col + p.width])
            _recursive_frame_repair(out_panel, p.row, p.col, p.height, p.width,
                                    axis, panel_bg, depth=0)
        if not np.array_equal(out_panel, grid):
            return out_panel

    return grid.copy()


def _recursive_frame_repair(
    grid: Grid,
    r0: int, c0: int, h: int, w: int,
    axis: str, bg: int, depth: int,
) -> None:
    """Recursively detect frames and repair periodic interiors."""
    from aria.decomposition import detect_framed_regions

    if depth > 5 or (h < 3 and w < 3):
        return

    sub = grid[r0:r0 + h, c0:c0 + w]
    sub_bg = detect_bg(sub) if depth > 0 else bg
    framed = detect_framed_regions(sub, sub_bg)

    if framed:
        for fr in framed:
            # Recurse into the interior
            ir, ic = r0 + fr.row, c0 + fr.col
            _recursive_frame_repair(grid, ir, ic, fr.height, fr.width,
                                    axis, fr.interior_bg, depth + 1)
    else:
        # Leaf: no more frames — repair lines in this region
        region = grid[r0:r0 + h, c0:c0 + w].copy()
        repaired = _repair_interior_periodic(region, axis, bg)
        grid[r0:r0 + h, c0:c0 + w] = repaired


def _repair_interior_periodic(interior: Grid, axis: str, interior_bg: int) -> Grid:
    """Repair periodic patterns within a framed interior.

    Each row/column of the interior may have its own frame border (e.g., color 2
    at the edges). Strip that inner border, repair the periodic core, restore.
    """
    out = interior.copy()
    rows, cols = interior.shape

    if axis == "row":
        for r in range(rows):
            row = interior[r, :]
            repaired = _repair_framed_interior_line(row)
            out[r, :] = repaired
    else:
        for c in range(cols):
            col = interior[:, c]
            repaired = _repair_framed_interior_line(col)
            out[:, c] = repaired

    return out


def _repair_framed_interior_line(line: np.ndarray) -> np.ndarray:
    """Repair a line that may have inner frame borders.

    Detects if the first and last cells are a different color from the
    core, strips them, repairs the core's periodic pattern, restores.
    """
    n = len(line)
    if n <= 2:
        return line.copy()

    result = line.copy()

    # Detect inner border: if first and last cells share a color
    # that differs from the majority of interior cells
    border_color = int(line[0])
    if int(line[-1]) != border_color:
        # No consistent border — try raw repair
        return _repair_line(line)

    # Find how deep the border goes (typically 1 cell each side)
    left = 0
    while left < n and int(line[left]) == border_color:
        left += 1
    right = n - 1
    while right >= 0 and int(line[right]) == border_color:
        right -= 1

    if left > right or right - left + 1 < 2:
        return line.copy()

    # Repair the core
    core = line[left:right + 1]
    repaired_core = _repair_line(core)
    result[left:right + 1] = repaired_core

    return result


def _repair_line(line: np.ndarray) -> np.ndarray:
    """Repair a 1D line to its dominant periodic pattern.

    Only repairs if there's a clear periodic pattern with a small
    number of anomalies. Lines that are already valid (period <= n/2
    with 0 violations) are returned unchanged.
    """
    n = len(line)
    if n <= 1:
        return line.copy()

    # First pass: check if the line already has a valid period (0 violations)
    for p in range(1, n // 2 + 1):
        tile = line[:p]
        perfect = True
        for i in range(p, n, p):
            chunk = line[i:i + p]
            if not np.array_equal(chunk, tile[:len(chunk)]):
                perfect = False
                break
        if perfect:
            return line.copy()  # already periodic, no repair needed

    # Second pass: find the best small-period repair
    best_repaired = None
    best_violations = n

    for p in range(1, n // 2 + 1):
        pattern = np.zeros(p, dtype=line.dtype)
        for phase in range(p):
            vals = [int(line[i]) for i in range(phase, n, p)]
            pattern[phase] = Counter(vals).most_common(1)[0][0]

        repaired = np.array([pattern[i % p] for i in range(n)], dtype=line.dtype)
        violations = int(np.sum(repaired != line))

        if violations == 0:
            continue  # handled in first pass

        # Accept only if violations are clearly anomalous (few relative to period)
        # Require: violations < period AND violations < half the repetitions
        n_reps = n // p
        if 0 < violations < p and violations <= max(1, n_reps // 2):
            if violations < best_violations:
                best_violations = violations
                best_repaired = repaired

    if best_repaired is not None:
        return best_repaired

    return line.copy()


# ===================================================================
# STRATEGY: RECOLOR LINE TO SINGLETON
# ===================================================================

def _strategy_recolor_line_to_singleton(
    demos: tuple[DemoPair, ...],
    preservations: list[PreservedSupport],
    task_id: str,
    trace: dict,
) -> SolveResult | None:
    """Each vertical/horizontal line gets recolored to match its endpoint singleton.

    Pattern: groups of (singleton of color A) + (vertical line of color B)
    connected in a column. The line gets recolored to A.

    Cross-demo: the structural relationship must hold across all demos.
    """
    all_ok = True
    for demo in demos:
        bg = detect_bg(demo.input)
        predicted = _apply_recolor_to_endpoint(demo.input, bg)
        if not np.array_equal(predicted, demo.output):
            all_ok = False
            break

    if all_ok:
        rule = UnifiedRule(
            rule=RewriteRule("recolor_line_to_endpoint", {},
                             "recolor line segment to endpoint singleton color"),
            train_verified=True,
            train_diff=0,
        )
        trace["strategy"] = "recolor_line_to_endpoint"
        return SolveResult(
            task_id=task_id,
            train_verified=True,
            test_outputs=[],
            unified_rule=rule,
            train_diff=0,
            details={"strategy": "recolor_line_to_endpoint"},
            stage_trace=trace,
        )

    return None


def _apply_recolor_to_endpoint(grid: Grid, bg: int) -> Grid:
    """For each non-bg column segment, recolor it to match its endpoint singleton."""
    out = grid.copy()
    rows, cols = grid.shape
    objs = extract_objects(grid, bg, connectivity=4)

    # Find "vertical line groups": objects in the same column, connected vertically
    # Group objects by column position
    col_groups: dict[int, list[RawObject]] = defaultdict(list)
    for obj in objs:
        if obj.bbox_w == 1:  # single-column object
            col_groups[obj.col].append(obj)

    for col, group in col_groups.items():
        if len(group) < 2:
            continue

        # Sort by row position
        group.sort(key=lambda o: o.row)

        # Find the endpoint pattern: top object is a singleton of color A,
        # bottom is a singleton of same or different color, middle is a line of color B
        # The line gets recolored to A

        # Check: do they form a continuous column?
        top = group[0]
        bottom = group[-1]

        # The endpoint singletons determine the target color for non-matching lines
        # Pattern: singleton of color A at one end, line of color B below/above it
        # If A != B, recolor B -> A. If A == B, leave alone.
        if top.is_singleton and bottom.is_singleton and top.color == bottom.color:
            target_color = top.color
            for obj in group:
                if not (obj is top or obj is bottom):
                    if obj.color != target_color:
                        for r in range(obj.bbox_h):
                            if obj.mask[r, 0]:
                                out[obj.row + r, obj.col] = target_color

    return out


# ===================================================================
# STRATEGY: FILL ENCLOSED BY LEGEND
# ===================================================================

def _strategy_fill_enclosed_by_legend(
    demos: tuple[DemoPair, ...],
    preservations: list[PreservedSupport],
    task_id: str,
    trace: dict,
) -> SolveResult | None:
    """Fill enclosed bg cells using a legend/color-key found in the input.

    Pattern: a small region of the input serves as a legend mapping
    frame colors to fill colors. Enclosed bg cells within each frame
    get filled with the legend's mapped color.
    """
    # Try to find a consistent legend + fill across all demos
    # For now, try the simpler variant: fill enclosed with the
    # most common non-bg neighbor color (which works when there's no legend)
    all_ok = True
    for demo in demos:
        bg = detect_bg(demo.input)
        predicted = _apply_fill_enclosed_majority(demo.input, bg)
        if not np.array_equal(predicted, demo.output):
            all_ok = False
            break

    if all_ok:
        rule = UnifiedRule(
            rule=RewriteRule("fill_enclosed_majority", {},
                             "fill enclosed bg with majority neighbor color"),
            train_verified=True,
            train_diff=0,
        )
        trace["strategy"] = "fill_enclosed_majority"
        return SolveResult(
            task_id=task_id,
            train_verified=True,
            test_outputs=[],
            unified_rule=rule,
            train_diff=0,
            details={"strategy": "fill_enclosed_majority"},
            stage_trace=trace,
        )
    return None


def _apply_fill_enclosed_majority(grid: Grid, bg: int) -> Grid:
    """Fill each enclosed bg region with the most common non-bg border color."""
    from collections import deque

    out = grid.copy()
    rows, cols = grid.shape

    # Flood from border through bg
    reachable = np.zeros((rows, cols), dtype=bool)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == bg:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == bg:
                reachable[nr, nc] = True
                q.append((nr, nc))

    enclosed = (grid == bg) & (~reachable)
    if not np.any(enclosed):
        return out

    # Label enclosed regions
    labeled, n = ndimage.label(enclosed, structure=np.ones((3, 3)))
    for label_id in range(1, n + 1):
        comp = labeled == label_id
        struct4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        dilated = ndimage.binary_dilation(comp, structure=struct4)
        border = dilated & ~comp
        border_vals = grid[border]
        non_bg_vals = border_vals[border_vals != bg]
        if len(non_bg_vals) > 0:
            fill_c = int(Counter(non_bg_vals.tolist()).most_common(1)[0][0])
            out[comp] = fill_c

    return out


# ===================================================================
# STRATEGY: SYMMETRY REPAIR
# ===================================================================

def _strategy_symmetry_repair(
    demos: tuple[DemoPair, ...],
    preservations: list[PreservedSupport],
    task_id: str,
    trace: dict,
) -> SolveResult | None:
    """Repair objects to be locally symmetric.

    Structural concept: each non-bg object should be symmetric along
    some axis. Find objects that are ALMOST symmetric, identify the
    violating pixels, and repair them to restore symmetry.

    This is not a named task family — it's a reusable structural concept:
    "an object's internal pattern should be symmetric, violations are anomalies."

    Cross-demo: the same symmetry type (H, V, or both) must explain all demos.
    """
    # Try uniform symmetry first (all objects same axis/mode)
    for sym_type in ["h", "v", "hv"]:
        for prefer_bg in [True, False]:
            all_ok = True
            for demo in demos:
                bg = detect_bg(demo.input)
                repaired = _apply_symmetry_repair(demo.input, bg, sym_type, prefer_bg)
                if not np.array_equal(repaired, demo.output):
                    all_ok = False
                    break

            if all_ok:
                mode = "remove" if prefer_bg else "fill"
                rule = UnifiedRule(
                    rule=RewriteRule("symmetry_repair",
                                     {"sym_type": sym_type, "prefer_bg": prefer_bg},
                                     f"symmetry repair ({sym_type}, {mode})"),
                    train_verified=True,
                    train_diff=0,
                )
                trace["strategy"] = f"symmetry_repair_{sym_type}_{mode}"
                return SolveResult(
                    task_id=task_id,
                    train_verified=True,
                    test_outputs=[],
                    unified_rule=rule,
                    train_diff=0,
                    details={"strategy": f"symmetry_repair_{sym_type}_{mode}"},
                    stage_trace=trace,
                )

    # Try per-object best symmetry (each object picks its own axis/mode)
    for prefer_bg in [True, False]:
        all_ok = True
        for demo in demos:
            bg = detect_bg(demo.input)
            repaired = _apply_per_object_best_symmetry(demo.input, bg, prefer_bg)
            if not np.array_equal(repaired, demo.output):
                all_ok = False
                break

        if all_ok:
            mode = "remove" if prefer_bg else "fill"
            rule = UnifiedRule(
                rule=RewriteRule("symmetry_repair",
                                 {"sym_type": "per_object_best", "prefer_bg": prefer_bg},
                                 f"per-object best symmetry repair ({mode})"),
                train_verified=True,
                train_diff=0,
            )
            trace["strategy"] = f"symmetry_repair_per_object_{mode}"
            return SolveResult(
                task_id=task_id,
                train_verified=True,
                test_outputs=[],
                unified_rule=rule,
                train_diff=0,
                details={"strategy": f"symmetry_repair_per_object_{mode}"},
                stage_trace=trace,
            )

    return None


def _apply_per_object_best_symmetry(grid: Grid, bg: int, prefer_bg: bool = True) -> Grid:
    """For each object, find which symmetry axis produces the best repair."""
    out = grid.copy()
    objs = extract_objects(grid, bg, connectivity=4)

    for obj in objs:
        r0, c0 = obj.row, obj.col
        h, w = obj.bbox_h, obj.bbox_w
        sub = out[r0:r0 + h, c0:c0 + w].copy()

        # Try each symmetry type and pick the one that makes the most improvement
        best_sub = sub
        best_improvement = 0

        for sym in ["h", "v"]:
            if sym == "h":
                orig_asym = int(np.sum(sub != sub[:, ::-1]))
            else:
                orig_asym = int(np.sum(sub != sub[::-1, :]))

            # Only repair objects that are MOSTLY symmetric (few anomalies)
            # Skip if already symmetric or if too asymmetric (>25% of pixels)
            if orig_asym == 0:
                continue
            if orig_asym > max(2, obj.size // 4):
                continue

            candidate = sub.copy()
            if sym == "h":
                candidate, ch = _symmetrize_h(candidate, obj.mask, bg, obj.color, prefer_bg)
            else:
                candidate, ch = _symmetrize_v(candidate, obj.mask, bg, obj.color, prefer_bg)

            if not ch:
                continue

            if sym == "h":
                new_asym = int(np.sum(candidate != candidate[:, ::-1]))
            else:
                new_asym = int(np.sum(candidate != candidate[::-1, :]))

            improvement = orig_asym - new_asym
            if improvement > best_improvement and new_asym == 0:
                best_improvement = improvement
                best_sub = candidate

        if best_improvement > 0:
            out[r0:r0 + h, c0:c0 + w] = best_sub

    return out


def _apply_symmetry_repair(grid: Grid, bg: int, sym_type: str, prefer_bg: bool = True) -> Grid:
    """Repair each object to be symmetric along the specified axis."""
    out = grid.copy()
    objs = extract_objects(grid, bg, connectivity=4)

    for obj in objs:
        r0, c0 = obj.row, obj.col
        h, w = obj.bbox_h, obj.bbox_w
        sub = out[r0:r0 + h, c0:c0 + w].copy()
        changed = False

        if sym_type in ("h", "hv"):
            sub, ch = _symmetrize_h(sub, obj.mask, bg, obj.color, prefer_bg)
            changed = changed or ch

        if sym_type in ("v", "hv"):
            sub, ch = _symmetrize_v(sub, obj.mask, bg, obj.color, prefer_bg)
            changed = changed or ch

        if changed:
            out[r0:r0 + h, c0:c0 + w] = sub

    return out


def _symmetrize_h(sub: Grid, mask: np.ndarray, bg: int, obj_color: int,
                   prefer_bg: bool = True) -> tuple[Grid, bool]:
    """Make sub horizontally symmetric by resolving disagreeing mirror pairs."""
    h, w = sub.shape
    out = sub.copy()
    changed = False

    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            if mc <= c:
                break
            left = int(sub[r, c])
            right = int(sub[r, mc])
            if left == right:
                continue

            left_in = mask[r, c] if r < mask.shape[0] and c < mask.shape[1] else False
            right_in = mask[r, mc] if r < mask.shape[0] and mc < mask.shape[1] else False

            winner = _pick_symmetric_value(left, right, left_in, right_in, bg, obj_color, prefer_bg)
            out[r, c] = winner
            out[r, mc] = winner
            changed = True

    return out, changed


def _symmetrize_v(sub: Grid, mask: np.ndarray, bg: int, obj_color: int,
                   prefer_bg: bool = True) -> tuple[Grid, bool]:
    """Make sub vertically symmetric by resolving disagreeing mirror pairs."""
    h, w = sub.shape
    out = sub.copy()
    changed = False

    for r in range(h // 2):
        mr = h - 1 - r
        if mr <= r:
            break
        for c in range(w):
            top = int(sub[r, c])
            bot = int(sub[mr, c])
            if top == bot:
                continue

            top_in = mask[r, c] if r < mask.shape[0] and c < mask.shape[1] else False
            bot_in = mask[mr, c] if mr < mask.shape[0] and c < mask.shape[1] else False

            winner = _pick_symmetric_value(top, bot, top_in, bot_in, bg, obj_color, prefer_bg)
            out[r, c] = winner
            out[mr, c] = winner
            changed = True

    return out, changed


def _pick_symmetric_value(
    a: int, b: int,
    a_in_mask: bool, b_in_mask: bool,
    bg: int, obj_color: int,
    prefer_bg: bool = True,
) -> int:
    """Pick which value to use when symmetrizing a disagreeing pair.

    When prefer_bg=True (anomaly removal mode):
    - Prefer bg (remove the anomalous pixel)
    When prefer_bg=False (anomaly fill mode):
    - Prefer non-bg (extend the pattern)
    """
    if prefer_bg:
        # Anomaly removal: the stray pixel is the anomaly
        if a == bg and b != bg:
            return bg
        if b == bg and a != bg:
            return bg
        # Both non-bg, different: prefer obj_color
        if a == obj_color:
            return a
        if b == obj_color:
            return b
        return a
    else:
        # Anomaly fill: the missing pixel is the anomaly
        if a == bg and b != bg:
            return b
        if b == bg and a != bg:
            return a
        if a == obj_color:
            return a
        if b == obj_color:
            return b
        return a


# ===================================================================
# STRATEGY: GRAVITY FILL
# ===================================================================

def _strategy_gravity_fill(
    demos: tuple[DemoPair, ...],
    preservations: list[PreservedSupport],
    task_id: str,
    trace: dict,
) -> SolveResult | None:
    """Objects or colors "flow" in a direction to fill available space.

    Pattern: a colored region expands in one direction (typically down)
    to fill bg cells until it hits a wall or grid boundary.
    """
    for direction in ["down", "up", "left", "right"]:
        for color_mode in ["any_nonbg", "specific"]:
            all_ok = True
            for demo in demos:
                bg = detect_bg(demo.input)
                predicted = _apply_gravity(demo.input, bg, direction)
                if not np.array_equal(predicted, demo.output):
                    all_ok = False
                    break

            if all_ok:
                rule = UnifiedRule(
                    rule=RewriteRule("gravity", {"direction": direction},
                                     f"gravity {direction}"),
                    train_verified=True,
                    train_diff=0,
                )
                trace["strategy"] = f"gravity_{direction}"
                return SolveResult(
                    task_id=task_id,
                    train_verified=True,
                    test_outputs=[],
                    unified_rule=rule,
                    train_diff=0,
                    details={"strategy": f"gravity_{direction}"},
                    stage_trace=trace,
                )
    return None


def _apply_gravity(grid: Grid, bg: int, direction: str) -> Grid:
    """Apply gravity: non-bg cells fall in the given direction."""
    out = grid.copy()
    rows, cols = grid.shape

    if direction == "down":
        for c in range(cols):
            col = list(grid[:, c])
            non_bg = [v for v in col if v != bg]
            n_bg = len(col) - len(non_bg)
            out[:, c] = np.array([bg] * n_bg + non_bg, dtype=np.uint8)
    elif direction == "up":
        for c in range(cols):
            col = list(grid[:, c])
            non_bg = [v for v in col if v != bg]
            n_bg = len(col) - len(non_bg)
            out[:, c] = np.array(non_bg + [bg] * n_bg, dtype=np.uint8)
    elif direction == "right":
        for r in range(rows):
            row = list(grid[r, :])
            non_bg = [v for v in row if v != bg]
            n_bg = len(row) - len(non_bg)
            out[r, :] = np.array([bg] * n_bg + non_bg, dtype=np.uint8)
    elif direction == "left":
        for r in range(rows):
            row = list(grid[r, :])
            non_bg = [v for v in row if v != bg]
            n_bg = len(row) - len(non_bg)
            out[r, :] = np.array(non_bg + [bg] * n_bg, dtype=np.uint8)

    return out


# ===================================================================
# EXECUTION (for test)
# ===================================================================

def lws_predict(unified: UnifiedRule, test_input: Grid) -> Grid | None:
    """Execute a unified rule on a test input."""
    bg = detect_bg(test_input)
    rtype = unified.rule.rule_type

    if rtype == "periodic_repair":
        axis = unified.rule.params.get("axis", "row")
        return _apply_periodic_repair(test_input, axis, bg)
    elif rtype == "symmetry_repair":
        sym_type = unified.rule.params.get("sym_type", "h")
        prefer_bg = unified.rule.params.get("prefer_bg", True)
        if sym_type == "per_object_best":
            return _apply_per_object_best_symmetry(test_input, bg, prefer_bg)
        return _apply_symmetry_repair(test_input, bg, sym_type, prefer_bg)
    elif rtype == "recolor_line_to_endpoint":
        return _apply_recolor_to_endpoint(test_input, bg)
    elif rtype == "fill_enclosed_majority":
        return _apply_fill_enclosed_majority(test_input, bg)
    elif rtype == "gravity":
        direction = unified.rule.params.get("direction", "down")
        return _apply_gravity(test_input, bg, direction)

    return None


# ===================================================================
# HELPERS
# ===================================================================

def _empty(task_id: str, details: dict | None = None) -> SolveResult:
    return SolveResult(
        task_id=task_id,
        train_verified=False,
        test_outputs=[],
        unified_rule=None,
        train_diff=0,
        details=details or {},
    )
