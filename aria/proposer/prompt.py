"""Prompt construction for the proposer model.

Builds structured prompts that include raw grids, state graph data,
available operations with type signatures, and example programs.
"""

from __future__ import annotations

import numpy as np

from aria.types import DemoPair, Delta, Grid, StateGraph


def _grid_to_text(grid: Grid) -> str:
    """Render a grid as a compact text block."""
    return "\n".join(
        " ".join(str(int(cell)) for cell in row)
        for row in grid
    )


def _render_demos(demos: tuple[DemoPair, ...]) -> str:
    """Render all demo pairs as text grids."""
    parts: list[str] = []
    for i, demo in enumerate(demos):
        parts.append(f"--- Demo {i} ---")
        parts.append(f"Input ({demo.input.shape[0]}x{demo.input.shape[1]}):")
        parts.append(_grid_to_text(demo.input))
        parts.append(f"Output ({demo.output.shape[0]}x{demo.output.shape[1]}):")
        parts.append(_grid_to_text(demo.output))
        parts.append("")
    return "\n".join(parts)


SYSTEM_PROMPT = """\
You are ARIA, a program synthesis engine for ARC-AGI tasks.

Each ARC task has demo pairs (input grid → output grid). Your job: find the \
transformation rule and express it as a program in the ARIA step language.

## Step Language Syntax

A program is a sequence of typed bindings ending with a yield:

```
bind <name> = <op>(<arg1>, <arg2>, ...)
yield <name>
```

The input grid is pre-bound as `input`. For context-reading tasks, `ctx` \
holds the demo pairs.

Higher-order ops use lambda syntax:

```
|x: INT| add(x, 1)
|row: INT, col: INT, val: INT| add(val, 1)
|val: INT, neighbors: INT_LIST| val
```

## Available Operations (with type signatures)

### Selection (querying objects in a grid)
find_objects(grid) → ObjectSet           -- CC-label non-background objects
by_color(color) → Predicate              -- returns Object→Bool predicate
by_shape(shape) → Predicate              -- shape: RECT, LINE, DOT, L, T, CROSS
by_size_rank(rank, objects) → Object     -- 0=largest, -1=smallest
where(pred, objects) → ObjectSet         -- filter by predicate
excluding(remove, from) → ObjectSet
singleton(objects) → Object              -- assert exactly one, return it
nth(idx, list) → Object

### Object Transforms
translate(dir, amount, obj) → Object     -- dir: UP, DOWN, LEFT, RIGHT
rotate(degrees, obj) → Object            -- 90, 180, 270
reflect(axis, obj) → Object              -- HORIZONTAL, VERTICAL
recolor(color, obj) → Object
extend(dir, amount, obj) → Object
resize_obj(factor, obj) → Object

### Grid Construction
new_grid(dims, fill_color) → Grid
crop(grid, (r, c, h, w)) → Grid
place_at(obj, (r, c), grid) → Grid      -- returns new grid with obj placed
overlay(top, bottom) → Grid              -- non-bg cells from top overwrite bottom
stack_h(a, b) → Grid                    -- horizontal concatenation
stack_v(a, b) → Grid                    -- vertical concatenation
embed(small, large, (r, c)) → Grid
apply_color_map({src: dst, ...}, grid) → Grid
fill_cells(grid, color_list) → Grid      -- fill in row-major order
flood_fill(grid, (r, c), color) → Grid

### Grid-level transforms
rotate_grid(degrees, grid) → Grid        -- 90, 180, 270 clockwise
reflect_grid(axis, grid) → Grid          -- HORIZONTAL, VERTICAL, DIAG_MAIN, DIAG_ANTI
transpose_grid(grid) → Grid
tile_grid(grid, rows, cols) → Grid       -- tile grid into rows x cols copies
upscale_grid(grid, factor) → Grid        -- scale each pixel to factor x factor
fill_enclosed(grid, color) → Grid        -- fill interior regions not reachable from border

### Dimensions
dims_of(grid) → Dims
dims_make(rows, cols) → Dims
scale_dims(dims, factor) → Dims
rows_of(dims) → Int
cols_of(dims) → Int

### Analysis
count(objects) → Int
length(list) → Int
group_by(property, objects) → List(ObjectSet)
sort_by(property, direction, objects) → ObjectList  -- direction: ASC, DESC
unique_colors(objects) → IntList
max_val(ints) → Int
min_val(ints) → Int
find_zones(grid) → ZoneList
zone_by_role(zones, role) → Zone         -- role: RULE, DATA
zone_to_grid(zone) → Grid
extract_map(zone) → ColorMap

### Arithmetic & Accessors
add(a, b), sub(a, b), mul(a, b), div(a, b), mod(a, b), isqrt(n)
get_color(obj) → Color
get_size(obj) → Int
get_width(obj) → Int, get_height(obj) → Int
get_pos_x(obj) → Int, get_pos_y(obj) → Int
eq(a, b), lt(a, b), gt(a, b) → Bool

### Higher-Order
map_obj(transform, objects) → ObjectSet
map_list(fn, list) → List
fold(fn, init, list) → Value
repeat_apply(n, grid_fn, grid) → Grid
if_then_else(cond, a, b) → Value

### Topological
flood_fill(grid, (r, c), color) → Grid
boundary(obj) → Region
hull(obj) → Object
connected_components(grid) → ObjectSet

### Cell-level operations
cell_map(grid, |row: INT, col: INT, val: INT| expr) → Grid
neighbor_map(grid, |val: INT, neighbors: INT_LIST| expr) → Grid
neighbor_map_8(grid, |val: INT, neighbors: INT_LIST| expr) → Grid
conditional_fill(grid, new_color, target_color) → Grid  -- replace target with new
fill_where_neighbor_count(grid, nbr_color, min_count, fill) → Grid
fill_between(grid, color, fill_color) → Grid    -- fill between same-colored cells (h+v)
fill_enclosed(grid, fill_color) → Grid           -- fill regions not reachable from border
propagate(grid, source_color, fill_color, bg) → Grid  -- BFS spread from source through bg

### Pattern matching
find_pattern(grid, pattern) → List(positions)    -- 0 in pattern = wildcard
replace_pattern(grid, pattern, replacement) → Grid

### Symmetry
complete_symmetry_h(grid) → Grid                 -- complete horizontal mirror symmetry
complete_symmetry_v(grid) → Grid                 -- complete vertical mirror symmetry

### Grid arithmetic (cell-wise)
grid_and(a, b) → Grid     -- non-zero only where both non-zero
grid_or(a, b) → Grid      -- union of non-zero cells
grid_xor(a, b) → Grid     -- cells that differ
grid_diff(a, b) → Grid    -- cells in a but not in b

### Row/column operations
get_row(grid, idx) → Grid
get_col(grid, idx) → Grid
set_row(grid, idx, row) → Grid
set_col(grid, idx, col) → Grid
sort_rows(grid) → Grid
sort_cols(grid) → Grid
unique_rows(grid) → Grid
unique_cols(grid) → Grid
most_common_color(grid) → Color
count_color(grid, color) → Int

### Cross-Demo Context (binds `ctx`)
demo_count(ctx) → Int
demo_at(ctx, idx) → Pair(Grid, Grid)
infer_map(ctx, prop_in, prop_out) → ColorMap
infer_step(ctx) → GridTransform
predict_dims(ctx, grid) → Dims

## Partial Application

Operations called with fewer args than expected return a function. Example:
`get_color` with 0 args returns an `Object → Color` function, usable with `map_list`.

## Example Programs

### Color remapping (entire grid)
```
bind mapping = infer_map(ctx, 0, 0)
bind result = apply_color_map(mapping, input)
yield result
```

### Computed output dimensions
```
bind objects = find_objects(input)
bind n = count(objects)
bind sorted = sort_by(SIZE, DESC, objects)
bind colors = map_list(get_color(), sorted)
bind side = isqrt(n)
bind out_dims = dims_make(side, side)
bind canvas = new_grid(out_dims, 0)
bind result = fill_cells(canvas, colors)
yield result
```

### Extract rule zone and apply
```
bind zones = find_zones(input)
bind key = zone_by_role(zones, RULE)
bind data = zone_by_role(zones, DATA)
bind mapping = extract_map(key)
bind result = apply_color_map(mapping, zone_to_grid(data))
yield result
```

### Direct color map (known mapping)
```
bind result = apply_color_map({1: 3, 2: 7, 5: 1}, input)
yield result
```

### Scale grid 3x
```
bind d = dims_of(input)
bind d2 = scale_dims(d, 3)
bind canvas = new_grid(d2, 0)
bind result = embed(input, canvas, (0, 0))
yield result
```

## Testing your program

You can test a program against the task's demo pairs by running:
```
python scripts/test_program.py <task_id> '<your program>'
```
This will show PASS/FAIL per demo, and on failure show the expected vs actual grid.
USE THIS TOOL. Write a candidate program, test it, and fix it before giving your final answer.

## Rules
1. Your FINAL output must be ONLY the program — no explanation, no commentary.
2. Each line must be: `bind <name> = <expr>` or `yield <name>`
3. The program must produce the correct output for ALL demo pairs.
4. Keep programs short (3-12 steps).
5. Study the grids carefully. Look for patterns in colors, shapes, positions, symmetry.
6. If output dims differ from input, you MUST compute them (dims_make, scale_dims, predict_dims).
7. TEST your program before submitting. Fix any errors.
"""

ROUND_1_INSTRUCTION = """\
Task ID: {task_id}

Study the demo pairs below. Find the transformation rule and emit a program.
Test it with: python scripts/test_program.py {task_id} '<your program>'

{demos}

{structured_input}

Emit the program:"""

ROUND_2_INSTRUCTION = """\
Task ID: {task_id}

Your previous attempt failed. Study the error and fix it.
Test with: python scripts/test_program.py {task_id} '<your program>'

{demos}

{structured_input}

{prior_attempts}

Fix the program:"""

ROUND_3_INSTRUCTION = """\
Task ID: {task_id}

Previous attempts got close — some passed most demos. Focus on the edge case.
Test with: python scripts/test_program.py {task_id} '<your program>'

{demos}

{structured_input}

{prior_attempts}

Emit a corrected program:"""

ROUND_4_INSTRUCTION = """\
Task ID: {task_id}

Final attempt. Try a different approach if prior ones failed.
Test with: python scripts/test_program.py {task_id} '<your program>'

{demos}

{structured_input}

{prior_attempts}

Emit the program:"""


def build_prompt(
    round_num: int,
    proposer_input: str,
    demos: tuple[DemoPair, ...] | None = None,
    prior_attempts: list[dict] | None = None,
    task_id: str = "unknown",
) -> str:
    """Build the full prompt for a given round."""
    demo_text = _render_demos(demos) if demos else ""

    prior_text = ""
    if prior_attempts:
        parts = ["[PRIOR ATTEMPTS — study these errors carefully]"]
        for att in prior_attempts:
            parts.append(f"\nAttempt {att.get('attempt_num', '?')}:")
            parts.append(f"  Program:")
            for line in att.get("program_text", "?").splitlines():
                parts.append(f"    {line}")

            etype = att.get("error_type", "")
            if etype == "execution_error":
                parts.append(f"  ERROR: {att.get('error_detail', 'unknown')}")
                parts.append(f"  FIX: Check op argument types. Common mistakes:")
                parts.append(f"    - overlay(ObjectSet, Grid) is wrong; overlay takes (Grid, Grid)")
                parts.append(f"    - gravity(dir, obj) takes an Object not a Grid")
                parts.append(f"    - map_obj(f, objects) needs f to be Object→Object")
                parts.append(f"    - translate/rotate/reflect take an Object, not a Grid")
                parts.append(f"    - singleton(set) fails if set has != 1 element")
            elif etype == "parse_error":
                parts.append(f"  PARSE ERROR: {att.get('error_detail', att.get('msg', '?'))}")
                parts.append(f"  FIX: Every program MUST end with 'yield <name>'. No explanation text.")
            elif etype == "wrong_output":
                parts.append(f"  WRONG OUTPUT on demo {att.get('failed_demo', '?')}")
                if att.get("diff"):
                    d = att["diff"]
                    parts.append(f"    Expected dims: {d.get('expected_dims')}")
                    parts.append(f"    Actual dims: {d.get('actual_dims')}")
                    if d.get("pixel_diff_count"):
                        parts.append(f"    Pixels wrong: {d['pixel_diff_count']}")
                    parts.append(f"    Summary: {d.get('pixel_diff_summary', '')}")
                parts.append(f"  FIX: Your program runs but produces the wrong grid.")
                parts.append(f"    - Re-read the demo pairs carefully")
                parts.append(f"    - Check if you're applying the transformation in the right order")
                parts.append(f"    - Consider using different ops (fill_enclosed, fill_between, propagate, grid_and/or/xor)")
                if att.get("step_trace"):
                    parts.append(f"  Step trace:")
                    for s in att["step_trace"][:8]:
                        status = "OK" if s["ok"] else f"SUSPECT: {s.get('suspect','')}"
                        parts.append(f"    {s['step_name']} = {s['value']} [{status}]")
            else:
                parts.append(f"  Error: {etype}")
        prior_text = "\n".join(parts)

    templates = {
        1: ROUND_1_INSTRUCTION,
        2: ROUND_2_INSTRUCTION,
        3: ROUND_3_INSTRUCTION,
        4: ROUND_4_INSTRUCTION,
    }
    template = templates.get(round_num, ROUND_1_INSTRUCTION)

    # Cap structured input to avoid enormous prompts on complex tasks
    if len(proposer_input) > 8000:
        proposer_input = proposer_input[:8000] + "\n  ... (truncated)"

    return SYSTEM_PROMPT + "\n\n" + template.format(
        task_id=task_id,
        demos=demo_text,
        structured_input=proposer_input,
        prior_attempts=prior_text,
    )
