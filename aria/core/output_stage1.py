"""Combined stage-1 output reasoning.

Stage 1 is intentionally narrow and ordered:

1. infer output size
2. if size verifies, try to infer a direct derivation relation

No background inference. No scene actions. No rendering beyond direct
region/object extraction checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

from aria.core.output_derivation import (
    SELECTOR_TO_CODE,
    OutputDerivationSpec,
    encode_output_derivation_spec,
    infer_output_derivation_spec,
)
from aria.core.output_size import (
    MODE_MARKER_STACKED_SELECTED_OBJECT,
    MODE_SCALED_BBOX_OF_SELECTED_OBJECT,
    MODE_SOLID_RECTANGLE_LAYOUT_SHAPE,
    OutputSizeSpec,
    infer_output_size_spec,
)
from aria.types import Bind, Call, DemoPair, Literal, Program, Ref, Type


@dataclass(frozen=True)
class OutputStage1Spec:
    size_spec: OutputSizeSpec
    derivation_spec: OutputDerivationSpec | None = None
    render_spec: Mapping[str, object] | None = None


    # _PerDemoPrograms removed: all families must compile to a single
    # shared Program with input-only inference. If a family's bindings
    # cannot be inferred from the input grid alone, it is not submission-ready.


TRANSFORM_TO_CODE = {
    "identity": 0,
    "flip_lr": 1,
    "flip_ud": 2,
    "rot180": 3,
}


def infer_output_stage1_spec(
    demos: tuple[DemoPair, ...],
) -> OutputStage1Spec | None:
    size_spec = infer_output_size_spec(demos)
    if size_spec is None:
        return None
    derivation_spec = infer_output_derivation_spec(demos)
    render_spec = _infer_stage1_render_spec(demos, size_spec)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_zone_summary_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_global_color_map_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_geometric_transform_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_partition_cell_select_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_fill_enclosed_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_mask_repair_render_spec(demos)
    if render_spec is None and derivation_spec is None:
        render_spec = _infer_scene_program_render_spec(demos)
    return OutputStage1Spec(
        size_spec=size_spec,
        derivation_spec=derivation_spec,
        render_spec=render_spec,
    )


def compile_stage1_derivation_program(
    spec: OutputStage1Spec,
) -> Program | None:
    """Compile a verified direct-derivation relation into a runtime Program."""
    import aria.runtime  # noqa: F401

    if spec.derivation_spec is None:
        return None

    encoded = encode_output_derivation_spec(spec.derivation_spec)
    if encoded is None:
        return None

    kind_code, relation_code, selector_code, arg0, arg1, arg2 = encoded
    return Program(
        steps=(
            Bind(
                "v0",
                Type.GRID,
                Call(
                    "derive_output_from_input",
                    (
                        Ref("input"),
                        Literal(kind_code, Type.INT),
                        Literal(relation_code, Type.INT),
                        Literal(selector_code, Type.INT),
                        Literal(arg0, Type.INT),
                        Literal(arg1, Type.INT),
                        Literal(arg2, Type.INT),
                    ),
                ),
            ),
        ),
        output="v0",
    )


def compile_stage1_program(
    spec: OutputStage1Spec,
) -> Program | None:
    """Compile a stage-1 explanation into an executable Program when possible."""
    derivation_program = compile_stage1_derivation_program(spec)
    if derivation_program is not None:
        return derivation_program

    render_program = _compile_stage1_render_program(spec)
    if render_program is not None:
        return render_program

    size_params = spec.size_spec.params

    if spec.size_spec.mode == MODE_SCALED_BBOX_OF_SELECTED_OBJECT:
        selector = size_params.get("selector")
        rank = size_params.get("rank")
        connectivity = size_params.get("connectivity")
        row_scale = size_params.get("row_scale")
        col_scale = size_params.get("col_scale")
        selector_code = SELECTOR_TO_CODE.get(selector) if isinstance(selector, str) else None
        selector_arg = size_params.get("color", -1)
        if (
            isinstance(selector_code, int)
            and isinstance(rank, int)
            and isinstance(connectivity, int)
            and isinstance(row_scale, int)
            and isinstance(col_scale, int)
            and isinstance(selector_arg, int)
        ):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "render_scaled_selected_object",
                            (
                                Ref("input"),
                                Literal(connectivity, Type.INT),
                                Literal(selector_code, Type.INT),
                                Literal(rank, Type.INT),
                                Literal(row_scale, Type.INT),
                                Literal(col_scale, Type.INT),
                                Literal(selector_arg, Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )

    if spec.size_spec.mode == MODE_MARKER_STACKED_SELECTED_OBJECT:
        selector = size_params.get("selector")
        rank = size_params.get("rank")
        connectivity = size_params.get("connectivity")
        selector_code = SELECTOR_TO_CODE.get(selector) if isinstance(selector, str) else None
        selector_arg = size_params.get("color", -1)
        if (
            isinstance(selector_code, int)
            and isinstance(rank, int)
            and isinstance(connectivity, int)
            and isinstance(selector_arg, int)
        ):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "render_marker_stacked_selected_object",
                            (
                                Ref("input"),
                                Literal(connectivity, Type.INT),
                                Literal(selector_code, Type.INT),
                                Literal(rank, Type.INT),
                                Literal(selector_arg, Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )

    if spec.size_spec.mode == MODE_SOLID_RECTANGLE_LAYOUT_SHAPE:
        return Program(
            steps=(
                Bind("v0", Type.GRID, Call("render_solid_rectangle_layout", (Ref("input"),))),
            ),
            output="v0",
        )

    # Global color map render
    render_spec = spec.render_spec
    if render_spec is not None and render_spec.get("kind") == "global_color_map":
        pairs = render_spec.get("pairs", [])
        n_pairs = len(pairs)
        if 1 <= n_pairs <= 10:
            # Pad to 10 pairs with sentinel (-1, -1)
            padded = list(pairs) + [(-1, -1)] * (10 - n_pairs)
            args = [Ref("input"), Literal(n_pairs, Type.INT)]
            for fc, tc in padded:
                args.append(Literal(fc, Type.INT))
                args.append(Literal(tc, Type.INT))
            return Program(
                steps=(
                    Bind("v0", Type.GRID, Call("apply_global_color_map", tuple(args))),
                ),
                output="v0",
            )

    # Zone/partition summary grid render
    if render_spec is not None and render_spec.get("kind") == "zone_summary_grid":
        property_code = render_spec.get("property_code")
        source_code = render_spec.get("source_code", 1)
        if isinstance(property_code, int) and isinstance(source_code, int):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "render_zone_summary_grid",
                            (
                                Ref("input"),
                                Literal(property_code, Type.INT),
                                Literal(source_code, Type.INT),
                            ),
                        ),
                    ),
                ),
                output="v0",
            )

    # Geometric transform render
    if render_spec is not None and render_spec.get("kind") == "geometric_transform":
        transform_code = render_spec.get("transform_code")
        if isinstance(transform_code, int):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "apply_geometric_transform",
                            (Ref("input"), Literal(transform_code, Type.INT)),
                        ),
                    ),
                ),
                output="v0",
            )

    # Partition cell selection render
    if render_spec is not None and render_spec.get("kind") == "partition_cell_select":
        selector_code = render_spec.get("selector_code")
        if isinstance(selector_code, int):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "select_partition_cell_by_property",
                            (Ref("input"), Literal(selector_code, Type.INT)),
                        ),
                    ),
                ),
                output="v0",
            )

    # Fill enclosed regions (fixed color)
    if render_spec is not None and render_spec.get("kind") == "fill_enclosed":
        fill_color = render_spec.get("fill_color")
        if isinstance(fill_color, int):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "fill_enclosed_regions",
                            (Ref("input"), Literal(fill_color, Type.INT)),
                        ),
                    ),
                ),
                output="v0",
            )

    # Fill enclosed regions (auto: boundary color)
    if render_spec is not None and render_spec.get("kind") == "fill_enclosed_auto":
        return Program(
            steps=(
                Bind(
                    "v0",
                    Type.GRID,
                    Call("fill_enclosed_regions_auto", (Ref("input"),)),
                ),
            ),
            output="v0",
        )

    # Scene program render
    if render_spec is not None and render_spec.get("kind") == "scene_program":
        import json as _json
        steps_json = render_spec.get("steps_json")
        if isinstance(steps_json, str):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "execute_scene_program_json",
                            (Ref("input"), Literal(steps_json, Type.INT)),
                        ),
                    ),
                ),
                output="v0",
            )

    # Mask repair: one shared Program. Source inferred at runtime from symmetry.
    if render_spec is not None and render_spec.get("kind") == "mask_repair":
        transform_code = render_spec.get("transform_code")
        if isinstance(transform_code, int):
            return Program(
                steps=(
                    Bind(
                        "v0",
                        Type.GRID,
                        Call(
                            "repair_masked_region",
                            (Ref("input"), Literal(transform_code, Type.INT)),
                        ),
                    ),
                ),
                output="v0",
            )

    return None


def _infer_stage1_render_spec(
    demos: tuple[DemoPair, ...],
    size_spec: OutputSizeSpec,
) -> Mapping[str, object] | None:
    if size_spec.mode != "scale_input":
        return None

    row_ratio = size_spec.params.get("row_ratio")
    col_ratio = size_spec.params.get("col_ratio")
    if not isinstance(row_ratio, Fraction) or not isinstance(col_ratio, Fraction):
        return None
    if row_ratio.denominator != 1 or col_ratio.denominator != 1:
        return None
    row_repeat = row_ratio.numerator
    col_repeat = col_ratio.numerator
    if row_repeat < 1 or col_repeat < 1:
        return None

    for odd_row_transform in TRANSFORM_TO_CODE:
        for odd_col_transform in TRANSFORM_TO_CODE:
            if all(
                _predict_tiled_input_pattern(
                    demo.input,
                    row_repeat=row_repeat,
                    col_repeat=col_repeat,
                    odd_row_transform=odd_row_transform,
                    odd_col_transform=odd_col_transform,
                ) is not None
                and _grid_equal(
                    _predict_tiled_input_pattern(
                        demo.input,
                        row_repeat=row_repeat,
                        col_repeat=col_repeat,
                        odd_row_transform=odd_row_transform,
                        odd_col_transform=odd_col_transform,
                    ),
                    demo.output,
                )
                for demo in demos
            ):
                return {
                    "kind": "tiled_input_pattern",
                    "row_repeat": row_repeat,
                    "col_repeat": col_repeat,
                    "odd_row_transform": odd_row_transform,
                    "odd_col_transform": odd_col_transform,
                    "rationale": (
                        "output tiles the input with alternating row/col transforms "
                        f"({odd_row_transform}, {odd_col_transform})"
                    ),
                }
    return None


def _compile_stage1_render_program(
    spec: OutputStage1Spec,
) -> Program | None:
    import aria.runtime  # noqa: F401

    render_spec = spec.render_spec
    if render_spec is None:
        return None
    if render_spec.get("kind") != "tiled_input_pattern":
        return None

    row_repeat = render_spec.get("row_repeat")
    col_repeat = render_spec.get("col_repeat")
    odd_row_transform = TRANSFORM_TO_CODE.get(render_spec.get("odd_row_transform"))
    odd_col_transform = TRANSFORM_TO_CODE.get(render_spec.get("odd_col_transform"))
    if not all(isinstance(v, int) for v in (row_repeat, col_repeat, odd_row_transform, odd_col_transform)):
        return None

    return Program(
        steps=(
            Bind(
                "v0",
                Type.GRID,
                Call(
                    "render_tiled_input_pattern",
                    (
                        Ref("input"),
                        Literal(row_repeat, Type.INT),
                        Literal(col_repeat, Type.INT),
                        Literal(odd_row_transform, Type.INT),
                        Literal(odd_col_transform, Type.INT),
                    ),
                ),
            ),
        ),
        output="v0",
    )


def _predict_tiled_input_pattern(
    grid,
    *,
    row_repeat: int,
    col_repeat: int,
    odd_row_transform: str,
    odd_col_transform: str,
):
    import numpy as np

    def apply_transform(arr, name: str):
        if name == "identity":
            return arr
        if name == "flip_lr":
            return np.fliplr(arr)
        if name == "flip_ud":
            return np.flipud(arr)
        if name == "rot180":
            return np.rot90(arr, k=2)
        return None

    row_blocks = []
    for row_idx in range(row_repeat):
        col_blocks = []
        for col_idx in range(col_repeat):
            tile = grid
            if row_idx % 2 == 1:
                tile = apply_transform(tile, odd_row_transform)
            if tile is None:
                return None
            if col_idx % 2 == 1:
                tile = apply_transform(tile, odd_col_transform)
            if tile is None:
                return None
            col_blocks.append(tile)
        row_blocks.append(np.hstack(col_blocks))
    return np.vstack(row_blocks).astype(grid.dtype, copy=False)


def _grid_equal(left, right) -> bool:
    import numpy as np

    return (
        left is not None
        and right is not None
        and left.shape == right.shape
        and np.array_equal(left, right)
    )


def _infer_zone_summary_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Check if the output is a zone/partition summary grid across all demos."""
    from aria.core.relations import verify_zone_summary_grid
    from aria.runtime.ops.zone_summary import SOURCE_PARTITION, SOURCE_ZONE, _PROPERTY_INDEX

    mapping = verify_zone_summary_grid(demos)
    if mapping is None:
        return None

    prop_name = mapping.params.get("property")
    if not isinstance(prop_name, str) or prop_name not in _PROPERTY_INDEX:
        return None

    source = mapping.params.get("source", "zone")
    source_code = SOURCE_PARTITION if source == "partition" else SOURCE_ZONE

    return {
        "kind": "zone_summary_grid",
        "property_code": _PROPERTY_INDEX[prop_name],
        "source_code": source_code,
        "rationale": f"output is a summary grid of {source} {prop_name}",
    }


def _infer_global_color_map_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Detect a consistent global color→color mapping across all demos.

    Applies when every pixel change in every demo follows the same
    color substitution table, and applying that table perfectly
    reproduces all outputs.
    """
    import numpy as np

    if not demos:
        return None

    # All demos must have same input/output dims
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    # Build a single global mapping
    global_map: dict[int, int] = {}
    any_diff = False
    for d in demos:
        diff = d.input != d.output
        if np.any(diff):
            any_diff = True
        rows, cols = np.where(diff)
        for r, c in zip(rows, cols):
            ic = int(d.input[r, c])
            oc = int(d.output[r, c])
            if ic in global_map:
                if global_map[ic] != oc:
                    return None
            else:
                global_map[ic] = oc

    if not global_map or len(global_map) > 10 or not any_diff:
        return None

    # Verify by simultaneous substitution (handles mutual swaps correctly)
    for d in demos:
        predicted = d.input.copy()
        temp = d.input.copy()
        for ic, oc in global_map.items():
            predicted[temp == ic] = oc
        if not np.array_equal(predicted, d.output):
            return None

    # Encode as sorted list of (from, to) pairs
    pairs = sorted(global_map.items())

    return {
        "kind": "global_color_map",
        "pairs": pairs,
        "rationale": f"output is input with global color substitution: {dict(pairs)}",
    }


def _infer_geometric_transform_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Detect a single geometric transform that maps input to output."""
    import numpy as np

    if not demos:
        return None

    from aria.runtime.ops.scene_transforms import TRANSFORM_CODES

    for code, name in TRANSFORM_CODES.items():
        all_match = True
        for d in demos:
            if name == "rot90":
                result = np.rot90(d.input, 1)
            elif name == "rot180":
                result = np.rot90(d.input, 2)
            elif name == "rot270":
                result = np.rot90(d.input, 3)
            elif name == "flip_lr":
                result = np.fliplr(d.input)
            elif name == "flip_ud":
                result = np.flipud(d.input)
            elif name == "transpose":
                result = d.input.T
            else:
                all_match = False
                break
            if result.shape != d.output.shape or not np.array_equal(result, d.output):
                all_match = False
                break
        if all_match:
            return {
                "kind": "geometric_transform",
                "transform_code": code,
                "rationale": f"output is {name} of input",
            }
    return None


def _infer_partition_cell_select_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Detect partition cell selection by a structural property."""
    import numpy as np

    from aria.core.grid_perception import perceive_grid
    from aria.runtime.ops.scene_transforms import CELL_SELECTOR_CODES

    if not demos:
        return None

    for code, name in CELL_SELECTOR_CODES.items():
        all_match = True
        for d in demos:
            state = perceive_grid(d.input)
            p = state.partition
            if p is None or len(p.cells) < 2:
                all_match = False
                break

            bg = state.bg_color
            oH, oW = d.output.shape

            # Build cell info for cells matching output dims
            candidates = [c for c in p.cells if c.dims == (oH, oW)]
            if not candidates:
                candidates = list(p.cells)
            if not candidates:
                all_match = False
                break

            infos = []
            for cell in candidates:
                r0, c0, r1, c1 = cell.bbox
                cg = d.input[r0 : r1 + 1, c0 : c1 + 1]
                non_bg = cg[cg != bg]
                n_non_bg = len(non_bg)
                n_colors = len(set(int(v) for v in non_bg)) if n_non_bg > 0 else 0
                infos.append((cell, cg, n_non_bg, n_colors))

            selected = None
            if code == 0:  # most_non_bg
                selected = max(infos, key=lambda x: x[2])
            elif code == 1:  # fewest_non_bg_gt0
                gt0 = [x for x in infos if x[2] > 0]
                if gt0:
                    selected = min(gt0, key=lambda x: x[2])
            elif code == 2:  # most_colors
                selected = max(infos, key=lambda x: x[3])
            elif code == 3:  # unique_non_empty
                non_empty = [x for x in infos if x[2] > 0]
                if len(non_empty) == 1:
                    selected = non_empty[0]

            if selected is None or not np.array_equal(selected[1], d.output):
                all_match = False
                break

        if all_match:
            return {
                "kind": "partition_cell_select",
                "selector_code": code,
                "rationale": f"output is partition cell selected by {name}",
            }

    return None


def _infer_fill_enclosed_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Detect fill-enclosed-regions pattern across all demos.

    Applies when:
    - output dims == input dims
    - every changed pixel was background in the input
    - every changed pixel is the same fill color per demo
    - the fill color equals the color of the enclosing boundary
    """
    import numpy as np

    from aria.decomposition import detect_bg

    if not demos:
        return None
    if any(d.input.shape != d.output.shape for d in demos):
        return None

    any_change = False
    for d in demos:
        diff = d.input != d.output
        if not np.any(diff):
            continue
        any_change = True

        bg = detect_bg(d.input)
        rows, cols = np.where(diff)

        # All changed pixels must have been bg in input
        if not all(int(d.input[r, c]) == bg for r, c in zip(rows, cols)):
            return None

        # All changed pixels must be the same color in output (per demo)
        fill_colors = set(int(d.output[r, c]) for r, c in zip(rows, cols))
        if len(fill_colors) != 1:
            return None

    if not any_change:
        return None

    # Verify by running fill_enclosed with each demo's fill color
    for fill_code in range(10):
        all_match = True
        for d in demos:
            bg = detect_bg(d.input)
            predicted = _apply_fill_enclosed(d.input, fill_code)
            if predicted is None or not np.array_equal(predicted, d.output):
                all_match = False
                break
        if all_match:
            return {
                "kind": "fill_enclosed",
                "fill_color": fill_code,
                "rationale": f"output is input with enclosed bg regions filled with color {fill_code}",
            }

    # Try fill with boundary color (auto-detect)
    all_match = True
    for d in demos:
        predicted = _apply_fill_enclosed_auto(d.input)
        if predicted is None or not np.array_equal(predicted, d.output):
            all_match = False
            break
    if all_match:
        return {
            "kind": "fill_enclosed_auto",
            "rationale": "output is input with enclosed bg regions filled by their boundary color",
        }

    return None


def _apply_fill_enclosed(grid: Grid, fill_color: int) -> Grid | None:
    """Fill enclosed background regions with a fixed color."""
    import numpy as np
    from scipy import ndimage

    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    mask = grid == bg

    # Label connected bg regions
    labeled, n_regions = ndimage.label(mask)
    if n_regions < 2:
        return None

    result = grid.copy()
    # Find which region touches the border (that's the exterior)
    border_labels = set()
    h, w = grid.shape
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    # Fill non-border regions
    for region_id in range(1, n_regions + 1):
        if region_id not in border_labels:
            result[labeled == region_id] = fill_color

    return result


def _apply_fill_enclosed_auto(grid: Grid) -> Grid | None:
    """Fill enclosed bg regions with the color of their enclosing boundary."""
    import numpy as np
    from scipy import ndimage

    bg = int(grid.flat[np.argmax(np.bincount(grid.ravel()))])
    mask = grid == bg

    labeled, n_regions = ndimage.label(mask)
    if n_regions < 2:
        return None

    result = grid.copy()
    h, w = grid.shape

    border_labels = set()
    for r in range(h):
        if labeled[r, 0] > 0:
            border_labels.add(labeled[r, 0])
        if labeled[r, w - 1] > 0:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if labeled[0, c] > 0:
            border_labels.add(labeled[0, c])
        if labeled[h - 1, c] > 0:
            border_labels.add(labeled[h - 1, c])

    for region_id in range(1, n_regions + 1):
        if region_id in border_labels:
            continue
        # Find boundary color: most common non-bg neighbor
        region_mask = labeled == region_id
        dilated = ndimage.binary_dilation(region_mask)
        boundary = dilated & ~region_mask
        boundary_colors = grid[boundary]
        boundary_non_bg = boundary_colors[boundary_colors != bg]
        if len(boundary_non_bg) == 0:
            continue
        vals, counts = np.unique(boundary_non_bg, return_counts=True)
        fill_color = int(vals[np.argmax(counts)])
        result[region_mask] = fill_color

    return result


def _infer_mask_repair_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Detect masked-region symmetry repair pattern.

    Searches for a solid marker rectangle (mask) whose bbox matches
    the output dims, then finds a source region + transform that
    exactly reproduces the output across all demos.
    """
    import numpy as np

    from aria.runtime.ops.mask_repair import _find_solid_marker, _TRANSFORMS

    if not demos:
        return None

    # All demos must have output smaller than input (extracting a patch)
    for d in demos:
        if d.input.shape[0] <= d.output.shape[0] and d.input.shape[1] <= d.output.shape[1]:
            if d.input.shape == d.output.shape:
                return None

    # For each (transform_code, source_r, source_c), check if it works on ALL demos
    # First, collect per-demo candidates
    per_demo_candidates: list[set[tuple[int, int, int]]] = []

    for d in demos:
        vals, counts = np.unique(d.input, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        marker = _find_solid_marker(d.input, bg)
        if marker is None:
            return None

        mask_color, mr0, mc0, mh, mw = marker
        oh, ow = d.output.shape
        if mh != oh or mw != ow:
            return None

        ih, iw = d.input.shape
        mr1 = mr0 + mh - 1
        mc1 = mc0 + mw - 1

        candidates: set[tuple[int, int, int]] = set()
        for ti, (_, fn) in enumerate(_TRANSFORMS):
            for sr0 in range(ih - mh + 1):
                for sc0 in range(iw - mw + 1):
                    if sr0 <= mr1 and sr0 + mh - 1 >= mr0 and sc0 <= mc1 and sc0 + mw - 1 >= mc0:
                        continue
                    source = d.input[sr0 : sr0 + mh, sc0 : sc0 + mw]
                    if np.any(source == mask_color):
                        continue
                    transformed = fn(source)
                    if transformed.shape == d.output.shape and np.array_equal(transformed, d.output):
                        candidates.add((ti, sr0, sc0))

        if not candidates:
            return None
        per_demo_candidates.append(candidates)

    # Find a (transform_code, source_r, source_c) that works on ALL demos?
    # The source position varies per demo, so we can't intersect directly.
    # Instead: find the transform_code that works for all demos (source varies).
    common_transforms: set[int] = set(range(len(_TRANSFORMS)))
    for cands in per_demo_candidates:
        demo_transforms = {t for t, _, _ in cands}
        common_transforms &= demo_transforms

    if not common_transforms:
        # No single transform works for all demos — try per-demo
        # Use the first candidate from each demo (verification will confirm)
        pass

    if not common_transforms:
        return None

    # Shared program: one transform code.
    # Source position inferred at runtime from symmetry center detection.
    tc = min(common_transforms)
    transform_name = _TRANSFORMS[tc][0]

    return {
        "kind": "mask_repair",
        "transform_code": tc,
        "rationale": f"mask_repair: shared transform={transform_name}, source from inferred symmetry center",
    }


def _infer_scene_program_render_spec(
    demos: tuple[DemoPair, ...],
) -> Mapping[str, object] | None:
    """Try to find a short scene program that verifies on all demos.

    Uses the scene-program proposer to generate bounded candidates
    and returns the first verified one as a render spec.
    """
    import json

    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    from aria.core.scene_propose import propose_and_verify

    result = propose_and_verify(demos)
    if result is None:
        return None

    template_name, prog = result

    # Serialize the scene program steps to JSON
    steps_data = []
    for step in prog.steps:
        sd: dict = {"op": step.op.value}
        if step.inputs:
            sd["inputs"] = list(step.inputs)
        if step.params:
            sd["params"] = dict(step.params)
        if step.output_id:
            sd["output_id"] = step.output_id
        steps_data.append(sd)

    return {
        "kind": "scene_program",
        "template": template_name,
        "steps_json": json.dumps(steps_data),
        "rationale": f"output matches scene program: {template_name}",
    }
