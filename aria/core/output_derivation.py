"""Stage-1 output-derivation inference.

This module answers a narrower question than full task solving:

- is the output an exact clone of some selected input candidate?
- or is it the interior of a selected framed/boxed region?

It operates on explicit structural candidates from perception and verifies
every hypothesis against all train demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from aria.core.grid_perception import GridPerceptionState, perceive_grid
from aria.decomposition import FramedRegion, RawObject
from aria.types import DemoPair, Grid


KIND_OBJECT = "object"
KIND_FRAME_REGION = "frame_region"
KIND_BOXED_REGION = "boxed_region"
KIND_PARTITION_CELL = "partition_cell"
KIND_STRIP_BLOCK = "strip_block"

RELATION_CLONE = "clone"
RELATION_INTERIOR = "interior"
RELATION_BORDER = "border"


KIND_TO_CODE = {
    KIND_OBJECT: 0,
    KIND_FRAME_REGION: 1,
    KIND_BOXED_REGION: 2,
    KIND_PARTITION_CELL: 3,
    KIND_STRIP_BLOCK: 4,
}
CODE_TO_KIND = {code: kind for kind, code in KIND_TO_CODE.items()}

RELATION_TO_CODE = {
    RELATION_CLONE: 0,
    RELATION_INTERIOR: 1,
    RELATION_BORDER: 2,
}
CODE_TO_RELATION = {code: relation for relation, code in RELATION_TO_CODE.items()}

SELECTOR_TO_CODE = {
    "bbox_area_desc": 0,
    "bbox_area_asc": 1,
    "pixel_size_desc": 2,
    "pixel_size_asc": 3,
    "color_bbox_area_desc": 4,
    "area_desc": 5,
    "area_asc": 6,
    "top_left_asc": 7,
    "bottom_right_desc": 8,
    "frame_color_area_desc": 9,
    "row_major": 10,
    "col_major": 11,
    "leading_tiled_axis": 12,
    "trailing_tiled_axis": 13,
    "leading_long_axis_by_count": 14,
    "trailing_long_axis_by_count": 15,
}
CODE_TO_SELECTOR = {code: selector for selector, code in SELECTOR_TO_CODE.items()}


@dataclass(frozen=True)
class OutputDerivationSpec:
    candidate_kind: str
    relation: str
    selector: str
    params: Mapping[str, object] = field(default_factory=dict)
    rationale: str = ""


def encode_output_derivation_spec(
    spec: OutputDerivationSpec,
) -> tuple[int, int, int, int, int, int] | None:
    kind = KIND_TO_CODE.get(spec.candidate_kind)
    relation = RELATION_TO_CODE.get(spec.relation)
    selector = SELECTOR_TO_CODE.get(spec.selector)
    if kind is None or relation is None or selector is None:
        return None

    rank = spec.params.get("rank", -1)
    color = spec.params.get("color", spec.params.get("frame_color", -1))
    index = spec.params.get("index", -1)
    block_count = spec.params.get("block_count", -1)

    if not isinstance(rank, int):
        rank = -1
    if not isinstance(color, int):
        color = -1
    if not isinstance(index, int):
        index = -1
    if not isinstance(block_count, int):
        block_count = -1

    return (kind, relation, selector, rank, color, index if index >= 0 else block_count)


def decode_output_derivation_spec(
    kind_code: int,
    relation_code: int,
    selector_code: int,
    arg0: int,
    arg1: int,
    arg2: int,
) -> OutputDerivationSpec | None:
    kind = CODE_TO_KIND.get(kind_code)
    relation = CODE_TO_RELATION.get(relation_code)
    selector = CODE_TO_SELECTOR.get(selector_code)
    if kind is None or relation is None or selector is None:
        return None

    params: dict[str, object] = {}
    if selector in {"bbox_area_desc", "bbox_area_asc", "pixel_size_desc", "pixel_size_asc",
                    "area_desc", "area_asc", "top_left_asc", "bottom_right_desc"}:
        if arg0 < 0:
            return None
        params["rank"] = arg0
    elif selector in {"color_bbox_area_desc", "frame_color_area_desc"}:
        if arg0 < 0 or arg1 < 0:
            return None
        params["rank"] = arg0
        if selector == "color_bbox_area_desc":
            params["color"] = arg1
        else:
            params["frame_color"] = arg1
    elif selector in {"row_major", "col_major"}:
        if arg2 < 0:
            return None
        params["index"] = arg2
    elif selector in {
        "leading_tiled_axis",
        "trailing_tiled_axis",
        "leading_long_axis_by_count",
        "trailing_long_axis_by_count",
    }:
        if arg2 <= 1:
            return None
        params["block_count"] = arg2

    return OutputDerivationSpec(
        candidate_kind=kind,
        relation=relation,
        selector=selector,
        params=params,
    )


def predict_output_derivation(spec: OutputDerivationSpec, grid: Grid) -> Grid | None:
    return predict_output_derivation_from_state(spec, perceive_grid(grid))


def predict_output_derivation_from_state(
    spec: OutputDerivationSpec,
    state: GridPerceptionState,
) -> Grid | None:
    if spec.candidate_kind == KIND_OBJECT and spec.relation == RELATION_CLONE:
        obj = _select_object(state, spec.selector, spec.params)
        if obj is None:
            return None
        return state.grid[
            obj.row:obj.row + obj.bbox_h,
            obj.col:obj.col + obj.bbox_w,
        ].copy()

    if spec.candidate_kind == KIND_OBJECT and spec.relation == RELATION_INTERIOR:
        obj = _select_object(state, spec.selector, spec.params)
        return _extract_object_grid(state, obj, RELATION_INTERIOR)

    if spec.candidate_kind == KIND_FRAME_REGION and spec.relation == RELATION_INTERIOR:
        region = _select_region(state.framed_regions, spec.selector, spec.params)
        return None if region is None else region.interior.copy()

    if spec.candidate_kind == KIND_FRAME_REGION and spec.relation == RELATION_BORDER:
        region = _select_region(state.framed_regions, spec.selector, spec.params)
        return _extract_region_grid(state, region, RELATION_BORDER)

    if spec.candidate_kind == KIND_BOXED_REGION and spec.relation == RELATION_INTERIOR:
        region = _select_region(state.boxed_regions, spec.selector, spec.params)
        return None if region is None else region.interior.copy()

    if spec.candidate_kind == KIND_BOXED_REGION and spec.relation == RELATION_BORDER:
        region = _select_region(state.boxed_regions, spec.selector, spec.params)
        return _extract_region_grid(state, region, RELATION_BORDER)

    if spec.candidate_kind == KIND_PARTITION_CELL and spec.relation == RELATION_CLONE:
        cell = _select_partition_cell(state, spec.selector, spec.params)
        if cell is None:
            return None
        r0, r1, c0, c1 = cell
        return state.grid[r0:r1 + 1, c0:c1 + 1].copy()

    if spec.candidate_kind == KIND_STRIP_BLOCK and spec.relation == RELATION_CLONE:
        block = _select_strip_block(state, spec.selector, spec.params)
        if block is None:
            return None
        r0, r1, c0, c1 = block
        return state.grid[r0:r1 + 1, c0:c1 + 1].copy()

    return None


def verify_output_derivation_spec(
    spec: OutputDerivationSpec,
    demos: tuple[DemoPair, ...],
) -> bool:
    if not demos:
        return False
    return all(
        _grid_equal(predict_output_derivation(spec, demo.input), demo.output)
        for demo in demos
    )


def infer_verified_output_derivation_specs(
    demos: tuple[DemoPair, ...],
) -> tuple[OutputDerivationSpec, ...]:
    candidates = (
        _candidate_selected_object_clone(demos),
        _candidate_selected_object_interior(demos),
        _candidate_selected_frame_interior(demos),
        _candidate_selected_frame_border(demos),
        _candidate_selected_boxed_region_interior(demos),
        _candidate_selected_boxed_region_border(demos),
        _candidate_selected_strip_block_clone(demos),
        _candidate_selected_partition_cell_clone(demos),
    )
    verified: list[OutputDerivationSpec] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if verify_output_derivation_spec(candidate, demos):
            verified.append(candidate)
    return tuple(verified)


def infer_output_derivation_spec(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    specs = infer_verified_output_derivation_specs(demos)
    return specs[0] if specs else None


def _candidate_selected_object_clone(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _object_selector_candidates(first_state, target, RELATION_CLONE):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_OBJECT,
            relation=RELATION_CLONE,
            selector=selector,
            params=params,
            rationale=f"output exactly clones selected object bbox ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_object_interior(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _object_selector_candidates(first_state, target, RELATION_INTERIOR):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_OBJECT,
            relation=RELATION_INTERIOR,
            selector=selector,
            params=params,
            rationale=f"output exactly equals selected object interior ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_frame_interior(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _region_selector_candidates(first_state, first_state.framed_regions, target, RELATION_INTERIOR):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_FRAME_REGION,
            relation=RELATION_INTERIOR,
            selector=selector,
            params=params,
            rationale=f"output exactly equals selected frame interior ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_frame_border(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _region_selector_candidates(first_state, first_state.framed_regions, target, RELATION_BORDER):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_FRAME_REGION,
            relation=RELATION_BORDER,
            selector=selector,
            params=params,
            rationale=f"output exactly equals selected frame border ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_boxed_region_interior(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _region_selector_candidates(first_state, first_state.boxed_regions, target, RELATION_INTERIOR):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_BOXED_REGION,
            relation=RELATION_INTERIOR,
            selector=selector,
            params=params,
            rationale=f"output exactly equals selected boxed-region interior ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_partition_cell_clone(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _partition_cell_selector_candidates(first_state, target):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_PARTITION_CELL,
            relation=RELATION_CLONE,
            selector=selector,
            params=params,
            rationale=f"output exactly clones selected partition cell ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_strip_block_clone(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _strip_block_selector_candidates(first_state, target):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_STRIP_BLOCK,
            relation=RELATION_CLONE,
            selector=selector,
            params=params,
            rationale=f"output exactly clones selected strip block ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _candidate_selected_boxed_region_border(
    demos: tuple[DemoPair, ...],
) -> OutputDerivationSpec | None:
    if not demos:
        return None
    first_state = perceive_grid(demos[0].input)
    target = demos[0].output
    for selector, params in _region_selector_candidates(first_state, first_state.boxed_regions, target, RELATION_BORDER):
        spec = OutputDerivationSpec(
            candidate_kind=KIND_BOXED_REGION,
            relation=RELATION_BORDER,
            selector=selector,
            params=params,
            rationale=f"output exactly equals selected boxed-region border ({selector}) on every demo",
        )
        if verify_output_derivation_spec(spec, demos):
            return spec
    return None


def _region_selector_candidates(
    state: GridPerceptionState,
    regions: tuple[FramedRegion, ...],
    target: Grid,
    relation: str,
) -> tuple[tuple[str, dict[str, object]], ...]:
    if not regions:
        return ()

    candidates: list[tuple[str, dict[str, object]]] = []
    seen: set[tuple[object, ...]] = set()

    by_area_desc = sorted(
        regions,
        key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
        reverse=True,
    )
    by_area_asc = sorted(
        regions,
        key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
    )
    by_top_left_asc = sorted(
        regions,
        key=lambda r: (r.row, r.col, r.height * r.width, r.frame_color),
    )
    by_bottom_right_desc = sorted(
        regions,
        key=lambda r: (r.row + r.height, r.col + r.width, r.height * r.width, r.frame_color),
        reverse=True,
    )

    for selector, ordered in (
        ("area_desc", by_area_desc),
        ("area_asc", by_area_asc),
        ("top_left_asc", by_top_left_asc),
        ("bottom_right_desc", by_bottom_right_desc),
    ):
        for rank, region in enumerate(ordered):
            if not _grid_equal(_extract_region_grid(state, region, relation), target):
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((selector, {"rank": rank}))

    for frame_color in sorted({region.frame_color for region in regions}):
        pool = [region for region in by_area_desc if region.frame_color == frame_color]
        for rank, region in enumerate(pool):
            if not _grid_equal(_extract_region_grid(state, region, relation), target):
                continue
            key = ("frame_color_area_desc", frame_color, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(("frame_color_area_desc", {"frame_color": frame_color, "rank": rank}))

    return tuple(candidates)


def _object_selector_candidates(
    state: GridPerceptionState,
    target: Grid,
    relation: str,
) -> tuple[tuple[str, dict[str, object]], ...]:
    objects = list(state.objects.objects)
    if not objects:
        return ()

    candidates: list[tuple[str, dict[str, object]]] = []
    seen: set[tuple[object, ...]] = set()

    orderings = (
        (
            "bbox_area_desc",
            sorted(
                objects,
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col, -obj.color),
                reverse=True,
            ),
        ),
        (
            "bbox_area_asc",
            sorted(
                objects,
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col, obj.color),
            ),
        ),
        (
            "pixel_size_desc",
            sorted(
                objects,
                key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col, -obj.color),
                reverse=True,
            ),
        ),
        (
            "pixel_size_asc",
            sorted(
                objects,
                key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col, obj.color),
            ),
        ),
    )

    for selector, ordered in orderings:
        for rank, obj in enumerate(ordered):
            crop = _extract_object_grid(state, obj, relation)
            if not _grid_equal(crop, target):
                continue
            key = (selector, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((selector, {"rank": rank}))

    for color in sorted({obj.color for obj in objects}):
        pool = [obj for obj in objects if obj.color == color]
        ordered = sorted(
            pool,
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
            reverse=True,
        )
        for rank, obj in enumerate(ordered):
            crop = _extract_object_grid(state, obj, relation)
            if not _grid_equal(crop, target):
                continue
            key = ("color_bbox_area_desc", color, rank)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(("color_bbox_area_desc", {"color": color, "rank": rank}))

    return tuple(candidates)


def _partition_cell_selector_candidates(
    state: GridPerceptionState,
    target: Grid,
) -> tuple[tuple[str, dict[str, object]], ...]:
    if state.partition is None:
        return ()

    cells = _partition_cells(state)
    if not cells:
        return ()

    candidates: list[tuple[str, dict[str, object]]] = []
    seen: set[tuple[object, ...]] = set()

    for selector, ordered in (
        ("row_major", cells),
        (
            "col_major",
            sorted(cells, key=lambda cell: (cell[2], cell[0], cell[3] - cell[2], cell[1] - cell[0])),
        ),
    ):
        for index, cell in enumerate(ordered):
            r0, r1, c0, c1 = cell
            crop = state.grid[r0:r1 + 1, c0:c1 + 1]
            if not _grid_equal(crop, target):
                continue
            key = (selector, index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((selector, {"index": index}))

    return tuple(candidates)


def _strip_block_selector_candidates(
    state: GridPerceptionState,
    target: Grid,
) -> tuple[tuple[str, dict[str, object]], ...]:
    candidates: list[tuple[str, dict[str, object]]] = []
    target_rows = int(target.shape[0])
    target_cols = int(target.shape[1])
    rows, cols = state.dims

    count_candidates: set[int] = set()
    if rows == target_rows and target_cols > 0 and cols % target_cols == 0 and cols > target_cols:
        count_candidates.add(cols // target_cols)
    if cols == target_cols and target_rows > 0 and rows % target_rows == 0 and rows > target_rows:
        count_candidates.add(rows // target_rows)

    for block_count in sorted(count_candidates):
        for selector in (
            "leading_tiled_axis",
            "trailing_tiled_axis",
            "leading_long_axis_by_count",
            "trailing_long_axis_by_count",
        ):
            params = {"block_count": block_count}
            block = _select_strip_block(state, selector, params)
            if block is None:
                continue
            r0, r1, c0, c1 = block
            crop = state.grid[r0:r1 + 1, c0:c1 + 1]
            if _grid_equal(crop, target):
                candidates.append((selector, params))
    return tuple(candidates)


def _select_region(
    regions: tuple[FramedRegion, ...],
    selector: str,
    params: Mapping[str, object],
) -> FramedRegion | None:
    if not regions:
        return None

    if selector == "area_desc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col, -r.frame_color),
            reverse=True,
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    if selector == "area_asc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.height * r.width, r.height, r.width, r.row, r.col, r.frame_color),
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    if selector == "top_left_asc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.row, r.col, r.height * r.width, r.frame_color),
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    if selector == "bottom_right_desc":
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        ordered = sorted(
            regions,
            key=lambda r: (r.row + r.height, r.col + r.width, r.height * r.width, r.frame_color),
            reverse=True,
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    if selector == "frame_color_area_desc":
        frame_color = params.get("frame_color")
        rank = params.get("rank")
        if not isinstance(frame_color, int) or not isinstance(rank, int):
            return None
        ordered = sorted(
            [region for region in regions if region.frame_color == frame_color],
            key=lambda r: (r.height * r.width, r.height, r.width, -r.row, -r.col),
            reverse=True,
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    return None


def _select_object(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> RawObject | None:
    objects = list(state.objects.objects)
    if not objects:
        return None

    if selector in {"bbox_area_desc", "bbox_area_asc", "pixel_size_desc", "pixel_size_asc"}:
        rank = params.get("rank")
        if not isinstance(rank, int):
            return None
        if selector == "bbox_area_desc":
            ordered = sorted(
                objects,
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col, -obj.color),
                reverse=True,
            )
        elif selector == "bbox_area_asc":
            ordered = sorted(
                objects,
                key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, obj.row, obj.col, obj.color),
            )
        elif selector == "pixel_size_desc":
            ordered = sorted(
                objects,
                key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, -obj.row, -obj.col, -obj.color),
                reverse=True,
            )
        else:
            ordered = sorted(
                objects,
                key=lambda obj: (obj.size, obj.bbox_h * obj.bbox_w, obj.row, obj.col, obj.color),
            )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    if selector == "color_bbox_area_desc":
        color = params.get("color")
        rank = params.get("rank")
        if not isinstance(color, int) or not isinstance(rank, int):
            return None
        ordered = sorted(
            [obj for obj in objects if obj.color == color],
            key=lambda obj: (obj.bbox_h * obj.bbox_w, obj.size, -obj.row, -obj.col),
            reverse=True,
        )
        return ordered[rank] if 0 <= rank < len(ordered) else None

    return None


def _select_partition_cell(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    cells = _partition_cells(state)
    if not cells:
        return None
    index = params.get("index")
    if not isinstance(index, int):
        return None
    if selector == "row_major":
        ordered = cells
    elif selector == "col_major":
        ordered = sorted(cells, key=lambda cell: (cell[2], cell[0], cell[3] - cell[2], cell[1] - cell[0]))
    else:
        return None
    return ordered[index] if 0 <= index < len(ordered) else None


def _select_strip_block(
    state: GridPerceptionState,
    selector: str,
    params: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    rows, cols = state.dims
    block_count = params.get("block_count")
    if not isinstance(block_count, int) or block_count <= 1:
        return None

    col_block = None
    row_block = None
    if cols % block_count == 0:
        block_width = cols // block_count
        if block_width < cols:
            if selector == "leading_tiled_axis":
                return (0, rows - 1, 0, block_width - 1)
            if selector == "trailing_tiled_axis":
                return (0, rows - 1, cols - block_width, cols - 1)
            col_block = {
                "leading_long_axis_by_count": (0, rows - 1, 0, block_width - 1),
                "trailing_long_axis_by_count": (0, rows - 1, cols - block_width, cols - 1),
            }

    if rows % block_count == 0:
        block_height = rows // block_count
        if block_height < rows:
            if selector == "leading_tiled_axis":
                return (0, block_height - 1, 0, cols - 1)
            if selector == "trailing_tiled_axis":
                return (rows - block_height, rows - 1, 0, cols - 1)
            row_block = {
                "leading_long_axis_by_count": (0, block_height - 1, 0, cols - 1),
                "trailing_long_axis_by_count": (rows - block_height, rows - 1, 0, cols - 1),
            }

    if selector in {"leading_long_axis_by_count", "trailing_long_axis_by_count"}:
        if col_block is None and row_block is None:
            return None
        if col_block is None:
            return row_block[selector]
        if row_block is None:
            return col_block[selector]
        if cols > rows:
            return col_block[selector]
        if rows > cols:
            return row_block[selector]
        return None

    return None
    return None


def _partition_cells(state: GridPerceptionState) -> list[tuple[int, int, int, int]]:
    return [
        (rr[0], rr[1], cc[0], cc[1])
        for rr in state.row_intervals
        for cc in state.col_intervals
        if rr[0] <= rr[1] and cc[0] <= cc[1]
    ]


def _grid_equal(left: Grid | None, right: Grid | None) -> bool:
    return (
        left is not None
        and right is not None
        and left.shape == right.shape
        and np.array_equal(left, right)
    )


def _extract_region_grid(
    state: GridPerceptionState,
    region: FramedRegion | None,
    relation: str,
) -> Grid | None:
    if region is None:
        return None
    if relation == RELATION_INTERIOR:
        return region.interior.copy()
    if relation == RELATION_BORDER:
        top = int(region.row) - 1
        left = int(region.col) - 1
        height = int(region.height) + 2
        width = int(region.width) + 2
        if top < 0 or left < 0:
            return None
        if top + height > state.grid.shape[0] or left + width > state.grid.shape[1]:
            return None
        return state.grid[top:top + height, left:left + width].copy()
    return None


def _extract_object_grid(
    state: GridPerceptionState,
    obj: RawObject | None,
    relation: str,
) -> Grid | None:
    if obj is None:
        return None
    if relation == RELATION_CLONE:
        return state.grid[
            obj.row:obj.row + obj.bbox_h,
            obj.col:obj.col + obj.bbox_w,
        ].copy()
    if relation == RELATION_INTERIOR:
        if obj.bbox_h < 3 or obj.bbox_w < 3:
            return None
        return state.grid[
            obj.row + 1:obj.row + obj.bbox_h - 1,
            obj.col + 1:obj.col + obj.bbox_w - 1,
        ].copy()
    return None
