"""Seed schema registry: program templates for search.

Each SeedSchema defines a program shape + parameter space.
Search instantiates schemas by binding parameters from perception.

Adapted from ericagi2's seeds.py, using aria's perception layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aria.search.sketch import SearchStep, SearchProgram, StepSelect


@dataclass
class SeedSchema:
    """A program template that search can instantiate.

    action: What the step does (maps to SearchStep.action)
    param_options: Static parameter dicts to try
    derive_params: Optional function(demos) → list[dict] for scene-derived params
    requires_same_shape: Guard — skip if input/output shapes differ
    requires_diff_shape: Guard — skip if shapes are the same
    select_options: List of StepSelect to try (None = whole grid)
    multi_step: For 2+ step schemas, list of (action, params, select) tuples
    """
    name: str
    action: str = ''
    param_options: list[dict[str, Any]] = field(default_factory=list)
    derive_params: Callable | None = None
    requires_same_shape: bool = True
    requires_diff_shape: bool = False
    select_options: list[StepSelect | None] = field(default_factory=lambda: [None])
    multi_step: list[tuple[str, dict, StepSelect | None]] | None = None

    def matches(self, demos: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        same = all(inp.shape == out.shape for inp, out in demos)
        if self.requires_same_shape and not same:
            return False
        if self.requires_diff_shape and same:
            return False
        return True

    def get_param_options(self, demos) -> list[dict[str, Any]]:
        derived = []
        if self.derive_params is not None:
            try:
                derived = self.derive_params(demos)
            except Exception:
                derived = []
        return derived + self.param_options

    def enumerate(self, demos) -> list[SearchProgram]:
        """Generate all candidate SearchPrograms from this schema."""
        if not self.matches(demos):
            return []

        if self.multi_step:
            return self._enumerate_multi(demos)

        programs = []
        params_list = self.get_param_options(demos)
        if not params_list:
            params_list = [{}]  # parameterless actions need one empty dict
        for params in params_list:
            # Derived selector overrides static select_options
            if '_select' in params:
                p = dict(params)  # copy to avoid mutation
                sel = p.pop('_select')
                action = 'crop_interior' if p.pop('_interior', False) else self.action
                step = SearchStep(action=action, params=p, select=sel)
                prog = SearchProgram(steps=[step], provenance=self.name)
                programs.append(prog)
            else:
                for sel in self.select_options:
                    step = SearchStep(action=self.action, params=params, select=sel)
                    prog = SearchProgram(steps=[step], provenance=self.name)
                    programs.append(prog)
        return programs

    def _enumerate_multi(self, demos):
        programs = []
        steps = []
        for action, params, sel in self.multi_step:
            steps.append(SearchStep(action=action, params=params, select=sel))
        programs.append(SearchProgram(steps=steps, provenance=self.name))
        return programs


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

def _common_selects() -> list[StepSelect | None]:
    """Common object selectors used across many schemas."""
    return [
        None,  # whole grid
        StepSelect('largest'),
        StepSelect('smallest'),
        StepSelect('unique_color'),
    ]


def _color_selects(demos) -> list[StepSelect]:
    """Generate color-based selectors from demo inputs."""
    from aria.guided.perceive import perceive
    colors = set()
    for inp, _ in demos:
        facts = perceive(inp)
        for obj in facts.objects:
            colors.add(obj.color)
    return [StepSelect('by_color', {'color': c}) for c in sorted(colors)]


def _derive_gravity_params(demos):
    """Derive gravity parameters from correspondence."""
    return [{'direction': d} for d in ('down', 'up', 'left', 'right')]


def _derive_recolor_params(demos):
    """Derive recolor target colors from output."""
    colors = set()
    for _, out in demos:
        colors.update(int(v) for v in np.unique(out))
    return [{'color': c} for c in sorted(colors)]


def _derive_tile_params(demos):
    """Derive tile parameters from input/output size ratios."""
    results = []
    tile_sizes = set()
    for inp, out in demos:
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih == 0 and ow % iw == 0:
            tile_sizes.add((oh // ih, ow // iw))
    for tr, tc in tile_sizes:
        if tr > 1 or tc > 1:
            results.append({'rows': tr, 'cols': tc, 'transforms': {}})
    return results


def _derive_combine_params(demos):
    """Derive split+combine+render parameters."""
    results = []
    for op in ('and', 'or', 'xor', 'diff', 'rdiff'):
        colors = set()
        for _, out in demos:
            colors.update(int(v) for v in np.unique(out) if v != 0)
        for c in sorted(colors):
            results.append({'op': op, 'color': c})
    return results


def _derive_output_construct_params(demos):
    """Derive output construction by finding which object bbox matches output dims.

    For each demo, find objects whose bbox matches the output shape.
    Return selectors for those objects.
    """
    from aria.guided.perceive import perceive
    from aria.guided.dsl import prim_find_frame, prim_crop_bbox, prim_crop_interior

    results = []
    inp0, out0 = demos[0]
    facts0 = perceive(inp0)
    oh, ow = out0.shape

    # Find objects whose bbox matches output dims
    for obj in facts0.objects:
        if obj.height == oh and obj.width == ow:
            crop = prim_crop_bbox(inp0, obj)
            if np.array_equal(crop, out0):
                # Find a selector for this object
                sel = _find_obj_selector(obj, facts0)
                if sel:
                    results.append({'_select': sel})

        # Check frame interior
        frame = prim_find_frame(obj, inp0)
        if frame:
            r0, c0, r1, c1 = frame
            ih, iw = r1 - r0 - 1, c1 - c0 - 1
            if (ih, iw) == (oh, ow):
                interior = prim_crop_interior(inp0, frame)
                if np.array_equal(interior, out0):
                    sel = _find_obj_selector(obj, facts0)
                    if sel:
                        results.append({'_select': sel, '_interior': True})

    return results


def _find_obj_selector(obj, facts):
    """Find a StepSelect that identifies this specific object."""
    from aria.guided.dsl import prim_select
    from aria.guided.clause import Predicate, Pred

    _pred_map = {
        Pred.IS_LARGEST: 'largest',
        Pred.IS_SMALLEST: 'smallest',
        Pred.UNIQUE_COLOR: 'unique_color',
        Pred.IS_TOPMOST: 'topmost',
        Pred.IS_BOTTOMMOST: 'bottommost',
        Pred.IS_LEFTMOST: 'leftmost',
        Pred.IS_RIGHTMOST: 'rightmost',
        Pred.NOT_TOUCHES_BORDER: 'interior',
    }

    for pred, role in _pred_map.items():
        sel_preds = [Predicate(pred)]
        selected = prim_select(facts, sel_preds)
        if len(selected) == 1 and selected[0].oid == obj.oid:
            return StepSelect(role)

    # Color-based
    sel_preds = [Predicate(Pred.COLOR_EQ, obj.color)]
    selected = prim_select(facts, sel_preds)
    if len(selected) == 1 and selected[0].oid == obj.oid:
        return StepSelect('by_color', {'color': obj.color})

    return None


def build_seed_registry() -> list[SeedSchema]:
    """Build the full registry of seed schemas."""
    seeds: list[SeedSchema] = []

    # --- Grid-level transforms ---
    for xform in ('flip_h', 'flip_v', 'flip_hv', 'rot90', 'rot180'):
        seeds.append(SeedSchema(name=xform, action=xform))

    # --- Object actions ---
    seeds.append(SeedSchema(
        name='gravity',
        action='gravity',
        derive_params=_derive_gravity_params,
        select_options=_common_selects(),
    ))

    seeds.append(SeedSchema(
        name='recolor',
        action='recolor',
        derive_params=_derive_recolor_params,
        select_options=_common_selects(),
    ))

    seeds.append(SeedSchema(
        name='remove',
        action='remove',
        select_options=_common_selects(),
        param_options=[{}],
    ))

    seeds.append(SeedSchema(
        name='fill_enclosed',
        action='fill_enclosed',
        derive_params=_derive_recolor_params,
    ))

    # --- Extraction / output construction ---
    # crop_bbox with structural selectors
    seeds.append(SeedSchema(
        name='crop_bbox',
        action='crop_bbox',
        requires_same_shape=False,
        requires_diff_shape=True,
        select_options=[
            StepSelect('largest'), StepSelect('smallest'),
            StepSelect('unique_color'),
            StepSelect('topmost'), StepSelect('bottommost'),
            StepSelect('leftmost'), StepSelect('rightmost'),
            StepSelect('interior'),
        ],
        param_options=[{}],
    ))

    # crop_bbox with color selectors (output = bbox of specific color object)
    seeds.append(SeedSchema(
        name='crop_bbox_by_color',
        action='crop_bbox',
        requires_same_shape=False,
        requires_diff_shape=True,
        derive_params=lambda demos: [{}],
        select_options=[],  # filled dynamically
    ))
    # Override enumerate to add color selectors
    seeds[-1].select_options = []
    seeds[-1]._dynamic_selects = True

    seeds.append(SeedSchema(
        name='crop_interior',
        action='crop_interior',
        requires_same_shape=False,
        requires_diff_shape=True,
        select_options=[
            StepSelect('largest'), StepSelect('smallest'),
            StepSelect('interior'),
        ],
        param_options=[{}],
    ))

    # Output construction: crop to object that matches output dims
    seeds.append(SeedSchema(
        name='output_construct_crop',
        action='crop_bbox',
        requires_same_shape=False,
        requires_diff_shape=True,
        derive_params=_derive_output_construct_params,
    ))

    # --- Region ops ---
    seeds.append(SeedSchema(
        name='split_combine_render',
        action='split_combine_render',
        requires_same_shape=False,
        derive_params=_derive_combine_params,
    ))

    # --- Grid constructors ---
    seeds.append(SeedSchema(
        name='tile',
        action='tile',
        requires_same_shape=False,
        requires_diff_shape=True,
        derive_params=_derive_tile_params,
    ))

    seeds.append(SeedSchema(
        name='repair_frames',
        action='repair_frames',
        param_options=[{}],
    ))

    # --- Mirror ---
    seeds.append(SeedSchema(
        name='mirror',
        action='mirror',
        param_options=[{'axis': a} for a in ('col', 'col_rtl', 'row', 'row_btt')],
    ))

    return seeds
