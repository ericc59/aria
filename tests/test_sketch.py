"""Tests for the sketch IR — task-local hypothesis representation."""

from __future__ import annotations

import json

from aria.sketch import (
    PrimitiveFamily,
    RoleKind,
    RoleVar,
    Sketch,
    SketchStep,
    Slot,
    SlotType,
    make_canvas_layout,
    make_composite_role_alignment,
    make_identify_roles,
    make_object_move_by_relation,
    make_region_periodic_repair,
    sketch_from_dict,
    sketch_to_dict,
    sketch_to_text,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _periodic_repair_sketch() -> Sketch:
    """Sketch for 135a2760-style: framed region periodic pattern repair."""
    return Sketch(
        task_id="135a2760",
        steps=(
            make_identify_roles(bg_colors=[3, 4]),
            make_region_periodic_repair(frame_role="frame", period_axis="col"),
        ),
        output_ref="repair",
        description="inside framed regions, detect col-periodic pattern and fix deviations",
    )


def _composite_alignment_sketch() -> Sketch:
    """Sketch for 581f7754-style: composite motifs align to anchor."""
    return Sketch(
        task_id="581f7754",
        steps=(
            make_identify_roles(bg_colors=[1, 3, 8]),
            SketchStep(
                name="composites",
                primitive=PrimitiveFamily.FIND_COMPOSITES,
                roles=(
                    RoleVar("center", RoleKind.CENTER, "singleton center pixel"),
                    RoleVar("frame", RoleKind.FRAME, "frame CCs around center"),
                ),
                input_refs=("roles",),
                description="group center singletons with adjacent frame CCs",
            ),
            make_composite_role_alignment(composites_ref="composites", axis="col"),
        ),
        output_ref="aligned",
        description="align composite motifs to anchor column",
    )


def _canvas_layout_sketch() -> Sketch:
    """Sketch for 269e22fb-style: dims-change canvas construction."""
    return Sketch(
        task_id="269e22fb",
        steps=(
            make_identify_roles(bg_colors=[0, 8]),
            make_canvas_layout(dims_evidence=(20, 20), fixed_output=True),
        ),
        output_ref="canvas",
        description="construct 20x20 output from input content by expansion rule",
    )


def _object_move_sketch() -> Sketch:
    """Sketch for movement task with anchor-relative displacement."""
    return Sketch(
        task_id="test_move",
        steps=(
            SketchStep(
                name="objects",
                primitive=PrimitiveFamily.FIND_OBJECTS,
                input_refs=("input",),
                description="extract connected components",
            ),
            make_object_move_by_relation(
                objects_ref="objects",
                predicate_desc="non-anchor objects of content color",
            ),
        ),
        output_ref="moved",
        description="move content objects by anchor-relative displacement",
    )


# ---------------------------------------------------------------------------
# Type / field tests
# ---------------------------------------------------------------------------


def test_sketch_has_required_fields():
    s = _periodic_repair_sketch()
    assert s.task_id == "135a2760"
    assert len(s.steps) == 2
    assert s.output_ref == "repair"
    assert s.description != ""


def test_step_has_required_fields():
    step = make_identify_roles(bg_colors=[0])
    assert step.name == "roles"
    assert step.primitive == PrimitiveFamily.IDENTIFY_ROLES
    assert len(step.roles) >= 1
    assert step.evidence.get("observed_bg_colors") == [0]


def test_role_var_construction():
    rv = RoleVar("bg", RoleKind.BG, "background")
    assert rv.name == "bg"
    assert rv.kind == RoleKind.BG
    assert repr(rv) == "$bg"


def test_slot_construction():
    slot = Slot("period", SlotType.INT, constraint="positive")
    assert slot.name == "period"
    assert slot.typ == SlotType.INT
    assert repr(slot) == "?period:INT"


def test_primitive_families():
    s = _periodic_repair_sketch()
    assert s.primitive_families == (
        PrimitiveFamily.IDENTIFY_ROLES,
        PrimitiveFamily.REGION_PERIODIC_REPAIR,
    )


def test_role_vars_collected():
    s = _periodic_repair_sketch()
    names = {rv.name for rv in s.role_vars}
    assert "bg" in names
    assert "frame" in names


def test_open_slots_collected():
    s = _periodic_repair_sketch()
    slot_names = {slot.name for slot in s.open_slots}
    assert "axis" in slot_names
    assert "period" in slot_names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_sketch_passes_validation():
    s = _periodic_repair_sketch()
    errors = s.validate()
    assert errors == [], f"unexpected errors: {errors}"


def test_invalid_ref_detected():
    s = Sketch(
        task_id="bad",
        steps=(
            SketchStep(
                name="s1",
                primitive=PrimitiveFamily.FILL_BY_RULE,
                input_refs=("nonexistent",),
            ),
        ),
        output_ref="s1",
    )
    errors = s.validate()
    assert any("nonexistent" in e for e in errors)


def test_duplicate_name_detected():
    s = Sketch(
        task_id="bad",
        steps=(
            SketchStep(name="s1", primitive=PrimitiveFamily.FIND_OBJECTS),
            SketchStep(name="s1", primitive=PrimitiveFamily.FILL_BY_RULE),
        ),
        output_ref="s1",
    )
    errors = s.validate()
    assert any("duplicate" in e for e in errors)


def test_bad_output_ref_detected():
    s = Sketch(
        task_id="bad",
        steps=(
            SketchStep(name="s1", primitive=PrimitiveFamily.FIND_OBJECTS),
        ),
        output_ref="missing",
    )
    errors = s.validate()
    assert any("missing" in e for e in errors)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_roundtrip():
    original = _composite_alignment_sketch()
    d = sketch_to_dict(original)
    restored = sketch_from_dict(d)
    assert restored.task_id == original.task_id
    assert len(restored.steps) == len(original.steps)
    assert restored.output_ref == original.output_ref
    for orig_step, rest_step in zip(original.steps, restored.steps):
        assert rest_step.name == orig_step.name
        assert rest_step.primitive == orig_step.primitive
        assert len(rest_step.roles) == len(orig_step.roles)
        assert len(rest_step.slots) == len(orig_step.slots)


def test_serialize_json_roundtrip():
    original = _canvas_layout_sketch()
    d = sketch_to_dict(original)
    json_str = json.dumps(d)
    restored_dict = json.loads(json_str)
    restored = sketch_from_dict(restored_dict)
    assert restored.task_id == original.task_id
    assert restored.steps[0].primitive == original.steps[0].primitive


def test_serialize_all_primitives():
    """All four sketch families serialize and deserialize correctly."""
    for sketch in [
        _periodic_repair_sketch(),
        _composite_alignment_sketch(),
        _canvas_layout_sketch(),
        _object_move_sketch(),
    ]:
        d = sketch_to_dict(sketch)
        restored = sketch_from_dict(d)
        assert restored.task_id == sketch.task_id
        assert len(restored.steps) == len(sketch.steps)
        assert restored.validate() == []


def test_serialize_preserves_evidence():
    s = _periodic_repair_sketch()
    d = sketch_to_dict(s)
    # The identify_roles step should carry observed_bg_colors
    roles_step = d["steps"][0]
    assert roles_step["evidence"]["observed_bg_colors"] == [3, 4]


def test_serialize_preserves_slot_evidence():
    step = make_region_periodic_repair(period_axis="col")
    d = sketch_to_dict(Sketch(
        task_id="test",
        steps=(step,),
        output_ref="repair",
    ))
    axis_slot = next(s for s in d["steps"][0]["slots"] if s["name"] == "axis")
    assert axis_slot["evidence"] == "col"


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def test_text_rendering():
    s = _periodic_repair_sketch()
    text = sketch_to_text(s)
    assert "sketch 135a2760" in text
    assert "$bg" in text
    assert "$frame" in text
    assert "REGION_PERIODIC_REPAIR" in text
    assert "-> repair" in text


def test_text_rendering_composite():
    s = _composite_alignment_sketch()
    text = sketch_to_text(s)
    assert "COMPOSITE_ROLE_ALIGNMENT" in text
    assert "$anchor" in text
    assert "$center" in text
    assert "?axis" in text


def test_text_rendering_canvas():
    s = _canvas_layout_sketch()
    text = sketch_to_text(s)
    assert "CANVAS_LAYOUT" in text
    assert "?output_dims" in text


def test_text_shows_confidence():
    step = SketchStep(
        name="s1",
        primitive=PrimitiveFamily.FILL_BY_RULE,
        confidence=0.7,
    )
    s = Sketch(task_id="test", steps=(step,), output_ref="s1")
    text = sketch_to_text(s)
    assert "70%" in text


# ---------------------------------------------------------------------------
# Builder convenience tests
# ---------------------------------------------------------------------------


def test_make_identify_roles():
    step = make_identify_roles(bg_colors=[0, 3, 5])
    assert step.primitive == PrimitiveFamily.IDENTIFY_ROLES
    assert any(r.kind == RoleKind.BG for r in step.roles)
    assert step.evidence["observed_bg_colors"] == [0, 3, 5]


def test_make_region_periodic_repair():
    step = make_region_periodic_repair(period_axis="row")
    assert step.primitive == PrimitiveFamily.REGION_PERIODIC_REPAIR
    axis_slot = next(s for s in step.slots if s.name == "axis")
    assert axis_slot.evidence == "row"
    period_slot = next(s for s in step.slots if s.name == "period")
    assert period_slot.constraint == "positive, infer from content"


def test_make_object_move_by_relation():
    step = make_object_move_by_relation(predicate_desc="non-anchor content objects")
    assert step.primitive == PrimitiveFamily.OBJECT_MOVE_BY_RELATION
    sel_slot = next(s for s in step.slots if s.name == "selection")
    assert "non-anchor" in sel_slot.constraint


def test_make_composite_role_alignment():
    step = make_composite_role_alignment(axis="col")
    assert step.primitive == PrimitiveFamily.COMPOSITE_ROLE_ALIGNMENT
    assert any(r.kind == RoleKind.ANCHOR for r in step.roles)
    assert any(r.kind == RoleKind.CENTER for r in step.roles)
    axis_slot = next(s for s in step.slots if s.name == "axis")
    assert axis_slot.evidence == "col"


def test_make_canvas_layout_fixed():
    step = make_canvas_layout(dims_evidence=(20, 20), fixed_output=True)
    assert step.primitive == PrimitiveFamily.CANVAS_LAYOUT
    dims_slot = next(s for s in step.slots if s.name == "output_dims")
    assert dims_slot.constraint == "fixed"
    assert dims_slot.evidence == (20, 20)


def test_make_canvas_layout_inferred():
    step = make_canvas_layout()
    dims_slot = next(s for s in step.slots if s.name == "output_dims")
    assert dims_slot.constraint == "infer from input structure"
    assert dims_slot.evidence is None


# ---------------------------------------------------------------------------
# Sketch diversity — different tasks produce different sketches
# ---------------------------------------------------------------------------


def test_sketches_are_distinct():
    sketches = [
        _periodic_repair_sketch(),
        _composite_alignment_sketch(),
        _canvas_layout_sketch(),
        _object_move_sketch(),
    ]
    families = [s.primitive_families for s in sketches]
    # No two sketches should have identical primitive sequences
    assert len(set(families)) == len(families)
