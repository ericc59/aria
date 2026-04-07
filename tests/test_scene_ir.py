from aria.scene_ir import (
    EntityKind,
    OutputGridSpec,
    RelationKind,
    Scene,
    SceneEntity,
    SceneProgram,
    SceneRelation,
    SceneStep,
    StepOp,
)


def test_scene_ir_can_model_separator_boolean_task():
    left = SceneEntity(
        id="panel:left",
        kind=EntityKind.PANEL,
        bbox=(0, 0, 2, 2),
        colors=(0, 1),
    )
    right = SceneEntity(
        id="panel:right",
        kind=EntityKind.PANEL,
        bbox=(0, 4, 2, 6),
        colors=(0, 1),
    )
    sep = SceneEntity(
        id="sep:0",
        kind=EntityKind.SEPARATOR,
        bbox=(0, 3, 2, 3),
        colors=(5,),
    )
    relation = SceneRelation(
        kind=RelationKind.CORRESPONDS_TO,
        source_id="panel:left",
        target_id="panel:right",
        attrs={"mode": "paired_panels"},
    )
    scene = Scene(shape=(3, 7), background=0, entities=(left, right, sep), relations=(relation,))
    assert scene.shape == (3, 7)
    assert {entity.kind for entity in scene.entities} == {EntityKind.PANEL, EntityKind.SEPARATOR}


def test_scene_program_preserves_step_order():
    program = SceneProgram(
        steps=(
            SceneStep(StepOp.INFER_OUTPUT_SIZE, params={"mode": "separator_cell_shape"}, output_id="out:size"),
            SceneStep(
                StepOp.INFER_OUTPUT_BACKGROUND,
                params={"mode": "constant", "value": 0},
                output_id="out:bg",
            ),
            SceneStep(
                StepOp.INITIALIZE_OUTPUT_SCENE,
                inputs=("out:size", "out:bg"),
                output_id="scene:out",
            ),
            SceneStep(StepOp.PARSE_SCENE, output_id="scene:in"),
            SceneStep(StepOp.SPLIT_BY_SEPARATOR, inputs=("scene:in",), output_id="panel_pair"),
            SceneStep(
                StepOp.BOOLEAN_COMBINE_PANELS,
                inputs=("panel_pair", "scene:out"),
                params={"mode": "intersect", "fill_color": 2},
                output_id="result",
            ),
            SceneStep(StepOp.RENDER_SCENE, inputs=("result",)),
        )
    )
    assert program.step_names() == (
        "infer_output_size",
        "infer_output_background",
        "initialize_output_scene",
        "parse_scene",
        "split_by_separator",
        "boolean_combine_panels",
        "render_scene",
    )
    assert program.starts_with_output_spec()


def test_output_grid_spec_carries_shape_and_background():
    spec = OutputGridSpec(shape=(3, 3), background=0, attrs={"source": "separator_grid"})
    assert spec.shape == (3, 3)
    assert spec.background == 0
    assert spec.attrs["source"] == "separator_grid"
