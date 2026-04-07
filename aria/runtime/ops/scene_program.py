"""Runtime op wrapper for executing serialized scene programs."""

from __future__ import annotations

import json

from aria.core.scene_executor import execute_scene_program
from aria.runtime.ops import OpSignature, register
from aria.scene_ir import SceneProgram, SceneStep, StepOp
from aria.types import Grid, Type


def _execute_scene_program_json(grid: Grid, steps_json: str) -> Grid:
    """Execute a scene program from its JSON-serialized step list."""
    step_data = json.loads(steps_json)
    steps = []
    for sd in step_data:
        op = StepOp(sd["op"])
        inputs = tuple(sd.get("inputs", []))
        params = sd.get("params", {})
        output_id = sd.get("output_id")
        steps.append(SceneStep(op=op, inputs=inputs, params=params, output_id=output_id))
    program = SceneProgram(steps=tuple(steps))
    return execute_scene_program(program, grid)


register(
    "execute_scene_program_json",
    OpSignature(
        params=(("grid", Type.GRID), ("steps_json", Type.INT)),
        return_type=Type.GRID,
    ),
    _execute_scene_program_json,
)
