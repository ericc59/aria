"""Bridge: re-exports AST types from aria.search for backward compatibility.

The canonical AST IR now lives in aria/search/ast.py.
The executor lives in aria/search/executor.py.
"""

from aria.search.ast import Op, ASTNode, ASTProgram  # noqa: F401
from aria.search.executor import execute_ast  # noqa: F401


def program_to_ast(desc: str) -> ASTNode | None:
    """Best-effort conversion of a program description string to AST.

    This is a bridge for existing closure-based programs.
    New programs should be built as ASTs directly.
    """
    desc = desc.strip()

    # Grid transforms
    for xform, op in [('grid_flip_h', Op.FLIP_H), ('grid_flip_v', Op.FLIP_V),
                       ('grid_flip_hv', Op.FLIP_HV), ('grid_rot90', Op.ROT90),
                       ('grid_rot180', Op.ROT180)]:
        if desc.startswith(f'synth: {xform}('):
            inner = desc[len(f'synth: {xform}('):-1]
            child = program_to_ast(inner) or ASTNode(Op.INPUT)
            return ASTNode(op, [child])

    # Crop
    if desc.startswith('synth: crop_bbox('):
        return ASTNode(Op.CROP_BBOX, [ASTNode(Op.INPUT), ASTNode(Op.HOLE)])
    if desc.startswith('synth: crop_interior('):
        return ASTNode(Op.CROP_INTERIOR, [ASTNode(Op.INPUT), ASTNode(Op.HOLE)])

    # Repair
    if 'repair_all_frames' in desc:
        return ASTNode(Op.REPAIR_FRAMES, [ASTNode(Op.INPUT)])

    # Render(combine(split))
    if 'render(combine_' in desc:
        return ASTNode(Op.RENDER, [
            ASTNode(Op.COMBINE, [ASTNode(Op.SPLIT, [ASTNode(Op.INPUT)])]),
            ASTNode(Op.HOLE, param='color'),
        ])

    # Tile
    if desc.startswith('tile '):
        return ASTNode(Op.TILE, [ASTNode(Op.INPUT)], param=desc)

    # Clause programs
    if desc.startswith('clause: '):
        return ASTNode(Op.COMPOSE, [], param=desc)

    # Transition programs
    if desc.startswith('synth: transition'):
        return ASTNode(Op.COMPOSE, [], param=desc)

    # Periodic extend
    if desc.startswith('periodic extend'):
        return ASTNode(Op.PERIODIC_EXTEND, [ASTNode(Op.INPUT)], param=desc)

    # Self-tile
    if desc.startswith('self-tile'):
        return ASTNode(Op.TILE, [ASTNode(Op.INPUT)], param=desc)

    return None
