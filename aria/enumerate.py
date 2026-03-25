"""Program enumerator for ARIA Phase 1 — corpus construction.

Systematically searches for correct multi-step programs in the ARIA step
language by enumerating step sequences, executing them against demo pairs,
and verifying pixel-perfect output.

Strategy:
- Breadth-first: try shortest programs first (length 1, then 2, ...)
- Type-guided: only attempt ops whose input types are satisfiable from
  the current environment
- Pattern-seeded: start from common program skeletons rather than
  purely blind combinatorics
- Early pruning: skip partial programs that cannot produce a GRID output
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from aria.types import (
    Axis,
    Bind,
    Call,
    Color,
    DemoPair,
    Dir,
    Expr,
    Grid,
    Literal,
    Program,
    Property,
    Ref,
    Shape,
    SortDir,
    Step,
    Task,
    Type,
    grid_eq,
    grid_from_list,
    make_grid,
)
from aria.runtime.executor import ExecutionError, execute
from aria.runtime.ops import OpSignature, all_ops, get_op
from aria.verify.verifier import verify_stateless


# ---------------------------------------------------------------------------
# Type compatibility helpers
# ---------------------------------------------------------------------------

# Map from Type enum to the set of Type enums that are assignment-compatible.
# E.g. a COLOR can be used where an INT is expected.
_TYPE_COMPAT: dict[Type, set[Type]] = {
    Type.COLOR: {Type.COLOR, Type.INT},
    Type.INT: {Type.INT, Type.COLOR},
}


def _type_satisfies(available: Type, required: Type) -> bool:
    """Check if a value of type `available` can fill a parameter of type `required`."""
    if available == required:
        return True
    return required in _TYPE_COMPAT.get(available, set())


# ---------------------------------------------------------------------------
# Literal value pools — small sets of constants to try for each type
# ---------------------------------------------------------------------------

_LITERAL_POOL: dict[Type, list[Any]] = {
    Type.COLOR: list(range(10)),
    Type.INT: [0, 1, 2, 3, -1],
    Type.BOOL: [True, False],
    Type.DIR: [Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT],
    Type.AXIS: [Axis.HORIZONTAL, Axis.VERTICAL],
    Type.SHAPE: [Shape.RECT, Shape.LINE, Shape.DOT],
    Type.PROPERTY: [Property.COLOR, Property.SIZE, Property.SHAPE,
                    Property.POS_X, Property.POS_Y],
    Type.SORT_DIR: [SortDir.ASC, SortDir.DESC],
    Type.DIMS: [(1, 1), (2, 2), (3, 3)],
}


def _literals_for_type(typ: Type) -> list[Literal]:
    """Return candidate Literal nodes for a given type."""
    if typ not in _LITERAL_POOL:
        return []
    return [Literal(value=v, typ=typ) for v in _LITERAL_POOL[typ]]


# ---------------------------------------------------------------------------
# Environment type tracking
# ---------------------------------------------------------------------------

class TypeEnv:
    """Tracks name->Type bindings available at a given point in a program."""

    __slots__ = ("_bindings",)

    def __init__(self) -> None:
        self._bindings: dict[str, Type] = {"input": Type.GRID}

    def copy(self) -> TypeEnv:
        te = TypeEnv()
        te._bindings = dict(self._bindings)
        return te

    def bind(self, name: str, typ: Type) -> None:
        self._bindings[name] = typ

    def refs_for_type(self, typ: Type) -> list[Ref]:
        """Return all Ref nodes whose type satisfies `typ`."""
        return [Ref(name=n) for n, t in self._bindings.items()
                if _type_satisfies(t, typ)]

    def has_type(self, typ: Type) -> bool:
        """Check if any binding can satisfy `typ`."""
        return any(_type_satisfies(t, typ) for t in self._bindings.values())

    def args_for_type(self, typ: Type) -> list[Expr]:
        """Return all possible argument expressions for a parameter type.

        Combines Ref lookups with small literal pools.
        """
        candidates: list[Expr] = self.refs_for_type(typ)
        candidates.extend(_literals_for_type(typ))
        return candidates

    @property
    def bindings(self) -> dict[str, Type]:
        return dict(self._bindings)


# ---------------------------------------------------------------------------
# Op filtering
# ---------------------------------------------------------------------------

# Ops to skip during enumeration: context ops (need TaskContext),
# higher-order ops (need lambdas — handled separately), stubs.
_SKIP_OPS = frozenset({
    "compose", "map_obj", "map_list", "fold", "if_then_else",
    "repeat_apply", "for_each_place",
    "demo_count", "demo_at", "infer_step", "infer_map",
    "disambiguate", "infer_iteration", "predict_dims",
    "related_to",  # stub
})

# Ops that are good first steps (produce OBJECT_SET from GRID).
_STARTER_OPS = frozenset({"find_objects", "connected_components"})


def _enumerable_ops() -> dict[str, OpSignature]:
    """Return ops suitable for enumeration (no higher-order, no context)."""
    return {name: sig for name, sig in all_ops().items()
            if name not in _SKIP_OPS}


def _can_fill_params(sig: OpSignature, env: TypeEnv) -> bool:
    """Check if every parameter of an op can be filled from env or literals."""
    for _, ptype in sig.params:
        if not env.has_type(ptype) and ptype not in _LITERAL_POOL:
            return False
    return True


# ---------------------------------------------------------------------------
# Argument enumeration for an op
# ---------------------------------------------------------------------------

def _arg_candidates(sig: OpSignature, env: TypeEnv,
                    max_per_param: int = 6) -> list[tuple[Expr, ...]]:
    """Enumerate all feasible argument tuples for an op, capped per param."""
    per_param: list[list[Expr]] = []
    for _, ptype in sig.params:
        candidates = env.args_for_type(ptype)
        if not candidates:
            return []
        # Limit to avoid combinatorial explosion
        per_param.append(candidates[:max_per_param])
    # Cartesian product, capped
    results: list[tuple[Expr, ...]] = []
    for combo in itertools.product(*per_param):
        results.append(combo)
        if len(results) >= 200:
            break
    return results


# ---------------------------------------------------------------------------
# Task-adaptive literal pools
# ---------------------------------------------------------------------------

def _extract_task_colors(task: Task) -> list[int]:
    """Extract all colors that appear in a task's demo grids."""
    colors: set[int] = set()
    for demo in task.train:
        colors.update(int(c) for c in np.unique(demo.input))
        colors.update(int(c) for c in np.unique(demo.output))
    return sorted(colors)


def _task_adaptive_literals(task: Task) -> dict[Type, list[Any]]:
    """Build a task-specific literal pool based on demo analysis."""
    colors = _extract_task_colors(task)
    pool = dict(_LITERAL_POOL)
    pool[Type.COLOR] = colors
    # Add task-relevant integers (grid dims, object counts)
    ints = set(pool[Type.INT])
    for demo in task.train:
        ints.add(demo.input.shape[0])
        ints.add(demo.input.shape[1])
        ints.add(demo.output.shape[0])
        ints.add(demo.output.shape[1])
    pool[Type.INT] = sorted(ints)
    return pool


# ---------------------------------------------------------------------------
# Core enumeration engine
# ---------------------------------------------------------------------------

def _verify_candidate(program: Program, demos: tuple[DemoPair, ...]) -> bool:
    """Quick verification: execute on all demos, check pixel-perfect match."""
    for demo in demos:
        try:
            result = execute(program, demo.input)
        except (ExecutionError, Exception):
            return False
        if not grid_eq(result, demo.output):
            return False
    return True


def _make_step(name: str, typ: Type, op_name: str,
               args: tuple[Expr, ...]) -> Bind:
    """Construct a Bind step."""
    return Bind(name=name, typ=typ, expr=Call(op=op_name, args=args))


def _program_hash(program: Program) -> int:
    """Cheap structural hash for deduplication."""
    parts: list[str] = []
    for step in program.steps:
        if isinstance(step, Bind):
            parts.append(f"{step.name}={_expr_str(step.expr)}")
    parts.append(f"out={program.output}")
    return hash(tuple(parts))


def _expr_str(expr: Expr) -> str:
    """String representation of an expression for hashing."""
    if isinstance(expr, Ref):
        return f"${expr.name}"
    if isinstance(expr, Literal):
        return f"#{expr.value}"
    if isinstance(expr, Call):
        args = ",".join(_expr_str(a) for a in expr.args)
        return f"{expr.op}({args})"
    return "?"


class _EnumState:
    """Mutable state for one enumeration run."""

    def __init__(self, demos: tuple[DemoPair, ...],
                 max_candidates: int, deadline: float,
                 task_pool: dict[Type, list[Any]] | None = None):
        self.demos = demos
        self.max_candidates = max_candidates
        self.deadline = deadline
        self.found: list[Program] = []
        self.seen_hashes: set[int] = set()
        self.candidates_tried = 0
        self.ops = _enumerable_ops()
        # Override global literal pool with task-specific one if provided
        self._task_pool = task_pool

    def timed_out(self) -> bool:
        return time.monotonic() > self.deadline

    def budget_exhausted(self) -> bool:
        return self.candidates_tried >= self.max_candidates

    def should_stop(self) -> bool:
        return self.timed_out() or self.budget_exhausted()

    def try_program(self, program: Program) -> bool:
        """Test a candidate. Returns True if it verified."""
        self.candidates_tried += 1
        h = _program_hash(program)
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)

        if _verify_candidate(program, self.demos):
            self.found.append(program)
            return True
        return False

    def literals_for(self, typ: Type) -> list[Literal]:
        """Task-aware literal generation."""
        pool = self._task_pool or _LITERAL_POOL
        vals = pool.get(typ, [])
        return [Literal(value=v, typ=typ) for v in vals]

    def args_for_type_in_env(self, typ: Type, env: TypeEnv) -> list[Expr]:
        """Refs from env + task-aware literals."""
        candidates: list[Expr] = env.refs_for_type(typ)
        candidates.extend(self.literals_for(typ))
        return candidates

    def arg_candidates_for_op(self, sig: OpSignature, env: TypeEnv,
                              max_per_param: int = 5) -> list[tuple[Expr, ...]]:
        """Enumerate argument tuples for an op using task-aware pools."""
        per_param: list[list[Expr]] = []
        for _, ptype in sig.params:
            candidates = self.args_for_type_in_env(ptype, env)
            if not candidates:
                return []
            per_param.append(candidates[:max_per_param])
        results: list[tuple[Expr, ...]] = []
        for combo in itertools.product(*per_param):
            results.append(combo)
            if len(results) >= 150:
                break
        return results


# ---------------------------------------------------------------------------
# Enumeration strategies
# ---------------------------------------------------------------------------

def _enumerate_depth(state: _EnumState, steps_so_far: tuple[Step, ...],
                     env: TypeEnv, remaining_depth: int,
                     step_counter: int) -> int:
    """Depth-first enumeration with type-guided pruning.

    Returns updated step_counter.
    """
    if state.should_stop():
        return step_counter

    # At each depth, check if we can already yield a GRID as output.
    # Try making a program from what we have.
    if steps_so_far:
        for name, typ in env.bindings.items():
            if typ == Type.GRID and name != "input":
                prog = Program(steps=steps_so_far, output=name)
                state.try_program(prog)
                if state.should_stop():
                    return step_counter

    if remaining_depth <= 0:
        return step_counter

    # Try each op whose params can be filled.
    for op_name, sig in state.ops.items():
        if state.should_stop():
            return step_counter
        if not _can_fill_params(sig, env):
            continue

        arg_combos = state.arg_candidates_for_op(sig, env)
        for args in arg_combos:
            if state.should_stop():
                return step_counter

            var_name = f"v{step_counter}"
            step = _make_step(var_name, sig.return_type, op_name, args)
            new_steps = steps_so_far + (step,)
            new_env = env.copy()
            new_env.bind(var_name, sig.return_type)

            step_counter += 1

            step_counter = _enumerate_depth(
                state, new_steps, new_env,
                remaining_depth - 1, step_counter
            )

    return step_counter


def _enumerate_identity(state: _EnumState) -> None:
    """Special case: test if output == input (identity program)."""
    prog = Program(steps=(), output="input")
    state.try_program(prog)


def _enumerate_single_op(state: _EnumState) -> None:
    """Enumerate 1-step programs: input -> op -> output."""
    env = TypeEnv()
    for op_name, sig in state.ops.items():
        if state.should_stop():
            return
        if sig.return_type != Type.GRID:
            continue
        if not _can_fill_params(sig, env):
            continue
        arg_combos = state.arg_candidates_for_op(sig, env)
        for args in arg_combos:
            if state.should_stop():
                return
            step = _make_step("result", sig.return_type, op_name, args)
            prog = Program(steps=(step,), output="result")
            state.try_program(prog)


def _enumerate_color_map_programs(state: _EnumState) -> None:
    """Enumerate programs of the form: apply_color_map({a: b, ...}, input).

    Color remapping is one of the most common ARC patterns. We try all
    single-color and two-color mappings derived from the task's palette.
    """
    colors = set()
    for demo in state.demos:
        colors.update(int(c) for c in np.unique(demo.input))
        colors.update(int(c) for c in np.unique(demo.output))
    colors_list = sorted(colors)

    # Single color mappings: one color changes to another
    for src in colors_list:
        for dst in colors_list:
            if src == dst:
                continue
            if state.should_stop():
                return
            mapping = {src: dst}
            step = Bind(
                name="result",
                typ=Type.GRID,
                expr=Call(
                    op="apply_color_map",
                    args=(Literal(value=mapping, typ=Type.COLOR_MAP),
                          Ref(name="input")),
                ),
            )
            prog = Program(steps=(step,), output="result")
            state.try_program(prog)

    # Two-color swaps
    for i, c1 in enumerate(colors_list):
        for c2 in colors_list[i + 1:]:
            if state.should_stop():
                return
            mapping = {c1: c2, c2: c1}
            step = Bind(
                name="result",
                typ=Type.GRID,
                expr=Call(
                    op="apply_color_map",
                    args=(Literal(value=mapping, typ=Type.COLOR_MAP),
                          Ref(name="input")),
                ),
            )
            prog = Program(steps=(step,), output="result")
            state.try_program(prog)


def _enumerate_inferred_color_map(state: _EnumState) -> None:
    """Infer the full color mapping from demo pairs and test it.

    Compares input/output pixel-by-pixel across all demos to build
    a consistent mapping. Handles the common case where every pixel
    changes color according to a fixed codebook.
    """
    mapping: dict[int, set[int]] = {}
    for demo in state.demos:
        if demo.input.shape != demo.output.shape:
            return  # Can't do pixel-level mapping with different shapes
        inp, out = demo.input, demo.output
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ic, oc = int(inp[r, c]), int(out[r, c])
                mapping.setdefault(ic, set()).add(oc)

    # Build deterministic mapping (each input color -> exactly one output color)
    cmap: dict[int, int] = {}
    for ic, ocs in mapping.items():
        if len(ocs) == 1:
            cmap[ic] = next(iter(ocs))
        else:
            return  # Ambiguous mapping

    if not cmap:
        return

    # Remove identity mappings to keep it clean
    cmap = {k: v for k, v in cmap.items() if k != v}
    if not cmap:
        return  # All identity = no change needed

    step = Bind(
        name="result",
        typ=Type.GRID,
        expr=Call(
            op="apply_color_map",
            args=(Literal(value=cmap, typ=Type.COLOR_MAP),
                  Ref(name="input")),
        ),
    )
    prog = Program(steps=(step,), output="result")
    state.try_program(prog)


def _enumerate_grid_transforms(state: _EnumState) -> None:
    """Try common whole-grid transforms: rotate, reflect, transpose."""
    for deg in [90, 180, 270]:
        step = Bind(
            name="result",
            typ=Type.GRID,
            expr=Call(op="rotate", args=(
                Literal(value=deg, typ=Type.INT),
                # rotate takes an Object, but we can try on the full grid
                # This won't work directly — rotate is Object->Object
                # Skip for now, need a grid-level rotate op
            )),
        )

    # Try flood_fill at various positions
    for demo in state.demos:
        if demo.input.shape != demo.output.shape:
            continue
        diff = demo.input != demo.output
        if not diff.any():
            continue
        # Find first differing pixel
        changed = np.argwhere(diff)
        if len(changed) > 0:
            r, c = int(changed[0][0]), int(changed[0][1])
            new_color = int(demo.output[r, c])
            step = Bind(
                name="result",
                typ=Type.GRID,
                expr=Call(op="flood_fill", args=(
                    Ref(name="input"),
                    Literal(value=(r, c), typ=Type.PAIR),
                    Literal(value=new_color, typ=Type.INT),
                )),
            )
            prog = Program(steps=(step,), output="result")
            state.try_program(prog)
            if state.should_stop():
                return


def _enumerate_find_recolor_place(state: _EnumState) -> None:
    """Enumerate: find_objects -> select/filter -> recolor -> place back.

    This covers a large family of ARC tasks: recolor objects matching
    some criterion, then reconstruct the grid.
    """
    colors = set()
    for demo in state.demos:
        colors.update(int(c) for c in np.unique(demo.input))
        colors.update(int(c) for c in np.unique(demo.output))

    # Step 1: find_objects(input)
    s1 = Bind(
        name="objs",
        typ=Type.OBJECT_SET,
        expr=Call(op="find_objects", args=(Ref(name="input"),)),
    )

    # For each source color -> target color, try:
    # where(by_color(src), objs) -> recolor(dst) via map_obj -> overlay
    # But map_obj is in _SKIP_OPS (higher-order). So we use a different
    # approach: build a color map and apply it.
    # Actually the most direct approach: apply_color_map on input.
    # Already covered by _enumerate_color_map_programs.

    # Alternative: find objects of one color, recolor each, place on new grid.
    # This is too complex for direct enumeration without higher-order ops.
    # Instead, enumerate 2-step programs starting with find_objects.
    env = TypeEnv()
    env.bind("objs", Type.OBJECT_SET)

    for op_name, sig in state.ops.items():
        if state.should_stop():
            return
        if sig.return_type != Type.GRID:
            continue
        if not _can_fill_params(sig, env):
            continue
        arg_combos = state.arg_candidates_for_op(sig, env, max_per_param=4)
        for args in arg_combos:
            if state.should_stop():
                return
            s2 = Bind(
                name="result",
                typ=Type.GRID,
                expr=Call(op=op_name, args=args),
            )
            prog = Program(steps=(s1, s2), output="result")
            state.try_program(prog)


def _enumerate_two_step_patterns(state: _EnumState) -> None:
    """Enumerate 2-step programs using common first-step ops."""
    # Common first steps and their return types
    first_steps: list[tuple[str, Bind, Type]] = []

    # find_objects(input)
    first_steps.append((
        "objs",
        Bind(name="objs", typ=Type.OBJECT_SET,
             expr=Call(op="find_objects", args=(Ref(name="input"),))),
        Type.OBJECT_SET,
    ))

    # dims_of(input)
    first_steps.append((
        "d",
        Bind(name="d", typ=Type.DIMS,
             expr=Call(op="dims_of", args=(Ref(name="input"),))),
        Type.DIMS,
    ))

    # find_zones(input)
    first_steps.append((
        "zones",
        Bind(name="zones", typ=Type.ZONE_LIST,
             expr=Call(op="find_zones", args=(Ref(name="input"),))),
        Type.ZONE_LIST,
    ))

    # connected_components(input)
    first_steps.append((
        "ccs",
        Bind(name="ccs", typ=Type.OBJECT_SET,
             expr=Call(op="connected_components", args=(Ref(name="input"),))),
        Type.OBJECT_SET,
    ))

    for var_name, s1, ret_type in first_steps:
        if state.should_stop():
            return
        env = TypeEnv()
        env.bind(var_name, ret_type)

        for op_name, sig in state.ops.items():
            if state.should_stop():
                return
            if sig.return_type != Type.GRID:
                continue
            if not _can_fill_params(sig, env):
                continue
            arg_combos = state.arg_candidates_for_op(sig, env, max_per_param=4)
            for args in arg_combos:
                if state.should_stop():
                    return
                s2 = Bind(
                    name="result",
                    typ=sig.return_type,
                    expr=Call(op=op_name, args=args),
                )
                prog = Program(steps=(s1, s2), output="result")
                state.try_program(prog)


def _enumerate_three_step_chains(state: _EnumState) -> None:
    """Enumerate 3-step programs with guided patterns.

    Pattern: first_op -> intermediate_op -> grid_producing_op
    """
    # Only do this if we have budget
    if state.candidates_tried > state.max_candidates // 2:
        return

    # find_objects -> filter/select -> grid op
    s1 = Bind(
        name="objs",
        typ=Type.OBJECT_SET,
        expr=Call(op="find_objects", args=(Ref(name="input"),)),
    )

    # Second step: ops that take OBJECT_SET and produce something useful
    mid_ops = {name: sig for name, sig in state.ops.items()
               if sig.params and any(pt == Type.OBJECT_SET
                                     for _, pt in sig.params)}

    env_after_s1 = TypeEnv()
    env_after_s1.bind("objs", Type.OBJECT_SET)

    for mid_name, mid_sig in mid_ops.items():
        if state.should_stop():
            return
        mid_combos = state.arg_candidates_for_op(mid_sig, env_after_s1,
                                                  max_per_param=3)
        for mid_args in mid_combos:
            if state.should_stop():
                return

            s2 = Bind(
                name="mid",
                typ=mid_sig.return_type,
                expr=Call(op=mid_name, args=mid_args),
            )

            env_after_s2 = env_after_s1.copy()
            env_after_s2.bind("mid", mid_sig.return_type)

            # If mid already produces GRID, try it
            if mid_sig.return_type == Type.GRID:
                prog = Program(steps=(s1, s2), output="mid")
                state.try_program(prog)

            # Third step: produce a GRID
            for fin_name, fin_sig in state.ops.items():
                if state.should_stop():
                    return
                if fin_sig.return_type != Type.GRID:
                    continue
                if not _can_fill_params(fin_sig, env_after_s2):
                    continue
                fin_combos = state.arg_candidates_for_op(
                    fin_sig, env_after_s2, max_per_param=3
                )
                for fin_args in fin_combos:
                    if state.should_stop():
                        return
                    s3 = Bind(
                        name="result",
                        typ=Type.GRID,
                        expr=Call(op=fin_name, args=fin_args),
                    )
                    prog = Program(steps=(s1, s2, s3), output="result")
                    state.try_program(prog)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enumerate_programs(
    task: Task,
    max_steps: int = 8,
    max_candidates: int = 10_000,
    timeout_sec: float = 30.0,
) -> list[Program]:
    """Search for correct programs for a task by bounded enumeration.

    Parameters
    ----------
    task : Task
        The ARC task with train demo pairs.
    max_steps : int
        Maximum number of steps in a program (up to 8 by default).
    max_candidates : int
        Maximum number of candidate programs to evaluate.
    timeout_sec : float
        Wall-clock time limit in seconds.

    Returns
    -------
    list[Program]
        All verified-correct programs found within budget.
    """
    demos = task.train
    if not demos:
        return []

    deadline = time.monotonic() + timeout_sec
    task_pool = _task_adaptive_literals(task)

    state = _EnumState(demos, max_candidates, deadline, task_pool)

    # Phase 0: identity
    _enumerate_identity(state)

    # Phase 0.5: inferred full color map (very cheap, very common)
    if not state.should_stop():
        _enumerate_inferred_color_map(state)

    # Phase 0.7: grid transforms (flood fill, etc.)
    if not state.should_stop():
        _enumerate_grid_transforms(state)

    # Phase 1: single-op programs (individual color maps)
    if not state.should_stop():
        _enumerate_color_map_programs(state)

    # Phase 2: other single-op programs
    if not state.should_stop():
        _enumerate_single_op(state)

    # Phase 3: two-step patterns
    if not state.should_stop():
        _enumerate_two_step_patterns(state)

    # Phase 4: find -> filter/transform -> place (3 steps)
    if not state.should_stop() and max_steps >= 3:
        _enumerate_three_step_chains(state)

    # Phase 5: deeper BFS if budget remains and max_steps > 3
    if not state.should_stop() and max_steps > 3:
        env = TypeEnv()
        _enumerate_depth(
            state, (), env,
            remaining_depth=min(max_steps, 4),  # cap depth to avoid explosion
            step_counter=100,
        )

    return state.found


def generate_synthetic_task(
    min_steps: int = 3,
    max_steps: int = 8,
    rng: random.Random | None = None,
) -> tuple[Task, Program] | None:
    """Generate a synthetic task by sampling a random valid program.

    Samples a random step sequence, executes it on random grids, and
    returns the resulting task and the known-correct program. Returns
    None if no valid task could be generated after several attempts.

    Parameters
    ----------
    min_steps : int
        Minimum number of meaningful ops in the program.
    max_steps : int
        Maximum number of steps.
    rng : random.Random or None
        Random number generator for reproducibility.

    Returns
    -------
    tuple[Task, Program] or None
        (task, program) if successful, None otherwise.
    """
    if rng is None:
        rng = random.Random()

    ops = _enumerable_ops()
    # Filter to ops that can be started from just a GRID (input)
    grid_only_types = {Type.GRID, Type.COLOR, Type.INT, Type.BOOL, Type.DIR,
                       Type.AXIS, Type.SHAPE, Type.PROPERTY, Type.SORT_DIR,
                       Type.DIMS}

    max_attempts = 50

    for _ in range(max_attempts):
        # Generate a random input grid
        rows = rng.randint(3, 10)
        cols = rng.randint(3, 10)
        n_colors = rng.randint(2, 5)
        palette = rng.sample(range(10), n_colors)
        grid_data = np.array(
            [[rng.choice(palette) for _ in range(cols)] for _ in range(rows)],
            dtype=np.uint8,
        )

        # Build a random program
        env = TypeEnv()
        steps: list[Step] = []
        step_idx = 0
        meaningful_ops = 0

        # Always start with find_objects
        s0 = Bind(
            name="v0",
            typ=Type.OBJECT_SET,
            expr=Call(op="find_objects", args=(Ref(name="input"),)),
        )
        steps.append(s0)
        env.bind("v0", Type.OBJECT_SET)
        step_idx = 1
        meaningful_ops = 1

        target_len = rng.randint(min_steps, max_steps)

        for _ in range(target_len - 1):
            # Pick a random op whose params we can fill
            feasible = [
                (name, sig) for name, sig in ops.items()
                if _can_fill_params(sig, env)
            ]
            if not feasible:
                break

            op_name, sig = rng.choice(feasible)

            # Pick random args
            args: list[Expr] = []
            ok = True
            for _, ptype in sig.params:
                candidates = env.args_for_type(ptype)
                if not candidates:
                    ok = False
                    break
                args.append(rng.choice(candidates))
            if not ok:
                continue

            var_name = f"v{step_idx}"
            step = _make_step(var_name, sig.return_type, op_name, tuple(args))
            steps.append(step)
            env.bind(var_name, sig.return_type)
            step_idx += 1
            meaningful_ops += 1

        if meaningful_ops < min_steps:
            continue

        # Find the last GRID binding
        output_name = None
        for s in reversed(steps):
            if isinstance(s, Bind) and s.typ == Type.GRID:
                output_name = s.name
                break

        # If no GRID output, try adding apply_color_map or new_grid
        if output_name is None:
            # Try adding a grid-producing step
            grid_ops = [(n, s) for n, s in ops.items()
                        if s.return_type == Type.GRID
                        and _can_fill_params(s, env)]
            if grid_ops:
                op_name, sig = rng.choice(grid_ops)
                args_list: list[Expr] = []
                ok = True
                for _, ptype in sig.params:
                    candidates = env.args_for_type(ptype)
                    if not candidates:
                        ok = False
                        break
                    args_list.append(rng.choice(candidates))
                if ok:
                    var_name = f"v{step_idx}"
                    step = _make_step(var_name, Type.GRID, op_name,
                                      tuple(args_list))
                    steps.append(step)
                    output_name = var_name

        if output_name is None:
            continue

        program = Program(steps=tuple(steps), output=output_name)

        # Execute on multiple random grids to build demos
        demos: list[DemoPair] = []
        failed = False
        for _ in range(4):  # 3 train + 1 test
            r = rng.randint(3, 10)
            c = rng.randint(3, 10)
            inp = np.array(
                [[rng.choice(palette) for _ in range(c)] for _ in range(r)],
                dtype=np.uint8,
            )
            try:
                out = execute(program, inp)
            except (ExecutionError, Exception):
                failed = True
                break
            if not isinstance(out, np.ndarray) or out.ndim != 2:
                failed = True
                break
            # Non-triviality check: output should differ from input
            if grid_eq(inp, out):
                failed = True
                break
            demos.append(DemoPair(input=inp, output=out))

        if failed or len(demos) < 4:
            continue

        task = Task(
            train=tuple(demos[:3]),
            test=tuple(demos[3:]),
        )
        return task, program

    return None


def build_corpus(
    tasks: list[Task],
    max_steps: int = 8,
    timeout_per_task: float = 30.0,
    max_candidates_per_task: int = 10_000,
) -> dict[str, Any]:
    """Run enumeration on a list of tasks and collect all correct programs.

    Parameters
    ----------
    tasks : list[Task]
        The tasks to solve.
    max_steps : int
        Max program length.
    timeout_per_task : float
        Timeout per task in seconds.
    max_candidates_per_task : int
        Max candidates per task.

    Returns
    -------
    dict with keys:
        - "programs": dict mapping task index to list of Programs
        - "tasks_solved": number of tasks with at least one program
        - "total_programs": total programs found
        - "time_sec": total wall-clock time
        - "per_task": list of dicts with per-task stats
    """
    t0 = time.monotonic()
    programs: dict[int, list[Program]] = {}
    per_task: list[dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        task_t0 = time.monotonic()
        found = enumerate_programs(
            task,
            max_steps=max_steps,
            max_candidates=max_candidates_per_task,
            timeout_sec=timeout_per_task,
        )
        task_elapsed = time.monotonic() - task_t0

        if found:
            programs[idx] = found

        per_task.append({
            "task_idx": idx,
            "programs_found": len(found),
            "time_sec": round(task_elapsed, 3),
        })

    total_time = time.monotonic() - t0
    tasks_solved = sum(1 for v in programs.values() if v)
    total_programs = sum(len(v) for v in programs.values())

    return {
        "programs": programs,
        "tasks_solved": tasks_solved,
        "total_programs": total_programs,
        "time_sec": round(total_time, 3),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# ARC data loading utility
# ---------------------------------------------------------------------------

def load_arc_task(path: str | Path) -> Task:
    """Load an ARC task from a JSON file.

    Expected format: {"train": [{"input": [[int]], "output": [[int]]}], "test": [...]}
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    train = tuple(
        DemoPair(
            input=grid_from_list(pair["input"]),
            output=grid_from_list(pair["output"]),
        )
        for pair in data["train"]
    )
    test = tuple(
        DemoPair(
            input=grid_from_list(pair["input"]),
            output=grid_from_list(pair["output"]),
        )
        for pair in data.get("test", [])
    )
    return Task(train=train, test=test)
