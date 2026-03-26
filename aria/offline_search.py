"""Deterministic program search for the offline solver path."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

from aria.graph.signatures import compute_task_signatures
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.reporting import extract_op_names
from aria.runtime.ops import OpSignature, all_ops
from aria.runtime.program import program_to_text
from aria.runtime.type_system import type_check
from aria.types import (
    Axis,
    Bind,
    Call,
    DemoPair,
    Dir,
    Expr,
    Literal,
    Program,
    Property,
    Ref,
    Shape,
    SortDir,
    Type,
    ZoneRole,
)
from aria.verify.verifier import verify


_CALLABLE_TYPES = frozenset({
    Type.PREDICATE,
    Type.OBJ_TRANSFORM,
    Type.GRID_TRANSFORM,
    Type.CALLABLE,
})


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    winning_program: Program | None
    candidates_tried: int


@dataclass(frozen=True)
class SearchTraceEntry:
    candidate_num: int
    depth: int
    program_text: str
    passed: bool
    failed_demo: int | None = None
    error_type: str | None = None
    diff: dict[str, Any] | None = None
    score: float | None = None
    score_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Binding:
    name: str
    typ: Type
    expr: Expr


@dataclass(frozen=True)
class _State:
    steps: tuple[Bind, ...]
    bindings: tuple[_Binding, ...]


def search_program(
    demos: tuple[DemoPair, ...],
    library: Library,
    *,
    program_store: ProgramStore | None = None,
    max_steps: int = 3,
    max_candidates: int = 5000,
    include_core_ops: bool = True,
    allowed_ops: frozenset[str] | None = None,
    preferred_ops: frozenset[str] | None = None,
    excluded_ops: frozenset[str] | None = None,
    depth_preferred_fn: Callable[[int, int], frozenset[str]] | None = None,
    observer: Callable[[SearchTraceEntry], None] | None = None,
) -> SearchResult:
    """Enumerate short typed programs using core ops and library entries.

    *preferred_ops*: op names that get a ranking boost at every depth.
    *excluded_ops*: op names to hard-exclude from the search space.
    *depth_preferred_fn*: callable(depth, max_steps) → frozenset[str]
        returning per-depth preferred ops (from decomposition sub-goals).
    """
    task_signatures = compute_task_signatures(demos)
    literal_pool = _build_literal_pool(demos)

    candidates_tried = 0
    seen_programs: set[str] = set()
    seen_states: set[tuple[tuple[str, str], ...]] = {()}
    frontier = [_State(steps=(), bindings=())]

    for depth in range(1, max_steps + 1):
        # Compute per-depth preferred ops by merging global + depth-specific
        depth_preferred = preferred_ops or frozenset()
        if depth_preferred_fn is not None:
            depth_preferred = depth_preferred | depth_preferred_fn(depth, max_steps)

        op_signatures = _ranked_op_signatures(
            library,
            program_store,
            include_core_ops=include_core_ops,
            task_signatures=task_signatures,
            allowed_ops=allowed_ops,
            preferred_ops=depth_preferred,
            excluded_ops=excluded_ops,
        )

        frontier_budget = _depth_frontier_budget(
            depth=depth,
            max_steps=max_steps,
            max_candidates=max_candidates,
            candidates_tried=candidates_tried,
        )
        frontier = _prioritize_frontier(frontier, task_signatures)[:frontier_budget]
        next_frontier: list[_State] = []
        verified_this_depth = 0
        depth_budget = _depth_candidate_budget(
            depth=depth,
            max_steps=max_steps,
            max_candidates=max_candidates,
            candidates_tried=candidates_tried,
        )

        for state in frontier:
            available = _available_bindings(state)
            next_index = len(state.bindings)

            for step in _iter_candidate_steps(
                available,
                next_index,
                op_signatures,
                literal_pool,
            ):
                program = Program(steps=state.steps + (step,), output=step.name)
                key = program_to_text(program)
                if key in seen_programs:
                    continue
                seen_programs.add(key)

                type_errors = type_check(
                    program,
                    initial_env={"input": Type.GRID, "ctx": Type.TASK_CTX},
                )
                if type_errors:
                    continue

                new_binding = _Binding(name=step.name, typ=step.typ, expr=step.expr)
                new_state = _State(
                    steps=program.steps,
                    bindings=state.bindings + (new_binding,),
                )
                state_key = tuple((binding.typ.name, repr(binding.expr)) for binding in new_state.bindings)
                if state_key not in seen_states and len(new_state.steps) < max_steps:
                    seen_states.add(state_key)
                    next_frontier.append(new_state)

                if step.typ != Type.GRID:
                    continue
                if depth < max_steps and verified_this_depth >= depth_budget:
                    continue

                candidates_tried += 1
                verified_this_depth += 1
                verify_result = verify(program, demos)
                if observer is not None:
                    observer(SearchTraceEntry(
                        candidate_num=candidates_tried,
                        depth=len(program.steps),
                        program_text=key,
                        passed=verify_result.passed,
                        failed_demo=verify_result.failed_demo,
                        error_type=verify_result.error_type,
                        diff=verify_result.diff,
                    ))
                if verify_result.passed:
                    return SearchResult(
                        solved=True,
                        winning_program=program,
                        candidates_tried=candidates_tried,
                    )
                if candidates_tried >= max_candidates:
                    return SearchResult(
                        solved=False,
                        winning_program=None,
                        candidates_tried=candidates_tried,
                    )

        frontier = next_frontier
        if not frontier:
            break

    return SearchResult(
        solved=False,
        winning_program=None,
        candidates_tried=candidates_tried,
    )


def _iter_candidate_steps(
    available: tuple[_Binding, ...],
    next_index: int,
    op_signatures: list[tuple[str, OpSignature]],
    literal_pool: dict[Type, tuple[Literal, ...]],
) -> Iterable[Bind]:
    binding_name = f"v{next_index}"
    derived_names = {binding.name for binding in available if binding.name not in {"input", "ctx"}}

    for op_name, sig in op_signatures:
        arg_choices = [
            _choices_for_param(op_name, param_name, param_type, available, literal_pool)
            for param_name, param_type in sig.params
        ]
        if not arg_choices or any(not choices for choices in arg_choices):
            continue

        product_cap = _combo_cap(len(arg_choices))
        for args in _iter_arg_products(arg_choices, product_cap):
            if derived_names and sig.return_type == Type.GRID and not _uses_derived_binding(args, derived_names):
                continue
            yield Bind(
                name=binding_name,
                typ=sig.return_type,
                expr=Call(op=op_name, args=tuple(args)),
            )

    for binding in available:
        if binding.typ == Type.GRID_TRANSFORM:
            for arg in _choices_for_param(binding.name, "arg0", Type.GRID, available, literal_pool):
                yield Bind(
                    name=binding_name,
                    typ=Type.GRID,
                    expr=Call(op=binding.name, args=(arg,)),
                )
        elif binding.typ == Type.OBJ_TRANSFORM:
            for arg in _choices_for_param(binding.name, "arg0", Type.OBJECT, available, literal_pool):
                yield Bind(
                    name=binding_name,
                    typ=Type.OBJECT,
                    expr=Call(op=binding.name, args=(arg,)),
                )
        elif binding.typ == Type.PREDICATE:
            for arg in _choices_for_param(binding.name, "arg0", Type.OBJECT, available, literal_pool):
                yield Bind(
                    name=binding_name,
                    typ=Type.BOOL,
                    expr=Call(op=binding.name, args=(arg,)),
                )


def _available_bindings(state: _State) -> tuple[_Binding, ...]:
    return (
        _Binding("input", Type.GRID, Ref("input")),
        _Binding("ctx", Type.TASK_CTX, Ref("ctx")),
        *state.bindings,
    )


def _choices_for_param(
    op_name: str,
    param_name: str,
    param_type: Type,
    available: tuple[_Binding, ...],
    literal_pool: dict[Type, tuple[Literal, ...]],
) -> tuple[Expr, ...]:
    refs = [
        Ref(binding.name)
        for binding in reversed(available)
        if _types_compatible(param_type, binding.typ)
    ]
    literals = list(_literals_for_param(op_name, param_name, param_type, literal_pool))
    return tuple(refs + literals)


def _types_compatible(expected: Type, actual: Type) -> bool:
    if expected == actual:
        return True
    if expected == Type.CALLABLE and actual in _CALLABLE_TYPES:
        return True
    if actual == Type.CALLABLE and expected in _CALLABLE_TYPES:
        return True
    if {expected, actual} == {Type.INT, Type.COLOR}:
        return True
    return False


def _iter_arg_products(arg_choices: list[tuple[Expr, ...]], cap: int) -> Iterable[tuple[Expr, ...]]:
    seen: set[tuple[str, ...]] = set()
    yielded = 0
    for args in product(*arg_choices):
        key = tuple(repr(arg) for arg in args)
        if key in seen:
            continue
        seen.add(key)
        yield args
        yielded += 1
        if yielded >= cap:
            break


def _combo_cap(arity: int) -> int:
    if arity <= 1:
        return 32
    if arity == 2:
        return 96
    return 48


def _uses_derived_binding(args: tuple[Expr, ...], derived_names: set[str]) -> bool:
    return any(isinstance(arg, Ref) and arg.name in derived_names for arg in args)


def _depth_candidate_budget(
    *,
    depth: int,
    max_steps: int,
    max_candidates: int,
    candidates_tried: int,
) -> int:
    remaining_budget = max(max_candidates - candidates_tried, 0)
    if depth >= max_steps:
        return remaining_budget
    remaining_depths = max_steps - depth + 1
    return max(1, remaining_budget // remaining_depths)


def _depth_frontier_budget(
    *,
    depth: int,
    max_steps: int,
    max_candidates: int,
    candidates_tried: int,
) -> int:
    depth_budget = _depth_candidate_budget(
        depth=depth,
        max_steps=max_steps,
        max_candidates=max_candidates,
        candidates_tried=candidates_tried,
    )
    if depth >= max_steps:
        return max(8, depth_budget)
    return max(32, depth_budget * 8)


def _prioritize_frontier(
    frontier: list[_State],
    task_signatures: frozenset[str],
) -> list[_State]:
    return sorted(
        frontier,
        key=lambda state: (
            -_state_bonus(state, task_signatures),
            -len(state.bindings),
            program_to_text(Program(steps=state.steps, output=state.bindings[-1].name if state.bindings else "input")),
        ),
    )


def _state_bonus(state: _State, task_signatures: frozenset[str]) -> int:
    bonus = 0
    marker_additive = (
        "change:additive" in task_signatures
        and "role:has_marker" in task_signatures
        and "dims:same" in task_signatures
    )
    if not marker_additive:
        return bonus

    type_counts = Counter(binding.typ for binding in state.bindings)
    bonus += 12 * type_counts.get(Type.OBJECT_SET, 0)
    bonus += 10 * type_counts.get(Type.PREDICATE, 0)
    bonus += 8 * type_counts.get(Type.OBJECT, 0)
    bonus += 4 * type_counts.get(Type.INT, 0)
    bonus -= 6 * type_counts.get(Type.GRID, 0)

    op_names = {
        binding.expr.op
        for binding in state.bindings
        if isinstance(binding.expr, Call)
    }
    if "find_objects" in op_names:
        bonus += 12
    if "where" in op_names:
        bonus += 10
    if "by_color" in op_names:
        bonus += 10
    if "nearest_to" in op_names:
        bonus += 8
    if "chebyshev_distance" in op_names:
        bonus += 6

    return bonus


def _ranked_op_signatures(
    library: Library,
    program_store: ProgramStore | None,
    *,
    include_core_ops: bool,
    task_signatures: frozenset[str],
    allowed_ops: frozenset[str] | None = None,
    preferred_ops: frozenset[str] | None = None,
    excluded_ops: frozenset[str] | None = None,
) -> list[tuple[str, OpSignature]]:
    library_scores = {entry.name: entry.use_count for entry in library.all_entries()}
    corpus_scores = Counter()
    preferred = preferred_ops or frozenset()
    excluded = excluded_ops or frozenset()

    if program_store is not None:
        for record in program_store.ranked_records():
            for op_name in extract_op_names(record.program_text):
                corpus_scores[op_name] += max(record.use_count, 1)

    ranked: list[tuple[str, OpSignature]] = []
    for name, sig in all_ops().items():
        if name in excluded:
            continue
        if allowed_ops is not None and name not in allowed_ops:
            continue
        if not include_core_ops and name not in library_scores:
            continue
        if not _is_searchable_sig(sig):
            continue
        ranked.append((name, sig))

    ranked.sort(
        key=lambda item: (
            -(30 if item[0] in preferred else 0),
            -_signature_bonus(item[0], task_signatures),
            -corpus_scores.get(item[0], 0),
            len(item[1].params),
            1 if item[0] in library_scores else 0,
            -library_scores.get(item[0], 0),
            item[0],
        )
    )
    return ranked


def _is_searchable_sig(sig: OpSignature) -> bool:
    if len(sig.params) == 0 or len(sig.params) > 3:
        return False
    if sig.return_type in {Type.BOOL, Type.OBJ_TRANSFORM, Type.CALLABLE, Type.PAIR}:
        return False
    unsupported = {Type.OBJ_TRANSFORM, Type.CALLABLE}
    if any(param_type in unsupported for _, param_type in sig.params):
        return False
    return True


def _build_literal_pool(demos: tuple[DemoPair, ...]) -> dict[Type, tuple[Literal, ...]]:
    colors_seen: list[int] = []
    ints_seen: list[int] = [-1, 0, 1, 2, 3, 4]

    for demo in demos:
        for grid in (demo.input, demo.output):
            for value in grid.ravel().tolist():
                colors_seen.append(int(value))
            ints_seen.extend([int(grid.shape[0]), int(grid.shape[1])])

    color_values = _dedupe_preserve_order(colors_seen + list(range(10)))
    int_values = _dedupe_preserve_order(ints_seen + [len(demos)])

    return {
        Type.COLOR: tuple(Literal(value, Type.COLOR) for value in color_values),
        Type.INT: tuple(Literal(value, Type.INT) for value in int_values),
        Type.BOOL: (
            Literal(False, Type.BOOL),
            Literal(True, Type.BOOL),
        ),
        Type.AXIS: tuple(Literal(axis, Type.AXIS) for axis in Axis),
        Type.DIR: tuple(Literal(direction, Type.DIR) for direction in Dir),
        Type.SHAPE: tuple(Literal(shape, Type.SHAPE) for shape in Shape),
        Type.PROPERTY: tuple(Literal(prop, Type.PROPERTY) for prop in Property),
        Type.SORT_DIR: tuple(Literal(direction, Type.SORT_DIR) for direction in SortDir),
        Type.ZONE_ROLE: tuple(Literal(role, Type.ZONE_ROLE) for role in ZoneRole),
    }


def _dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def excluded_ops_from_signatures(task_signatures: frozenset[str]) -> frozenset[str]:
    """Derive hard op exclusions from task signatures.

    This is the type-level constraint step: if signatures establish that
    the task preserves dimensions, exclude ops that change dimensions, and
    vice versa. This narrows the search space without losing correctness.
    """
    excluded: set[str] = set()

    if "dims:same" in task_signatures:
        # Same-size tasks: exclude ops that change grid dimensions
        excluded.update({
            "tile_grid", "upscale_grid", "scale_dims", "stack_h", "stack_v",
            "embed",
        })

    if "dims:different" in task_signatures:
        # Different-size tasks: exclude ops that only make sense for same-size
        # (nothing to exclude here — different-size tasks need flexible ops)
        pass

    return frozenset(excluded)


def build_literal_pool(demos: tuple[DemoPair, ...]) -> dict[Type, tuple[Literal, ...]]:
    """Public access to the literal pool builder."""
    return _build_literal_pool(demos)


def _literals_for_param(
    op_name: str,
    param_name: str,
    param_type: Type,
    literal_pool: dict[Type, tuple[Literal, ...]],
) -> tuple[Literal, ...]:
    if param_type != Type.INT:
        return literal_pool.get(param_type, ())

    base_ints = [int(literal.value) for literal in literal_pool.get(Type.INT, ())]

    if op_name in {"rotate", "rotate_grid"} and param_name == "degrees":
        values = [90, 180, 270]
    elif param_name in {"rows", "cols", "factor", "r", "c"}:
        values = [value for value in base_ints if 1 <= value <= 30]
    elif param_name in {"idx", "index", "rank"}:
        values = [value for value in base_ints if -1 <= value <= 4]
    else:
        values = base_ints

    return tuple(Literal(value, Type.INT) for value in _dedupe_preserve_order(values))


def _signature_bonus(op_name: str, task_signatures: frozenset[str]) -> int:
    bonus = 0

    if "size:multiplicative" in task_signatures:
        if op_name == "upscale_grid":
            bonus += 20
        elif op_name == "predict_dims":
            bonus += 10
        elif op_name == "scale_dims":
            bonus += 9
        elif op_name == "tile_grid":
            bonus += 8
        elif op_name == "new_grid":
            bonus += 6

    if "size:scale_2x" in task_signatures:
        if op_name == "upscale_grid":
            bonus += 12
        elif op_name == "tile_grid":
            bonus += 4

    if "size:additive" in task_signatures or "size:fixed_output_shape" in task_signatures:
        if op_name == "predict_dims":
            bonus += 12
        elif op_name == "dims_make":
            bonus += 9
        elif op_name == "new_grid":
            bonus += 8
        elif op_name == "rows_of" or op_name == "cols_of":
            bonus += 5

    if "dims:same" in task_signatures and op_name in {"tile_grid", "upscale_grid", "scale_dims", "predict_dims", "new_grid"}:
        bonus -= 8

    if "change:additive" in task_signatures or "color:new_in_output" in task_signatures:
        if op_name in {"fill_region", "box_region", "square_region"}:
            bonus += 10
        elif op_name in {"find_objects", "where", "by_color", "nearest_to"}:
            bonus += 8
        elif op_name in {"chebyshev_distance", "center_x", "center_y"}:
            bonus += 6

    if "role:has_marker" in task_signatures:
        if op_name in {"find_objects", "where", "by_color", "nearest_to"}:
            bonus += 8

    return bonus
