"""Build a persistent Level-1 library from a verified program corpus."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from aria.library.admission import check_admission
from aria.library.mining import extract_candidates, parameterize_candidate
from aria.library.mdl import mdl_improvement
from aria.library.store import Library
from aria.program_store import ProgramStore
from aria.proposer.parser import ParseError, parse_program
from aria.runtime.program import program_to_text
from aria.types import Assert, Bind, Call, Expr, Lambda, LibraryEntry, Literal, Program, Ref, Step, Type


@dataclass(frozen=True)
class CandidateAggregate:
    entry: LibraryEntry
    program_indexes: frozenset[int]
    support_task_ids: frozenset[str]
    support_program_texts: frozenset[str]
    mdl_gain: int


@dataclass(frozen=True)
class LibraryBuildReport:
    corpus_programs: int
    corpus_tasks: int
    candidates_mined: int
    aggregated_candidates: int
    admitted_entries: tuple[LibraryEntry, ...]
    rejected_reasons: dict[str, int] = field(default_factory=dict)


def programs_from_store(program_store: ProgramStore) -> list[Program]:
    programs: list[Program] = []
    for record in program_store.ranked_records():
        try:
            program = parse_program(record.program_text)
        except ParseError:
            continue
        # Distinct task ids are the right notion of reuse evidence here.
        # Replaying the same verified program across multiple tasks should count
        # as multiple corpus observations when mining/promoting abstractions.
        weight = max(record.distinct_task_count, 1)
        programs.extend(program for _ in range(weight))
    return programs


def build_library_from_store(
    program_store: ProgramStore,
    *,
    min_length: int = 2,
    max_length: int = 6,
    min_uses: int = 2,
    max_entries: int = 0,
) -> tuple[Library, LibraryBuildReport]:
    weighted_programs = programs_from_store(program_store)
    _AggValue = tuple[LibraryEntry, set[int], set[str], set[str], set[str]]
    aggregates: dict[str, _AggValue] = {}
    candidates_mined = 0

    for prog_idx, record in enumerate(program_store.ranked_records()):
        try:
            program = parse_program(record.program_text)
        except ParseError:
            continue

        for start, end in extract_candidates(program, min_length=min_length, max_length=max_length):
            result = parameterize_candidate(program, start, end)
            if result is None:
                continue
            params, steps, output_name = result
            return_type = _infer_output_type(steps, output_name)
            normalized = _normalize_entry(
                params=params,
                steps=steps,
                output=output_name,
                return_type=return_type,
            )
            candidates_mined += 1
            key = _candidate_key(normalized)
            existing = aggregates.get(key)
            if existing is None:
                aggregates[key] = (
                    normalized,
                    {prog_idx},
                    set(record.task_ids),
                    {record.program_text},
                    set(record.signatures),
                )
            else:
                existing_entry, indexes, support_task_ids, support_program_texts, sigs = existing
                indexes.add(prog_idx)
                support_task_ids.update(record.task_ids)
                support_program_texts.add(record.program_text)
                sigs.update(record.signatures)
                aggregates[key] = (
                    existing_entry,
                    indexes,
                    support_task_ids,
                    support_program_texts,
                    sigs,
                )

    ranked: list[CandidateAggregate] = []
    for key, (entry, program_indexes, support_task_ids, support_program_texts, sigs) in aggregates.items():
        candidate = LibraryEntry(
            name=_stable_name(entry, key),
            params=entry.params,
            return_type=entry.return_type,
            steps=entry.steps,
            output=entry.output,
            level=1,
            use_count=max(len(support_task_ids), len(program_indexes)),
            support_task_ids=tuple(sorted(support_task_ids)),
            support_program_count=len(support_program_texts),
            signatures=tuple(sorted(sigs)),
        )
        ranked.append(CandidateAggregate(
            entry=candidate,
            program_indexes=frozenset(program_indexes),
            support_task_ids=frozenset(support_task_ids),
            support_program_texts=frozenset(support_program_texts),
            mdl_gain=mdl_improvement(weighted_programs, candidate),
        ))

    return _admit_ranked_candidates(
        weighted_programs,
        ranked,
        min_uses=min_uses,
        max_entries=max_entries,
        corpus_tasks=len({
            task_id
            for record in program_store.all_records()
            for task_id in record.task_ids
        }),
        candidates_mined=candidates_mined,
    )


def build_library(
    programs: list[Program],
    *,
    min_length: int = 2,
    max_length: int = 6,
    min_uses: int = 2,
    max_entries: int = 0,
) -> tuple[Library, LibraryBuildReport]:
    """Mine and admit reusable abstractions from verified programs."""
    aggregates: dict[str, tuple[LibraryEntry, set[int]]] = {}
    candidates_mined = 0

    for prog_idx, program in enumerate(programs):
        for start, end in extract_candidates(program, min_length=min_length, max_length=max_length):
            result = parameterize_candidate(program, start, end)
            if result is None:
                continue
            params, steps, output_name = result
            return_type = _infer_output_type(steps, output_name)
            normalized = _normalize_entry(
                params=params,
                steps=steps,
                output=output_name,
                return_type=return_type,
            )
            candidates_mined += 1
            key = _candidate_key(normalized)
            existing = aggregates.get(key)
            if existing is None:
                aggregates[key] = (normalized, {prog_idx})
            else:
                existing_entry, indexes = existing
                indexes.add(prog_idx)
                aggregates[key] = (existing_entry, indexes)

    ranked: list[CandidateAggregate] = []
    for key, (entry, program_indexes) in aggregates.items():
        candidate = LibraryEntry(
            name=_stable_name(entry, key),
            params=entry.params,
            return_type=entry.return_type,
            steps=entry.steps,
            output=entry.output,
            level=1,
            use_count=len(program_indexes),
        )
        ranked.append(CandidateAggregate(
            entry=candidate,
            program_indexes=frozenset(program_indexes),
            support_task_ids=frozenset(),
            support_program_texts=frozenset(),
            mdl_gain=mdl_improvement(programs, candidate),
        ))
    return _admit_ranked_candidates(
        programs,
        ranked,
        min_uses=min_uses,
        max_entries=max_entries,
        corpus_tasks=0,
        candidates_mined=candidates_mined,
    )


def _admit_ranked_candidates(
    programs: list[Program],
    ranked: list[CandidateAggregate],
    *,
    min_uses: int,
    max_entries: int,
    corpus_tasks: int,
    candidates_mined: int,
) -> tuple[Library, LibraryBuildReport]:
    ranked.sort(
        key=lambda aggregate: (
            -aggregate.entry.use_count,
            -aggregate.mdl_gain,
            len(aggregate.entry.steps),
            aggregate.entry.name,
        )
    )

    library = Library()
    admitted_entries: list[LibraryEntry] = []
    rejected: dict[str, int] = {}

    for aggregate in ranked:
        admitted, reason = check_admission(
            aggregate.entry,
            programs,
            library,
            min_uses=min_uses,
        )
        if admitted:
            entry = LibraryEntry(
                name=aggregate.entry.name,
                params=aggregate.entry.params,
                return_type=aggregate.entry.return_type,
                steps=aggregate.entry.steps,
                output=aggregate.entry.output,
                level=aggregate.entry.level,
                use_count=aggregate.entry.use_count,
                support_task_ids=tuple(sorted(aggregate.support_task_ids)),
                support_program_count=(
                    len(aggregate.support_program_texts)
                    if aggregate.support_program_texts
                    else len(aggregate.program_indexes)
                ),
                mdl_gain=aggregate.mdl_gain,
                signatures=aggregate.entry.signatures,
            )
            library.add(entry)
            admitted_entries.append(entry)
            if max_entries and len(admitted_entries) >= max_entries:
                break
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    report = LibraryBuildReport(
        corpus_programs=len(programs),
        corpus_tasks=corpus_tasks,
        candidates_mined=candidates_mined,
        aggregated_candidates=len(ranked),
        admitted_entries=tuple(admitted_entries),
        rejected_reasons=rejected,
    )
    return library, report


def _infer_output_type(steps: tuple[Step, ...], output_name: str) -> Type:
    for step in steps:
        if isinstance(step, Bind) and step.name == output_name:
            return step.typ
    return Type.GRID


def _candidate_key(entry: LibraryEntry) -> str:
    signature = (
        tuple(param_type.name for _, param_type in entry.params),
        entry.return_type.name,
        program_to_text(Program(steps=entry.steps, output=entry.output)),
    )
    return repr(signature)


def _stable_name(entry: LibraryEntry, key: str) -> str:
    first_op = "recipe"
    for step in entry.steps:
        if isinstance(step, Bind) and isinstance(step.expr, Call):
            first_op = step.expr.op
            break
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"lib_{first_op}_{digest}"


def _normalize_entry(
    *,
    params: tuple[tuple[str, Type], ...],
    steps: tuple[Step, ...],
    output: str,
    return_type: Type,
) -> LibraryEntry:
    env = {name: f"arg{i}" for i, (name, _) in enumerate(params)}
    normalized_params = tuple((f"arg{i}", typ) for i, (_, typ) in enumerate(params))
    normalized_steps: list[Step] = []
    bind_counter = 0
    lambda_counter = 0

    def normalize_expr(expr: Expr, local_env: dict[str, str]) -> Expr:
        nonlocal lambda_counter
        match expr:
            case Ref(name=name):
                return Ref(name=local_env.get(name, name))
            case Literal():
                return expr
            case Call(op=op, args=args):
                return Call(op=op, args=tuple(normalize_expr(arg, local_env) for arg in args))
            case Lambda(param=param, param_type=param_type, body=body):
                alias = f"lam{lambda_counter}"
                lambda_counter += 1
                nested_env = dict(local_env)
                nested_env[param] = alias
                return Lambda(
                    param=alias,
                    param_type=param_type,
                    body=normalize_expr(body, nested_env),
                )
            case _:
                return expr

    for step in steps:
        if isinstance(step, Bind):
            expr = normalize_expr(step.expr, env)
            alias = f"v{bind_counter}"
            bind_counter += 1
            env = dict(env)
            env[step.name] = alias
            normalized_steps.append(Bind(
                name=alias,
                typ=step.typ,
                expr=expr,
                declared=step.declared,
            ))
        elif isinstance(step, Assert):
            normalized_steps.append(Assert(pred=normalize_expr(step.pred, env)))

    return LibraryEntry(
        name="candidate",
        params=normalized_params,
        return_type=return_type,
        steps=tuple(normalized_steps),
        output=env.get(output, output),
        level=1,
        use_count=0,
    )
