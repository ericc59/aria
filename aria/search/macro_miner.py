"""First exact macro miner for aria/search.

Groups solved traces by structural keys and produces Macro objects
for patterns that repeat across multiple tasks.

Macros are:
- exact structural groupings (no fuzzy clustering)
- reducible to SearchProgram steps
- discardable if useless
- not runtime ontology

See docs/raw/aria_learning_roadmap.md Phase 3.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from aria.search.trace_schema import SolveTrace
from aria.search.macros import Macro, MacroLibrary


# ---------------------------------------------------------------------------
# Grouping keys
# ---------------------------------------------------------------------------

def _action_key(trace: SolveTrace) -> str:
    """Primary grouping: provenance + action sequence."""
    return f'{trace.provenance}|{trace.signature()}'


def _selector_key(trace: SolveTrace) -> str:
    """Secondary grouping: provenance + full selector-aware signature."""
    return f'{trace.provenance}|{trace.selector_signature()}'


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def mine_macros(
    traces: list[SolveTrace],
    *,
    min_frequency: int = 2,
    min_steps: int = 1,
    require_test_correct: bool = False,
) -> MacroLibrary:
    """Mine exact repeated compositions from solved traces.

    Groups traces by (provenance, action_signature). For groups that
    meet the frequency threshold, produces a Macro with a representative
    program template.

    Args:
        traces: solved trace records to mine
        min_frequency: minimum group size to produce a macro
        min_steps: minimum number of steps (filters trivial patterns)
        require_test_correct: if True, only include traces with test_correct=True

    Returns:
        MacroLibrary with the mined macros, ordered by frequency desc.
    """
    # Filter
    filtered = traces
    if require_test_correct:
        filtered = [t for t in filtered if t.test_correct is True]

    # Group by primary key
    groups: dict[str, list[SolveTrace]] = defaultdict(list)
    for trace in filtered:
        key = _action_key(trace)
        groups[key].append(trace)

    # Produce macros for qualifying groups
    macros: list[Macro] = []
    for key, group in groups.items():
        if len(group) < min_frequency:
            continue

        representative = group[0]
        if representative.n_steps < min_steps:
            continue

        # Collect metadata
        provenances = sorted(set(t.provenance for t in group))
        task_ids = sorted(set(t.task_id for t in group))
        test_correct_count = sum(1 for t in group if t.test_correct is True)
        solve_rate = test_correct_count / len(group) if group else 0.0

        # Build selector pattern from the most common selector signature
        sel_sig_counts: dict[str, int] = defaultdict(int)
        for t in group:
            sel_sig_counts[t.selector_signature()] += 1
        most_common_sel_sig = max(sel_sig_counts, key=sel_sig_counts.get)

        # Name: structural, not task-specific
        name = _macro_name(representative)

        template, param_schema = _parameterize_program(representative.program_dict)
        macro = Macro(
            name=name,
            description=f'{len(group)} tasks, {representative.signature()}',
            program_template=template,
            param_schema=param_schema,
            source_provenances=provenances,
            source_task_count=len(task_ids),
            frequency=len(group),
            solve_rate=solve_rate,
            action_signature=representative.signature(),
            selector_pattern=most_common_sel_sig,
        )
        macros.append(macro)

    # Sort by frequency descending
    macros.sort(key=lambda m: -m.frequency)

    lib = MacroLibrary()
    for m in macros:
        lib.add(m)
    return lib


def _macro_name(trace: SolveTrace) -> str:
    """Generate a structural macro name from a trace."""
    # Use provenance stem + action count
    prov = trace.provenance.split(':')[-1] if ':' in trace.provenance else trace.provenance
    n = trace.n_steps
    actions = set(trace.step_actions)
    if len(actions) == 1:
        action_part = f'{next(iter(actions))}_{n}'
    else:
        action_part = '_'.join(sorted(actions))
    return f'{prov}__{action_part}'


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

_PARAM_KEYS: dict[str, str] = {
    'color': 'color',
    'bg': 'color',
    'fill_color': 'color',
    'on_color': 'color',
    'off_color': 'color',
    'color_map': 'color_map',
    'stencils': 'stencils',
    'r0': 'index',
    'c0': 'index',
    'h': 'size',
    'w': 'size',
    'dr': 'delta',
    'dc': 'delta',
    'factor': 'scale',
    'rows': 'count',
    'cols': 'count',
    'n': 'count',
}


def _parameterize_program(program_dict: dict) -> tuple[dict, list[dict[str, Any]]]:
    """Replace literal params with placeholders and emit a param schema."""
    template = deepcopy(program_dict)
    param_schema: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)

    steps = template.get('steps', [])
    for step_idx, step in enumerate(steps):
        params = step.get('params')
        if isinstance(params, dict):
            _parameterize_dict(params, ['steps', step_idx, 'params'], param_schema, counters)

        sel = step.get('select')
        if isinstance(sel, dict):
            sparams = sel.get('params')
            if isinstance(sparams, dict):
                _parameterize_dict(
                    sparams,
                    ['steps', step_idx, 'select', 'params'],
                    param_schema,
                    counters,
                    only_keys={'color'},
                )

    return template, param_schema


def _parameterize_dict(
    params: dict,
    base_path: list[Any],
    param_schema: list[dict[str, Any]],
    counters: dict[str, int],
    *,
    only_keys: set[str] | None = None,
) -> None:
    for key, value in list(params.items()):
        if only_keys is not None and key not in only_keys:
            continue
        kind = _PARAM_KEYS.get(key)
        if kind is None:
            continue

        if key in {'color_map', 'stencils'}:
            name = f'{kind}{counters[kind]}'
            counters[kind] += 1
            params[key] = f'${name}'
            param_schema.append({
                'name': name,
                'kind': kind,
                'path': base_path + [key],
            })
            continue

        if not isinstance(value, int):
            continue
        name = f'{kind}{counters[kind]}'
        counters[kind] += 1
        params[key] = f'${name}'
        param_schema.append({
            'name': name,
            'kind': kind,
            'path': base_path + [key],
        })


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces_jsonl(path: str) -> list[SolveTrace]:
    """Load SolveTrace records from a JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(SolveTrace.from_json(line))
    return traces


def load_traces_from_eval_reports(paths: list[str]) -> list[SolveTrace]:
    """Load SolveTrace records from eval report JSON files."""
    import json
    traces = []
    seen: set[str] = set()
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for task_result in data.get('tasks', []):
            trace_dict = task_result.get('solve_trace')
            if trace_dict is None:
                continue
            tid = trace_dict.get('task_id', '')
            if tid in seen:
                continue
            seen.add(tid)
            traces.append(SolveTrace.from_dict(trace_dict))
    return traces
