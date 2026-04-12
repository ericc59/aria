"""Macro schema for learned compositions above primitives.

A Macro is a named, stored composition of low-level search steps
that was discovered through repeated successful solves. Macros are:

- learned/stored compositions, NOT new runtime ontology
- reducible to lower-level SearchProgram steps
- discardable if useless
- rankable by utility/frequency

This module defines the schema. Mining, storage, and retrieval
are handled separately (trace_capture, build_search_traces).

See docs/raw/aria_learning_roadmap.md Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json


@dataclass
class Macro:
    """A reusable learned composition of search steps.

    Not a primitive. Not an AST op. A stored pattern that the search
    engine can propose as a candidate, then lower to existing primitives.
    """
    # Identity
    name: str                          # structural name, NOT task-id
    description: str = ''

    # Structure: the composition as a SearchProgram dict
    program_template: dict = field(default_factory=dict)
    param_schema: list[dict[str, Any]] = field(default_factory=list)

    # Provenance: how this was discovered
    source_provenances: list[str] = field(default_factory=list)
    source_task_count: int = 0         # how many tasks produced this pattern

    # Utility
    frequency: int = 0                 # times this pattern was observed
    solve_rate: float = 0.0            # fraction of attempts that verified

    # Signature for matching
    action_signature: str = ''         # e.g. 'recolor -> recolor -> recolor'
    selector_pattern: str = ''         # e.g. 'largest -> rule -> rule -> smallest'

    def to_dict(self) -> dict:
        return {
            'schema_version': 1,
            'name': self.name,
            'description': self.description,
            'program_template': self.program_template,
            'param_schema': self.param_schema,
            'source_provenances': self.source_provenances,
            'source_task_count': self.source_task_count,
            'frequency': self.frequency,
            'solve_rate': self.solve_rate,
            'action_signature': self.action_signature,
            'selector_pattern': self.selector_pattern,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Macro:
        return cls(
            name=d.get('name', ''),
            description=d.get('description', ''),
            program_template=d.get('program_template', {}),
            param_schema=d.get('param_schema', []),
            source_provenances=d.get('source_provenances', []),
            source_task_count=d.get('source_task_count', 0),
            frequency=d.get('frequency', 0),
            solve_rate=d.get('solve_rate', 0.0),
            action_signature=d.get('action_signature', ''),
            selector_pattern=d.get('selector_pattern', ''),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> Macro:
        return cls.from_dict(json.loads(s))


@dataclass
class MacroLibrary:
    """Collection of learned macros.

    Not a static ontology. Rebuilt from solve traces during consolidation.
    """
    macros: list[Macro] = field(default_factory=list)

    def add(self, macro: Macro) -> None:
        self.macros.append(macro)

    def find_by_signature(self, action_sig: str) -> list[Macro]:
        return [m for m in self.macros if m.action_signature == action_sig]

    def save_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({
                'schema_version': 1,
                'macros': [m.to_dict() for m in self.macros],
            }, f, indent=2)

    def score_candidate(
        self, action_sig: str, provenance: str = '',
        selector_sig: str = '',
    ) -> float:
        """Score a candidate by matching macro dimensions.

        Tiered matching (full weight → half → quarter):
        1. provenance + action_sig + selector_pattern → full
        2. provenance + action_sig → 0.5×
        3. action_sig only → 0.25×

        Returns 0.0 if no matching macro exists.
        """
        best = 0.0
        for m in self.macros:
            if m.action_signature != action_sig:
                continue
            score = m.frequency * m.solve_rate
            prov_match = provenance and provenance in m.source_provenances
            sel_match = selector_sig and m.selector_pattern == selector_sig

            if prov_match and sel_match:
                best = max(best, score)
            elif prov_match:
                best = max(best, score * 0.5)
            else:
                best = max(best, score * 0.25)
        return best

    @classmethod
    def load_json(cls, path: str) -> MacroLibrary:
        lib = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            for md in data.get('macros', []):
                lib.add(Macro.from_dict(md))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return lib


def _default_macro_library_path() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parents[2] / 'results' / 'search_macro_library.json')


_cached_macro_library: MacroLibrary | None = None


def load_default_macro_library() -> MacroLibrary:
    """Load the macro library from the default results path.

    Returns an empty library if the file doesn't exist.
    Cached after first load.
    """
    global _cached_macro_library
    if _cached_macro_library is not None:
        return _cached_macro_library
    _cached_macro_library = MacroLibrary.load_json(_default_macro_library_path())
    return _cached_macro_library
