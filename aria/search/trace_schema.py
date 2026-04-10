"""Structured trace records for solved search programs.

Captures enough structure from successful derive/search solves to
support later macro mining and pattern consolidation.

This is NOT a runtime execution layer. These records are produced
after verification and stored for offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json


@dataclass
class SolveTrace:
    """Structured record of one successful search solve.

    Captures the structural program, provenance, and selector details
    so repeated compositions can be mined into macros.
    """
    # --- Identity ---
    task_id: str
    task_signatures: tuple[str, ...]

    # --- Program structure ---
    provenance: str                    # derive strategy name
    step_actions: tuple[str, ...]      # ordered action names
    step_selectors: tuple[str, ...]    # selector summaries per step
    program_dict: dict                 # full SearchProgram.to_dict()

    # --- Metadata ---
    n_demos: int = 0
    n_steps: int = 0
    test_correct: bool | None = None   # None if no test data

    def signature(self) -> str:
        """Structural signature: action sequence."""
        return ' -> '.join(self.step_actions)

    def selector_signature(self) -> str:
        """Selector-aware signature: action(selector) sequence."""
        parts = []
        for act, sel in zip(self.step_actions, self.step_selectors):
            parts.append(f'{act}({sel})' if sel else act)
        return ' -> '.join(parts)

    def to_dict(self) -> dict:
        return {
            'schema_version': 1,
            'task_id': self.task_id,
            'task_signatures': list(self.task_signatures),
            'provenance': self.provenance,
            'step_actions': list(self.step_actions),
            'step_selectors': list(self.step_selectors),
            'program_dict': self.program_dict,
            'n_demos': self.n_demos,
            'n_steps': self.n_steps,
            'test_correct': self.test_correct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SolveTrace:
        return cls(
            task_id=d['task_id'],
            task_signatures=tuple(d.get('task_signatures', ())),
            provenance=d.get('provenance', ''),
            step_actions=tuple(d.get('step_actions', ())),
            step_selectors=tuple(d.get('step_selectors', ())),
            program_dict=d.get('program_dict', {}),
            n_demos=d.get('n_demos', 0),
            n_steps=d.get('n_steps', 0),
            test_correct=d.get('test_correct'),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SolveTrace:
        return cls.from_dict(json.loads(s))
