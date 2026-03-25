"""Library storage for admitted abstractions.

Manages the growing vocabulary of reusable sub-programs.
Level 1 = permanent (survived MDL gate with 2+ uses).
Level 2 = task-specific (temporary, single-use during test time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aria.proposer.parser import ParseError, parse_program
from aria.runtime.program import program_to_text
from aria.runtime.ops import OpSignature, register, has_op
from aria.types import LibraryEntry, Program, Step, Type


class Library:
    """In-memory library of admitted abstractions."""

    def __init__(self) -> None:
        self._entries: dict[str, LibraryEntry] = {}

    def add(self, entry: LibraryEntry) -> None:
        self._entries[entry.name] = entry
        # Register as a runtime op if not already there
        if not has_op(entry.name):
            sig = OpSignature(
                params=entry.params,
                return_type=entry.return_type,
            )
            impl = _make_library_impl(entry)
            register(entry.name, sig, impl, is_library=True)

    def clone(self) -> Library:
        clone = Library()
        for entry in self.all_entries():
            clone.add(entry)
        return clone

    def get(self, name: str) -> LibraryEntry | None:
        return self._entries.get(name)

    def all_entries(self) -> list[LibraryEntry]:
        return list(self._entries.values())

    def level1_entries(self) -> list[LibraryEntry]:
        return [e for e in self._entries.values() if e.level == 1]

    def level2_entries(self) -> list[LibraryEntry]:
        return [e for e in self._entries.values() if e.level == 2]

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)

    def clear_level2(self) -> None:
        """Remove all task-specific abstractions."""
        to_remove = [n for n, e in self._entries.items() if e.level == 2]
        for n in to_remove:
            del self._entries[n]

    def promote_to_level1(self, name: str) -> None:
        entry = self._entries.get(name)
        if entry and entry.level == 2:
            self._entries[name] = LibraryEntry(
                name=entry.name,
                params=entry.params,
                return_type=entry.return_type,
                steps=entry.steps,
                output=entry.output,
                level=1,
                use_count=entry.use_count,
            )

    def increment_use(self, name: str) -> None:
        entry = self._entries.get(name)
        if entry:
            self._entries[name] = LibraryEntry(
                name=entry.name,
                params=entry.params,
                return_type=entry.return_type,
                steps=entry.steps,
                output=entry.output,
                level=entry.level,
                use_count=entry.use_count + 1,
            )

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def index_for_proposer(self) -> list[dict[str, Any]]:
        """Produce a serializable index for the proposer's prompt."""
        return [
            {
                "name": e.name,
                "params": [(n, t.name) for n, t in e.params],
                "return_type": e.return_type.name,
                "level": e.level,
                "use_count": e.use_count,
            }
            for e in self._entries.values()
        ]

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "entries": [
                {
                    "name": entry.name,
                    "params": [[param_name, param_type.name] for param_name, param_type in entry.params],
                    "return_type": entry.return_type.name,
                    "program": program_to_text(Program(steps=entry.steps, output=entry.output)),
                    "level": entry.level,
                    "use_count": entry.use_count,
                }
                for entry in self.all_entries()
            ],
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> Library:
        source_path = Path(path)
        library = cls()
        if not source_path.exists():
            return library

        with open(source_path) as f:
            data = json.load(f)

        if data.get("version") != 1 or not isinstance(data.get("entries"), list):
            raise ValueError(f"Unsupported library format: {source_path}")

        for item in data["entries"]:
            name = item.get("name")
            program_text = item.get("program")
            params_data = item.get("params", [])
            return_type_name = item.get("return_type")
            if not isinstance(name, str) or not isinstance(program_text, str):
                continue
            if return_type_name not in Type.__members__:
                continue
            try:
                program = parse_program(program_text)
            except ParseError:
                continue

            params: list[tuple[str, Type]] = []
            malformed = False
            for raw_param in params_data:
                if not isinstance(raw_param, (list, tuple)) or len(raw_param) != 2:
                    malformed = True
                    break
                param_name, param_type_name = raw_param
                if not isinstance(param_name, str) or param_type_name not in Type.__members__:
                    malformed = True
                    break
                params.append((param_name, Type[param_type_name]))
            if malformed:
                continue

            library.add(LibraryEntry(
                name=name,
                params=tuple(params),
                return_type=Type[return_type_name],
                steps=program.steps,
                output=program.output,
                level=int(item.get("level", 1)),
                use_count=int(item.get("use_count", 0)),
            ))

        return library


def _make_library_impl(entry: LibraryEntry):
    """Create a callable implementation from a library entry's steps."""

    def impl(*args, **kwargs):
        from aria.runtime.executor import eval_expr
        from aria.types import Bind, Assert

        if len(args) > len(entry.params):
            raise RuntimeError(
                f"{entry.name} expected {len(entry.params)} args, got {len(args)}"
            )

        env = {
            param_name: arg
            for (param_name, _param_type), arg in zip(entry.params, args)
        }
        env.update(kwargs)

        missing = [
            param_name
            for param_name, _param_type in entry.params
            if param_name not in env
        ]
        if missing:
            raise RuntimeError(
                f"{entry.name} missing args: {', '.join(missing)}"
            )

        for step in entry.steps:
            match step:
                case Bind(name=name, expr=expr):
                    env[name] = eval_expr(expr, env)
                case Assert(pred=pred):
                    if not eval_expr(pred, env):
                        raise RuntimeError(f"Library assertion failed in {entry.name}")
        return env[entry.output]

    return impl
