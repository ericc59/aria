"""Persistent storage for verified programs.

The canonical on-disk format is a deduplicated JSON store. This module can
also ingest legacy solve reports and corpus reports so the runtime can use
previously verified programs as retrieval candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aria.proposer.parser import ParseError, parse_program
from aria.runtime.program import program_to_text
from aria.types import Program


@dataclass(frozen=True)
class StoredProgram:
    program_text: str
    task_ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    use_count: int = 1
    signatures: tuple[str, ...] = ()

    @property
    def step_count(self) -> int:
        return sum(
            1 for line in self.program_text.splitlines()
            if line.startswith(("let ", "bind ", "assert "))
        )


class ProgramStore:
    """Deduplicated set of verified programs, persisted to disk."""

    def __init__(self) -> None:
        self._records: dict[str, StoredProgram] = {}

    def __len__(self) -> int:
        return len(self._records)

    def clone(self) -> ProgramStore:
        clone = ProgramStore()
        clone._records = dict(self._records)
        return clone

    def all_records(self) -> list[StoredProgram]:
        return list(self._records.values())

    def ranked_records(
        self,
        signatures: frozenset[str] | None = None,
    ) -> list[StoredProgram]:
        """Rank likely-useful programs first.

        High-frequency and shorter programs are cheap to replay and more likely
        to transfer, so prefer them before long one-off programs.
        """
        query_signatures = signatures or frozenset()

        def rank_key(record: StoredProgram) -> tuple[int, int, int, int, int, str]:
            overlap = len(query_signatures & set(record.signatures)) if query_signatures else 0
            return (
                -overlap,
                -record.use_count,
                record.step_count,
                -len(record.signatures),
                -len(record.task_ids),
                record.program_text,
            )

        return sorted(
            self._records.values(),
            key=rank_key,
        )

    def add_program(
        self,
        program: Program,
        *,
        task_id: str | None = None,
        source: str = "runtime",
        signatures: frozenset[str] | None = None,
    ) -> StoredProgram:
        return self.add_text(
            program_to_text(program),
            task_id=task_id,
            source=source,
            signatures=signatures,
        )

    def add_text(
        self,
        text: str,
        *,
        task_id: str | None = None,
        source: str = "runtime",
        signatures: frozenset[str] | None = None,
    ) -> StoredProgram:
        canonical = _canonicalize_program_text(text)
        return self._merge_record(
            canonical,
            task_ids=(task_id,) if task_id else (),
            sources=(source,),
            use_count=1,
            signatures=tuple(sorted(signatures or ())),
        )

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "programs": [
                {
                    "program": record.program_text,
                    "task_ids": list(record.task_ids),
                    "sources": list(record.sources),
                    "use_count": record.use_count,
                    "signatures": list(record.signatures),
                }
                for record in self.ranked_records()
            ],
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> ProgramStore:
        store = cls()
        source_path = Path(path)
        if not source_path.exists():
            return store

        with open(source_path) as f:
            data = json.load(f)

        if data.get("version") not in {1, 2} or not isinstance(data.get("programs"), list):
            raise ValueError(f"Unsupported program store format: {source_path}")

        for item in data["programs"]:
            program_text = item.get("program")
            if not isinstance(program_text, str):
                continue
            try:
                canonical = _canonicalize_program_text(program_text)
            except ParseError:
                continue
            task_ids = tuple(
                str(task_id) for task_id in item.get("task_ids", [])
                if task_id
            )
            sources = tuple(
                str(source) for source in item.get("sources", [])
                if source
            )
            use_count = int(item.get("use_count", 1))
            store._merge_record(
                canonical,
                task_ids=task_ids,
                sources=sources,
                use_count=max(use_count, 1),
                signatures=tuple(
                    str(signature) for signature in item.get("signatures", [])
                    if signature
                ),
            )

        return store

    def import_path(self, path: str | Path) -> int:
        source_path = Path(path)
        if not source_path.exists():
            return 0

        with open(source_path) as f:
            data = json.load(f)

        if data.get("version") in {1, 2} and isinstance(data.get("programs"), list):
            return self._import_canonical_store(data)
        if isinstance(data.get("tasks"), list):
            return self._import_solve_report(data, source_path.name)
        if isinstance(data.get("tasks"), dict):
            return self._import_corpus_report(data, source_path.name)
        raise ValueError(f"Unrecognized program source: {source_path}")

    def _import_canonical_store(self, data: dict) -> int:
        imported = 0
        for item in data["programs"]:
            program_text = item.get("program")
            if not isinstance(program_text, str):
                continue
            try:
                canonical = _canonicalize_program_text(program_text)
            except ParseError:
                continue
            before = len(self._records)
            self._merge_record(
                canonical,
                task_ids=tuple(str(task_id) for task_id in item.get("task_ids", []) if task_id),
                sources=tuple(str(source) for source in item.get("sources", []) if source),
                use_count=max(int(item.get("use_count", 1)), 1),
                signatures=tuple(
                    str(signature) for signature in item.get("signatures", [])
                    if signature
                ),
            )
            if len(self._records) > before:
                imported += 1
        return imported

    def _import_solve_report(self, data: dict, source_name: str) -> int:
        imported = 0
        for task in data.get("tasks", []):
            program_text = task.get("program")
            if not isinstance(program_text, str):
                continue
            try:
                canonical = _canonicalize_program_text(program_text)
            except ParseError:
                continue
            before = len(self._records)
            self._merge_record(
                canonical,
                task_ids=(str(task.get("task_id")),) if task.get("task_id") else (),
                sources=(f"solve-report:{source_name}",),
                use_count=1,
                signatures=tuple(
                    str(signature) for signature in task.get("task_signatures", [])
                    if signature
                ),
            )
            if len(self._records) > before:
                imported += 1
        return imported

    def _import_corpus_report(self, data: dict, source_name: str) -> int:
        imported = 0
        tasks = data.get("tasks", {})
        for task_id, payload in tasks.items():
            for program_text in payload.get("programs", []):
                if not isinstance(program_text, str):
                    continue
                try:
                    canonical = _canonicalize_program_text(program_text)
                except ParseError:
                    continue
                before = len(self._records)
                self._merge_record(
                    canonical,
                    task_ids=(str(task_id),),
                    sources=(f"corpus-report:{source_name}",),
                    use_count=1,
                    signatures=tuple(
                        str(signature) for signature in payload.get("task_signatures", [])
                        if signature
                    ),
                )
                if len(self._records) > before:
                    imported += 1
        return imported

    def _merge_record(
        self,
        program_text: str,
        *,
        task_ids: tuple[str, ...],
        sources: tuple[str, ...],
        use_count: int,
        signatures: tuple[str, ...],
    ) -> StoredProgram:
        existing = self._records.get(program_text)
        merged_task_ids = tuple(sorted({
            *(existing.task_ids if existing else ()),
            *task_ids,
        }))
        merged_sources = tuple(sorted({
            *(existing.sources if existing else ()),
            *sources,
        }))
        merged_signatures = tuple(sorted({
            *(existing.signatures if existing else ()),
            *signatures,
        }))
        merged_use_count = (existing.use_count if existing else 0) + use_count
        record = StoredProgram(
            program_text=program_text,
            task_ids=merged_task_ids,
            sources=merged_sources,
            use_count=merged_use_count,
            signatures=merged_signatures,
        )
        self._records[program_text] = record
        return record


def _canonicalize_program_text(text: str) -> str:
    """Round-trip through the parser so all persisted programs share one format."""
    return program_to_text(parse_program(text))
