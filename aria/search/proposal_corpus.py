"""Offline corpus builder for solved `aria/search` runs.

This turns solved search outcomes from eval reports into a compact JSONL
training corpus for future learned proposal/ranking models.

The corpus is intentionally minimal:
- task signatures
- solved search family
- full search description
- task id / source report provenance
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


def _family_from_description(description: str) -> str | None:
    if not description.startswith("search:"):
        return None
    if "[" in description and "]" in description:
        start = description.rfind("[")
        end = description.rfind("]")
        if 0 <= start < end:
            family = description[start + 1:end].strip()
            if family:
                return family
    body = description.split(":", 1)[1].strip()
    return body or None


@dataclass(frozen=True)
class SearchProposalExample:
    schema_version: int
    task_id: str
    task_signatures: tuple[str, ...]
    family: str
    description: str
    source_report: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_signatures": list(self.task_signatures),
            "family": self.family,
            "description": self.description,
            "source_report": self.source_report,
        }


def examples_from_eval_reports(paths: list[Path]) -> list[SearchProposalExample]:
    examples: list[SearchProposalExample] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    for path in paths:
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue

        for task in payload.get("tasks", []):
            if not task.get("solved"):
                continue
            if task.get("solve_source") != "search":
                continue
            description = str(task.get("description", ""))
            family = _family_from_description(description)
            task_id = str(task.get("task_id", ""))
            task_signatures = tuple(sorted(
                sig for sig in task.get("task_signatures", [])
                if isinstance(sig, str) and sig
            ))
            if not family or not task_id:
                continue
            key = (task_id, family, task_signatures)
            if key in seen:
                continue
            seen.add(key)
            examples.append(SearchProposalExample(
                schema_version=1,
                task_id=task_id,
                task_signatures=task_signatures,
                family=family,
                description=description,
                source_report=path.name,
            ))

    return sorted(examples, key=lambda ex: (ex.task_id, ex.family, ex.source_report))


def write_jsonl(path: str | Path, examples: list[SearchProposalExample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True))
            f.write("\n")
