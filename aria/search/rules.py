"""Tiny symbolic rule induction over derived boolean facts.

This is a derive-time utility, not a learned model and not an AST layer.
It searches a very small space of boolean DNF rules and only keeps rules
that fit the provided examples exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping


FactRow = Mapping[str, Any]


@dataclass(frozen=True)
class BoolAtom:
    """One boolean test on a named field."""

    field: str
    value: bool = True

    def matches(self, row: FactRow) -> bool:
        return bool(row.get(self.field, False)) is self.value

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoolAtom":
        return cls(field=str(data["field"]), value=bool(data.get("value", True)))


@dataclass(frozen=True)
class ConjunctionRule:
    """All atoms must match."""

    atoms: tuple[BoolAtom, ...] = ()

    def matches(self, row: FactRow) -> bool:
        return all(atom.matches(row) for atom in self.atoms)

    def complexity(self) -> tuple[int, tuple[str, ...]]:
        return (
            len(self.atoms),
            tuple(f"{atom.field}={int(atom.value)}" for atom in self.atoms),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"atoms": [atom.to_dict() for atom in self.atoms]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConjunctionRule":
        atoms = tuple(BoolAtom.from_dict(atom) for atom in data.get("atoms", ()))
        return cls(atoms=atoms)


@dataclass(frozen=True)
class DNFRule:
    """Disjunction of conjunctions. Zero clauses means always false."""

    clauses: tuple[ConjunctionRule, ...] = ()

    def matches(self, row: FactRow) -> bool:
        return any(clause.matches(row) for clause in self.clauses)

    def complexity(self) -> tuple[int, int, tuple[tuple[str, ...], ...]]:
        return (
            sum(len(clause.atoms) for clause in self.clauses),
            len(self.clauses),
            tuple(clause.complexity()[1] for clause in self.clauses),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dnf",
            "clauses": [clause.to_dict() for clause in self.clauses],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DNFRule":
        if data.get("kind") != "dnf":
            raise ValueError(f"Unsupported rule kind: {data.get('kind')}")
        clauses = tuple(
            ConjunctionRule.from_dict(clause) for clause in data.get("clauses", ())
        )
        return cls(clauses=clauses)


def eval_rule(rule: DNFRule | Mapping[str, Any], row: FactRow) -> bool:
    """Evaluate a rule against one fact row."""
    compiled = rule if isinstance(rule, DNFRule) else DNFRule.from_dict(rule)
    return compiled.matches(row)


def induce_boolean_dnf(
    rows: list[FactRow],
    labels: list[bool],
    *,
    candidate_fields: list[str],
    max_clause_size: int = 2,
    max_clauses: int = 2,
) -> DNFRule | None:
    """Induce the simplest exact-fit boolean DNF over the candidate fields."""
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have the same length")

    positive_idxs = {idx for idx, label in enumerate(labels) if label}
    all_idxs = set(range(len(rows)))
    if not positive_idxs:
        return DNFRule(())
    if positive_idxs == all_idxs:
        return DNFRule((ConjunctionRule(()),))

    atoms = _candidate_atoms(candidate_fields)
    exact_clauses: list[tuple[ConjunctionRule, set[int]]] = []

    for clause in _enumerate_clauses(atoms, max_clause_size=max_clause_size):
        covered = {idx for idx, row in enumerate(rows) if clause.matches(row)}
        if covered and covered <= positive_idxs:
            exact_clauses.append((clause, covered))

    if not exact_clauses:
        return None

    exact_clauses.sort(key=lambda item: item[0].complexity())
    best: DNFRule | None = None
    best_score: tuple[int, int, tuple[tuple[str, ...], ...]] | None = None

    for n_clauses in range(1, max_clauses + 1):
        for combo in combinations(exact_clauses, n_clauses):
            covered = set().union(*(covered for _, covered in combo))
            if covered != positive_idxs:
                continue
            rule = DNFRule(tuple(clause for clause, _ in combo))
            score = rule.complexity()
            if best is None or score < best_score:
                best = rule
                best_score = score

    return best


def _candidate_atoms(fields: list[str]) -> tuple[BoolAtom, ...]:
    return tuple(
        atom
        for field in fields
        for atom in (BoolAtom(field, True), BoolAtom(field, False))
    )


def _enumerate_clauses(
    atoms: tuple[BoolAtom, ...],
    *,
    max_clause_size: int,
) -> list[ConjunctionRule]:
    clauses = [ConjunctionRule(())]
    for size in range(1, max_clause_size + 1):
        for combo in combinations(atoms, size):
            if _contradictory(combo):
                continue
            clauses.append(
                ConjunctionRule(
                    tuple(sorted(combo, key=lambda atom: (atom.field, atom.value)))
                )
            )
    return clauses


def _contradictory(atoms: tuple[BoolAtom, ...]) -> bool:
    seen: dict[str, bool] = {}
    for atom in atoms:
        if atom.field in seen and seen[atom.field] != atom.value:
            return True
        seen[atom.field] = atom.value
    return False
