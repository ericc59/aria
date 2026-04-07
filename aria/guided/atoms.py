"""Atom learning: extract reusable concepts from training tasks.

An atom is a generalized rule verified on one training task and
potentially applicable to others. Atoms are stored in a library
and composed to solve harder tasks.

The learning loop:
1. For each training task, try all known atoms and compositions
2. If no atom solves it, analyze what NEW concept it teaches
3. Encode the concept as an atom
4. Add to library
5. Re-run — new atom may unlock more tasks

Atoms are NOT task-specific hacks. They are GENERAL structural
operations described in terms of roles, not literal values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from aria.guided.generalize import (
    GeneralizedRule, verify_rule,
    _apply_recolor_to_marker, _apply_recolor_to_adjacent,
    _apply_recolor_to_enclosing, _apply_fill_with_frame,
    _apply_fill_literal, _apply_move,
    _apply_canvas_recolor_marker, _apply_literal_recolor,
)
from aria.guided.workspace import _detect_bg, _extract_objects
from aria.guided.residual_objects import analyze_residual_objects
from aria.guided.construct import construct_canvas
from aria.types import Grid


# ---------------------------------------------------------------------------
# Atom: a reusable structural concept
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    name: str
    description: str
    apply_fn: Callable[[Grid], Grid]
    source_task: str = ""     # which task taught us this
    n_tasks_solved: int = 0   # how many tasks this atom solves


# ---------------------------------------------------------------------------
# Atom library
# ---------------------------------------------------------------------------

class AtomLibrary:
    def __init__(self):
        self.atoms: list[Atom] = []
        self._init_builtin_atoms()

    def _init_builtin_atoms(self):
        """Seed with atoms we already know work."""
        # Recolor atoms
        self.atoms.append(Atom(
            "recolor_to_marker",
            "recolor non-singleton objects to singleton marker's color",
            _apply_recolor_to_marker,
        ))
        self.atoms.append(Atom(
            "recolor_to_adjacent",
            "recolor objects to adjacent singleton's color",
            _apply_recolor_to_adjacent,
        ))
        self.atoms.append(Atom(
            "recolor_to_enclosing",
            "recolor objects to enclosing object's color",
            _apply_recolor_to_enclosing,
        ))

        # Fill atoms
        self.atoms.append(Atom(
            "fill_enclosed_frame",
            "fill enclosed bg with enclosing frame's color",
            _apply_fill_with_frame,
        ))
        for c in range(10):
            self.atoms.append(Atom(
                f"fill_enclosed_{c}",
                f"fill enclosed bg with color {c}",
                lambda inp, color=c: _apply_fill_literal(inp, color),
            ))

        # Move atoms
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                self.atoms.append(Atom(
                    f"move_{dr}_{dc}",
                    f"move all objects by ({dr},{dc})",
                    lambda inp, r=dr, c=dc: _apply_move(inp, r, c),
                ))

        # Canvas-based atoms
        self.atoms.append(Atom(
            "canvas_recolor_marker",
            "build canvas + recolor targets to marker color",
            _apply_canvas_recolor_marker,
        ))

        # Delete atoms
        for c in range(10):
            self.atoms.append(Atom(
                f"delete_color_{c}",
                f"delete all pixels of color {c}",
                lambda inp, color=c: _delete_color(inp, color),
            ))

        # Anomaly deletion: delete the pixel that breaks symmetry/pattern
        self.atoms.append(Atom(
            "delete_anomaly",
            "delete the pixel that doesn't fit the pattern",
            _delete_anomaly,
        ))

        # Gravity atoms
        for direction in ['down', 'up', 'left', 'right']:
            self.atoms.append(Atom(
                f"gravity_{direction}",
                f"slide non-bg pixels {direction}",
                lambda inp, d=direction: _apply_gravity(inp, d),
            ))

        # Sort/reorder atoms
        self.atoms.append(Atom(
            "sort_objects_by_size",
            "sort objects by size (reorder rows/cols)",
            _sort_by_size,
        ))

    def add_atom(self, atom: Atom):
        """Add a newly learned atom."""
        # Don't add duplicates
        if any(a.name == atom.name for a in self.atoms):
            return
        self.atoms.append(atom)

    def try_all(self, demos: list[tuple[Grid, Grid]]) -> Atom | None:
        """Try each atom on the demos. Return first that verifies."""
        for atom in self.atoms:
            rule = GeneralizedRule(atom.description, atom.apply_fn)
            ok, diff = verify_rule(rule, demos)
            if ok:
                atom.n_tasks_solved += 1
                return atom
        return None

    def try_all_with_best(self, demos: list[tuple[Grid, Grid]]) -> tuple[Atom | None, int]:
        """Try all atoms, return (best_atom, best_diff)."""
        orig = sum(
            int(np.sum(inp != out)) if inp.shape == out.shape else out.size
            for inp, out in demos
        )
        best_atom = None
        best_diff = orig

        for atom in self.atoms:
            rule = GeneralizedRule(atom.description, atom.apply_fn)
            ok, diff = verify_rule(rule, demos)
            if ok:
                atom.n_tasks_solved += 1
                return atom, 0
            if diff < best_diff:
                best_diff = diff
                best_atom = atom

        return best_atom, best_diff

    def try_compositions(self, demos: list[tuple[Grid, Grid]], max_depth: int = 2) -> Atom | None:
        """Try compositions of 2 atoms."""
        if max_depth < 2:
            return None

        # First find atoms that improve (reduce diff)
        orig = sum(
            int(np.sum(inp != out)) if inp.shape == out.shape else out.size
            for inp, out in demos
        )
        improving = []
        for atom in self.atoms:
            rule = GeneralizedRule(atom.description, atom.apply_fn)
            _, diff = verify_rule(rule, demos)
            if diff < orig:
                improving.append((atom, diff))

        improving.sort(key=lambda x: x[1])

        # Try composing top improvers with all atoms
        for first_atom, _ in improving[:10]:
            for second_atom in self.atoms:
                def _composed(inp, a1=first_atom, a2=second_atom):
                    return a2.apply_fn(a1.apply_fn(inp))

                composed_rule = GeneralizedRule(
                    f"{first_atom.name} + {second_atom.name}",
                    _composed,
                )
                ok, diff = verify_rule(composed_rule, demos)
                if ok:
                    # Create a new composite atom
                    return Atom(
                        f"{first_atom.name}_then_{second_atom.name}",
                        f"{first_atom.description}, then {second_atom.description}",
                        _composed,
                    )

        return None

    def learn_from_task(self, tid: str, demos: list[tuple[Grid, Grid]]) -> Atom | None:
        """Try to learn a new atom from an unsolved task.

        Analyzes what the task teaches and creates a generalized atom.
        """
        try:
            residuals = analyze_residual_objects(demos)
        except Exception:
            return None

        # Collect observations across demos
        observations = []
        for di, (dr, (inp, out)) in enumerate(zip(residuals, demos)):
            bg = _detect_bg(inp)
            objs = _extract_objects(inp, bg)
            for robj in dr.objects:
                observations.append({
                    'cause': robj.cause,
                    'source_color': robj.source_color,
                    'new_color': robj.new_color,
                    'size': robj.size,
                    'demo_idx': di,
                    'bg': bg,
                    'n_singletons': sum(1 for o in objs if o.is_singleton),
                    'n_objects': len(objs),
                })

        if not observations:
            return None

        # Try to derive a new atom from consistent observations
        causes = set(o['cause'] for o in observations)

        # Pattern: all recolored, and new_color = bg across all demos
        # → "delete specific objects" (already covered by delete_color)

        # Pattern: all FILLED, consistent fill color derived from structure
        if causes == {'FILLED'}:
            return self._learn_fill_atom(tid, observations, demos)

        # Pattern: SAME_POS_RECOLOR with color from specific structural source
        if 'SAME_POS_RECOLOR' in causes:
            return self._learn_recolor_atom(tid, observations, demos)

        return None

    def _learn_fill_atom(self, tid, observations, demos):
        """Learn a fill atom from observations."""
        # Already covered by existing atoms
        return None

    def _learn_recolor_atom(self, tid, observations, demos):
        """Learn a recolor atom: what determines the new color?"""
        recolors = [o for o in observations if o['cause'] == 'SAME_POS_RECOLOR']
        if not recolors:
            return None

        # Check: new_color is consistent across all demos (literal map)
        cmap = {}
        consistent = True
        for o in recolors:
            if o['source_color'] in cmap and cmap[o['source_color']] != o['new_color']:
                consistent = False
                break
            cmap[o['source_color']] = o['new_color']

        if consistent and cmap:
            def _fn(inp, m=dict(cmap)):
                return _apply_literal_recolor(inp, m)
            atom = Atom(
                f"recolor_map_{tid}",
                f"recolor by map {cmap}",
                _fn,
                source_task=tid,
            )
            rule = GeneralizedRule(atom.description, atom.apply_fn)
            ok, _ = verify_rule(rule, demos)
            if ok:
                return atom

        return None


# ---------------------------------------------------------------------------
# Iterative learning loop
# ---------------------------------------------------------------------------

def learn_atoms_iteratively(
    library: AtomLibrary,
    tasks: list[tuple[str, list[tuple[Grid, Grid]], Any]],
    max_rounds: int = 3,
) -> dict:
    """Iteratively try atoms, learn new ones, re-run.

    Returns stats about what was solved.
    """
    solved = set()
    stats = {'rounds': [], 'total_solved': 0}

    for round_idx in range(max_rounds):
        new_solves = 0
        new_atoms = 0

        for tid, demos, test_pairs in tasks:
            if tid in solved:
                continue

            # Try single atoms
            atom = library.try_all(demos)
            if atom:
                solved.add(tid)
                new_solves += 1
                continue

            # Try compositions
            comp = library.try_compositions(demos)
            if comp:
                library.add_atom(comp)
                solved.add(tid)
                new_solves += 1
                new_atoms += 1
                continue

            # Try to learn a new atom
            learned = library.learn_from_task(tid, demos)
            if learned:
                rule = GeneralizedRule(learned.description, learned.apply_fn)
                ok, _ = verify_rule(rule, demos)
                if ok:
                    library.add_atom(learned)
                    solved.add(tid)
                    new_solves += 1
                    new_atoms += 1

        stats['rounds'].append({
            'round': round_idx,
            'new_solves': new_solves,
            'new_atoms': new_atoms,
            'total_solved': len(solved),
        })

        if new_solves == 0:
            break  # no progress

    stats['total_solved'] = len(solved)
    return stats


# ---------------------------------------------------------------------------
# Atom implementations
# ---------------------------------------------------------------------------

def _delete_color(inp, color):
    bg = _detect_bg(inp)
    out = inp.copy()
    out[out == color] = bg
    return out


def _delete_anomaly(inp):
    """Delete the pixel that breaks the dominant pattern."""
    bg = _detect_bg(inp)
    out = inp.copy()
    objs = _extract_objects(inp, bg)

    for obj in objs:
        if obj.size < 3:
            continue
        sub = inp[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width].copy()
        # Check H symmetry
        h_asym = np.sum(sub != sub[:, ::-1])
        if 0 < h_asym <= 2:
            repaired = sub.copy()
            for r in range(obj.height):
                for c in range(obj.width // 2):
                    mc = obj.width - 1 - c
                    if repaired[r, c] != repaired[r, mc]:
                        if repaired[r, c] == bg:
                            repaired[r, c] = repaired[r, mc]
                        else:
                            repaired[r, mc] = repaired[r, c]
            out[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] = repaired
            continue
        # Check V symmetry
        v_asym = np.sum(sub != sub[::-1, :])
        if 0 < v_asym <= 2:
            repaired = sub.copy()
            for r in range(obj.height // 2):
                mr = obj.height - 1 - r
                for c in range(obj.width):
                    if repaired[r, c] != repaired[mr, c]:
                        if repaired[r, c] == bg:
                            repaired[r, c] = repaired[mr, c]
                        else:
                            repaired[mr, c] = repaired[r, c]
            out[obj.row:obj.row + obj.height, obj.col:obj.col + obj.width] = repaired

    return out


def _apply_gravity(inp, direction):
    bg = _detect_bg(inp)
    out = inp.copy()
    rows, cols = inp.shape
    if direction == 'down':
        for c in range(cols):
            col = list(inp[:, c])
            non_bg = [v for v in col if v != bg]
            out[:, c] = np.array([bg] * (rows - len(non_bg)) + non_bg, dtype=np.uint8)
    elif direction == 'up':
        for c in range(cols):
            col = list(inp[:, c])
            non_bg = [v for v in col if v != bg]
            out[:, c] = np.array(non_bg + [bg] * (rows - len(non_bg)), dtype=np.uint8)
    elif direction == 'right':
        for r in range(rows):
            row = list(inp[r, :])
            non_bg = [v for v in row if v != bg]
            out[r, :] = np.array([bg] * (cols - len(non_bg)) + non_bg, dtype=np.uint8)
    elif direction == 'left':
        for r in range(rows):
            row = list(inp[r, :])
            non_bg = [v for v in row if v != bg]
            out[r, :] = np.array(non_bg + [bg] * (cols - len(non_bg)), dtype=np.uint8)
    return out


def _sort_by_size(inp):
    """Sort rows by the number of non-bg pixels."""
    bg = _detect_bg(inp)
    rows = list(range(inp.shape[0]))
    rows.sort(key=lambda r: -np.sum(inp[r, :] != bg))
    return inp[rows, :]
