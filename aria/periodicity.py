"""Generalized 1D periodicity detection, completion, and extension.

Reusable operators for detecting repeating patterns in color sequences,
completing broken patterns, extending patterns into empty space, and
finding localized anomalies against a periodic background.

These operate on generic 1D integer sequences — not tied to any
specific grid decomposition or structural view.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from aria.types import Grid


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodicPattern:
    """A detected repeating pattern in a 1D sequence."""
    pattern: tuple[int, ...]     # the repeating unit
    period: int                  # length of one cycle
    violations: tuple[int, ...]  # indices where the sequence deviates
    confidence: float            # 0-1, based on fraction of matching positions
    start: int                   # first index included in the analysis
    end: int                     # last index (exclusive) included


# ---------------------------------------------------------------------------
# 1D period detection
# ---------------------------------------------------------------------------


def detect_1d_period(
    seq: np.ndarray,
    *,
    min_period: int = 1,
    max_period: int | None = None,
    bg: int | None = None,
    require_violations: bool = False,
) -> PeriodicPattern | None:
    """Detect the best periodic pattern in a 1D integer sequence.

    Args:
        seq: 1D array of integers.
        min_period: minimum period to consider.
        max_period: maximum period (default: len//2).
        bg: if set, ignore positions where seq[i]==bg when computing the pattern.
        require_violations: if True, skip perfect patterns (nothing to repair).

    Returns the pattern with highest confidence, or None.
    """
    n = len(seq)
    if n < 2:
        return None
    if max_period is None:
        max_period = max(n // 2, min_period)

    best: PeriodicPattern | None = None

    for period in range(min_period, max_period + 1):
        if period > n:
            break

        # Infer pattern by majority vote at each phase position
        pattern = []
        for phase in range(period):
            positions = list(range(phase, n, period))
            if bg is not None:
                vals = [int(seq[i]) for i in positions if int(seq[i]) != bg]
            else:
                vals = [int(seq[i]) for i in positions]
            if not vals:
                pattern.append(bg if bg is not None else 0)
            else:
                counts = Counter(vals)
                pattern.append(counts.most_common(1)[0][0])

        # Count violations
        violations = []
        for i in range(n):
            expected = pattern[i % period]
            actual = int(seq[i])
            if actual != expected:
                if bg is not None and actual == bg:
                    continue  # bg positions are "empty", not violations
                violations.append(i)

        n_matched = n - len(violations)
        confidence = n_matched / n if n > 0 else 0.0

        if require_violations and not violations:
            continue
        if confidence < 0.5:
            continue

        candidate = PeriodicPattern(
            pattern=tuple(pattern),
            period=period,
            violations=tuple(violations),
            confidence=confidence,
            start=0,
            end=n,
        )

        if best is None:
            best = candidate
        elif len(violations) < len(best.violations):
            best = candidate
        elif len(violations) == len(best.violations) and period < best.period:
            best = candidate

    return best


# ---------------------------------------------------------------------------
# Completion and extension
# ---------------------------------------------------------------------------


def complete_sequence(seq: np.ndarray, pattern: PeriodicPattern) -> np.ndarray:
    """Repair violations in a sequence to match the detected pattern."""
    result = seq.copy()
    for i in pattern.violations:
        if 0 <= i < len(result):
            result[i] = pattern.pattern[i % pattern.period]
    return result


def extend_sequence(
    pattern: PeriodicPattern,
    length: int,
    *,
    offset: int = 0,
) -> np.ndarray:
    """Generate a periodic sequence of the given length.

    Args:
        pattern: the detected periodic pattern.
        length: desired output length.
        offset: phase offset (for aligning with a specific grid position).
    """
    result = np.zeros(length, dtype=np.int32)
    for i in range(length):
        result[i] = pattern.pattern[(i + offset) % pattern.period]
    return result


# ---------------------------------------------------------------------------
# Seed-based extension (for separated grids)
# ---------------------------------------------------------------------------


def detect_seed_and_extend_row(
    row: np.ndarray,
    separator_col: int,
    bg: int = 0,
) -> np.ndarray | None:
    """Detect the pattern on one side of a separator and extend it to the other.

    Reads the left side (cols 0..separator_col-1) as the "seed pattern",
    detects its period, and tiles it across the right side.

    Returns the completed row or None if no pattern found.
    """
    left = row[:separator_col]
    right_len = len(row) - separator_col - 1

    if right_len <= 0 or len(left) < 2:
        return None

    # Extract non-bg pattern from left side
    non_bg_positions = [i for i in range(len(left)) if int(left[i]) != bg]
    if not non_bg_positions:
        return None

    # The seed is the left-side content
    period_pattern = detect_1d_period(left, bg=bg)

    result = row.copy()

    if period_pattern is not None and period_pattern.confidence >= 0.8:
        # Use detected period to extend
        for c in range(separator_col + 1, len(row)):
            result[c] = period_pattern.pattern[c % period_pattern.period]
    else:
        # Fallback: tile the left pattern directly
        for c in range(separator_col + 1, len(row)):
            src = c % len(left) if len(left) > 0 else 0
            if src < len(left):
                result[c] = int(left[src])

    return result


# ---------------------------------------------------------------------------
# Column-wise seed tiling
# ---------------------------------------------------------------------------


def detect_column_seed_row(grid: Grid, bg: int = 0) -> int | None:
    """Find the row that serves as the "seed" for column-wise periodic tiling.

    The seed row is the row with the most non-bg content that, when tiled
    vertically, best explains the rest of the grid.
    """
    rows, cols = grid.shape
    if rows < 2:
        return None

    # Find rows with non-bg content
    row_counts = []
    for r in range(rows):
        n = int(np.sum(grid[r] != bg))
        row_counts.append((r, n))

    # The seed row is the one with the most non-bg values
    # (often the last non-empty row)
    best_r = max(row_counts, key=lambda x: x[1])
    if best_r[1] == 0:
        return None

    return best_r[0]


def tile_from_seed_row(
    grid: Grid,
    seed_row: int,
    bg: int = 0,
) -> Grid:
    """Tile the grid column-wise using per-column periods from the seed row.

    For each column, finds all non-bg values in that column, detects
    the vertical period, and tiles it through all rows.
    """
    rows, cols = grid.shape
    result = np.full_like(grid, bg)

    # Copy seed row
    result[seed_row] = grid[seed_row].copy()

    for c in range(cols):
        col = grid[:, c]
        non_bg = [(r, int(col[r])) for r in range(rows) if int(col[r]) != bg]

        if not non_bg:
            continue

        # Detect column period
        col_pattern = detect_1d_period(col, bg=bg)
        if col_pattern is not None and col_pattern.confidence >= 0.7:
            for r in range(rows):
                result[r, c] = col_pattern.pattern[r % col_pattern.period]
        else:
            # Fallback: just place the seed row value
            seed_val = int(grid[seed_row, c])
            if seed_val != bg:
                # Find period from the non-bg positions
                positions = [r for r, v in non_bg]
                if len(positions) >= 2:
                    diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    if diffs and all(d == diffs[0] for d in diffs):
                        period = diffs[0]
                        phase = positions[0] % period
                        for r in range(rows):
                            if r % period == phase:
                                result[r, c] = seed_val
                else:
                    result[positions[0], c] = non_bg[0][1]

    return result


# ---------------------------------------------------------------------------
# 2D periodic anomaly detection
# ---------------------------------------------------------------------------


def detect_2d_anomaly(
    grid: Grid,
    bg: int | None = None,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    """Detect a 2D periodic motif and find anomalous positions.

    Returns (motif_grid, anomaly_positions) or None.
    The motif_grid has the same shape as the input and contains
    the expected periodic value at each position.
    anomaly_positions is a set of (row, col) where the actual value
    deviates from the motif.
    """
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return None

    # Try row periods × col periods
    best_anomalies = None
    best_motif = None
    best_count = float('inf')

    for rp in range(1, rows // 2 + 1):
        for cp in range(1, cols // 2 + 1):
            if rp * cp > rows * cols // 2:
                continue

            # Infer motif by majority vote
            motif = np.zeros((rp, cp), dtype=np.int32)
            for mr in range(rp):
                for mc in range(cp):
                    vals = []
                    for r in range(mr, rows, rp):
                        for c in range(mc, cols, cp):
                            vals.append(int(grid[r, c]))
                    if vals:
                        motif[mr, mc] = Counter(vals).most_common(1)[0][0]

            # Count anomalies
            anomalies = set()
            for r in range(rows):
                for c in range(cols):
                    if int(grid[r, c]) != int(motif[r % rp, c % cp]):
                        anomalies.add((r, c))

            if not anomalies:
                continue  # perfect period, nothing to do
            if len(anomalies) > rows * cols // 4:
                continue  # too many anomalies

            if len(anomalies) < best_count:
                best_count = len(anomalies)
                best_anomalies = anomalies
                # Expand motif to full grid size
                full = np.zeros_like(grid)
                for r in range(rows):
                    for c in range(cols):
                        full[r, c] = motif[r % rp, c % cp]
                best_motif = full

    if best_motif is None:
        return None
    return best_motif, best_anomalies
