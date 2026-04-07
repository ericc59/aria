"""Pixel-level rule induction — supporting tool.

NOT part of the canonical architecture (ComputationGraph + protocol).
Learns decision-tree pixel rules from context. May be useful as a
feature for the graph editor's state encoding or as a fallback solver
for simple pixel-map tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from aria.types import DemoPair, Grid, Program, Bind, Call, Literal, Ref, Type


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------


def _extract_pixel_context(
    grid: Grid,
    r: int,
    c: int,
) -> np.ndarray:
    """Extract a feature vector for one pixel from its grid context.

    Features (30-dimensional):
      [0]     current color
      [1]     row position (normalized 0-1)
      [2]     col position (normalized 0-1)
      [3-10]  8-neighbor colors (up, down, left, right, ul, ur, dl, dr)
              -1 for out-of-bounds
      [11]    count of non-bg neighbors
      [12]    count of distinct neighbor colors
      [13]    most common neighbor color
      [14]    is on border (0/1)
      [15-24] count of each color (0-9) in same row
      [25]    count of distinct colors in same row
      [26]    count of distinct colors in same col
      [27]    distance to nearest non-bg pixel in row (left)
      [28]    distance to nearest non-bg pixel in row (right)
      [29]    distance to nearest non-bg pixel in col (up)
    """
    rows, cols = grid.shape
    features = np.zeros(30, dtype=np.float32)

    # Basic position
    features[0] = float(grid[r, c])
    features[1] = r / max(rows - 1, 1)
    features[2] = c / max(cols - 1, 1)

    # 8-neighbors
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbor_colors = []
    for i, (dr, dc) in enumerate(deltas):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            features[3 + i] = float(grid[nr, nc])
            neighbor_colors.append(int(grid[nr, nc]))
        else:
            features[3 + i] = -1.0

    # Neighbor statistics
    bg = _detect_bg_fast(grid)
    features[11] = sum(1 for nc in neighbor_colors if nc != bg)
    features[12] = len(set(neighbor_colors))
    if neighbor_colors:
        from collections import Counter
        features[13] = Counter(neighbor_colors).most_common(1)[0][0]

    # Border
    features[14] = float(r == 0 or r == rows - 1 or c == 0 or c == cols - 1)

    # Row color counts
    row = grid[r]
    for color in range(10):
        features[15 + color] = float(np.sum(row == color))
    features[25] = float(len(np.unique(row)))

    # Col color diversity
    col = grid[:, c]
    features[26] = float(len(np.unique(col)))

    # Distance to nearest non-bg in row (left)
    dist_left = 0
    for cc in range(c - 1, -1, -1):
        dist_left += 1
        if int(grid[r, cc]) != bg:
            break
    features[27] = float(dist_left) if dist_left > 0 else float(cols)

    # Distance to nearest non-bg in row (right)
    dist_right = 0
    for cc in range(c + 1, cols):
        dist_right += 1
        if int(grid[r, cc]) != bg:
            break
    features[28] = float(dist_right) if dist_right > 0 else float(cols)

    # Distance to nearest non-bg in col (up)
    dist_up = 0
    for rr in range(r - 1, -1, -1):
        dist_up += 1
        if int(grid[rr, c]) != bg:
            break
    features[29] = float(dist_up) if dist_up > 0 else float(rows)

    return features


def _detect_bg_fast(grid: Grid) -> int:
    unique, counts = np.unique(grid, return_counts=True)
    return int(unique[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Rule induction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelRule:
    """A learned pixel transformation rule."""
    tree: Any                    # fitted DecisionTreeClassifier
    accuracy: float              # fraction of pixels correctly predicted
    n_training_pixels: int
    n_changed_pixels: int
    tree_depth: int
    description: str = ""


def induce_pixel_rule(
    demos: tuple[DemoPair, ...],
    *,
    max_depth: int = 8,
    include_unchanged: bool = True,
) -> PixelRule | None:
    """Learn a decision tree that predicts each pixel's output color from context.

    Trains on changed pixels from all demos.  If include_unchanged, also
    samples unchanged pixels (to teach the tree to leave them alone).

    Returns a PixelRule if the tree achieves perfect accuracy on the
    training data, None otherwise.
    """
    if not demos:
        return None
    if not all(d.input.shape == d.output.shape for d in demos):
        return None

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    n_changed = 0

    for demo in demos:
        rows, cols = demo.input.shape
        diff_mask = demo.input != demo.output

        # Changed pixels — always include all
        for r, c in zip(*np.where(diff_mask)):
            ctx = _extract_pixel_context(demo.input, int(r), int(c))
            X_list.append(ctx)
            y_list.append(int(demo.output[r, c]))
            n_changed += 1

        # Unchanged pixels — sample proportionally
        if include_unchanged:
            unchanged = list(zip(*np.where(~diff_mask)))
            # Sample up to 2x the number of changed pixels
            n_sample = min(len(unchanged), max(n_changed * 2, 50))
            if n_sample > 0:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(unchanged), size=n_sample, replace=False)
                for idx in indices:
                    r, c = unchanged[idx]
                    ctx = _extract_pixel_context(demo.input, int(r), int(c))
                    X_list.append(ctx)
                    y_list.append(int(demo.output[r, c]))

    if n_changed == 0:
        return None

    X = np.array(X_list)
    y = np.array(y_list)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=1,
    )
    tree.fit(X, y)

    # Check: does the tree perfectly predict ALL pixels (changed + unchanged)?
    accuracy = _verify_rule_on_demos(tree, demos)

    if accuracy < 1.0:
        return None

    return PixelRule(
        tree=tree,
        accuracy=accuracy,
        n_training_pixels=len(X),
        n_changed_pixels=n_changed,
        tree_depth=tree.get_depth(),
        description=f"pixel_rule(depth={tree.get_depth()}, changed={n_changed})",
    )


def _verify_rule_on_demos(
    tree: DecisionTreeClassifier,
    demos: tuple[DemoPair, ...],
) -> float:
    """Check if the tree correctly predicts EVERY pixel in ALL demos."""
    total = 0
    correct = 0

    for demo in demos:
        rows, cols = demo.input.shape
        for r in range(rows):
            for c in range(cols):
                ctx = _extract_pixel_context(demo.input, r, c)
                predicted = tree.predict(ctx.reshape(1, -1))[0]
                expected = int(demo.output[r, c])
                total += 1
                if predicted == expected:
                    correct += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Apply the rule to produce output
# ---------------------------------------------------------------------------


def apply_pixel_rule(rule: PixelRule, grid: Grid) -> Grid:
    """Apply a learned pixel rule to transform a grid."""
    rows, cols = grid.shape
    result = grid.copy()

    for r in range(rows):
        for c in range(cols):
            ctx = _extract_pixel_context(grid, r, c)
            predicted = int(rule.tree.predict(ctx.reshape(1, -1))[0])
            result[r, c] = predicted

    return result


# ---------------------------------------------------------------------------
# Multi-step pixel rule induction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiStepPixelResult:
    """Result of multi-step pixel rule induction."""
    solved: bool
    rules: tuple[PixelRule, ...]
    n_steps: int
    description: str = ""


def induce_multi_step_pixel_rules(
    demos: tuple[DemoPair, ...],
    *,
    max_steps: int = 5,
    max_depth: int = 6,
) -> MultiStepPixelResult:
    """Iteratively learn pixel rules, applying each to the residual.

    Step 1: Learn a rule on (input, output).  Apply it.
    Step 2: If residual diff > 0, learn a rule on (step1_result, output).
    Repeat until diff is 0 or no rule makes progress.

    This handles multi-step transformations where each step's context
    depends on the previous step's output.
    """
    if not demos:
        return MultiStepPixelResult(solved=False, rules=(), n_steps=0)
    if not all(d.input.shape == d.output.shape for d in demos):
        return MultiStepPixelResult(solved=False, rules=(), n_steps=0)

    # Current state for each demo
    states = [d.input.copy() for d in demos]
    targets = [d.output for d in demos]
    rules: list[PixelRule] = []

    total_diff = sum(
        int(np.sum(s != t)) for s, t in zip(states, targets)
    )
    if total_diff == 0:
        return MultiStepPixelResult(solved=True, rules=(), n_steps=0)

    for step in range(max_steps):
        # Build intermediate demo pairs from current states
        intermediate_demos = tuple(
            DemoPair(input=s, output=t)
            for s, t in zip(states, targets)
        )

        rule = induce_pixel_rule(intermediate_demos, max_depth=max_depth)
        if rule is None:
            break

        # Apply the rule to get new states
        new_states = [apply_pixel_rule(rule, s) for s in states]
        new_diff = sum(
            int(np.sum(s != t)) for s, t in zip(new_states, targets)
        )

        if new_diff >= total_diff:
            break  # no progress

        rules.append(rule)
        states = new_states
        total_diff = new_diff

        if total_diff == 0:
            break

    solved = total_diff == 0
    return MultiStepPixelResult(
        solved=solved,
        rules=tuple(rules),
        n_steps=len(rules),
        description=f"pixel_rules({len(rules)} steps)" if solved else "",
    )
