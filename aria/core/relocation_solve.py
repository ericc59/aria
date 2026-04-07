"""Relocation solver — move/swap pixels by structural rule.

Handles tasks where the output equals the input with some pixels
relocated (erased from one position, added at another, same colors
conserved).

Approach: for each pixel that should move, try bounded relocation
rules and verify exactly across all demos.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from aria.decomposition import detect_bg
from aria.types import DemoPair, Grid


def solve_relocation(demos: tuple[DemoPair, ...]) -> Grid | None:
    """Try to solve a relocation task by finding a shared relocation rule.

    Returns a function that transforms input→output, or None.
    """
    if not demos or not all(d.input.shape == d.output.shape for d in demos):
        return None

    # Check: is this a relocation task? (erased == added by color)
    for d in demos:
        bg = detect_bg(d.input)
        diff = d.input != d.output
        if not np.any(diff):
            return None
        from collections import Counter
        erased = Counter()
        added = Counter()
        for r, c in zip(*np.where(diff)):
            ic, oc = int(d.input[r, c]), int(d.output[r, c])
            if ic != bg:
                erased[ic] += 1
            if oc != bg:
                added[oc] += 1
        if erased != added:
            return None

    # Try bounded relocation rules
    for rule_fn in [
        _try_horizontal_flip_selective,
        _try_vertical_flip_selective,
        _try_color_swap_in_place,
    ]:
        result = rule_fn(demos)
        if result is not None:
            return result

    return None


def _try_horizontal_flip_selective(demos: tuple[DemoPair, ...]) -> Any | None:
    """Try: for each pixel of a specific color, flip its column position."""
    for d in demos:
        bg = detect_bg(d.input)
        h, w = d.input.shape
        diff = d.input != d.output

        # Find which colors move
        from collections import Counter
        moved_colors = set()
        for r, c in zip(*np.where(diff)):
            if int(d.input[r, c]) != bg:
                moved_colors.add(int(d.input[r, c]))

        for mc in moved_colors:
            # Try: flip all pixels of color mc across the vertical center
            result = d.input.copy()
            for r in range(h):
                for c in range(w):
                    if int(d.input[r, c]) == mc:
                        new_c = w - 1 - c
                        result[r, c] = bg  # erase old
                        result[r, new_c] = mc  # place new

            if np.array_equal(result, d.output):
                # Verify across all demos
                all_ok = True
                for d2 in demos[1:]:
                    bg2 = detect_bg(d2.input)
                    h2, w2 = d2.input.shape
                    r2 = d2.input.copy()
                    for rr in range(h2):
                        for cc in range(w2):
                            if int(d2.input[rr, cc]) == mc:
                                new_cc = w2 - 1 - cc
                                r2[rr, cc] = bg2
                                r2[rr, new_cc] = mc
                    if not np.array_equal(r2, d2.output):
                        all_ok = False
                        break
                if all_ok:
                    return ("hflip_color", mc)

    return None


def _try_vertical_flip_selective(demos: tuple[DemoPair, ...]) -> Any | None:
    """Try: for each pixel of a specific color, flip its row position."""
    for d in demos:
        bg = detect_bg(d.input)
        h, w = d.input.shape
        diff = d.input != d.output

        from collections import Counter
        moved_colors = set()
        for r, c in zip(*np.where(diff)):
            if int(d.input[r, c]) != bg:
                moved_colors.add(int(d.input[r, c]))

        for mc in moved_colors:
            result = d.input.copy()
            for r in range(h):
                for c in range(w):
                    if int(d.input[r, c]) == mc:
                        new_r = h - 1 - r
                        result[r, c] = bg
                        result[new_r, c] = mc

            if np.array_equal(result, d.output):
                all_ok = True
                for d2 in demos[1:]:
                    bg2 = detect_bg(d2.input)
                    h2, w2 = d2.input.shape
                    r2 = d2.input.copy()
                    for rr in range(h2):
                        for cc in range(w2):
                            if int(d2.input[rr, cc]) == mc:
                                new_rr = h2 - 1 - rr
                                r2[rr, cc] = bg2
                                r2[new_rr, cc] = mc
                    if not np.array_equal(r2, d2.output):
                        all_ok = False
                        break
                if all_ok:
                    return ("vflip_color", mc)

    return None


def _try_color_swap_in_place(demos: tuple[DemoPair, ...]) -> Any | None:
    """Try: swap two colors at their positions (no spatial movement)."""
    d = demos[0]
    bg = detect_bg(d.input)
    diff = d.input != d.output

    from collections import Counter
    pairs = Counter()
    for r, c in zip(*np.where(diff)):
        pairs[(int(d.input[r, c]), int(d.output[r, c]))] += 1

    # Check if it's a symmetric swap
    swap_pairs = []
    for (a, b), cnt in pairs.items():
        if (b, a) in pairs:
            if a < b:
                swap_pairs.append((a, b))

    for a, b in swap_pairs:
        # Try swapping a↔b everywhere
        all_ok = True
        for d2 in demos:
            result = d2.input.copy()
            mask_a = d2.input == a
            mask_b = d2.input == b
            result[mask_a] = b
            result[mask_b] = a
            if not np.array_equal(result, d2.output):
                all_ok = False
                break
        if all_ok:
            return ("swap_colors", a, b)

    return None
