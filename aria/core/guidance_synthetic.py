"""Synthetic-data hooks — generate training variants from solved traces.

Provides hooks for creating perturbations and family variants to support
future training on relation detection, correspondence, role prediction,
and stage-1 program family selection.

Lightweight — no heavy generation, just the interfaces and simple transforms.
No solver changes. No task-id logic.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aria.types import Grid


# ---------------------------------------------------------------------------
# Grid perturbations
# ---------------------------------------------------------------------------


def rotate_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    """Remap colors in a grid according to the given mapping.

    Colors not in the mapping are left unchanged.
    """
    result = grid.copy()
    for src, dst in mapping.items():
        result[grid == src] = dst
    return result


def random_color_rotation(grid: Grid, rng: np.random.Generator | None = None) -> tuple[Grid, dict[int, int]]:
    """Apply a random color permutation to a grid.

    Returns (new_grid, color_mapping).
    """
    if rng is None:
        rng = np.random.default_rng()

    colors = sorted(set(int(v) for v in np.unique(grid)))
    shuffled = list(colors)
    rng.shuffle(shuffled)
    mapping = dict(zip(colors, shuffled))
    return rotate_colors(grid, mapping), mapping


def flip_grid(grid: Grid, axis: str) -> Grid:
    """Flip a grid along an axis: 'lr', 'ud', 'rot90', 'rot180'."""
    if axis == "lr":
        return np.fliplr(grid).copy()
    elif axis == "ud":
        return np.flipud(grid).copy()
    elif axis == "rot90":
        return np.rot90(grid).copy()
    elif axis == "rot180":
        return np.rot90(grid, 2).copy()
    else:
        raise ValueError(f"Unknown axis: {axis}")


def inject_noise(grid: Grid, fraction: float = 0.05, rng: np.random.Generator | None = None) -> Grid:
    """Randomly change a fraction of pixels to random colors."""
    if rng is None:
        rng = np.random.default_rng()

    result = grid.copy()
    n_pixels = grid.size
    n_noise = max(1, int(n_pixels * fraction))
    indices = rng.choice(n_pixels, size=n_noise, replace=False)
    rows, cols = np.unravel_index(indices, grid.shape)
    result[rows, cols] = rng.integers(0, 10, size=n_noise).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Record-level augmentation
# ---------------------------------------------------------------------------


def augment_color_rotation(
    record: dict,
    rng: np.random.Generator | None = None,
) -> dict:
    """Produce a color-rotated variant of an exported guidance record.

    Applies the same random color permutation to all demo grids (input and output).
    Updates perception summaries and role labels accordingly.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine color mapping from all grids
    all_colors: set[int] = set()
    for demo in record.get("train_demos", []):
        for grid_data in [demo.get("input", []), demo.get("output", [])]:
            for row in grid_data:
                all_colors.update(int(c) for c in row)

    colors = sorted(all_colors)
    shuffled = list(colors)
    rng.shuffle(shuffled)
    mapping = dict(zip(colors, shuffled))

    # Apply to all demo grids
    new_demos = []
    for demo in record.get("train_demos", []):
        new_demo = {}
        for key in ("input", "output"):
            grid_data = demo.get(key, [])
            new_grid = [[mapping.get(c, c) for c in row] for row in grid_data]
            new_demo[key] = new_grid
        new_demos.append(new_demo)

    new_record = dict(record)
    new_record["train_demos"] = new_demos

    # Update bg_color in perception summaries
    new_perceptions = []
    for ps in record.get("perception_summaries", []):
        new_ps = dict(ps)
        if "bg_color" in new_ps:
            new_ps["bg_color"] = mapping.get(new_ps["bg_color"], new_ps["bg_color"])
        if "palette" in new_ps:
            new_ps["palette"] = sorted(mapping.get(c, c) for c in new_ps["palette"])
        if "non_bg_colors" in new_ps:
            new_ps["non_bg_colors"] = sorted(mapping.get(c, c) for c in new_ps["non_bg_colors"])
        new_perceptions.append(new_ps)
    new_record["perception_summaries"] = new_perceptions

    # Update roles
    new_roles = []
    for role in record.get("roles", []):
        new_role = dict(role)
        if "color" in new_role and new_role["color"] is not None:
            new_role["color"] = mapping.get(new_role["color"], new_role["color"])
        new_roles.append(new_role)
    new_record["roles"] = new_roles

    # Update legend
    if record.get("legend"):
        new_legend = dict(record["legend"])
        if "key_to_value" in new_legend:
            new_legend["key_to_value"] = {
                str(mapping.get(int(k), int(k))): mapping.get(int(v), int(v))
                for k, v in new_legend["key_to_value"].items()
            }
        new_record["legend"] = new_legend

    return new_record


def augment_spatial_flip(
    record: dict,
    axis: str = "lr",
) -> dict:
    """Produce a spatially flipped variant of an exported guidance record."""
    new_demos = []
    for demo in record.get("train_demos", []):
        new_demo = {}
        for key in ("input", "output"):
            grid_data = demo.get(key, [])
            arr = np.array(grid_data, dtype=np.uint8)
            flipped = flip_grid(arr, axis)
            new_demo[key] = flipped.tolist()
        new_demos.append(new_demo)

    new_record = dict(record)
    new_record["train_demos"] = new_demos
    return new_record


def augment_correspondence_shuffle(
    record: dict,
    rng: np.random.Generator | None = None,
) -> dict:
    """Shuffle correspondence order in an exported record.

    Useful for training models that should be order-invariant
    over correspondence sets.
    """
    if rng is None:
        rng = np.random.default_rng()

    corr = record.get("correspondences")
    if not corr:
        return record

    new_record = dict(record)
    new_corr = list(corr)
    rng.shuffle(new_corr)
    new_record["correspondences"] = new_corr
    return new_record


# ---------------------------------------------------------------------------
# Family variant generation
# ---------------------------------------------------------------------------


def generate_family_variants(
    record: dict,
    n_variants: int = 3,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate n augmented variants of a solved record.

    Applies a mix of color rotation and spatial flips.
    """
    if rng is None:
        rng = np.random.default_rng()

    variants = []
    axes = ["lr", "ud", "rot180"]

    for i in range(n_variants):
        # Always do color rotation
        variant = augment_color_rotation(record, rng=rng)

        # Optionally add spatial flip
        if i < len(axes):
            variant = augment_spatial_flip(variant, axis=axes[i])

        # Shuffle correspondences
        variant = augment_correspondence_shuffle(variant, rng=rng)

        variants.append(variant)

    return variants
