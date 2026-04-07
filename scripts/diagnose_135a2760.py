"""Diagnose ARC task 135a2760: periodic core + boundary conditions."""

import numpy as np
from aria.datasets import get_dataset, load_arc_task

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "maroon"
}

def color_name(c):
    return COLOR_NAMES.get(c, f"c{c}")

def find_frames(grid):
    """Find rectangular frames (borders of a single non-black color).

    Strategy: look for rectangles where the border is a single color
    and is at least 1 cell thick.
    """
    h, w = grid.shape
    frames = []

    # Try every possible rectangle
    for r1 in range(h):
        for c1 in range(w):
            for r2 in range(r1 + 2, h):  # need at least 3 rows
                for c2 in range(c1 + 2, w):  # need at least 3 cols
                    # Check if this rectangle has a uniform-color border
                    top = grid[r1, c1:c2+1]
                    bottom = grid[r2, c1:c2+1]
                    left = grid[r1:r2+1, c1]
                    right = grid[r1:r2+1, c2]

                    border = np.concatenate([top, bottom, left, right])
                    unique = np.unique(border)

                    if len(unique) == 1 and unique[0] != 0:
                        # Check interior is not all the same color as border
                        interior = grid[r1+1:r2, c1+1:c2]
                        if interior.size > 0:
                            frames.append({
                                'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
                                'color': int(unique[0]),
                                'interior': interior.copy()
                            })

    # Remove frames that are subsets of larger frames with same color
    # Keep the largest frames
    filtered = []
    for f in frames:
        is_subset = False
        for g in frames:
            if f is g:
                continue
            if (g['r1'] <= f['r1'] and g['c1'] <= f['c1'] and
                g['r2'] >= f['r2'] and g['c2'] >= f['c2'] and
                (g['r1'] < f['r1'] or g['c1'] < f['c1'] or
                 g['r2'] > f['r2'] or g['c2'] > f['c2'])):
                is_subset = True
                break
        if not is_subset:
            filtered.append(f)

    return filtered

def analyze_columns(interior, label=""):
    """Analyze interior column-by-column for periodic patterns."""
    h, w = interior.shape
    print(f"  Interior shape: {h}x{w}")
    print(f"  Full interior:")
    for r in range(h):
        row_str = " ".join(f"{color_name(interior[r,c]):>7}" for c in range(w))
        print(f"    row {r}: {row_str}")

    print(f"\n  Column-by-column analysis:")
    for c in range(w):
        col = interior[:, c]
        col_names = [color_name(v) for v in col]
        print(f"    col {c}: {col_names}")

        # Try to find repeating body pattern
        if h >= 3:
            # Check cap + body* + cap pattern
            top_cap = col[0]
            bottom_cap = col[-1]
            body = col[1:-1]

            # Check if body has a period
            for period in range(1, len(body) + 1):
                tile = body[:period]
                repeats = True
                for i in range(len(body)):
                    if body[i] != tile[i % period]:
                        repeats = False
                        break
                if repeats:
                    tile_names = [color_name(v) for v in tile]
                    cap_match = "same" if top_cap == bottom_cap else "different"
                    body_matches_cap = all(v == top_cap for v in body)
                    print(f"           top_cap={color_name(top_cap)}, body period={period} {tile_names}, bottom_cap={color_name(bottom_cap)} (caps: {cap_match})")
                    if body_matches_cap:
                        print(f"           (body matches cap — uniform column)")
                    break

def compare_columns(inp_interior, out_interior):
    """Compare input vs output interior column-by-column."""
    ih, iw = inp_interior.shape
    oh, ow = out_interior.shape
    print(f"\n  Comparison: input interior {ih}x{iw} vs output interior {oh}x{ow}")

    if iw != ow:
        print(f"  WARNING: different widths!")
        return

    for c in range(iw):
        in_col = inp_interior[:, c]
        out_col = out_interior[:, c]
        in_names = [color_name(v) for v in in_col]
        out_names = [color_name(v) for v in out_col]

        changed = "CHANGED" if not np.array_equal(in_col, out_col) else "same"
        print(f"    col {c}: input={in_names}")
        print(f"            output={out_names}  [{changed}]")

        if changed == "CHANGED":
            # Show diffs
            if ih == oh:
                for r in range(ih):
                    if in_col[r] != out_col[r]:
                        print(f"            row {r}: {color_name(in_col[r])} -> {color_name(out_col[r])}")
            else:
                print(f"            height changed: {ih} -> {oh}")

# Load task
ds = get_dataset('v2-eval')
task = load_arc_task(ds, '135a2760')

print("=" * 80)
print("TASK 135a2760 DIAGNOSIS")
print("=" * 80)

# First, just print the raw grids
for i, p in enumerate(task.train):
    inp = np.array(p.input)
    out = np.array(p.output)
    print(f"\n{'='*80}")
    print(f"DEMO {i}: input {inp.shape}, output {out.shape}")
    print(f"{'='*80}")

    print("\nInput grid:")
    for r in range(inp.shape[0]):
        print("  " + " ".join(f"{v}" for v in inp[r]))

    print("\nOutput grid:")
    for r in range(out.shape[0]):
        print("  " + " ".join(f"{v}" for v in out[r]))

    # Find frames in input
    print("\n--- Input frames ---")
    in_frames = find_frames(inp)
    if not in_frames:
        print("  No frames found! Trying whole grid as region...")
        # Maybe the whole grid is the region
        analyze_columns(inp, "whole input")
    else:
        for j, f in enumerate(in_frames):
            print(f"\n  Frame {j}: rows [{f['r1']}..{f['r2']}], cols [{f['c1']}..{f['c2']}], border={color_name(f['color'])}")
            analyze_columns(f['interior'], f"input frame {j}")

    # Find frames in output
    print("\n--- Output frames ---")
    out_frames = find_frames(out)
    if not out_frames:
        print("  No frames found!")
        analyze_columns(out, "whole output")
    else:
        for j, f in enumerate(out_frames):
            print(f"\n  Frame {j}: rows [{f['r1']}..{f['r2']}], cols [{f['c1']}..{f['c2']}], border={color_name(f['color'])}")
            analyze_columns(f['interior'], f"output frame {j}")

    # Compare matching frames
    if in_frames and out_frames:
        print("\n--- Column comparison (input vs output) ---")
        # Match frames by position
        for j, (inf, outf) in enumerate(zip(in_frames, out_frames)):
            print(f"\n  Frame {j}:")
            compare_columns(inf['interior'], outf['interior'])

# Test inputs
for i, t in enumerate(task.test):
    inp = np.array(t.input)
    print(f"\n{'='*80}")
    print(f"TEST {i}: input {inp.shape}")
    print(f"{'='*80}")

    print("\nInput grid:")
    for r in range(inp.shape[0]):
        print("  " + " ".join(f"{v}" for v in inp[r]))

    print("\n--- Input frames ---")
    in_frames = find_frames(inp)
    if not in_frames:
        print("  No frames found!")
        analyze_columns(inp, "whole input")
    else:
        for j, f in enumerate(in_frames):
            print(f"\n  Frame {j}: rows [{f['r1']}..{f['r2']}], cols [{f['c1']}..{f['c2']}], border={color_name(f['color'])}")
            analyze_columns(f['interior'], f"input frame {j}")
