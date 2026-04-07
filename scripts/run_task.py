#!/usr/bin/env python3
"""Run the solver on a single ARC task with debug output.

Usage:
    python scripts/run_task.py TASK_ID
    python scripts/run_task.py TASK_ID --dataset v2-eval
    python scripts/run_task.py TASK_ID --verbose
    python scripts/run_task.py TASK_ID --phase clause   # only run clause engine
    python scripts/run_task.py TASK_ID --phase synth    # only run typed synthesis
    python scripts/run_task.py TASK_ID --show-grids      # print grids in color
"""
import argparse
import sys
import time
import numpy as np

from aria.datasets import get_dataset, load_arc_task
from aria.guided.perceive import perceive
from aria.guided.correspond import map_output_to_input
from aria.guided.output_size import infer_output_size
from aria.guided.dsl import synthesize_program, _verify, Program
from aria.types import Grid


# ---------------------------------------------------------------------------
# Grid visualization
# ---------------------------------------------------------------------------

# ARC color palette → ANSI 256-color codes
_ARC_ANSI = {
    0: 232,   # black (near-black gray)
    1: 21,    # blue
    2: 196,   # red
    3: 28,    # green
    4: 226,   # yellow
    5: 244,   # gray
    6: 201,   # magenta/pink
    7: 208,   # orange
    8: 51,    # cyan/light blue
    9: 88,    # maroon/dark red
}


def _cell(val: int) -> str:
    """Render a single cell as a colored block with the digit."""
    code = _ARC_ANSI.get(int(val), 255)
    # Use bright white text on dark colors, black text on bright colors
    fg = 15 if val in (0, 1, 6, 9) else 0
    return f"\033[38;5;{fg};48;5;{code}m {val} \033[0m"


def _diff_cell(val: int, expected: int) -> str:
    """Render a cell, highlighted red if it differs from expected."""
    if val == expected:
        return _cell(val)
    # Wrong cell: red background with X
    return f"\033[97;41m {val} \033[0m"


def print_grid(grid: np.ndarray, label: str = "", indent: int = 2):
    prefix = " " * indent
    if label:
        print(f"{prefix}{label} ({grid.shape[0]}x{grid.shape[1]}):")
    for r in range(grid.shape[0]):
        row_str = "".join(_cell(grid[r, c]) for c in range(grid.shape[1]))
        print(f"{prefix}  {row_str}")


def print_grid_diff(predicted: np.ndarray, expected: np.ndarray, label: str = "", indent: int = 2):
    """Print predicted grid with wrong cells highlighted."""
    prefix = " " * indent
    if predicted.shape != expected.shape:
        print(f"{prefix}{label}: SHAPE MISMATCH pred={predicted.shape} expected={expected.shape}")
        print_grid(predicted, "predicted", indent)
        print_grid(expected, "expected", indent)
        return
    n_diff = int(np.sum(predicted != expected))
    print(f"{prefix}{label} ({n_diff} wrong cells):")
    for r in range(predicted.shape[0]):
        row_str = "".join(_diff_cell(predicted[r, c], expected[r, c])
                          for c in range(predicted.shape[1]))
        print(f"{prefix}  {row_str}")


def print_pair(inp: np.ndarray, out: np.ndarray, label: str = ""):
    """Print input → output side by side (or stacked if too wide)."""
    print(f"  {label}")
    # Side by side if narrow enough
    if inp.shape[1] + out.shape[1] <= 15:
        max_rows = max(inp.shape[0], out.shape[0])
        for r in range(max_rows):
            left = ""
            if r < inp.shape[0]:
                left = "".join(_cell(inp[r, c]) for c in range(inp.shape[1]))
            else:
                left = " " * (inp.shape[1] * 3)
            right = ""
            if r < out.shape[0]:
                right = "".join(_cell(out[r, c]) for c in range(out.shape[1]))
            sep = " → " if r == max_rows // 2 else "   "
            print(f"    {left}{sep}{right}")
    else:
        print_grid(inp, "input", 4)
        print_grid(out, "output", 4)


# ---------------------------------------------------------------------------
# Debug phases
# ---------------------------------------------------------------------------

def run_perception(demos, test_pairs, verbose=False):
    print("\n=== Perception ===")
    for i, (inp, out) in enumerate(demos):
        in_facts = perceive(inp)
        out_facts = perceive(out)
        print(f"  demo {i}: input {inp.shape} bg={in_facts.bg} "
              f"objs={in_facts.n_objects} seps={len(in_facts.separators)}"
              f" | output {out.shape} bg={out_facts.bg} objs={out_facts.n_objects}")
        if verbose:
            for j, o in enumerate(in_facts.objects):
                rels = []
                if hasattr(o, 'is_rectangular') and o.is_rectangular:
                    rels.append('rect')
                if hasattr(o, 'is_line') and o.is_line:
                    rels.append('line')
                print(f"    obj{j}: color={o.color} {o.height}x{o.width} "
                      f"at ({o.row},{o.col}) size={o.size} {' '.join(rels)}")
            for s in in_facts.separators:
                print(f"    sep: axis={s.axis} index={s.index} color={s.color}")
    for i, (ti, _) in enumerate(test_pairs):
        in_facts = perceive(ti)
        print(f"  test {i}: input {ti.shape} bg={in_facts.bg} "
              f"objs={in_facts.n_objects} seps={len(in_facts.separators)}")


def run_correspondence(demos, verbose=False):
    print("\n=== Correspondence ===")
    for i, (inp, out) in enumerate(demos):
        in_facts = perceive(inp)
        out_facts = perceive(out)
        mappings = map_output_to_input(out_facts, in_facts)
        for m in mappings:
            src = (f"in(color={m.color_from} {m.in_obj.height}x{m.in_obj.width})"
                   if m.in_obj else "none")
            print(f"  demo {i}: {m.match_type:16s} {src} → "
                  f"out(color={m.color_to} {m.out_obj.height}x{m.out_obj.width})")


def run_output_size(demos, test_pairs):
    print("\n=== Output Size ===")
    rule = infer_output_size(demos)
    if rule:
        print(f"  rule: {rule.description} (mode={rule.mode})")
        for i, (ti, to) in enumerate(test_pairs):
            pred = rule.predict(perceive(ti))
            ok = pred == to.shape
            status = "ok" if ok else "WRONG"
            print(f"  test {i}: pred={pred} actual={to.shape} [{status}]")
    else:
        print("  no size rule found")
    return rule


def run_clause_engine(demos, verbose=False):
    """Run just the clause engine, return (program, elapsed)."""
    print("\n=== Phase 1: Clause Engine ===")
    from aria.guided.induce import induce_program
    t0 = time.time()
    clause_prog = induce_program(demos)
    elapsed = time.time() - t0
    if clause_prog is None:
        print(f"  no clause program found ({elapsed:.2f}s)")
        return None, elapsed

    print(f"  clauses ({elapsed:.2f}s):")
    for c in clause_prog.clauses:
        print(f"    {c.description}")

    # Verify on demos
    all_ok = True
    for i, (inp, out) in enumerate(demos):
        try:
            pred = clause_prog.execute(inp)
            if np.array_equal(pred, out):
                print(f"  demo {i}: ok")
            else:
                n_diff = int(np.sum(pred != out)) if pred.shape == out.shape else -1
                print(f"  demo {i}: WRONG ({n_diff} cells)")
                all_ok = False
        except Exception as e:
            print(f"  demo {i}: ERROR {e}")
            all_ok = False

    if all_ok:
        from aria.guided.dsl import _make_program
        desc = '; '.join(c.description for c in clause_prog.clauses)
        return _make_program(clause_prog.execute, f"clause: {desc}"), elapsed
    else:
        print("  clause program does NOT verify on all demos")
        return None, elapsed


def run_typed_synthesis(demos, verbose=False):
    """Run bottom-up typed synthesis (skipping clause engine)."""
    print("\n=== Phase 2: Typed Synthesis ===")
    from aria.guided.synthesize import _build_op_library, _bottom_up_search
    t0 = time.time()
    demo_bg = perceive(demos[0][0]).bg
    op_lib = _build_op_library(bg=demo_bg)
    prog = _bottom_up_search(demos, op_lib, max_depth=3)
    elapsed = time.time() - t0
    if prog:
        print(f"  found: {prog.description} ({elapsed:.2f}s)")
    else:
        print(f"  no program found ({elapsed:.2f}s)")
    return prog, elapsed


def run_full_solve(demos):
    """Run the full solver pipeline."""
    print("\n=== Full Solve ===")
    t0 = time.time()
    prog = synthesize_program(demos)
    elapsed = time.time() - t0
    if prog:
        print(f"  program: {prog.description} ({elapsed:.2f}s)")
    else:
        print(f"  no program found ({elapsed:.2f}s)")
    return prog, elapsed


def verify_program(prog: Program, demos, test_pairs, show_grids=False):
    """Verify a program on demos and test pairs."""
    print("\n=== Verification ===")
    train_ok = True
    for i, (inp, out) in enumerate(demos):
        try:
            pred = prog.execute(inp)
            match = np.array_equal(pred, out)
            if match:
                print(f"  demo {i}: ok")
            else:
                n_diff = int(np.sum(pred != out)) if pred.shape == out.shape else -1
                print(f"  demo {i}: FAIL ({n_diff} wrong cells)")
                train_ok = False
                if show_grids:
                    print_grid_diff(pred, out, "predicted vs expected", 4)
        except Exception as e:
            print(f"  demo {i}: ERROR {e}")
            train_ok = False

    test_ok = True
    for i, (ti, to) in enumerate(test_pairs):
        try:
            pred = prog.execute(ti)
            match = np.array_equal(pred, to)
            if match:
                print(f"  test {i}: ok")
            else:
                n_diff = int(np.sum(pred != to)) if pred.shape == to.shape else -1
                print(f"  test {i}: FAIL ({n_diff} wrong cells)")
                test_ok = False
                if show_grids:
                    print_grid_diff(pred, to, "predicted vs expected", 4)
        except Exception as e:
            print(f"  test {i}: ERROR {e}")
            test_ok = False

    if train_ok and test_ok:
        print("\n  SOLVED")
    elif train_ok:
        print("\n  TRAIN VERIFIED (test failures)")
    else:
        print("\n  NOT VERIFIED")

    return train_ok, test_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run solver on a single ARC task")
    parser.add_argument("task_id", help="ARC task ID (hex string)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset (v2-train, v2-eval). Auto-detects if omitted.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed perception/correspondence")
    parser.add_argument("--show-grids", "-g", action="store_true",
                        help="Show colored grid output")
    parser.add_argument("--phase", choices=["clause", "synth", "all"], default="all",
                        help="Which synthesis phase to run")
    args = parser.parse_args()

    # Load task (auto-detect dataset)
    task = None
    dataset_name = args.dataset
    if dataset_name:
        ds = get_dataset(dataset_name)
        task = load_arc_task(ds, args.task_id)
    else:
        for name in ("v2-train", "v2-eval"):
            try:
                ds = get_dataset(name)
                task = load_arc_task(ds, args.task_id)
                dataset_name = name
                break
            except (FileNotFoundError, KeyError):
                continue
        if task is None:
            print(f"Task {args.task_id} not found in any dataset")
            sys.exit(1)

    demos = [(np.array(p.input), np.array(p.output)) for p in task.train]
    test_pairs = [(np.array(p.input), np.array(p.output)) for p in task.test]

    print(f"Task: {args.task_id} ({dataset_name})")
    print(f"Demos: {len(demos)}, Tests: {len(test_pairs)}")
    for i, (inp, out) in enumerate(demos):
        same = "same" if inp.shape == out.shape else "DIFFERENT"
        print(f"  train {i}: {inp.shape} → {out.shape} [{same}]")
    for i, (ti, to) in enumerate(test_pairs):
        print(f"  test  {i}: {ti.shape} → {to.shape}")

    # Show grids
    if args.show_grids:
        print("\n=== Grids ===")
        for i, (inp, out) in enumerate(demos):
            print_pair(inp, out, f"demo {i}")
        for i, (ti, _) in enumerate(test_pairs):
            print_grid(ti, f"test {i} input")

    # Perception
    run_perception(demos, test_pairs, verbose=args.verbose)

    # Correspondence
    if args.verbose:
        run_correspondence(demos, verbose=args.verbose)

    # Output size
    run_output_size(demos, test_pairs)

    # Synthesis
    prog = None
    if args.phase == "clause":
        prog, _ = run_clause_engine(demos, verbose=args.verbose)
    elif args.phase == "synth":
        prog, _ = run_typed_synthesis(demos, verbose=args.verbose)
    else:
        prog, _ = run_full_solve(demos)

    if prog is None:
        print("\nNo program found.")
        sys.exit(1)

    # Verify
    train_ok, test_ok = verify_program(prog, demos, test_pairs,
                                        show_grids=args.show_grids)
    sys.exit(0 if (train_ok and test_ok) else 1)


if __name__ == "__main__":
    main()
