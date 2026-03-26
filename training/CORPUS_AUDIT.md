# Corpus Audit Report

## 1. Verified Program Corpus

- **Total programs**: 47
- **Unique task IDs**: 34
- **Programs with signatures**: 0
- **Unique signatures**: 0
- **Tasks with multiple programs**: 10

### Step-count distribution

| Steps | Count |
|-------|-------|
| 1 | 14 |
| 2 | 8 |
| 3 | 4 |
| 4 | 8 |
| 5 | 3 |
| 6 | 1 |
| 7 | 1 |
| 8 | 1 |
| 9 | 1 |
| 10 | 2 |
| 12 | 1 |
| 13 | 2 |
| 14 | 1 |

### Source distribution

- `corpus-report:corpus_report.json`: 24
- `solve-report:v1-train.json`: 23

### Library-op usage (top 20)

| Op | Count |
|-----|-------|
| `find_objects` | 22 |
| `apply_color_map` | 20 |
| `stack_h` | 13 |
| `from_object` | 12 |
| `dims_make` | 12 |
| `new_grid` | 10 |
| `by_size_rank` | 8 |
| `overlay` | 7 |
| `crop` | 7 |
| `reflect_grid` | 6 |
| `nth` | 6 |
| `rotate_grid` | 5 |
| `embed` | 5 |
| `map_list` | 5 |
| `if_then_else` | 5 |
| `stack_v` | 4 |
| `tile_grid` | 4 |
| `dims_of` | 4 |
| `sub` | 4 |
| `sort_by` | 4 |

Total unique ops: 53
  Ops used only once: background_obj, count, excluding, get_pos_x, get_pos_y, get_size, get_width, gt, infer_map, length, place_at, transpose_grid, unique_colors

## 2. Solve Results

### Train (v1-train)

- Tasks: 83
- Solved: 25 (30.1%)
- Tasks with signatures: 0
- Rounds distribution: {1: 13, 2: 70}
- Solve sources: {'search': 25}

### Eval (v1-eval)

- Tasks: 6
- Solved: 0 (0.0%)
- Tasks with signatures: 6
- Rounds distribution: {0: 6}

## 3. Refinement Traces

**No refinement trace store found.**

## 4. Library

- Entries: 12
- Total use count: 29
- Max level: 1
- Return types: {'GRID': 7, 'OBJECT': 4, 'OBJECT_SET': 1}

## 5. Weaknesses

1. CRITICAL: Only 47 verified programs (target: 800-1000 for Phase 2). Far too few for any model training.
2. Size skew: 22/47 programs (47%) are trivial (<=2 steps). Model will overfit to short programs.
3. No programs have task signatures attached. Cannot train NEXT_FOCUS without signature labels.
4. No refinement traces found. Cannot train NEXT_EDIT without (feedback, edit) pairs.
5. Eval solve rate is 0% (6 tasks). Retrieval-transfer is not working yet.
6. Library has only 12 entries (target: 80-120). Abstractions are thin.
7. Only 8 programs with >=8 steps. Need complex multi-step examples for SKETCH training.

## 6. Recommendations

1. PRIORITY 1: Run batch solve on full ARC-1 training set (400 tasks) with higher search budgets (max_candidates=50000, max_steps=5, max_refinement_rounds=4). Target: 200+ verified programs.
2. PRIORITY 2: Backfill task signatures into program_store.json. Re-run solve with signature persistence enabled, or write a migration script to recompute signatures for existing programs.
3. PRIORITY 3: Enable trace persistence in solve.py (RefinementTraceStore.save_json). Every solve attempt — successful or not — should emit a trace.
4. NOT READY for SKETCH training. Need at least 200 verified programs with 80+ having >=3 steps. Current: 47 programs total.
5. NOT READY for NEXT_EDIT training. Need 100+ refinement traces with 20+ near-misses. Current: 0 traces, 0 near-misses.
6. Train NEXT_FOCUS first — it requires less data and improves refinement quality for collecting NEXT_EDIT pairs. Need: 100 programs with signatures and 3+ focus labels. Current: 0 with signatures, 0 focus labels.
7. Generate synthetic tasks (Phase 2 in DESIGN.md) to multiply training volume. Target: 200K random step sequences length 3-10 for clean, high-volume training data.
8. Grow the library before training. More library entries = more compositional programs in training data. Run library mining on a larger program corpus first.

## 7. Verdict

**Do not train yet.** Collect more data first.

Priority actions before training:
- CRITICAL: Only 47 verified programs (target: 800-1000 for Phase 2). Far too few for any model training.
