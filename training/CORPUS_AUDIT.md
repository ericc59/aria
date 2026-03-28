# Corpus Audit Report

## 1. Verified Program Corpus

- **Total programs**: 66
- **Unique task IDs**: 60
- **Programs with signatures**: 35
- **Unique signatures**: 57
- **Tasks with multiple programs**: 10

### Step-count distribution

| Steps | Count |
|-------|-------|
| 1 | 25 |
| 2 | 12 |
| 3 | 4 |
| 4 | 8 |
| 5 | 3 |
| 6 | 1 |
| 7 | 2 |
| 8 | 1 |
| 9 | 1 |
| 10 | 2 |
| 12 | 1 |
| 13 | 2 |
| 14 | 1 |
| 27 | 1 |
| 40 | 1 |
| 56 | 1 |

### Source distribution

- `corpus-report:corpus_report.json`: 24
- `offline-retrieval`: 19
- `offline-search`: 12
- `solve-report:v1-eval-sweep2.json`: 2
- `solve-report:v1-train-sweep2.json`: 34
- `solve-report:v1-train.json`: 40

### Library-op usage (top 20)

| Op | Count |
|-----|-------|
| `by_color` | 58 |
| `fill_around` | 30 |
| `find_objects` | 24 |
| `apply_color_map` | 23 |
| `stack_h` | 13 |
| `from_object` | 12 |
| `dims_make` | 12 |
| `new_grid` | 10 |
| `fill_diagonal` | 10 |
| `fill_cardinal` | 10 |
| `by_size_rank` | 8 |
| `where` | 8 |
| `rotate_grid` | 7 |
| `reflect_grid` | 7 |
| `overlay` | 7 |
| `crop` | 7 |
| `nth` | 6 |
| `paint_objects` | 6 |
| `translate_delta` | 5 |
| `map_obj` | 5 |

Total unique ops: 65
  Ops used only once: background_obj, complete_symmetry_v, count, get_pos_x, get_pos_y, get_size, get_width, gt, infer_map, length, lib_reflect_grid_b23704e3, lib_reflect_grid_bdc95704, place_at, shift_grid, transpose_grid

## 2. Solve Results

### Train (v1-train)

- Tasks: 400
- Solved: 46 (11.5%)
- Tasks with signatures: 400
- Rounds distribution: {0: 400}
- Solve sources: {'retrieval': 28, 'search': 18}

### Eval (v1-eval)

- Tasks: 400
- Solved: 2 (0.5%)
- Tasks with signatures: 400
- Rounds distribution: {0: 400}
- Solve sources: {'retrieval': 1, 'search': 1}

## 3. Refinement Traces

- Total records: 403
- Solved: 36
- Failed: 367
- Total rounds: 380
- Total candidates tried: 786469
- Avg candidates/record: 1951.5
- Max trace depth: 5
- Records with near-miss (score>=350.0): 330

### Focus label distribution

- `color_map`: 163
- `size`: 126
- `marker_geometry`: 91

## 4. Library

- Entries: 27
- Total use count: 63
- Max level: 1
- Return types: {'GRID': 22, 'OBJECT': 4, 'OBJECT_SET': 1}

## 5. Weaknesses

1. CRITICAL: Only 66 verified programs (target: 800-1000 for Phase 2). Far too few for any model training.
2. Size skew: 37/66 programs (56%) are trivial (<=2 steps). Model will overfit to short programs.

## 6. Recommendations

1. PRIORITY 1: Run batch solve on full ARC-1 training set (400 tasks) with higher search budgets (max_candidates=50000, max_steps=5, max_refinement_rounds=4). Target: 200+ verified programs.
2. NOT READY for SKETCH training. Need at least 200 verified programs with 80+ having >=3 steps. Current: 66 programs total.
3. NEXT_EDIT training is feasible. Proceed with (feedback, current_program) -> refined_program pairs.
4. Train NEXT_FOCUS first — it requires less data and improves refinement quality for collecting NEXT_EDIT pairs. Need: 100 programs with signatures and 3+ focus labels. Current: 35 with signatures, 3 focus labels.
5. Generate synthetic tasks (Phase 2 in DESIGN.md) to multiply training volume. Target: 200K random step sequences length 3-10 for clean, high-volume training data.
6. Grow the library before training. More library entries = more compositional programs in training data. Run library mining on a larger program corpus first.

## 7. Verdict

**Do not train yet.** Collect more data first.

Priority actions before training:
- CRITICAL: Only 66 verified programs (target: 800-1000 for Phase 2). Far too few for any model training.
