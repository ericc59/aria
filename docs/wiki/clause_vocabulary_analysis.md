# Clause Vocabulary Analysis

Gap analysis and refactoring recommendations for `aria/guided/clause.py`.

---

## Current Inventory

### Predicates (16)

| Predicate | What it does | Redundancy |
|---|---|---|
| IS_SMALLEST | size == min across all objects | Special case of RANK_BY(size, 0) |
| IS_LARGEST | size == max across all objects | Special case of RANK_BY(size, -1) |
| SIZE_EQ(n) | size == n | Special case of COMPARE(size, ==, n) |
| SIZE_LT(n) | size < n | Special case of COMPARE(size, <, n) |
| SIZE_GT(n) | size > n | Special case of COMPARE(size, >, n) |
| IS_SINGLETON | size == 1 | SIZE_EQ(1) |
| COLOR_EQ(c) | color == c | Could be COMPARE(color, ==, c) but common enough to keep |
| UNIQUE_COLOR | only object with this color | Keep — hard to express otherwise |
| IS_RECTANGULAR | mask fills bbox | Special case of SHAPE_IS(rect) |
| IS_LINE | h==1 or w==1 | Special case of SHAPE_IS(line) |
| IS_SQUARE | h==w and rectangular | Special case of SHAPE_IS(square) |
| TOUCHES_BORDER | any edge touches grid edge | Special case of POSITION(any_edge) |
| NOT_TOUCHES_BORDER | no edge touches grid edge | NOT(TOUCHES_BORDER) |

**Problems:**
- 6 predicates are special cases of 2 parameterized predicates (COMPARE, RANK_BY)
- 3 shape predicates are special cases of 1 parameterized predicate (SHAPE_IS)
- 2 position predicates are a predicate and its negation
- No relational predicates despite perceive.py computing all relationships
- No conjunction support despite `_generate_target_predicates` needing it
- No negation combinator

### Aggregations (8)

| Aggregation | What it does | Redundancy |
|---|---|---|
| COLOR_OF | support[0].color | ATTR(color) |
| SIZE_OF | support[0].size | ATTR(size) |
| ROW_OF | support[0].row | ATTR(row) |
| COL_OF | support[0].col | ATTR(col) |
| SHAPE_OF | support[0].mask | ATTR(mask) |
| COUNT | len(support) | Keep |
| MOST_COMMON_COLOR | most frequent non-bg color in a region | REGION_STAT(mode_color) |
| FRAME_COLOR | most common adjacent non-bg color around enclosed bg | REGION_STAT(frame_color) |

**Problems:**
- 5 aggregations are the same operation (attribute extraction) with different field names
- Only operates on support[0] — no multi-object aggregation (majority, minority, unique)
- No positional aggregations across multiple support objects (centroid, bbox, spread)
- MOST_COMMON_COLOR and FRAME_COLOR are region-level, not object-level — different semantics shoehorned into the same interface

### Actions (10)

| Action | What it does | Redundancy |
|---|---|---|
| RECOLOR | set target pixels to attr | Identical to PLACE |
| PLACE | set target pixels to attr | Identical to RECOLOR |
| PLACE_AT | place target mask at offset | Same as STAMP but erases original |
| REMOVE | don't include target in output | Keep |
| KEEP | copy target as-is | Keep |
| PLACE_PIXEL | place single pixel at position | PLACE with singleton target |
| FILL_ENCLOSED | fill enclosed bg with derived color | Region-level RECOLOR |
| STAMP | place target mask at offset, preserve original | PLACE_AT with preserve=True |
| GRAVITY | move target to grid border | MOVE(dir, stop=border) |
| SLIDE | move target until hitting non-bg | MOVE(dir, stop=collision) |

**Problems:**
- RECOLOR and PLACE have identical executor code paths
- GRAVITY and SLIDE are the same action with different stop conditions
- PLACE_AT and STAMP differ only in whether the original is preserved
- PLACE_PIXEL is PLACE for singletons with a position override
- FILL_ENCLOSED is a region-level action mixed into the object-level action vocabulary
- No geometric transforms (rotate, flip, scale, tile)
- No conditional action dispatch

---

## Proposed Vocabulary

### Predicates

8 parameterized predicate types replacing 16 special-cased ones, plus relational and combinators:

| Predicate | Parameters | Replaces | New? |
|---|---|---|---|
| `COMPARE(attr, op, value)` | attr: size/height/width/aspect/n_same_color; op: ==/</>/<=/>=; value: int/float | SIZE_EQ, SIZE_LT, SIZE_GT, IS_SINGLETON | Generalized |
| `RANK(attr, index)` | attr: size/row/col; index: 0=smallest, -1=largest, 1=2nd smallest, etc. | IS_SMALLEST, IS_LARGEST | Generalized |
| `COLOR_EQ(c)` | c: int | — | Keep |
| `UNIQUE_COLOR` | — | — | Keep |
| `SHAPE_IS(type)` | type: rect/line/square/any | IS_RECTANGULAR, IS_LINE, IS_SQUARE | Generalized |
| `POSITION(spec)` | spec: touches_top/touches_bottom/touches_left/touches_right/touches_any/interior | TOUCHES_BORDER, NOT_TOUCHES_BORDER | Generalized |
| `RELATION(rel, other_pred)` | rel: adjacent_to/contained_by/contains/aligned_h/aligned_v/same_shape_as; other_pred: any Predicate | — | **New** |
| `NOT(pred)` | pred: any Predicate | NOT_TOUCHES_BORDER | **New** |

**Why RELATION matters:** It's the single biggest unlock. Examples of what becomes expressible:
- "objects adjacent to a singleton" → `RELATION(adjacent_to, IS_SINGLETON)`
- "objects contained by the largest" → `RELATION(contained_by, RANK(size, -1))`
- "objects with same shape as the unique-color one" → `RELATION(same_shape_as, UNIQUE_COLOR)`

The data for all of these already exists in `perceive.py`'s `PairFact`. The predicate language just can't express it.

**Why NOT matters:** Currently NOT_TOUCHES_BORDER is a separate predicate. With NOT as a combinator: `NOT(COLOR_EQ(3))`, `NOT(RELATION(adjacent_to, IS_SINGLETON))`, `NOT(IS_LARGEST)`. Doubles the expressiveness for free.

### Aggregations

5 aggregation types replacing 8, plus new multi-object and positional aggregations:

| Aggregation | Parameters | Replaces | New? |
|---|---|---|---|
| `ATTR(field)` | field: color/size/row/col/height/width/mask/center_row/center_col | COLOR_OF, SIZE_OF, ROW_OF, COL_OF, SHAPE_OF | Generalized |
| `COUNT` | — | — | Keep |
| `STAT(field, reducer)` | field: color/size; reducer: mode/min/max/unique/minority | MOST_COMMON_COLOR | Generalized |
| `REGION(method)` | method: frame_color/fill_color/region_mode | FRAME_COLOR | Generalized |
| `COLLECT(field)` | field: color/size/row/col → returns set/list | — | **New** |

**Why STAT matters:** Enables "recolor to the majority color of adjacent objects", "recolor to the unique color not shared by any neighbor", etc. These are common ARC patterns.

**Why COLLECT matters:** Returns a set instead of a scalar. Needed for set operations: "colors in support but not in target", "intersection of colors across regions". Many legend/decode tasks need this.

### Actions

7 action types replacing 10, plus geometric transforms and conditional dispatch:

| Action | Parameters | Replaces | New? |
|---|---|---|---|
| `RECOLOR(source)` | source: derived attr or literal | RECOLOR, PLACE, FILL_ENCLOSED, PLACE_PIXEL | Unified |
| `MOVE(direction, stop)` | direction: up/down/left/right; stop: border/collision/N_cells | GRAVITY, SLIDE | Unified |
| `PLACE_AT(offset, preserve)` | offset: (dr,dc); preserve: bool | PLACE_AT, STAMP | Unified |
| `REMOVE` | — | — | Keep |
| `KEEP` | — | — | Keep |
| `TRANSFORM(op)` | op: rotate_cw/rotate_ccw/rotate_180/flip_h/flip_v | — | **New** |
| `SCALE(factor)` | factor: 2/3/0.5 or (fh, fw) | — | **New** |

**Why TRANSFORM matters:** Pure geometric transforms of selected objects are common in ARC. "Rotate all small objects 90 degrees", "flip the non-border objects horizontally". Currently completely unreachable.

### Conditional Dispatch (New Concept)

Not a new action — a new **clause structure**. Currently:

```
SELECT targets WHERE preds
FOR EACH target:
    APPLY action
```

Proposed:

```
SELECT targets WHERE preds
FOR EACH target:
    IF condition(target, support) THEN action_A
    ELSE action_B
```

This is the entity-conditional composition gap. Implementation options:

**Option A: ConditionalClause**
```python
@dataclass
class ConditionalClause:
    target_preds: list[Predicate]
    condition: Predicate           # tested per-target
    then_clause: Clause            # action if condition true
    else_clause: Clause            # action if condition false
```

**Option B: Clause with per-object action dispatch**
```python
@dataclass
class Clause:
    target_preds: list[Predicate]
    action_dispatch: list[tuple[Predicate, Act, Agg]]  # (condition, action, agg)
    default_action: Act
```

**Option C: Multiple clauses with disjoint predicates**
Already supported by ClauseProgram — just needs the inducer to emit them. If clause 1 selects `COLOR_EQ(red)` with `MOVE(left)` and clause 2 selects `COLOR_EQ(blue)` with `MOVE(right)`, the composition IS conditional dispatch. The gap is the inducer, not the language.

**Recommendation: Option C first.** It requires zero new language features — just make the inducer smarter about emitting complementary clause pairs. Option A or B only if Option C proves insufficient.

---

## Inducer Gaps

The clause language defines what's expressible. The inducer defines what's reachable. Several inducer gaps limit the effective vocabulary even where the language has capacity:

### 1. Single-predicate target selection

`_generate_target_predicates` emits ~7 single-predicate sets per object. Never emits conjunctions.

**Fix:** Generate 2-predicate conjunctions from the Cartesian product. `[IS_SINGLETON, COLOR_EQ(3)]` selects red singletons — much more selective than either alone. Cap at ~20 combinations to control search.

### 2. No relational predicate generation

Even after adding RELATION to the Pred enum, the inducer needs to generate relational predicates as candidates.

**Fix:** For each changed object, check which relationships it has (from PairFact). If all changed objects across all demos share a relationship (e.g., all are adjacent to a singleton), emit that as a candidate predicate.

### 3. Two-clause composition is limited

`_compose_two_clauses` tries 15 partials × 15 partials (Strategy A) and 3 first-clauses × 100 residual candidates (Strategy B). No 3+ clause composition.

**Fix (incremental):** After Strategy B, try 3-clause by re-running residual composition on the best 2-clause partial. Recursive but bounded.

**Fix (structural):** The inducer should decompose the residual into independent groups (e.g., "red objects changed, blue objects changed") and induce one clause per group. This is how humans solve multi-rule tasks — they see that different things happen to different groups.

### 4. Greedy correspondence

`correspond.py` does greedy largest-first matching. If the largest output object matches the wrong input object, every downstream clause is wrong.

**Fix:** Generate top-K correspondence hypotheses (e.g., top 3 by score) and induce clauses under each. Verify across demos to select the correct correspondence.

### 5. No offset generalization

`_candidates_for_move` emits a literal offset `(dr, dc)`. This only works if the offset is constant across demos. If the offset depends on input structure (e.g., "move to the centroid of the enclosed region"), it's unreachable.

**Fix:** After detecting a move, check whether the destination position can be described structurally: "at the row of the singleton", "inside the enclosed region", "at the centroid of the same-color group". Emit structural move candidates alongside literal offsets.

---

## Priority Ordering

Revised 2026-04-05 after review. Priorities ordered by **new solves unlocked**, not code cleanliness. Output size inference is important but is a separate upstream module, not clause-vocabulary work — tracked elsewhere.

### Tier 1: Solve-count movers

| Priority | Change | Type | Why it's top tier |
|---|---|---|---|
| 1 | **RELATION(rel, other_pred)** predicate | Predicate | Biggest single unlock. Data exists in PairFact, language can't express it. Enables "adjacent to singleton", "contained by largest", "aligned with same-color". |
| 2 | **Generic creation vocabulary** for match_type=="new" | Action + Inducer | Biggest actual capability gap per match-type survey. Current STAMP handles copy-at-offset, but many tasks create objects from scratch — construct from template, fill region, draw line/shape, composite from parts. Need CREATE(method, params) actions and inducer logic to detect creation patterns. |
| 3 | **Multi-predicate conjunctions** in inducer | Inducer | Currently single-pred only. `[IS_SINGLETON, COLOR_EQ(3)]` is much more selective. Cheap to add, reduces false matches, directly improves solve rate. |
| 4 | **Structural offset/placement** generalization | Inducer | Literal `(dr, dc)` only works for constant offsets. Need "move to row of singleton", "place inside enclosed region", "align with centroid of group". Many move/place tasks are unreachable without this. |
| 5 | **Top-K correspondence** | Inducer | Greedy matching fails silently. Wrong correspondence → wrong clauses → no solve. Top-3 hypotheses with cross-demo verification. |

### Tier 2: Expressiveness multipliers

| Priority | Change | Type | Why it matters |
|---|---|---|---|
| 6 | **Conditional dispatch** via complementary clause pairs | Inducer | Entity-conditional rules ("red→left, blue→right") via the inducer emitting disjoint-predicate clause pairs. No new language needed — Option C. |
| 7 | **NOT(pred)** combinator | Predicate | Doubles predicate expressiveness. `NOT(COLOR_EQ(3))`, `NOT(RELATION(adjacent_to, ...))`. Trivial to implement. |
| 8 | **STAT(field, reducer)** aggregation | Aggregation | Majority/minority/unique color from support set. Common ARC patterns currently unreachable. |
| 9 | **COLLECT(field)** aggregation | Aggregation | Returns sets for set operations. Needed for legend/decode tasks: "colors in support but not in target". |
| 10 | **TRANSFORM(op)** action | Action | Rotate, flip selected objects. Common in ARC, completely missing. |

### Tier 3: Cleanup and extension (do when touching nearby code)

| Priority | Change | Type | Notes |
|---|---|---|---|
| 11 | COMPARE(attr, op, val) / RANK(attr, index) | Predicate | Good generalization but won't move solves alone. Do when refactoring predicates for RELATION. |
| 12 | ATTR(field) unification | Aggregation | Mechanical. Do when adding STAT/COLLECT. |
| 13 | MOVE(dir, stop) unification | Action | Merge GRAVITY + SLIDE. Do when adding TRANSFORM. |
| 14 | PLACE_AT(offset, preserve) unification | Action | Merge STAMP + PLACE_AT. Do when working on creation vocabulary. |
| 15 | SCALE(factor) action | Action | Upscale/downscale. Lower priority than rotate/flip. |
| 16 | 3+ clause composition | Inducer | High cost, defer until 2-clause composition is mature. |

### Notes on what changed from v1

- **Output size inference removed** from this list. It's critical but it's upstream preprocessing, not clause vocabulary. Tracked separately.
- **Generic creation vocabulary elevated to #2.** Match-type survey shows new-object creation is the biggest actual capability gap. The analysis v1 underweighted this.
- **Cleanup refactors (COMPARE/RANK/ATTR/MOVE unification) demoted to tier 3.** They're good engineering but won't move solve counts. Do them opportunistically.
- **TRANSFORM elevated above SCALE.** Rotation and flip are far more common in ARC than scaling.
