# Claude Prompt: Param Expression Layer (Per‑Object Parameterization)

Goal: add a small, deterministic expression language for per‑object parameters (color, row/col, rank, lookup, etc.) so derive can express “recolor by rank” or “move to lookup row” without code synthesis.

Constraints:
- This is *not* a new AST op. It’s a param expression evaluated at execution time.
- Must be deterministic and demo‑verifiable.
- Keep the expression vocabulary tiny and composable.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) Add `ParamExpr` schema

Add to `aria/search/sketch.py` (or new module if cleaner):

```python
@dataclass(frozen=True)
class ParamExpr:
    op: str  # 'const'|'field'|'rank'|'lookup'|'mod'|'count'
    args: tuple[Any, ...] = ()
```

Use in `SearchStep.params` where a literal int is expected.

### 2) Evaluation in executor

In `aria/search/executor.py`, add a helper:

```python
def eval_param_expr(expr: ParamExpr, obj, facts, context) -> int:
```

Supported ops:
- `const(v)`
- `field(name)` where name is from object (color, area, row, col, height, width)
- `rank(field)` rank among selected objects (largest = 1 by area or specified field)
- `mod(field, k)`
- `count(predicate)` (count of objects matching predicate; predicate from existing rule system)
- `lookup(field, table)` where table is derived from legend mapping (use existing legend map if present; otherwise skip)

Add minimal context plumbing so that when `SearchStep` executes on each object, param exprs can be resolved per object.

### 3) Derive support

In `aria/search/derive.py`, add a small derived strategy:
- `derive:rank_recolor_expr` (or extend existing rank_recolor):
  - Use a `ParamExpr('rank', 'area')` to compute color from rank via a mapping table (derived from demos).
  - Verify across demos.

Optionally add `lookup` case if legend mapping exists.

### 4) Tests

Add tests in `tests/test_search_param_expr.py`:
- `field` returns correct object property
- `rank` over a set of objects
- `mod` over object area
- `count` with a simple predicate
- `lookup` with a provided table

### 5) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` with a short “ParamExpr” section under DSL.

## Acceptance

- ParamExpr is evaluated deterministically in executor.
- Search/derive uses ParamExpr for at least one real strategy (rank recolor).
- No new AST ops.
- All tests pass.

