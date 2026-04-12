# Claude Prompt: ParamExpr-Based Derive Strategies

Goal: make ParamExpr useful by adding at least two derive strategies that emit ParamExpr instead of constants.

Constraints:
- Use existing primitives (no new AST ops).
- Strategies must be demo-verified.
- Prefer small, interpretable expressions.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) Rank-based recolor with ParamExpr

In `aria/search/derive.py`, extend or add a strategy that:
- infers a mapping from object size rank → output color
- emits `SearchStep('recolor', params={'color': ParamExpr('rank', ('size',))}, select=...)`
- verifies across demos

If a direct rank mapping isn’t consistent, skip.

### 2) Field-based move with ParamExpr

Add a small derive that detects per-object uniform moves based on object fields:
- e.g., row/col offsets proportional to object’s own row/col or size
- emit `ParamExpr('field', ('row',))` or `ParamExpr('mod', ('size', k))` if it fits

Keep it conservative and verify strictly.

### 3) Tests

Add `tests/test_search_param_expr_derives.py`:
- synthetic task for rank recolor via ParamExpr
- synthetic task for param-based move

### 4) Docs

Add a short note in `docs/ARIA_SYSTEM_OVERVIEW.md` describing ParamExpr derive usage.

## Acceptance

- ParamExpr is used in at least one real derive path.
- Tests pass.
- No regressions in existing derives.

