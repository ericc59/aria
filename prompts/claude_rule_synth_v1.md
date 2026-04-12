# Claude Prompt: Rule Synthesis v1 (ParamExpr Extensions)

Goal: extend per‑object parameterization to cover common rule patterns (lookup, rank by non‑size fields, neighbor count).

Constraints:
- Must remain deterministic and demo‑verifiable.
- No new AST ops.
- Keep expression vocabulary small.
- If stuck, use the codex CLI for review/help.

## Deliverables

### 1) Extend ParamExpr

Add two new ops:
- `neighbor_count(color|any)` → count of 4‑connected neighbors matching a color or non‑bg
- `rank(field, order)` → rank by a field with order (`asc`/`desc`)

Update `eval_param_expr` in `aria/search/executor.py`.

### 2) Derive strategies using ParamExpr

Add to `aria/search/derive.py`:
- `derive:recolor_by_neighbor_count`  
  Detects recolor patterns where output color depends on neighbor count or adjacency.
- `derive:rank_recolor_expr` enhancement  
  Allow rank by fields other than size (e.g., row, col, width, height).

### 3) Tests

Add `tests/test_search_rule_synth.py`:
- neighbor_count param expr evaluation
- recolor_by_neighbor_count on a synthetic demo
- rank recolor by row/col

### 4) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` to mention new ParamExpr ops and rule synthesis.

## Acceptance

- New ParamExpr ops work end‑to‑end.
- At least one new derive uses them and verifies on synthetic data.
- Tests pass.

