# Claude Prompt: Integrate Output-Dims Hypotheses into Search

Goal: make `output_dims.py` actually constrain search when `dims_change` is true, instead of just computing hypotheses.

Constraints:
- Must be deterministic and demo-verified.
- Do not hardcode task IDs or benchmark-specific patterns.
- Avoid adding new primitives.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) Search integration

In `aria/search/search.py`:
- After `analysis = analyze_task(demos)` and `dim_hypotheses = solve_output_dims(...)`,
  route `dims_change` tasks through a small, bounded loop:
  - for each `DimHypothesis`, try derive strategies that can produce that shape
  - if a program verifies, return it

At minimum, gate these paths:
- crop-related derives (output == object bbox, output == non-bg bbox)
- scale/tile/downscale derives (output = input * k or / k)
- panel/legend derives (output == panel size)

If you can’t plumb shape constraints directly, do a pre-filter:
- run a candidate program on demo 0
- reject if output shape != hypothesis shape

### 2) Shared helper

Add helper in `aria/search/search.py` or a small new module:

```python
def _matches_shape(prog, demos, shape) -> bool:
    # execute on demo 0, compare shape
```

### 3) Tests

Add `tests/test_search_output_dims_integration.py` with:
- A dims_change task where output dims are constant → only matching candidates allowed
- A scale-down task where dim hypothesis filters out non-matching derives

### 4) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` to say output-dims hypotheses now gate search.

## Acceptance

- `dims_change` tasks use dim hypotheses to prune candidates.
- Tests pass.
- No regression in `search_programs` for same-dims tasks.

