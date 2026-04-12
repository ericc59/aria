# Claude Prompt: Task Analysis + Output Dimensions Pre‑Solver

Goal: add a lightweight, analysis‑gated pre‑solver that predicts output size hypotheses before search. This is the highest‑ROI structural gate for `dims_change` tasks and the decomposition planner.

Constraints:
- Keep analysis minimal and cheap. It must run once per task and be reusable by other tracks.
- Do NOT hardcode task IDs or benchmark‑specific patterns.
- Keep everything deterministic and verifiable across demos.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) New module: `aria/search/task_analysis.py`

Implement:

```python
@dataclass(frozen=True)
class TaskAnalysis:
    dims_change: bool
    same_dims: bool
    diff_type: str  # 'recolor_only'|'additive'|'subtractive'|'rearrange'|'mixed'
    changed_pixel_fraction: float
    new_colors: set[int]
    removed_colors: set[int]
    has_separators: bool
    has_panels: bool
    is_extraction: bool  # output is subgrid of input in all demos
    is_construction: bool  # output has no overlap with input (heuristic)
```

Add:

```python
def analyze_task(demos: list[tuple[np.ndarray, np.ndarray]]) -> TaskAnalysis
```

Rules (keep simple):
- `dims_change`: any demo output shape != input shape
- `diff_type`:
  - `recolor_only`: same dims and output mask equals input mask across demos
  - `additive`: same dims and output has extra non‑bg where input had bg
  - `subtractive`: same dims and output missing non‑bg that input had
  - `rearrange`: same dims, changed positions, same color multiset
  - `mixed`: fallback
- `changed_pixel_fraction`: average fraction changed across demos
- `new_colors`, `removed_colors`: per demo sets unioned
- `has_separators`: use guided perceive (sep color or panel detection)
- `has_panels`: true if panel detection yields >=2
- `is_extraction`: output equals some subgrid of input across demos (exact match)
- `is_construction`: output shares <10% pixels/colors with input (cheap heuristic)

### 2) New module: `aria/search/output_dims.py`

Implement:

```python
@dataclass(frozen=True)
class DimHypothesis:
    rule: str
    shape: tuple[int, int] | None
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)
```

```python
def solve_output_dims(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
) -> list[DimHypothesis]
```

Hypotheses to include (cheap):
- constant output shape (if all demos equal)
- input scaled by factor k (k in 2..5 if all divisible and match)
- input divided by k (k in 2..5 if all divisible and match)
- output equals bbox of a detected object (any object whose bbox dims match)
- output equals panel size (if separators exist)

Sort by confidence desc. Only return hypotheses consistent across demos.

### 3) Integration in `aria/search/search.py`

At the start of `search_programs`, compute `analysis = analyze_task(demos)`.

If `analysis.dims_change` is true:
- compute `dim_hypotheses = solve_output_dims(demos, analysis)`
- attach to task_signatures or pass along to candidate ranker (minimal integration is fine for now)
- do NOT force a new pipeline yet (just compute and expose; wire into next prompt)

### 4) Tests

Add tests:
- `tests/test_search_task_analysis.py`
  - recolor_only, additive, subtractive, rearrange, mixed
  - dims_change detection
  - extraction detection (output is subgrid)
- `tests/test_search_output_dims.py`
  - constant output
  - scale up/down by k
  - object bbox match
  - panel size match (mock)

### 5) Update docs

Add a short section to `docs/ARIA_SYSTEM_OVERVIEW.md` under the search pipeline describing `TaskAnalysis` and output‑dims pre‑solver.

## Acceptance

- All tests pass.
- No new hardcoded task IDs.
- `search_programs` still works as before.
- New modules are deterministic and cheap.

