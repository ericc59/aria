# Claude Prompt: Multi‑Step Planner v1 (Goal‑Directed Decomposition)

Goal: replace shallow decomposition with a goal‑directed planner that can chain 3–5 steps without brute force.

Constraints:
- Must be analysis‑gated (TaskAnalysis).
- Must reason about intermediate goals derived from diff analysis.
- Must not enumerate all step combinations.
- If stuck, use the codex CLI for review/help.

## Deliverables

### 1) New module `aria/search/planner.py`

Implement a small planner:

```python
@dataclass(frozen=True)
class GoalState:
    target_shape: tuple[int, int] | None
    diff_type: str
    required_colors: frozenset[int]
    removed_colors: frozenset[int]
```

```python
def plan_search(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
    *,
    max_depth: int = 4,
) -> SearchProgram | None
```

Planner behavior:
- Derive an initial `GoalState` from TaskAnalysis + demo output.
- Use a small, gated set of “goal reducers” (splitters + transforms) that reduce diff complexity.
- After each reducer, re‑analyze the sub‑problem and update GoalState.
- Stop when a derive strategy verifies or depth is exhausted.

### 2) Goal reducers (initial set)

Keep it small (5–8 reducers), deterministic and analysis‑gated:
- `crop_nonbg` (dims_change/extraction)
- `extract_panel` (has_panels)
- `remove_color_c` (removed_colors)
- `grid_transform` (rearrange)
- `recolor_map` (recolor_only)

Each reducer must have:
- `apply(inp) -> np.ndarray`
- `program` as SearchProgram
- `compatible(analysis)` gate
- `improves(sub_analysis)` heuristic (e.g., lower changed_pixel_fraction or closer to target_shape)

### 3) Integration

In `aria/search/search.py`, insert `plan_search(...)` after Phase 0e, before seed enumeration.

### 4) Tests

Add `tests/test_search_planner.py`:
- synthetic 3‑step task solved by planner
- gating test (planner rejects reducer when incompatible)

### 5) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` to mention the planner.

## Acceptance

- Planner can solve at least one 3‑step synthetic task without brute force.
- All tests pass.

