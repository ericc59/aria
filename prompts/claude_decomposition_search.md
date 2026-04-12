# Claude Prompt: Decomposition Search (Top‑Down Multi‑Step)

Goal: replace the current flat 2‑step composition with a structured, analysis‑gated decomposition planner that can handle 3–5 step tasks without combinatorial explosion.

Constraints:
- Must be analysis‑gated (use TaskAnalysis from `task_analysis.py`).
- Avoid brute‑force N‑step enumeration.
- Keep splitters small and deterministic.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) New module `aria/search/decompose.py`

Implement:

```python
@dataclass(frozen=True)
class Splitter:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    program: SearchProgram  # a 1‑step program representing the splitter
    compatible: Callable[[TaskAnalysis], bool]
```

```python
def search_decomposed(
    demos: list[tuple[np.ndarray, np.ndarray]],
    analysis: TaskAnalysis,
    *,
    max_depth: int = 3,
) -> SearchProgram | None
```

Workflow:
- Use `analysis` to select compatible splitters.
- For each splitter:
  - Apply to all demo inputs → intermediate demos.
  - Run existing derive strategies on the sub‑problem.
  - If sub‑program verifies, compose `splitter.program + sub_program`.
- Allow 1 level of recursion (max_depth 3 total).

### 2) Splitter set (initial)

Keep it small (10–15 max). Examples:
- `crop_non_bg_bbox` (if `analysis.is_extraction`)
- `extract_panel_0` (if `analysis.has_panels`)
- `remove_color_c` (for colors in analysis.new_colors/removed_colors)
- `apply_color_map` (if diff_type == recolor_only)
- `mask_to_template` (if output is a template‑broadcast style)

Each splitter must have a deterministic `SearchProgram` representation.

### 3) Integration in `aria/search/search.py`

After Phase 0a‑0e derives fail, call `search_decomposed(...)` before seed enumeration.

### 4) Tests

Add `tests/test_search_decompose.py` with at least:
- A synthetic 2‑step task where splitter + derive solves it
- A task where analysis gating rejects splitters
- Composition correctness check

### 5) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` (search pipeline section) to mention decomposition search.

## Acceptance

- Decomposition search is analysis‑gated.
- 3–5 step tasks can be expressed without brute‑force enumeration.
- All tests pass.

