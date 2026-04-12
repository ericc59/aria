# Claude Prompt: Nearest‑Neighbor Proposer (No Neural)

Goal: implement a simple k‑NN proposer that uses `TaskAnalysis` + solved traces to propose candidate program families for unsolved tasks.

Constraints:
- No neural model required.
- Deterministic, cheap.
- Reuse existing SearchProgram verification.
- If stuck, use the codex CLI for review/help.

## Deliverables

### 1) Feature extraction

Create `aria/search/analysis_features.py`:

```python
def task_feature_vector(analysis: TaskAnalysis) -> np.ndarray:
    # Convert TaskAnalysis to a numeric vector for distance.
```

Include:
- dims_change, same_dims (0/1)
- diff_type one‑hot (recolor/additive/subtractive/rearrange/mixed)
- changed_pixel_fraction (float)
- has_separators, has_panels, is_extraction, is_construction (0/1)

### 2) k‑NN proposer

New module `aria/search/nn_proposer.py`:

```python
def propose_from_traces(
    analysis: TaskAnalysis,
    traces: list[SolveTrace],
    k: int = 3,
) -> list[SearchProgram]:
```

Process:
- compute feature vector for current task
- compute distance to each trace’s task features (store in trace or recompute from task_signatures if available)
- choose k nearest traces
- return their program_dicts as SearchPrograms (re‑derive params where possible if needed)

### 3) Integration

In `aria/search/search.py`:
- after Phase 0e (derive/binding), call `propose_from_traces`
- verify candidate programs in order (same as seed enumeration)

### 4) Tests

Add `tests/test_search_nn_proposer.py`:
- test vector extraction shape
- test that nearest neighbor ranking is stable
- test candidate programs returned in correct order

### 5) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` to mention NN proposer as a fallback route.

## Acceptance

- No external ML dependencies.
- k‑NN proposer produces candidates on solved traces.
- Tests pass.

