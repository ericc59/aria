# Claude Prompt: Refine Decomposition Splitters

Goal: refine and prune decomposition splitters to keep search tractable while increasing coverage. Focus on correctness and gating, not quantity.

Constraints:
- No benchmark-specific logic.
- Splitters must be deterministic and have a faithful SearchProgram representation.
- Gating must be based on TaskAnalysis fields.
- If stuck, use the codex CLI for code review / help.

## Deliverables

### 1) Audit and prune splitters

In `aria/search/decompose.py`:
- Remove any splitter whose program does not match apply() exactly.
- Remove any splitter that is always identity for many tasks (high false positives).

### 2) Add at most two new splitters

Candidates (pick only those that are safe + clearly gated):
- `extract_non_bg_color` (if `removed_colors` or `new_colors` suggests it)
- `crop_object_unique_color` (if analysis indicates a unique color appears)

Each must be gated by analysis and verified by demo execution.

### 3) Tests

Update `tests/test_search_decompose.py`:
- add one test for each new splitter
- add a test that a removed splitter is no longer used

### 4) Docs

Add one paragraph in `docs/ARIA_SYSTEM_OVERVIEW.md` about decomposition splitters being analysis-gated.

## Acceptance

- Decomposition search remains tractable.
- New splitters are correct and gated.
- Tests pass.

