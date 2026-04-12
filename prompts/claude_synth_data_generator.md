# Claude Prompt: Synthetic Task Generator for Proposal Training

Goal: generate synthetic ARC‑like tasks by sampling DSL programs, to train proposal routing.

Constraints:
- Deterministic seeding.
- Use only existing DSL ops.
- Output format should match existing training pipeline.
- If stuck, use the codex CLI for review/help.

## Deliverables

### 1) Generator

Create `aria/search/synth_data.py`:

```python
def generate_synthetic_tasks(
    n: int,
    *,
    seed: int = 0,
    max_steps: int = 3,
) -> list[tuple[np.ndarray, np.ndarray, SearchProgram]]:
```

Steps:
- sample a random SearchProgram from a small subset of safe ops
- sample a random input grid (small sizes 3–10)
- execute program to produce output
- return (input, output, program)

### 2) Export script

Add `scripts/build_synth_corpus.py`:
- generate N tasks
- write JSONL with (input, output, program_dict, analysis features)

### 3) Tests

Add `tests/test_search_synth_data.py`:
- generator returns valid grids
- output matches program execution
- serialization sanity

### 4) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` with a note about synthetic corpus generation.

## Acceptance

- Can generate at least 100 synthetic tasks deterministically.
- Tests pass.

