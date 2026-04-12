# Claude Prompt: Train Proposal Model on Synthetic + Trace Data

Goal: extend the proposal model training to use synthetic tasks and solved traces to predict program families.

Constraints:
- Must keep training offline.
- No GPU requirement.
- Use existing `proposal_model.py` or extend it minimally.
- If stuck, use the codex CLI for review/help.

## Deliverables

### 1) Training pipeline

Extend `scripts/build_search_model.py`:
- load synthetic corpus JSONL (from build_synth_corpus.py)
- load solved traces JSONL
- build feature vectors from TaskAnalysis
- train/update the proposal model (family classifier)

### 2) Proposal model update

In `aria/search/proposal_model.py`:
- add a method to train on (feature_vector → family label)
- store the feature mapping for inference

### 3) Tests

Add `tests/test_search_proposal_model_train.py`:
- train on a tiny synthetic dataset
- verify predict returns known family for a matching feature vector

### 4) Docs

Update `docs/ARIA_SYSTEM_OVERVIEW.md` with proposal training via synthetic + traces.

## Acceptance

- Training completes on CPU quickly.
- Model can rank families based on synthetic features.
- Tests pass.

