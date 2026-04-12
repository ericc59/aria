# Aria System Overview

This document describes the current `aria` ARC system end‑to‑end: architecture, perception, symbolic substrate, search, derive strategies, execution, evaluation, and consolidation. It is grounded in the canonical `aria/search` stack, not legacy or experimental paths.

References:
- `docs/AGI.md` (architectural framing)
- `docs/raw/aria_learning_roadmap.md` (learning/consolidation roadmap)
- `DESIGN.md` (system notes and historical context)

## 1) Design Principles

From the roadmap and AGI notes:
- Keep the permanent execution layer low‑level, composable, parameterizable, and domain‑general.
- Push mid‑level patterns into derive/search, traces, and learned macros.
- Separate derive‑time reasoning from runtime execution.
- Every solve should emit trace data suitable for replay and consolidation.

## 2) High‑Level Architecture

```
solve_task(demos)
  └─ search_programs(demos) ──> SearchProgram ──> ASTProgram
       ├─ derive strategies (structured)
       ├─ panel/legend decode
       ├─ binding‑guided decode
       ├─ seed schema enumeration
       └─ 2‑step compositions
```

Primary entrypoints:
- `aria/solve.py` → `solve_task()` (canonical search‑only path)
- `aria/search/search.py` → `search_programs()` (derivation + flat search)
- `aria/eval.py` → evaluation harness for v1/v2 runs

Key property: the system is **verification‑first**. All candidates are executed and verified against demos before returning a program.

## 3) Perception and Scene Substrate

Core perception is in the guided layer and search scene utilities:
- `aria/guided/perceive.py` (object decomposition, background, separators)
- `aria/search/binding.py` (roles, relations, typed entities)
- `aria/search/geometry.py`, `aria/search/frames.py`, `aria/search/windows.py`
- `aria/search/registration.py` (anchor/module correspondence substrate)
- `aria/search/selection_facts.py` (structural object facts for selection rules)

Objects are decomposed into:
- background color, separators
- connected components (objects)
- roles (frame/legend/marker/etc.)
- relational facts (encloses, adjacent, size rank, border contact)

These facts drive derive‑time selection and action parameterization.

## 4) Primitive Execution Layer

Canonical execution surface:
- `aria/search/ast.py` (AST ops)
- `aria/search/executor.py` (runtime execution)
- `aria/search/sketch.py` (SearchProgram/StepSelect to AST lowering)

Principles:
- AST ops are low‑level, composable primitives.
- Search programs are sequences of steps with optional selectors.
- Execution is deterministic and uses demo‑derived parameters.

## 5) Search Programs and Derive Strategies

Search programs are structured sequences:
- `SearchProgram` = steps + provenance
- `SearchStep` = action + params + selector (`StepSelect`)

Derive strategies (in `aria/search/derive.py`) attempt structured inference, e.g.:
- recolor variants (uniform, rank‑based, map‑based)
- grid/slot transfer and grid‑conditional rules
- registration transfer (module → frame openings)
- panel/legend routing and mapping
- symmetry repairs, template broadcast, cropping

Derive strategies do **not** bypass verification. They produce candidate programs that must satisfy all demos.

## 6) Selection and Rules

Selection is explicit and fact‑based:
- `aria/search/selection_facts.py` computes structural booleans per object.
- `StepSelect(role='by_rule', params={'rule': DNF})` selects objects using derived rules.
- `aria/search/rules.py` provides bounded DNF induction used at derive time.

This enables cross‑demo selectors that generalize beyond single predicate rules.

## 7) Correspondence and Transitions

Transition modeling attempts to explain object movements and changes:
- `aria/guided/correspond.py` matches input↔output objects.
- Matching tiers include exact, near‑shape, size‑tolerant, recolor‑only.

This powers derive strategies that depend on motion detection (move, swap, registration).

## 8) Panels, Legends, and Multi‑Region Tasks

Panel and region decoding in `aria/search/panels.py` and `aria/search/decode.py`:
- Separator‑based panel extraction
- Motif comparison across panels (common/unique)
- Legend→target color mapping (panel routing)

These are reasoning layers above perception, not new primitives.

## 9) Candidate Ranking and Priors

Candidate ranking combines:
- verifier features (demo pass count, pixel diffs)
- proposal prior (`aria/search/proposal_memory.py`)
- family model (`aria/search/proposal_model.py`)
- optional macro prior (`aria/search/macros.py`)

Ranking only changes order, not correctness. Exact verification remains the gate.

## 10) Traces and Macro Mining

Trace capture:
- `aria/search/trace_schema.py`
- `aria/search/trace_capture.py`

Macros:
- `aria/search/macros.py` (schema and library)
- `aria/search/macro_miner.py` (mines repeated patterns)
- `scripts/build_search_traces.py`, `scripts/build_macro_library.py`

Current direction: **macros should be parameterized templates**, not frozen instances. They are learned compositions, not new runtime ops.

## 11) Evaluation

Evaluation path:
- `aria/eval.py` → `solve_task()` per task
- `scripts/eval_arc2.py` for batch runs

Evaluation records:
- solves
- failure clusters (`selection`, `dims_change`, etc.)
- trace capture for solved tasks

## 12) Consolidation Loop (Current + Roadmap)

Current consolidation tools:
- build prior: `scripts/build_search_prior.py`
- build corpus: `scripts/build_search_corpus.py`
- build model: `scripts/build_search_model.py`
- build traces: `scripts/build_search_traces.py`
- build macros: `scripts/build_macro_library.py`

Roadmap target:
- unify into a single orchestration pass (Phase 5 in roadmap)
- move to fact‑conditioned routing and learned macro use

## 13) What’s Canonical vs Legacy

Canonical: `aria/search` stack, `solve_task()`, and `eval.py`.

Legacy/experimental:
- `aria/legacy/*`, `aria/core/*`, `aria/ngs/*` (kept for ablation/history)
- these are **not** in the default execution path

## 14) Known Gaps (Operational)

From current eval failure clusters:
- Per‑object **action parameterization** (destinations, anchors, per‑object colors)
- **dims_change** derivation (layout synthesis, grid‑based output construction)
- Deeper **correspondence/registration** for non‑trivial target sites
- Multi‑step structured reasoning beyond flat search (planning)

These are the highest‑leverage remaining architecture gaps for ARC v1/v2.

## 15) How This Maps to AGI/Architecture Goals

`aria` is a small domain‑world‑model subsystem:
- explicit entities, roles, relations
- symbolic operators
- exact verification
- trace capture for skill consolidation

Its long‑term value is the **pattern of architecture**, not ARC score alone.

