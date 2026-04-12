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

Perception outputs typically include:
- object masks, bounding boxes, area, color
- separator color and panel partitions when present
- frame/window candidates (for registration/legend tasks)
- grid hypotheses (explicit separators or implicit lattices)
- structural relations (touching, enclosing, adjacency, size rank)

## 4) DSL and Execution Model

This section is the DSL “contract” for `aria/search`.

### 4.1 SearchProgram DSL (symbolic programs)

Defined in `aria/search/sketch.py`:
- `SearchProgram` = list of `SearchStep` + provenance + helpers.
- `SearchStep` = `action` + `params` + optional `select`.
- `StepSelect` encodes a selector (`role`, `params`) used to pick objects.
- `SearchProgram.to_dict()` / `from_dict()` serialize the DSL used in traces/macros.
- `SearchProgram.to_ast()` lowers to an AST program when the action is expressible.
- `SearchProgram.execute()` runs the step sequence directly (used for non‑AST steps).

Example serialized program (shape only):

```json
{
  "provenance": "derive:rank_recolor",
  "steps": [
    {"action": "recolor", "params": {"color": 1}, "select": {"role": "largest"}},
    {"action": "recolor", "params": {"color": 2}, "select": {"role": "by_rule", "params": {"rule": {"kind": "dnf", "clauses": [{"atoms": [{"field": "is_largest", "value": false}]}]}}}}
  ]
}
```

Lowering rules:
- Most steps lower to a single AST op with encoded params.
- Some steps remain “search‑level” and execute via `SearchProgram.execute()` (e.g., registration transfer).
- `StepSelect` is lowered to selector predicates for object‑level AST ops.

Constraints:
- All programs must be demo‑verifiable.
- Search‑level steps must be deterministic and pure (no hidden state).

### 4.2 Selector DSL

Selectors are explicit and composable:
- `role='by_color'` with `params={'color': C}`
- `role='by_predicate'` with a list of guided predicates
- `role='by_rule'` with a DNF rule over structural object facts
- Roles like `largest`, `smallest`, `topmost`, `leftmost`, `boundary`, etc.

Predicate selectors:
- Predicates live in the guided layer (`aria/guided/clause.py`).
- They include shape/position/adjacency constraints and are evaluated per object.

Rule‑based selectors:
- DNF rules over fields from `aria/search/selection_facts.py`
- Atoms are simple boolean features (e.g., `is_largest`, `touches_top`)
- Rules are induced across demos in `aria/search/derive.py`

### 4.3 AST DSL (runtime primitives)

Defined in `aria/search/ast.py` and executed in `aria/search/executor.py`.

The AST is the *canonical runtime surface* and should stay low‑level. It includes:
- grid transforms: rotate/flip/transpose/scale/tile
- paint/recolor: recolor, recolor_map, fill, fill_enclosed, fill_interior
- object edits: crop, crop_object, remove, stamp, slide/move
- pattern ops: periodic_extend, template_broadcast, grid_fill_between
- composition helpers: boolean panel combine, stencil, symmetry repair (when canonical)

Parameter types (typical):
- colors: single colors or color maps (derived or enumerated)
- spatial: row/col offsets, bounding boxes, scale factors
- counts: grid rows/cols, repetition counts
- masks/patches: stencils or templates (from demos)

For the exhaustive list, see:
- `aria/search/ast.py` (op enum and node structure)
- `aria/search/executor.py` (implementation for each op)

#### 4.3.1 Op Catalog (grouped)

Leaves and constants:
- `INPUT`, `CONST_COLOR`, `CONST_INT`

Perception and selection:
- `PERCEIVE`, `SELECT`, `SELECT_IDX`

Extraction and region handling:
- `CROP_BBOX`, `CROP_INTERIOR`, `SPLIT`, `FIRST`, `SECOND`

Region ops:
- `COMBINE` (and/or/xor/diff/rdiff), `RENDER`

Geometric transforms:
- `FLIP_H`, `FLIP_V`, `FLIP_HV`, `ROT90`, `ROT180`, `TRANSPOSE`

Trace/geometry:
- `TRACE` (ray/trace spec)

Grid constructors and patterning:
- `TILE`, `PERIODIC_EXTEND`, `REPAIR_FRAMES`, `TEMPLATE_BROADCAST`

Panel operations:
- `PANEL_ODD_SELECT`, `PANEL_MAJORITY_SELECT`, `PANEL_REPAIR`, `PANEL_BOOLEAN`

Repair:
- `SYMMETRY_REPAIR`

Structured output / packing:
- `OBJECT_REPACK`, `FRAME_BBOX_PACK`

Region decode / domain helpers:
- `QUADRANT_TEMPLATE_DECODE`, `CROSS_STENCIL_RECOLOR`, `LEGEND_FRAME_FILL`

Object‑level actions:
- `RECOLOR`, `REMOVE`, `MOVE`, `GRAVITY`, `SLIDE`, `STAMP`
- `TRANSFORM_OBJ`, `FILL_INTERIOR`, `FILL_ENCLOSED`

Composition:
- `COMPOSE`, `FOR_EACH`, `IF_ELSE`

Mirror:
- `MIRROR`

Hole (partial programs):
- `HOLE`

Compatibility / noncanonical ops (kept for replay/macro use, typically quarantined from default derive):
- `ANOMALY_HALO`, `OBJECT_HIGHLIGHT`, `LEGEND_CHAIN_CONNECT`
- `DIAGONAL_COLLISION_TRACE`, `MASKED_PATCH_TRANSFER`
- `STACKED_GLYPH_TRACE`, `CORNER_DIAG_FILL`
- `SEPARATOR_MOTIF_BROADCAST`, `LINE_ARITH_BROADCAST`
- `BARRIER_PORT_TRANSFER`, `CAVITY_TRANSFER`

### 4.4 Execution Semantics

Execution is deterministic and demo‑parametrized:
- `ASTProgram.execute()` runs the lowered AST via the executor.
- `SearchProgram.execute()` handles steps that do not lower cleanly.
- All candidates are verified against demos before acceptance.
- No stochasticity, no external models at runtime.

### 4.5 Serialization, Signatures, and Traces

Key serialization surfaces:
- `SearchProgram.to_dict()` → stored in solve traces and macro templates.
- `SearchProgram.signature` / `selector_signature` → compact structural keys for ranking/mining.
- `StepSelect.to_dict()` preserves role + params; `by_rule` stores DNF rules.

Trace capture (`aria/search/trace_capture.py`) stores:
- action sequence + selector summaries
- full `program_dict`
- provenance and task signatures

These records are the basis for macro mining and consolidation.

## 5) Search Programs and Derive Strategies

Search programs are built from:
- `SearchProgram` = steps + provenance
- `SearchStep` = action + params + selector
- `StepSelect` = object selection logic

Derive strategies (in `aria/search/derive.py`) attempt structured inference, e.g.:
- recolor variants (uniform, rank‑based, map‑based)
- grid/slot transfer and grid‑conditional rules
- registration transfer (module → frame openings)
- panel/legend routing and mapping
- symmetry repairs, template broadcast, cropping

Derive strategies do **not** bypass verification. They produce candidate programs that must satisfy all demos.

### 5.1 Search Pipeline (current)

`aria/search/search.py` runs a fixed‑order pipeline:

Phase 0a: correspondence‑derived programs  
Phase 0b: panel structural reasoning (`aria/search/panels.py`)  
Phase 0c: panel algebra (`aria/search/panel_ops.py`)  
Phase 0d: region decode/transfer (`aria/search/decode.py`)  
Phase 0e: binding‑guided decode (`aria/search/binding_derive.py`)

If none verify, it falls back to:
- Phase 1: seed schema enumeration (single‑step programs from `aria/search/seeds.py`)
- Phase 2: 2‑step compositions of verified singles

All candidates are ranked before verification using `aria/search/candidate_rank.py`.

Verification:
- `SearchProgram.verify(demos)` executes each step and compares to demo outputs.
- Candidates are discarded immediately on mismatch.

Seed schemas:
- Each schema declares an action, parameter search space, and selector options.
- Enumeration produces `SearchProgram` candidates without using task IDs.
- Color‑based selector variations are injected for object‑level actions.

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

Ranking details:
- `score_search_program` executes on up to `max_demos` demos to compute partial metrics.
- `SearchCandidateScore.rank_key` prioritizes demos passed, dims correctness, low diff, and then priors.
- Macro prior (when enabled) is a *ranking hint only*, not a generator.

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
