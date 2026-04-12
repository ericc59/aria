## 2026-04-12

- added `aria/search/task_analysis.py`: `TaskAnalysis` dataclass with dims_change, diff_type (recolor_only/additive/subtractive/rearrange/mixed), extraction/construction detection, separator/panel flags; `analyze_task()` runs once per task
- added `aria/search/output_dims.py`: `DimHypothesis` and `solve_output_dims()` predicting output shapes (constant/scale_up/scale_down/object_bbox/panel_size); integrated at start of `search_programs`
- added `correspondence_transfer` strategy: correspondence-driven object placement for swap patterns; supports position swaps within same-shape groups (Hungarian assignment) and explicit color permutations (mapping stored in params) verified across demos
- upgraded `grid_slot_transfer` with tiered feature matching: exact content (cost 0) → near-shape mask overlap ≥0.5 (cost 1-10) → spatial distance (cost 20+) → incompatible (cost 1e6); deduplicated derive and exec definitions
- added `grid_conditional_transfer`: fills empty grid cells using verified row/col/mirror rules; four rule candidates (nearest_row, nearest_col, mirror_h, mirror_v) tested across all demos
- added `object_grid_pack`: packs input objects into output grid by ordering (row_major, size_asc, size_desc, color_asc); auto-infers grid dims and separator from output shape; works with different input/output sizes
- added `panel_legend_map`: detects legend panel split by separator, derives per-pixel color mapping from target changes, verifies legend colors appear in legend region; supports left/right/top/bottom legend placement
- cleaned up duplicate `_derive_grid_slot_transfer` and `_exec_grid_slot_transfer` definitions from prior incomplete merges
- strengthened `panel_legend_map` mapping validation: prefers deriving mapping from adjacent legend pairs (row-wise then column-wise) when legend is grid-like; falls back to target-diff with both source and target colors required in legend
- added `_extract_legend_pairs`: scans legend rows/cols for exactly-2-non-bg-color pairs, returns consistent mapping or None
- added feature-match tier to correspondence `_match_cost`: different-color objects with size ratio ≥ 0.5 and mask IoU ≥ 0.5 or perimeter similarity ≥ 0.5 get cost 20-30 (between near-shape and poor-match)
- added `_mask_perimeter` helper for perimeter-based shape similarity in correspondence
- updated `_classify_match` to label feature-matched pairs as `moved_recolored` instead of fallback
- added grid-conditional rule induction: `_induce_grid_cell_rule` tries coordinate mappings (mirror_hv, parity-conditional composites) beyond the 4 hardcoded rules; `_grid_cell_facts` extracts row/col/parity/emptiness per cell
- added `_CELL_MAPPINGS` dict with named coordinate transforms and `_apply_simple_rule` for exec dispatch
- extended `object_grid_pack` with `col_major` ordering, common-size-mode cell size, and `centered` placement; refactored into `_sort_objects`, `_place_patch`, `_try_grid_pack` helpers

## 2026-04-10

- fixed P1: `build_search_traces.py` now processes reports newest-first so fresh traces shadow stale ones instead of the reverse
- fixed P1: `build_macro_library.py` now defaults to `--require-test-correct` so known-bad solves are excluded from the macro library; all macros now 100% test-correct
- fixed P2: `MacroLibrary.score_candidate` now uses tiered matching on provenance + action_signature + selector_pattern (full/0.5×/0.25×) instead of collapsing to action_signature alone
- added `_derive_registration_transfer`: modules move into frame openings based on shape/mask fit; finds large non-rectangular objects (frames), detects their bg-cell openings (rectangular and non-rectangular), matches small objects by mask compatibility, emits a registration_transfer step; `228f6490` now solves (previously unsolved, test correct)
- added `_exec_registration_transfer` in sketch.py: execution-time frame/opening detection and shape-matched module placement
- ASTProgram.execute now prefers SearchProgram execution when available, enabling non-AST actions like registration_transfer to work through the full eval pipeline
- improved correspondence matching: `_match_cost` and `_classify_match` now have a near-shape same-color tier for objects that grow/shrink during movement (size ratio >= 0.5); catches 2 additional v2-eval movement detections (53→55); same-color near-size matches get cost 5-20 instead of 100+
- registration/correspondence analysis: quantified the structural gap — 42 v2-eval tasks have per-object variable-offset movements, 20 have zero-detected movements (mostly additive, correctly classified). The per-object destination inference layer is the next major capability gap.
- fixed P1: `Predicate` now has `to_dict`/`from_dict` for stable serialization; `StepSelect.to_dict` serializes predicates and nested selectors instead of silently dropping them; `from_dict` deserializes both back — `by_predicate` selectors now round-trip correctly through JSON
- fixed P1: `crop_object_rule` derive path now stores the selector in `SearchStep.select` (not raw in params); `_exec_crop_object` accepts the select argument and uses `select_objects()` for rule-based selection
- fixed P2: `MacroLibrary.score_candidate` (renamed from `score_signature`) now matches on provenance + action_signature — exact provenance match gets full weight, action-only match gets half weight, preserving the miner's grouping dimensions
- extended `_derive_rank_recolor` to handle size-group ties: objects with the same size now correctly get the same target color, using deduped size_group→color mappings; `6e82a1ae` now solves (previously unsolved); moved rank_recolor before conditional_dispatch to avoid accidental partition matches
- fixed `_derive_marker_stamp` pixel attribution: each added pixel is now assigned to its nearest marker (by L1 distance) instead of all markers within a half-grid radius, preventing template inconsistency when markers are close
- added `aria/search/selection_facts.py`: rich per-object boolean fact extraction (31 structural features + per-color) for derive-time selector rule induction
- added `StepSelect('by_rule')` with `select_objects()` at the search level: conjunction/DNF selectors induced via `induce_boolean_dnf` over pooled cross-demo object facts; rule evaluation stays in `search/` layer (never touches `guided/clause.py`), executor dispatches via `_select_targets` helper
- upgraded `_find_selector_for_group` with cross-demo verification: selectors found on demo 0 are now verified against all demos before use, falling back to cross-demo rule induction when simple predicates don't generalize
- upgraded `_find_selector` with rule induction fallback: when no single predicate exactly separates target objects, bounded conjunction search (up to 3 atoms) over structural facts is attempted
- added `_derive_action_first_dispatch`: groups changed objects by observed action (match_type) across all demos, then finds cross-demo selectors for each action group — handles cases where selector-first partitioning fails to generalize
- added `_derive_rank_recolor`: detects when all changed objects are recolored to distinct colors by size rank, decomposes into per-rank recolor steps with induced selectors; `08ed6ac7` now solves (previously unsolved)
- added `_find_selector_for_oid_sets`: shared cross-demo selector finder that takes explicit OID sets per demo, used by both action-first dispatch and rank-recolor
- added `aria/search/trace_schema.py`: `SolveTrace` dataclass capturing task_id, task_signatures, provenance, step_actions, step_selectors, full program_dict, and test_correct — the structured record for future macro mining
- added `aria/search/trace_capture.py`: `capture_solve_trace()` bridges SearchProgram → SolveTrace; `_summarize_selector` produces readable one-line selector descriptions
- added `aria/search/macros.py`: `Macro` and `MacroLibrary` schema for learned compositions above primitives — stored patterns that are reducible to SearchProgram steps, not new runtime ontology (Phase 3 groundwork)
- integrated trace capture into search/eval: `ASTProgram.search_program` carries the SearchProgram through `search_programs` → `evaluate_task`, which now stores `solve_trace` in eval outcomes
- added `scripts/build_search_traces.py`: offline export of solved traces from eval reports to JSONL for macro mining
- added `aria/search/macro_miner.py`: first exact macro miner — groups traces by `(provenance, action_signature)`, filters by frequency/step count, produces `Macro` objects with structural names and solve-rate metadata
- added `scripts/build_macro_library.py`: offline macro library builder from trace JSONL; first run on v1-train produced 6 macros (color_map freq=5, color_stencil freq=3, uniform_recolor freq=3, crop_object freq=2, crop_fixed freq=2, pixel_scale freq=2)
- fixed `StepSelect.to_dict` serialization: `by_predicate` selectors with `Predicate` objects now skip non-serializable params instead of crashing JSON encoding
- closed the first consolidation loop: `MacroLibrary.score_signature()` provides a ranking signal in `candidate_rank.py`; `search_programs` loads the macro library via `load_default_macro_library()` and passes it through all ranking calls; macro-matched candidates rank higher without bypassing exact verification
- added canonical `TEMPLATE_BROADCAST`: mask-driven blockwise template placement (out = kron(input != bg, input)); `007bbfb7` now solves via search stack in 0.02s
- filled 6 missing low-level AST executor cases: `SLIDE`, `STAMP`, `TRANSFORM_OBJ`, `FILL_INTERIOR`, `FILL_ENCLOSED`, `PERIODIC_EXTEND` — all had AST lowering paths from search but no executor dispatch
- admission audit: quarantined `STACKED_GLYPH_TRACE` from canonical `aria/search`; it depended on a task-local glyph codebook rather than a reusable execution primitive
- admission audit: quarantined `CORNER_DIAG_FILL` from canonical `aria/search`; hidden-eval success depended on a room-level fallback that did not meet the generality bar
- retained `DIAGONAL_COLLISION_TRACE`, `MASKED_PATCH_TRANSFER`, `SEPARATOR_MOTIF_BROADCAST`, `LINE_ARITH_BROADCAST`, and `BARRIER_PORT_TRANSFER` as the current defensible additions from this wave
- started G07 substrate with exact rectangular frame-item extraction and frame-family grouping/column compaction helpers for frame-pack tasks like `b5ca7ac4` and `2ba387bc`
- refactored `frame_bbox_pack` to use explicit rectangular item extraction (hollow frames + solid blocks) instead of the older object/frame heuristic
- extended `frame_bbox_pack` with family-side lane packing for G07: same-color rectangular families now get overlap-aware column clustering and side placement without a new task op
- added `registration.py` substrate for anchor-conditioned transfer tasks: anchored shape extraction and movable-module clustering now expose the missing representation behind `20270e3b`
- extended `registration.py` with base target-site enumeration and exact anchored overlay candidate generation; `20270e3b` now has a clean search space over plausible module placements instead of hand analysis
- added the first persistent `aria/search` proposal prior: search now mines past solved eval reports by task signatures and uses that memory to rank derive families and seed schemas before exact verification
- eval outcomes and trace-store search records now persist computed `task_signatures`, so future runs add better proposal memory instead of only reporting solved/unsolved
- added an explicit replay/consolidation path for proposal memory via `scripts/build_search_prior.py`; proposal ordering can now be rebuilt into a persisted JSON prior instead of implicitly rescanning eval reports every run
- added a small candidate-ranking layer for `aria/search`: seed and composition candidates are now reordered by generic verifier-style partial scores plus proposal prior strength before exact verification
- added `proposal_corpus.py` and `scripts/build_search_corpus.py` to mine solved search outcomes into a JSONL corpus keyed by task signatures and solved families, so future learned proposal/ranking work has an explicit offline dataset
- added a tiny trainable `SearchFamilyModel` (Bernoulli NB over task signatures) plus `scripts/build_search_model.py`; candidate ranking can now use a persisted family model as an additional amortized proposal signal
- fixed the AST contract for `recolor_map`: global color substitution now has its own canonical AST op/executor path instead of mislowering to object-level `RECOLOR`
- tightened public eval accounting: `aria/eval.py` now treats a task as solved only when known test outputs are correct, preventing false solves from polluting refreshed reports and learned search priors
- filled another canonical AST surface gap: `Op.TILE` now executes in `aria/search/executor.py`, so derived tile programs fail honestly as unsolved instead of crashing mid-eval refresh
- added `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`, a concrete implementation roadmap for moving `aria` toward low-level primitives + learned macros + learned routing + replay/consolidation
- added `/Users/ericc59/Dev/aria/docs/raw/looped_models_vs_aria.md`, a short architecture note on what looped language model work suggests for `aria` and what not to over-infer from benchmark tables

# 2026-04-11

- added anchor-based registration transfer derive path (`registration_anchor_transfer`) with nearest-anchor matching and search-level execution
- added `module_anchor_origin` helper to keep anchor selection consistent between derive and execution
- added synthetic regression for anchor-based registration transfer
- generalized anchor registration transfer to multiple modules using nearest-anchor assignment, with shared derive/exec movement logic
- added `module_anchor_centroid` helper to support assignment
- added separator/implicit grid detection (`grid_detect.py`) and `grid_fill_between` strategy: fills empty grid cells between same-color blocks or identical cell patterns along rows/cols; implicit grids are inferred from repeated object lattices; added `fill_all` mode to fill all empty cells with the dominant template; `06df4c85` now solves via derive
- added `grid_cell_pack` strategy: packs non-empty grid-cell contents into row/col/color order within the detected grid
- added `grid_slot_transfer` strategy: moves grid cell contents into empty slots using nearest assignment (separator or implicit grids)
- quarantined mid-level benchmark-shaped derive strategies from default routing (marker_stamp, anomaly_halo, cavity_transfer, cross_stencil_recolor, diagonal_collision_trace, separator_motif_broadcast, line_arith_broadcast, barrier_port_transfer, legend_frame_fill); kept only for macro/replay use
- added `grid_slot_transfer` strategy: moves cell contents from occupied source cells to empty target cells in a detected grid using Hungarian assignment for matching; handles both separator and implicit grids
- consolidation refresh: v1-train 45/400 (up from 34), v2-eval 11/120; 11 new v1-train solves including `228f6490` (registration_transfer), plus cumulative crop/stencil/recolor gains

## 2026-04-09

- added canonical `DIAGONAL_COLLISION_TRACE` for `142ca369`
- geometry/search now supports corner and point emitters with diagonal bounce through bar/point reflectors
- search solve confirmed for `142ca369`, `38007db0`, `3e6067c3`, `3dc255db`, and `d8e07eb2`
- added canonical `MASKED_PATCH_TRANSFER` for `0934a4d8`
- masked rectangular occlusions can now be recovered by transformed source-patch retrieval using surrounding-ring agreement
- investigated `16b78196`: missing layer is still barrier-port family induction when multiple compatible ports remain
- added canonical `LINE_ARITH_BROADCAST` for `16de56c4`
- sparse repeated markers on rows/columns can now induce arithmetic progression fills, with aligned singletons inheriting the scaffold step
- added canonical `SEPARATOR_MOTIF_BROADCAST` for `1ae2feb7`
- separator-split lines can now broadcast source-side color runs into the empty opposite side using run-length periods and separator-adjacent precedence
- added canonical `BARRIER_PORT_TRANSFER` for `16b78196`
- spanning barrier strips can now expose boundary-touching port families, then repack objects by cross-barrier extent with collision-based packing through the chosen opening
- refined `BARRIER_PORT_TRANSFER` with port-fit scoring from occupied opening cells, so ambiguous compatible ports now resolve canonically on hidden eval structure
- investigated `195c6913` and `20270e3b`: both still block on clean anchor-registered overlay/union rather than a missing leaf execution primitive
- added `bar-window` extraction for G06 frame-role scenes
- search binding can now expose side-anchored window entities, interior/content facts, and conservative source/target/workspace hints for tasks like `271d71e2`
- investigated `28a6681f`: the current local fact language is still too weak for an exact neighborhood-rule cover, so that repair layer needs a broader design pass rather than a forced task solve
