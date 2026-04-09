# ARC-2 V2-Eval Gap Analysis

Source of truth: `docs/raw/arc2_v2_eval_gap_ledger.md` (117 unsolved tasks).
No solver family labels. No inferred lore. Everything below is derived from the ledger.

## First-break distribution

| Layer | Count |
|-------|-------|
| binding | 45 |
| execution | 38 |
| output | 21 |
| derivation | 9 |
| representation | 4 |

---

## 1. Ranked Gap Table

### G01 — Typed region-role decode

| Field | Value |
|-------|-------|
| gap_id | G01 |
| kind | `binding` |
| short_description | Scene regions (panels, cells, strips) need typed roles (legend/query/answer); a learned mapping transfers content from source role to target role |
| first_break_frequency | **13** (12 binding, 1 derivation) |
| task_ids | `3e6067c3` `5545f144` `58490d8a` `58f5dbd5` `5dbc8537` `8698868d` `898e7135` `a32d8b75` `abc82100` `d8e07eb2` `bf45cf4b` `b10624e5` `9385bd28` |
| closest_existing_aria_substrate | `PanelSet`, `PANEL_BOOLEAN`, panel/cell/separator extraction |
| smallest_principled_addition | Typed panel/region role assignment (source/legend/query/answer) + source-to-target decode/transfer operation over `PanelSet` or `CellGrid` |
| canonical_or_task_shaped | **canonical** |
| priority | **high** |

Sub-flavors in the ledger (all share: role assignment + cross-region transfer):
- Pure codebook lookup (color/symbol -> pattern): `3e6067c3` `58490d8a` `a32d8b75` `abc82100` `d8e07eb2`
- Cell-grid positional transfer: `58f5dbd5` `b10624e5`
- Exemplar/pattern-match decode: `5dbc8537` `8698868d` `898e7135` `bf45cf4b` `9385bd28`
- Typed multi-panel role decode: `5545f144`

### G02 — Endpoint-anchored rectilinear path routing

| Field | Value |
|-------|-------|
| gap_id | G02 |
| kind | `execution` |
| short_description | Route/complete rectilinear paths between anchored endpoints with junction and stop constraints |
| first_break_frequency | **9** (8 execution, 1 binding) |
| task_ids | `332f06d7` `36a08778` `5961cc34` `7b80bb43` `8b7bacbf` `8b9c3697` `cb2d8a2c` `cbebaa4b` `a47bf94d` |
| closest_existing_aria_substrate | `prim_trace`, `prim_cast_ray`, cell generators |
| smallest_principled_addition | Rectilinear pathfinding (BFS/A*-style) between detected anchor positions with divider/junction constraints and stop rules |
| canonical_or_task_shaped | **canonical** |
| priority | **high** |

Note: `a47bf94d` adds symbol placement on top of routing. The core operation is still path routing; placement is a composable add-on.

### G03 — Direction-from-geometry inference + guided emission

| Field | Value |
|-------|-------|
| gap_id | G03 |
| kind | `derivation` |
| short_description | Seed/component geometry implies propagation direction; a resolver reads shape -> direction, then emits rays/projections |
| first_break_frequency | **5** (3 derivation, 2 execution) |
| task_ids | `142ca369` `409aa875` `4a21e3da` `88bcf3b4` `db695cfb` |
| closest_existing_aria_substrate | `prim_trace`, `prim_cast_ray`, `_ray_cells` |
| smallest_principled_addition | Direction resolver from seed object geometry (orientation, endpoints, elbows) + parameterized emission with stop rules |
| canonical_or_task_shaped | **canonical** |
| priority | **medium** |

Codex review removed `53fb4810` (multi-color propagation with crossing policy) and `80a900e0` (parity-aware diagonal propagation) — these need propagation-policy additions, not direction inference.

### G04 — Prototype-anchor registration and transfer

| Field | Value |
|-------|-------|
| gap_id | G04 |
| kind | `binding` |
| short_description | One entity serves as prototype; others are anchors. Prototype is aligned and transferred/stamped to each anchor position. |
| first_break_frequency | **10** (8 binding, 2 execution) |
| task_ids | `20270e3b` `247ef758` `4c416de3` `4e34c42c` `7c66cb00` `9aaea919` `db0c5428` `195c6913` `221dfab4` `3dc255db` |
| closest_existing_aria_substrate | Correspondence (`ObjMapping`), object detection, crop, prior stamp experiments |
| smallest_principled_addition | Prototype vs anchor role detection + aligned template transfer (stamp/overlay at each anchor with local substitution rules) |
| canonical_or_task_shaped | **canonical** |
| priority | **high** |

Codex review removed `1ae2feb7` (separator-row broadcast — broadcast semantics, not registration/transfer). `3dc255db` (scaffold render with anchored inserts) is borderline: scaffold-guided placement is related but not purely prototype-anchor.

### G05 — Local stencil/pattern match with conditional rewrite

| Field | Value |
|-------|-------|
| gap_id | G05 |
| kind | `execution` |
| short_description | Scan grid for small recurring patterns/motifs; conditionally rewrite matching locations |
| first_break_frequency | **6** (4 execution, 2 binding) |
| task_ids | `1818057f` `28a6681f` `581f7754` `dbff022c` `faa9f03d` `dd6b8c4b` |
| closest_existing_aria_substrate | Object detection, `RECOLOR`, `REMOVE` |
| smallest_principled_addition | Small-pattern (stencil) match primitive: scan for recurring local motifs independent of connected-component boundaries + conditional rewrite/recolor of matches |
| canonical_or_task_shaped | **canonical** |
| priority | **medium** |

### G06 — Frame-role classification and content routing

| Field | Value |
|-------|-------|
| gap_id | G06 |
| kind | `binding` |
| short_description | Multiple framed/bordered objects need role classification; roles determine which content is transferred, selected, or rewritten |
| first_break_frequency | **5** (all binding) |
| task_ids | `271d71e2` `6e4f6532` `6ffbe589` `88e364bc` `8f215267` |
| closest_existing_aria_substrate | Frame extraction (`prim_find_frame`), `CROP_INTERIOR` |
| smallest_principled_addition | Frame-role classifier (source/target/filler/control) over detected frames + role-conditioned content select/transfer |
| canonical_or_task_shaped | **canonical** |
| priority | **medium** |

Codex review removed `a6f40cea` (overlap-priority composition), `dfadab01` (hierarchical part-role rewrite), `e87109e9` (shell/band removal + rebasing) — these are different additions from frame-role classification.

### G07 — Frame-interior extraction with ordered pack

| Field | Value |
|-------|-------|
| gap_id | G07 |
| kind | `output` |
| short_description | Extract content from framed/bordered/slotted objects as ordered items; pack into canonical output layout |
| first_break_frequency | **8** (all output) |
| task_ids | `89565ca0` `a251c730` `b5ca7ac4` `b6f77b65` `fc7cae8d` `446ef5d2` `4c3d4a41` `2ba387bc` |
| closest_existing_aria_substrate | Frame extraction, `CROP_INTERIOR`, `OBJECT_REPACK` |
| smallest_principled_addition | Frame-content extraction as ordered items (not just single objects) + multi-object spatial compositing/packing into a new canvas |
| canonical_or_task_shaped | **canonical** |
| priority | **medium** |

Codex review removed `6e453dd6` (panel-to-strip packing) and `7b3084d4` (strip-select render) — panel/strip packing is not frame-interior extraction.

### G08 — Relational/grouped object selection and summary pack (heterogeneous)

| Field | Value |
|-------|-------|
| gap_id | G08 |
| kind | `output` |
| short_description | Objects must be selected, grouped, counted, or classified, then packed into compact summary outputs. **Heterogeneous** — not one addition. |
| first_break_frequency | **14** (10 output, 4 binding) |
| task_ids | `62593bfd` `a395ee82` `edb79dae` `e8686506` `f560132c` `67e490f4` `31f7f899` `291dc1e1` `97d7923e` `c4d067a0` `f931b4a8` `21897d95` `c7f57c3e` `8e5c0c38` |
| closest_existing_aria_substrate | `OBJECT_REPACK`, `prim_select`, correspondence |
| smallest_principled_addition | No single addition covers this gap. Three sub-needs: |
| canonical_or_task_shaped | **general-ish** |
| priority | **medium** (discounted for heterogeneity) |

**Sub-gaps per Codex review:**
- **G08a: Relational/role object selection** (5 tasks): `f560132c` `8e5c0c38` `a395ee82` `62593bfd` `e8686506` — need relational predicates as object selection criterion
- **G08b: Grouped/counted summary render** (5 tasks): `97d7923e` `c4d067a0` `291dc1e1` `f931b4a8` `21897d95` — need group/count derivation + compact render
- **G08c: Multi-part spatial assembly** (4 tasks): `edb79dae` `c7f57c3e` `31f7f899` `67e490f4` — need part-to-composite spatial registration

### G09 — Symmetry repair extensions (mask/fold/scaffold)

| Field | Value |
|-------|-------|
| gap_id | G09 |
| kind | `execution` |
| short_description | Extend `SYMMETRY_REPAIR` beyond single damage color: damage masks, fold/hinge axes, scaffold-guided completion |
| first_break_frequency | **3** (all execution) |
| task_ids | `0934a4d8` `981571dc` `e12f9a14` |
| closest_existing_aria_substrate | `SYMMETRY_REPAIR` |
| smallest_principled_addition | Three related-but-distinct sub-additions: (a) damage mask parameter (`981571dc`), (b) fold/hinge axis (`0934a4d8`), (c) scaffold-guided multi-color completion (`e12f9a14`) |
| canonical_or_task_shaped | **canonical** (per sub-addition) |
| priority | **low** (only 3 tasks, each a different sub-addition) |

### G10 — Enclosed-region denoise / void fill

| Field | Value |
|-------|-------|
| gap_id | G10 |
| kind | `execution` |
| short_description | Identify enclosed regions; detect noise/voids inside them; fill or repair using majority/structural rules |
| first_break_frequency | **3** (all execution) |
| task_ids | `d59b0160` `e3721c99` `71e489b6` |
| closest_existing_aria_substrate | `SYMMETRY_REPAIR`, flood fill, frame detection |
| smallest_principled_addition | Enclosed-region identification + majority/structural denoise within region bounds |
| canonical_or_task_shaped | **canonical** |
| priority | **low** |

### G11 — Stateful iterative growth from seeds

| Field | Value |
|-------|-------|
| gap_id | G11 |
| kind | `execution` |
| short_description | Iterative construction from seeds with turning, branching, or fill rules |
| first_break_frequency | **4** (all execution) |
| task_ids | `20a9e565` `b9e38dc0` `d35bdbdc` `da515329` |
| closest_existing_aria_substrate | `prim_trace` |
| smallest_principled_addition | Stateful fill/grow primitive with per-step turning/branching policy |
| canonical_or_task_shaped | **general-ish** |
| priority | **low** |

### G12 — Sparse-part composite assembly

| Field | Value |
|-------|-------|
| gap_id | G12 |
| kind | `binding` |
| short_description | Multiple sparse parts need spatial registration and compositing into one canonical output |
| first_break_frequency | **5** (3 binding, 2 execution) |
| task_ids | `2c181942` `7b0280bc` `7ed72f31` `64efde09` `a25697e4` |
| closest_existing_aria_substrate | Object detection, crop, correspondence |
| smallest_principled_addition | Part-key classification + spatial registration into composite canvas |
| canonical_or_task_shaped | **general-ish** |
| priority | **low** |

### G13 — Sparse-hint shape/frame completion

| Field | Value |
|-------|-------|
| gap_id | G13 |
| kind | `derivation` |
| short_description | Sparse corner/edge hints imply a full rectilinear enclosure or frame; infer and render it |
| first_break_frequency | **2** (1 derivation, 1 execution) |
| task_ids | `de809cff` `8f3a5a89` |
| closest_existing_aria_substrate | Frame detection, trace fragments |
| smallest_principled_addition | Sparse-hint to complete frame/rectangle inference |
| canonical_or_task_shaped | **canonical** |
| priority | **low** (only 2 tasks) |

### G14 — Noise-robust structured extraction

| Field | Value |
|-------|-------|
| gap_id | G14 |
| kind | `representation` |
| short_description | Extract structured windows/regions from scenes with heavy noise or patterned backgrounds |
| first_break_frequency | **2** (both representation) |
| task_ids | `4c7dc4dd` `65b59efc` |
| closest_existing_aria_substrate | Panel/frame extraction |
| smallest_principled_addition | Robust multi-level extraction that filters noise from signal regions |
| canonical_or_task_shaped | **general-ish** |
| priority | **low** (representation-layer rewrite, only 2 tasks) |

### Ungrouped tasks (28)

Tasks that do not merge cleanly into the gaps above:

| task_id | first_break | ledger addition | why ungrouped |
|---------|-------------|-----------------|---------------|
| `53fb4810` | execution | multi-color propagation with crossing policy | propagation policy, not direction inference |
| `80a900e0` | execution | parity-aware diagonal propagation | lattice parity handling, not direction inference |
| `1ae2feb7` | execution | separator-row broadcast render | broadcast semantics, not prototype-anchor transfer |
| `a6f40cea` | binding | overlap-priority host-slot render | layered composition by z-order, not frame-role classification |
| `dfadab01` | binding | hierarchical part-role rewrite | nested part hierarchy, not frame-level roles |
| `e87109e9` | binding | band/frame role selection + rebasing | shell removal, not frame-role routing |
| `6e453dd6` | output | panel-to-strip packing | panel packing, not frame-interior extraction |
| `7b3084d4` | output | ordered strip-select render | strip extraction, not frame-interior extraction |
| `136b0064` | execution | panel motif extraction + canonical path placement | hybrid panel + path |
| `13e47133` | execution | region-template fill | unique nested rectangular fill |
| `16b78196` | binding | barrier-aware object transfer | unique structural barrier semantics |
| `16de56c4` | derivation | partition-cell line-completion | unique arithmetic derivation |
| `2d0172a1` | execution | shell/spiral normalization | unique contour operation |
| `35ab12c3` | representation | sparse codebook from point patterns | representation gap, task-shaped |
| `2b83f449` | representation | band-level boolean algebra | extends PANEL_BOOLEAN to 1D bands, only 1 task |
| `3a25b0d8` | binding | hierarchical motif decomposition | unique part hierarchy |
| `45a5af55` | execution | concentric band-to-shell render | unique band construction |
| `7491f3cf` | derivation | analogical tile completion | inter-tile transform inference |
| `7666fa5d` | execution | line-family intersection fill | unique intersection geometry |
| `78332cb0` | output | path-skeleton compact render | unique path normalization |
| `800d221b` | binding | junction-conditioned graph rewrite | unique graph semantics |
| `9bbf930d` | execution | stripe/band routing overlay | unique band + path hybrid |
| `aa4ec2a5` | execution | framed-object template render | creates frames around objects (inverse of G07) |
| `b0039139` | binding | periodic stencil decode/render | unique overlap/trim policy |
| `b99e7126` | binding | tile-grid anomaly select-and-summarize | unique anomaly detection |
| `eee78d87` | execution | tile-to-macro expansion | unique macro render |
| `e376de54` | derivation | oriented stripe-structure transform | unique stripe direction |
| `269e22fb` | execution | D4 tile-transform compose | unique tile layout with transforms |

---

## 2. Top 5 Canonical Additions

### 1. Typed region-role decode (G01)

| Field | Value |
|-------|-------|
| name | `typed_region_role_decode` |
| why_it_recurs | 13 tasks structure their scene as distinct regions with source/legend/query/answer roles. The ledger consistently records "typed panel/region roles" and "source-to-target mapping" as the missing binding across these tasks. |
| why_it_is_or_is_not_clean | **Clean.** Role assignment is a thin layer on top of `PanelSet` (panels already extracted; role = function of position/content). Decode is a single cross-region transfer op parameterized by role type and transfer mode. However, the ledger notes that some tasks also need new perception (e.g. `3e6067c3` needs "legend cells and query cells in one shared grid model") — so perception may need minor extension. |
| which_existing_ops_or_substrates_it_builds_on | `PanelSet`, panel/cell/separator extraction, `PANEL_BOOLEAN` (already does cross-panel operations) |
| task_ids | `3e6067c3` `5545f144` `58490d8a` `58f5dbd5` `5dbc8537` `8698868d` `898e7135` `a32d8b75` `abc82100` `d8e07eb2` `bf45cf4b` `b10624e5` `9385bd28` |

### 2. Prototype-anchor registration and transfer (G04)

| Field | Value |
|-------|-------|
| name | `prototype_anchor_transfer` |
| why_it_recurs | 10 tasks have one entity as prototype and others as anchors needing stamped copies. The ledger consistently records "prototype-vs-anchor roles" and "aligned transfer/stamp" as the missing binding. |
| why_it_is_or_is_not_clean | **Mostly clean.** Role detection (prototype vs anchor) builds on correspondence. Stamp/overlay builds on existing crop + grid write. The derivation of alignment rules (rotation, color substitution) adds complexity. `3dc255db` is borderline (scaffold semantics, not pure prototype transfer). |
| which_existing_ops_or_substrates_it_builds_on | Correspondence (`ObjMapping`), object detection, crop, prior stamp experiments |
| task_ids | `20270e3b` `247ef758` `4c416de3` `4e34c42c` `7c66cb00` `9aaea919` `db0c5428` `195c6913` `221dfab4` `3dc255db` |

### 3. Endpoint-anchored rectilinear path routing (G02)

| Field | Value |
|-------|-------|
| name | `rectilinear_path_routing` |
| why_it_recurs | 9 tasks need path/corridor completion between anchored endpoints. The ledger consistently records "orthogonal path/corridor" and "junction/stop constraints" as the missing execution. |
| why_it_is_or_is_not_clean | **Clean execution primitive.** Rectilinear pathfinding between endpoints is well-studied. Output is a set of cells to fill — composable with existing trace/render. However, the ledger notes some tasks need new perception (e.g. `8b7bacbf` needs "graph nodes/edges and path targets"). |
| which_existing_ops_or_substrates_it_builds_on | `prim_trace`, `_ray_cells`, cell generators (already handle cell-by-cell writing) |
| task_ids | `332f06d7` `36a08778` `5961cc34` `7b80bb43` `8b7bacbf` `8b9c3697` `cb2d8a2c` `cbebaa4b` `a47bf94d` |

### 4. Frame-interior extraction with ordered pack (G07)

| Field | Value |
|-------|-------|
| name | `frame_interior_ordered_pack` |
| why_it_recurs | 8 tasks need to extract content from framed/bordered objects and pack into canonical output. The ledger consistently records "frame-slot/interior extraction" and "ordered pack" as the missing output construction. |
| why_it_is_or_is_not_clean | **Clean extension.** Frame-interior extraction builds directly on existing `prim_find_frame` + `CROP_INTERIOR`. The missing piece is: treat multiple frame interiors as an ordered list and pack them into a new canvas. This extends `OBJECT_REPACK` with frame-content inputs. |
| which_existing_ops_or_substrates_it_builds_on | Frame extraction, `CROP_INTERIOR`, `OBJECT_REPACK` |
| task_ids | `89565ca0` `a251c730` `b5ca7ac4` `b6f77b65` `fc7cae8d` `446ef5d2` `4c3d4a41` `2ba387bc` |

### 5. Local stencil/pattern match with conditional rewrite (G05)

| Field | Value |
|-------|-------|
| name | `local_stencil_rewrite` |
| why_it_recurs | 6 tasks need scanning for small motifs (plus shapes, neighborhoods) and rewriting them. The ledger consistently records "local motif" and "conditional rewrite" as the missing execution. |
| why_it_is_or_is_not_clean | **Very clean.** One primitive: stencil match scan + conditional rewrite. Parameterized by stencil shape and rewrite rule. Independent of connected-component boundaries. Composes with any upstream perception. |
| which_existing_ops_or_substrates_it_builds_on | Grid scanning, `RECOLOR`/`REMOVE` (cell-level rewrite already exists for objects) |
| task_ids | `1818057f` `28a6681f` `581f7754` `dbff022c` `faa9f03d` `dd6b8c4b` |

---

## 3. Extensibility Check

### PANEL_BOOLEAN
- **Already covers**: Aligned panel occupancy + boolean ops (or/xor/and/etc.) over separator-extracted panels
- **Ledger supports extending to**: Band-level boolean algebra for 1D band structures (`2b83f449`); stripe/band overlay routing (`9bbf930d`)
- **Ledger does NOT support**: Legend/codebook decode (role-based transfer, not boolean); path routing; template transfer

### PANEL_REPAIR
- **Already covers**: Periodic pattern repair within panels (separator-based and frame-based)
- **Ledger supports extending to**: Robust periodic detection under noise; anomalous tile detection in repeated grids
- **Ledger does NOT support**: General codebook decode; non-periodic repair; path completion

### SYMMETRY_REPAIR
- **Already covers**: Single damage-color D4 symmetry reconstruction
- **Ledger supports extending to**: Explicit damage masks (`981571dc`); fold/hinge axes (`0934a4d8`); scaffold-guided multi-color completion (`e12f9a14`)
- **Ledger does NOT support**: Neighborhood-rule repair (G05 — local pattern match); enclosed-region denoise (G10 — majority fill); multi-color propagation policies

### OBJECT_REPACK
- **Already covers**: All-object ordering (chain/spatial/size) x layout (column/row) x payload (color_by_area/bbox)
- **Ledger supports extending to**: Frame-interior extraction as input source (G07, 8 tasks); relational/role filtering of input objects (G08a, 5 tasks); grouped/counted summary (G08b, 5 tasks); multi-part spatial compositing (G08c, 4 tasks)
- **Ledger does NOT support**: Legend decode; path routing; template transfer; local pattern match

### crop/frame/panel extraction
- **Already covers**: Frame detection (`prim_find_frame`), panel extraction from separators, crop to bbox/interior
- **Ledger supports extending to**: Robust extraction under noise (`4c7dc4dd`, `65b59efc`); multi-level scene parsing; frame-role classification (G06, 5 tasks)
- **Ledger does NOT support**: Pathfinding; propagation; codebook decode (extraction is necessary but not sufficient)

### tracing / propagation infrastructure
- **Already covers**: `prim_trace` with `_ray_cells`, `_line_cells`, `_path_cells`; ray casting with stop rules
- **Ledger supports extending to**: Rectilinear pathfinding between endpoints (G02, 9 tasks); direction-from-geometry emission (G03, 5 tasks); multi-color propagation with crossing policy (`53fb4810`); parity-aware propagation (`80a900e0`)
- **Ledger does NOT support**: Codebook decode; frame roles; object packing/summary

---

## 4. Anti-Targets

| gap_id | why_it_looks_bespoke | task_ids |
|--------|---------------------|----------|
| AT01 | Shell/spiral normalization — very specific geometric contour operation; no shared structure with other tasks | `2d0172a1` |
| AT02 | Sparse point codebook — the "codebook" is spatial arrangement of points, not a reusable decode; representation-layer gap | `35ab12c3` |
| AT03 | Periodic stencil with overlap/trim policy — overlap semantics are task-specific decode | `b0039139` |
| AT04 | Analogical tile completion — needs inter-tile transform inference, a derivation of transformation sequences, not a single canonical op | `7491f3cf` |
| AT05 | Junction-conditioned graph rewrite — unique graph semantics with typed junction roles | `800d221b` |
| AT06 | Hierarchical motif decomposition — needs recursive part-role decomposition within multicolor objects | `3a25b0d8` |
| AT07 | Line-family intersection fill — niche diagonal geometry | `7666fa5d` |
| AT08 | Concentric band-to-shell render — unique band construction geometry | `45a5af55` |

---

## 5. Recommendation

### Best next canonical addition: Typed region-role decode (G01)

- **13 tasks** — highest recurrence in the ledger
- Builds directly on existing `PanelSet` (panels already extracted; add role typing + decode transfer)
- The ledger consistently records the same structural need across all 13 tasks: typed region roles + source-to-target mapping
- Composes with existing substrate without new perception infrastructure (though some tasks note minor perception extensions needed)
- **Distinct from all other gaps** — no overlap with G02-G12

**Representative tasks**: `3e6067c3` `5545f144` `58f5dbd5` `d8e07eb2` `a32d8b75` `b10624e5`

### Best second choice: Rectilinear path routing (G02)

- **9 tasks** — second-highest recurrence among clean gaps
- Builds directly on existing trace infrastructure (`prim_trace`, cell generators)
- Rectilinear pathfinding between endpoints is a well-studied substrate
- **Orthogonal to G01** — different substrate (trace vs panels), different operation (pathfinding vs decode), no shared tasks
- G01 + G02 together cover 22 tasks across independent substrates

**Representative tasks**: `332f06d7` `8b7bacbf` `8b9c3697` `cbebaa4b` `5961cc34` `36a08778`

### Why these beat alternatives

**G01 over G04**: Both have similar recurrence (13 vs 10). G01 builds on `PanelSet` which already exists as typed infrastructure. G04 needs new prototype/anchor role detection as a derivation step — more moving parts. (This is an engineering judgment, not a ledger claim.)

**G02 over G03**: G02 has higher recurrence (9 vs 5 after Codex review tightened G03). Both build on trace infrastructure, but G02's pathfinding is a single well-understood algorithm while G03 needs a novel direction resolver from shape geometry. (This is an engineering judgment, not a ledger claim.)

**G07 as alternative**: 8 tasks, clean extension of `OBJECT_REPACK`. If G01 or G02 hits unexpected walls, G07 is the strongest fallback — purely output-layer, low derivation complexity.

---

## 6. Codex Review Delta

| Field | Value |
|-------|-------|
| codex_available | **yes** — Codex CLI v0.46.0, model gpt-5.4 |

### Main challenges Codex raised

1. **G03 over-groups** direction inference with propagation-policy (`53fb4810`, `80a900e0`)
2. **G04 over-groups** registration/transfer with broadcast (`1ae2feb7`) and scaffold (`3dc255db`)
3. **G06 over-groups** frame-role routing with overlap-priority (`a6f40cea`), hierarchical rewrite (`dfadab01`), shell removal (`e87109e9`)
4. **G07 over-groups** frame-interior extraction with panel/strip packing (`6e453dd6`, `7b3084d4`)
5. **G08 bundles 3+ different additions** (relational select, group summary, composite assembly)
6. **G09's 3 tasks are 3 different sub-additions** (fold, mask, scaffold)
7. **Recommendation claims unsupported by ledger**: "one operation" for G01, "one algorithm" for G02, "dominant ARC pattern", implementation lift comparisons, "existing perception suffices"
8. **Missing patterns**: noise-robust extraction (`4c7dc4dd` + `65b59efc`), sparse enclosure completion (`de809cff` + `8f3a5a89`)

### Challenges accepted

| # | Challenge | Action taken |
|---|-----------|-------------|
| 1 | G03 over-groups propagation-policy tasks | Removed `53fb4810`, `80a900e0` from G03; G03 now 5 tasks |
| 2 | G04 includes broadcast task | Removed `1ae2feb7`; noted `3dc255db` as borderline; G04 now 10 tasks |
| 3 | G06 includes 3 unrelated tasks | Removed `a6f40cea`, `dfadab01`, `e87109e9`; G06 now 5 tasks |
| 4 | G07 includes panel/strip tasks | Removed `6e453dd6`, `7b3084d4`; G07 now 8 tasks |
| 5 | G08 is heterogeneous | Kept as one gap but decomposed into 3 sub-gaps (G08a/b/c); noted no single addition covers it |
| 6 | G09 has 3 different sub-additions | Downgraded priority to low; noted sub-additions explicitly |
| 7 | Unsupported claims in recommendation | Removed "one operation"/"one algorithm" claims; removed "dominant ARC pattern"; marked implementation comparisons as engineering judgment; acknowledged some tasks need perception extensions |
| 8 | Missing patterns | Added G13 (sparse enclosure) and G14 (noise-robust extraction) |

### Challenges rejected

| # | Challenge | Why rejected |
|---|-----------|-------------|
| A1 | G01 marginal over-grouping | All 13 tasks share the same structural need: typed region roles + source-to-target transfer. The ledger names different transfer modes (lookup, match, analogy) but the core structural primitive is the same. A parameterized decode op covers all. |
| A2 | G02 marginal over-grouping | Path-skeleton reconstruction, line-completion, and corridor routing are all rectilinear pathfinding between endpoints. The ledger names different scene geometries but the algorithmic substrate is the same. |
| A5 | G05 over-groups motif match variants | Stencil match, neighborhood repair, control-marker rewrite, and orientation-based decode all share: scan for small pattern + conditional rewrite. Pattern size/shape varies but the primitive is one convolution-and-rewrite operation. |
| B12 | G10 marginal over-grouping | Denoise, noise-color cleanup, and void fill all need enclosed-region identification + cleanup. The cleanup method varies but the structural primitive (find enclosed region + fill) is shared. |
| B13 | G12 marginal over-grouping | Register-and-assemble, segment assembly, and fragment compose all need multi-part spatial registration into a composite. The specific parts differ but the operation is the same. |

### How the final output changed after review

- G03: 7 -> 5 tasks (2 removed)
- G04: 11 -> 10 tasks (1 removed, 1 borderline-noted)
- G06: 8 -> 5 tasks (3 removed)
- G07: 10 -> 8 tasks (2 removed)
- G08: decomposed into 3 explicit sub-gaps
- G09: priority downgraded low
- Added G13, G14 as mini-gaps
- Ungrouped: 24 -> 28 tasks
- Recommendation: removed all unsupported implementation-lift claims; added perception-gap caveats
- Top 5: G07 (frame-interior pack) replaced G03 (direction inference, now 5 tasks) at position 4
