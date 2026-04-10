## 2026-04-10

- admission audit: quarantined `STACKED_GLYPH_TRACE` from canonical `aria/search`; it depended on a task-local glyph codebook rather than a reusable execution primitive
- admission audit: quarantined `CORNER_DIAG_FILL` from canonical `aria/search`; hidden-eval success depended on a room-level fallback that did not meet the generality bar
- retained `DIAGONAL_COLLISION_TRACE`, `MASKED_PATCH_TRANSFER`, `SEPARATOR_MOTIF_BROADCAST`, `LINE_ARITH_BROADCAST`, and `BARRIER_PORT_TRANSFER` as the current defensible additions from this wave
- started G07 substrate with exact rectangular frame-item extraction and frame-family grouping/column compaction helpers for frame-pack tasks like `b5ca7ac4` and `2ba387bc`
- refactored `frame_bbox_pack` to use explicit rectangular item extraction (hollow frames + solid blocks) instead of the older object/frame heuristic
- extended `frame_bbox_pack` with family-side lane packing for G07: same-color rectangular families now get overlap-aware column clustering and side placement without a new task op

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
