# ARC-2 V2-Eval Unsolved Gap Ledger

Source of truth for the task-by-task `aria/search` stack-gap audit of all unsolved ARC-2 `v2-eval` tasks.

Excluded solved tasks:
- `135a2760`
- `38007db0`
- `7b5033c1`

Each task records:
- what the task appears to require
- what `aria` currently has that is relevant
- first failed layer
- missing representation
- missing derivation
- missing canonical operation
- smallest principled addition
- whether it looks general or task-shaped

## Tasks 1–20

### 1. `0934a4d8`
- What the task appears to require: mask-guided fold reconstruction from partial symmetric evidence
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`, `CROP_*`
- First failed layer: `execution`
- Missing representation: fold mask and hinge axes
- Missing derivation: axis inference from the masked rectangle
- Missing canonical operation: masked fold/unfold reconstruction
- Smallest principled addition: fold-reconstruct op
- Likely general or task-shaped: `general`

### 2. `136b0064`
- What the task appears to require: separator-panel motif extraction plus path render
- What `aria` currently has that is relevant: `PANEL_*`
- First failed layer: `execution`
- Missing representation: panel-local motif plus path scaffold
- Missing derivation: which panel supplies which path segment
- Missing canonical operation: motif-path render
- Smallest principled addition: panel motif extraction with canonical path placement
- Likely general or task-shaped: `general-ish`

### 3. `13e47133`
- What the task appears to require: partition/frame-local region rewrite with nested rectangular fill keyed by local seeds
- What `aria` currently has that is relevant: frame/panel extraction, symmetry repair
- First failed layer: `execution`
- Missing representation: region-local template fill
- Missing derivation: mapping each region’s markers to fill colors
- Missing canonical operation: rectangular band/template render inside regions
- Smallest principled addition: region-template fill
- Likely general or task-shaped: `general-ish`

### 4. `142ca369`
- What the task appears to require: diagonal ray emission from oriented components
- What `aria` currently has that is relevant: `prim_trace`-style tracing only
- First failed layer: `derivation`
- Missing representation: oriented component endpoints and elbows
- Missing derivation: direction from component geometry
- Missing canonical operation: reusable ray emission with derived direction
- Smallest principled addition: canonical ray op with component-derived direction resolver
- Likely general or task-shaped: `general`

### 5. `16b78196`
- What the task appears to require: barrier-conditioned relocation/repacking of objects across a horizontal structural strip
- What `aria` currently has that is relevant: crop and panel/frame extraction
- First failed layer: `binding`
- Missing representation: top/bottom roles relative to barrier and target slots
- Missing derivation: which object moves where
- Missing canonical operation: structural relocation/packing relative to a barrier
- Smallest principled addition: barrier-aware object transfer
- Likely general or task-shaped: `general-ish`

### 6. `16de56c4`
- What the task appears to require: arithmetic line completion across partition cells
- What `aria` currently has that is relevant: partition/panel extraction
- First failed layer: `derivation`
- Missing representation: color/marker to line-length relation
- Missing derivation: arithmetic rule over cells
- Missing canonical operation: axis projection/broadcast with learned length
- Smallest principled addition: partition-cell line-completion
- Likely general or task-shaped: `general-ish`

### 7. `1818057f`
- What the task appears to require: local plus-motif detection and conditional recolor
- What `aria` currently has that is relevant: object detection only
- First failed layer: `execution`
- Missing representation: recurring local motifs independent of connected components
- Missing derivation: which motif color maps to which recolor
- Missing canonical operation: motif-conditioned local rewrite
- Smallest principled addition: stencil/motif match plus recolor
- Likely general or task-shaped: `general`

### 8. `195c6913`
- What the task appears to require: region-registered overlay from small control markers into large shaped regions
- What `aria` currently has that is relevant: crop and panels
- First failed layer: `binding`
- Missing representation: marker-to-region registration
- Missing derivation: how control chips determine the overlay path
- Missing canonical operation: registered border/path overlay
- Smallest principled addition: anchor-registered overlay
- Likely general or task-shaped: `task-shaped leaning`

### 9. `1ae2feb7`
- What the task appears to require: row-wise motif broadcast across a separator structure
- What `aria` currently has that is relevant: `PANEL_*`
- First failed layer: `execution`
- Missing representation: reference motif per row and a target axis
- Missing derivation: broadcast order
- Missing canonical operation: row motif broadcast
- Smallest principled addition: separator-row broadcast render
- Likely general or task-shaped: `general-ish`

### 10. `20270e3b`
- What the task appears to require: crop-align-union of components at marker-registered positions
- What `aria` currently has that is relevant: crop and object detection
- First failed layer: `binding`
- Missing representation: marker-to-component registration frame
- Missing derivation: which crop aligns to which marker
- Missing canonical operation: registered crop-plus-overlay union
- Smallest principled addition: canonical registration plus union
- Likely general or task-shaped: `general`

### 11. `20a9e565`
- What the task appears to require: staged motif growth or symmetric stacking from seed pieces
- What `aria` currently has that is relevant: symmetry repair only
- First failed layer: `execution`
- Missing representation: generative sequence state
- Missing derivation: growth order and symmetry policy
- Missing canonical operation: staged copy/reflect placement
- Smallest principled addition: iterative motif growth
- Likely general or task-shaped: `task-shaped leaning`

### 12. `21897d95`
- What the task appears to require: cleanup of large color regions plus compact summary/repacking of the dominant structures
- What `aria` currently has that is relevant: crop and object summary only
- First failed layer: `output`
- Missing representation: ordered macro-regions after denoising
- Missing derivation: which regions survive and in what arrangement
- Missing canonical operation: region summary/repack
- Smallest principled addition: region-to-summary pack
- Likely general or task-shaped: `task-shaped leaning`

### 13. `221dfab4`
- What the task appears to require: row-aligned component rewrite along a central track or boundary
- What `aria` currently has that is relevant: frame/panel extraction
- First failed layer: `binding`
- Missing representation: row objects registered to side track
- Missing derivation: how the track encodes inserted colors
- Missing canonical operation: component-to-track render
- Smallest principled addition: axis-local registered overlay
- Likely general or task-shaped: `general-ish`

### 14. `247ef758`
- What the task appears to require: extracting an interior motif and stamping it into a bordered frame at a derived position
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: frame role, interior template, and target anchor
- Missing derivation: placement inside the frame
- Missing canonical operation: frame-local template transfer
- Smallest principled addition: anchored template stamp
- Likely general or task-shaped: `general-ish`

### 15. `269e22fb`
- What the task appears to require: D4-style template completion or transformed tiling onto a canonical output canvas
- What `aria` currently has that is relevant: `CROP_*`
- First failed layer: `execution`
- Missing representation: canonical template plus transform layout
- Missing derivation: layout of rotations and flips
- Missing canonical operation: tile-transform compose/render
- Smallest principled addition: `tile_transform`
- Likely general or task-shaped: `general`

### 16. `271d71e2`
- What the task appears to require: transferring inner content among framed windows based on frame roles
- What `aria` currently has that is relevant: frame extraction
- First failed layer: `binding`
- Missing representation: source frame, target frame, and filler frame roles
- Missing derivation: role assignment among multiple frames
- Missing canonical operation: frame-role transfer
- Smallest principled addition: frame-to-frame content transfer
- Likely general or task-shaped: `general-ish`

### 17. `28a6681f`
- What the task appears to require: local repair or conditional recolor under neighborhood continuity constraints
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`
- First failed layer: `execution`
- Missing representation: local neighborhood predicates
- Missing derivation: which color is repairable and from which neighbors
- Missing canonical operation: local conditional rewrite
- Smallest principled addition: neighborhood-rule repair
- Likely general or task-shaped: `general`

### 18. `291dc1e1`
- What the task appears to require: ordered component summary plus vertical repacking into a compact output strip
- What `aria` currently has that is relevant: `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: ordered component blocks rather than just color counts
- Missing derivation: block role/order within the source strip
- Missing canonical operation: component-to-strip summary render
- Smallest principled addition: structured component repack
- Likely general or task-shaped: `general-ish`

### 19. `2b83f449`
- What the task appears to require: boolean-like combination over separator bands or stripe patterns, but not the simple aligned panel occupancy case already solved
- What `aria` currently has that is relevant: `PANEL_BOOLEAN`
- First failed layer: `representation`
- Missing representation: 1D band structure with alternating content rows, not clean 2D panels
- Missing derivation: which bands combine and where writes land
- Missing canonical operation: band boolean or stripe overlay
- Smallest principled addition: band-level boolean algebra
- Likely general or task-shaped: `general-ish`

### 20. `2ba387bc`
- What the task appears to require: cropping multiple boxed objects and packing them into a compact output layout
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `output`
- Missing representation: a set of boxed objects with stable ordering
- Missing derivation: sort/pack order
- Missing canonical operation: crop-box-and-concatenate
- Smallest principled addition: ordered box packing
- Likely general or task-shaped: `general`

## Tasks 21–40

### 21. `2c181942`
- What the task appears to require: compose several small colored motifs into one registered composite shape
- What `aria` currently has that is relevant: object detection, crop, simple panel/frame ops
- First failed layer: `binding`
- Missing representation: source motifs with stable anchor/orientation roles
- Missing derivation: how each small motif aligns into the final composite
- Missing canonical operation: anchored multi-object assembly
- Smallest principled addition: canonical `register_and_assemble` over a small motif set
- Likely general or task-shaped: `general-ish`

### 22. `2d0172a1`
- What the task appears to require: extract a spiral/shell object and canonicalize it to a compact centered form
- What `aria` currently has that is relevant: crop, symmetry repair, basic object extraction
- First failed layer: `execution`
- Missing representation: ordered contour/band structure within one object
- Missing derivation: how to normalize shell thickness/orientation
- Missing canonical operation: contour/shell canonicalization render
- Smallest principled addition: canonical shell/spiral normalization
- Likely general or task-shaped: `general-ish`

### 23. `31f7f899`
- What the task appears to require: register colored bars around a reference axis and emit a normalized axis-centered arrangement
- What `aria` currently has that is relevant: object detection, crop, separator/frame hints
- First failed layer: `binding`
- Missing representation: reference axis plus side-specific bar roles
- Missing derivation: which bar goes left/right of the axis and in what order
- Missing canonical operation: axis-registered bar assembly
- Smallest principled addition: axis-anchored packing/overlay
- Likely general or task-shaped: `general`

### 24. `332f06d7`
- What the task appears to require: recover a path/maze-like shape from a marked partial path and preserve corner semantics
- What `aria` currently has that is relevant: panel ops and simple traces only
- First failed layer: `execution`
- Missing representation: directed path skeleton with entry/exit markers
- Missing derivation: which turns/segments are preserved or completed
- Missing canonical operation: path continuation or corridor completion
- Smallest principled addition: canonical path-skeleton reconstruction
- Likely general or task-shaped: `general-ish`

### 25. `35ab12c3`
- What the task appears to require: convert sparse point configurations into structured emblem/frame outputs
- What `aria` currently has that is relevant: object detection and crop only
- First failed layer: `representation`
- Missing representation: sparse points as code specifying a higher-order template
- Missing derivation: how point arrangement chooses the emblem/frame type
- Missing canonical operation: codebook/template decode render
- Smallest principled addition: sparse codebook decode from point patterns
- Likely general or task-shaped: `task-shaped leaning`

### 26. `36a08778`
- What the task appears to require: infer orthogonal corridor/line completions from sparse horizontal fragments and corner seeds
- What `aria` currently has that is relevant: panel extraction, limited tracing
- First failed layer: `execution`
- Missing representation: corridor graph with junction constraints
- Missing derivation: where vertical joins are inserted to close the pattern
- Missing canonical operation: rectilinear completion over sparse line hints
- Smallest principled addition: orthogonal line-completion primitive
- Likely general or task-shaped: `general`

### 27. `3a25b0d8`
- What the task appears to require: transform a complex colored emblem into a normalized symbolic output using internal part roles
- What `aria` currently has that is relevant: object extraction and crop
- First failed layer: `binding`
- Missing representation: internal subparts of a multicolor motif as labeled roles
- Missing derivation: which subparts are preserved, recolored, or duplicated
- Missing canonical operation: part-aware motif rewrite
- Smallest principled addition: hierarchical motif decomposition plus rewrite
- Likely general or task-shaped: `task-shaped leaning`

### 28. `3dc255db`
- What the task appears to require: place small glyph parts into a larger symmetric arch/handle structure
- What `aria` currently has that is relevant: crop, object detection, limited symmetry
- First failed layer: `execution`
- Missing representation: skeletal scaffold that receives placed parts
- Missing derivation: attachment points for each glyph piece
- Missing canonical operation: scaffold-guided placement
- Smallest principled addition: canonical scaffold render with anchored inserts
- Likely general or task-shaped: `general-ish`

### 29. `3e6067c3`
- What the task appears to require: decode a cell grid using a bottom legend row and rewrite cell contents accordingly
- What `aria` currently has that is relevant: panel/grid extraction, `PANEL_BOOLEAN`, crop
- First failed layer: `binding`
- Missing representation: legend cells and query cells in one shared grid model
- Missing derivation: mapping legend symbols/colors to output rewrites
- Missing canonical operation: separator-grid legend rewrite
- Smallest principled addition: cell-grid decode via legend mapping
- Likely general or task-shaped: `general`

### 30. `409aa875`
- What the task appears to require: directional projection from small corner/triomino seeds
- What `aria` currently has that is relevant: trace infrastructure only
- First failed layer: `derivation`
- Missing representation: seed orientation and projection direction
- Missing derivation: which direction each seed projects and where it stops
- Missing canonical operation: directional projection or emission
- Smallest principled addition: seed-oriented projection op
- Likely general or task-shaped: `general`

### 31. `446ef5d2`
- What the task appears to require: extract slot/bar structures from framed objects and repack them into normalized outputs
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `output`
- Missing representation: framed objects with ordered slot contents
- Missing derivation: packing order and normalization of the extracted bars
- Missing canonical operation: slot/bar repack render
- Smallest principled addition: frame-slot extraction plus ordered pack
- Likely general or task-shaped: `general-ish`

### 32. `45a5af55`
- What the task appears to require: build concentric bands/shells from a stripe/band description
- What `aria` currently has that is relevant: symmetry repair and crop only
- First failed layer: `execution`
- Missing representation: ordered band stack with inside/outside nesting
- Missing derivation: band order and target square extent
- Missing canonical operation: concentric shell construction
- Smallest principled addition: band-to-shell render
- Likely general or task-shaped: `general`

### 33. `4a21e3da`
- What the task appears to require: emit cross/axis extensions from compact seed objects
- What `aria` currently has that is relevant: trace/projection fragments
- First failed layer: `derivation`
- Missing representation: seed arm lengths and axis choices
- Missing derivation: which arms extend and how far
- Missing canonical operation: cross-axis extension
- Smallest principled addition: object-centered axis emission
- Likely general or task-shaped: `general`

### 34. `4c3d4a41`
- What the task appears to require: reorder colored bars from a frame/slot scene into a compact canonical layout
- What `aria` currently has that is relevant: frame extraction, crop, limited pack-like summary
- First failed layer: `output`
- Missing representation: slot contents as ordered sequence items
- Missing derivation: destination order and layout geometry
- Missing canonical operation: compact bar repack
- Smallest principled addition: ordered strip/slot packing
- Likely general or task-shaped: `general-ish`

### 35. `4c416de3`
- What the task appears to require: use a prototype square to clone/attach colored markers at new anchor positions
- What `aria` currently has that is relevant: object detection, frame extraction, simple stamp experiments
- First failed layer: `binding`
- Missing representation: prototype/anchor roles plus attachment offsets
- Missing derivation: where prototype corners/markers transfer in each scene
- Missing canonical operation: role-conditioned clone/place
- Smallest principled addition: prototype-guided anchored transfer
- Likely general or task-shaped: `general`

### 36. `4c7dc4dd`
- What the task appears to require: read framed regions in a noisy checkerboard and rewrite them according to corner/motif cues
- What `aria` currently has that is relevant: panel extraction, frame extraction
- First failed layer: `representation`
- Missing representation: multi-level scene with noisy background, framed windows, and corner motifs
- Missing derivation: which local motif controls which window rewrite
- Missing canonical operation: noisy-panel decode and rewrite
- Smallest principled addition: robust window extraction plus local code decode
- Likely general or task-shaped: `task-shaped leaning`

### 37. `4e34c42c`
- What the task appears to require: register small source structures to colored anchors and emit larger connected assemblies
- What `aria` currently has that is relevant: crop and object detection
- First failed layer: `binding`
- Missing representation: source-anchor pairs with a shared coordinate frame
- Missing derivation: alignment rule between the small prototype and the target anchor
- Missing canonical operation: registered transfer/assembly
- Smallest principled addition: anchor-registered template transfer
- Likely general or task-shaped: `general-ish`

### 38. `53fb4810`
- What the task appears to require: propagate colored signals along orthogonal axes from plus-like seeds
- What `aria` currently has that is relevant: limited trace infrastructure
- First failed layer: `execution`
- Missing representation: seed type plus propagation colors/policies
- Missing derivation: which channels propagate and how they interact at crossings
- Missing canonical operation: colored axis propagation
- Smallest principled addition: multi-color propagation with crossing policy
- Likely general or task-shaped: `general`

### 39. `5545f144`
- What the task appears to require: panel/legend decode with several control regions, then render a sparse output from that code
- What `aria` currently has that is relevant: panel extraction, frame extraction, `PANEL_BOOLEAN`
- First failed layer: `binding`
- Missing representation: distinct control panels with typed roles such as legend/query/output
- Missing derivation: the mapping from panel-local code to output pixels
- Missing canonical operation: multi-panel legend decode and transfer
- Smallest principled addition: typed panel-role decode over a `PanelSet`
- Likely general or task-shaped: `general`

### 40. `581f7754`
- What the task appears to require: marker-conditioned local transforms of small framed shapes, preserving relative placement
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: local shape plus side markers as control signals
- Missing derivation: how each marker changes the adjacent shape
- Missing canonical operation: marker-conditioned local rewrite/clone
- Smallest principled addition: local control-marker rewrite
- Likely general or task-shaped: `general-ish`

## Tasks 41–60

### 41. `58490d8a`
- What the task appears to require: decode a small legend/code panel and render matching larger motifs in a target scene
- What `aria` currently has that is relevant: panel extraction, crop, simple rewrite ops
- First failed layer: `binding`
- Missing representation: legend panel with typed symbol entries
- Missing derivation: mapping legend glyphs to target motif classes
- Missing canonical operation: codebook-driven motif rewrite
- Smallest principled addition: legend decode over a typed panel/cell set
- Likely general or task-shaped: `general-ish`

### 42. `58f5dbd5`
- What the task appears to require: map small symbolic/color patterns to larger rendered forms inside a grid
- What `aria` currently has that is relevant: panel/grid extraction and crop
- First failed layer: `binding`
- Missing representation: compact code cells paired with rendered target cells
- Missing derivation: which local pattern selects which output motif
- Missing canonical operation: cell codebook decode/render
- Smallest principled addition: grid-cell symbol decode
- Likely general or task-shaped: `general`

### 43. `5961cc34`
- What the task appears to require: connect marked objects with rectilinear segments from a seed/anchor axis
- What `aria` currently has that is relevant: primitive tracing only
- First failed layer: `execution`
- Missing representation: object centers plus connector seed
- Missing derivation: connector topology and join order
- Missing canonical operation: rectilinear connector/tree render
- Smallest principled addition: object-center orthogonal connection op
- Likely general or task-shaped: `general`

### 44. `5dbc8537`
- What the task appears to require: use one region as a codebook to rewrite or summarize another region into a compact canonical output
- What `aria` currently has that is relevant: panel extraction, crop, `OBJECT_REPACK`
- First failed layer: `binding`
- Missing representation: source/query/legend region roles
- Missing derivation: which subregion controls which output block
- Missing canonical operation: region-to-region codebook transfer
- Smallest principled addition: typed region-role decode and render
- Likely general or task-shaped: `general-ish`

### 45. `62593bfd`
- What the task appears to require: sort several sparse objects into canonical corner/row positions while preserving internal shape
- What `aria` currently has that is relevant: object detection and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: stable object ordering across scenes
- Missing derivation: target slot assignment
- Missing canonical operation: object sort-and-place into canonical slots
- Smallest principled addition: ordered object packing by role
- Likely general or task-shaped: `general`

### 46. `64efde09`
- What the task appears to require: assemble separate colored segments into a combined frame/path arrangement
- What `aria` currently has that is relevant: tracing and object detection
- First failed layer: `binding`
- Missing representation: segment endpoints and shared anchors
- Missing derivation: which segments join into one structure
- Missing canonical operation: segment registration and merge
- Smallest principled addition: endpoint-aware segment assembly
- Likely general or task-shaped: `general-ish`

### 47. `65b59efc`
- What the task appears to require: decode a noisy multi-region symbolic scene and emit a cleaner structured result
- What `aria` currently has that is relevant: panel extraction and crop
- First failed layer: `representation`
- Missing representation: robust partition under heavy noise or patterned backgrounds
- Missing derivation: which local quadrants are signal vs distractor
- Missing canonical operation: noisy partition decode
- Smallest principled addition: robust region extraction with local codebook reading
- Likely general or task-shaped: `task-shaped leaning`

### 48. `67e490f4`
- What the task appears to require: read scattered symbols against a reference panel and place them into a canonical layout/grid
- What `aria` currently has that is relevant: object detection and simple packing
- First failed layer: `binding`
- Missing representation: symbol classes and target grid slots
- Missing derivation: map free symbols to canonical cells
- Missing canonical operation: symbol-to-grid packing
- Smallest principled addition: classified object pack into fixed grid
- Likely general or task-shaped: `general`

### 49. `6e453dd6`
- What the task appears to require: extract panel-local glyphs and repack them into narrow canonical strips
- What `aria` currently has that is relevant: panel extraction and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: per-panel glyph stack as ordered items
- Missing derivation: strip layout and side-color handling
- Missing canonical operation: panel-local strip repack
- Smallest principled addition: panel-to-strip packing
- Likely general or task-shaped: `general-ish`

### 50. `6e4f6532`
- What the task appears to require: use frame colors and local markers to rewrite or select the important subobject inside each framed scene
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: frame role plus internal control marker role
- Missing derivation: how frame color controls target behavior
- Missing canonical operation: frame-conditioned local rewrite
- Smallest principled addition: frame-role rewrite over extracted interiors
- Likely general or task-shaped: `general-ish`

### 51. `6ffbe589`
- What the task appears to require: identify one framed/checkered source motif and render a canonical extracted version
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: candidate framed motifs with salience/selection role
- Missing derivation: which frame is the answer and how much border is canonical
- Missing canonical operation: salient framed-motif extraction
- Smallest principled addition: framed motif selection plus canonical crop
- Likely general or task-shaped: `general-ish`

### 52. `71e489b6`
- What the task appears to require: detect voids/holes or local anomalies in large regions and decorate them with a standard repair/halo pattern
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`, frame extraction
- First failed layer: `execution`
- Missing representation: hole/void regions as entities
- Missing derivation: which voids get highlighted and with what shape
- Missing canonical operation: local hole halo / void decoration
- Smallest principled addition: enclosed-void pattern fill
- Likely general or task-shaped: `general`

### 53. `7491f3cf`
- What the task appears to require: infer a transformation sequence across a row of tiles and complete the missing/final tile
- What `aria` currently has that is relevant: panel extraction and limited transform handling
- First failed layer: `derivation`
- Missing representation: ordered tile sequence with per-step relation
- Missing derivation: analogical transformation between neighboring tiles
- Missing canonical operation: sequence completion over aligned tiles
- Smallest principled addition: row-wise analogical tile completion
- Likely general or task-shaped: `general`

### 54. `7666fa5d`
- What the task appears to require: fill the overlap/interaction region between diagonal stripe sets
- What `aria` currently has that is relevant: primitive tracing only
- First failed layer: `execution`
- Missing representation: stripe families as extended lines, not just observed pixels
- Missing derivation: overlap polygon between families
- Missing canonical operation: diagonal overlap fill
- Smallest principled addition: line-family intersection/fill
- Likely general or task-shaped: `general`

### 55. `78332cb0`
- What the task appears to require: normalize maze/path motifs from several panels into canonical strip-like outputs
- What `aria` currently has that is relevant: panel extraction and tracing
- First failed layer: `output`
- Missing representation: path skeleton with ordered branches
- Missing derivation: which part of the path becomes the compact output
- Missing canonical operation: path normalization/rebase
- Smallest principled addition: path-skeleton extract and compact render
- Likely general or task-shaped: `general-ish`

### 56. `7b0280bc`
- What the task appears to require: match sparse shapes by type/color and compose a canonical multi-part output
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `binding`
- Missing representation: shape-key classes across objects
- Missing derivation: which matched pieces belong together
- Missing canonical operation: shape-match compose
- Smallest principled addition: shape-key grouping and assembly
- Likely general or task-shaped: `general`

### 57. `7b3084d4`
- What the task appears to require: extract and re-express a structured colored strip/subpanel from a larger scene
- What `aria` currently has that is relevant: panel extraction and crop
- First failed layer: `output`
- Missing representation: target strip as a selected substructure with canonical orientation
- Missing derivation: which strip/segment is retained
- Missing canonical operation: strip extraction and rebasing
- Smallest principled addition: ordered strip-select render
- Likely general or task-shaped: `general-ish`

### 58. `7b80bb43`
- What the task appears to require: complete or regularize orthogonal path networks relative to dividers and sparse corner hints
- What `aria` currently has that is relevant: tracing and panel extraction
- First failed layer: `execution`
- Missing representation: orthogonal network with junction rules
- Missing derivation: which segments extend and where they turn/stop
- Missing canonical operation: orthogonal path completion
- Smallest principled addition: network completion with divider constraints
- Likely general or task-shaped: `general`

### 59. `7c66cb00`
- What the task appears to require: stamp or copy small motifs into larger horizontal bands based on band-local examples
- What `aria` currently has that is relevant: panel/band extraction and `OBJECT_REPACK`
- First failed layer: `binding`
- Missing representation: band roles and exemplar motifs per band
- Missing derivation: where each motif repeats inside its band
- Missing canonical operation: band-local motif broadcast
- Smallest principled addition: exemplar-to-band stamp
- Likely general or task-shaped: `general`

### 60. `7ed72f31`
- What the task appears to require: assemble sparse colored parts into a compact composite emblem on a new canvas
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `execution`
- Missing representation: part anchors and relative geometry of the assembled emblem
- Missing derivation: how sparse pieces align into one structure
- Missing canonical operation: part assembly into canonical composite
- Smallest principled addition: sparse-part assembly render
- Likely general or task-shaped: `general-ish`

## Tasks 61–80

### 61. `800d221b`
- What the task appears to require: rewrite a branching rectilinear structure based on local color-role interactions at junctions and subframes
- What `aria` currently has that is relevant: crop, object detection, simple frame/panel extraction
- First failed layer: `binding`
- Missing representation: typed junction/subframe roles inside one connected structure
- Missing derivation: which local colors control the rewritten branch colors
- Missing canonical operation: graph-local structural rewrite
- Smallest principled addition: canonical junction-conditioned graph rewrite
- Likely general or task-shaped: `general-ish`

### 62. `80a900e0`
- What the task appears to require: propagate colored diagonals over a checkerboard/parity lattice from a seed motif
- What `aria` currently has that is relevant: basic tracing only
- First failed layer: `execution`
- Missing representation: alternating lattice/parity as first-class structure
- Missing derivation: which rays follow which parity class and color
- Missing canonical operation: diagonal propagation on a periodic lattice
- Smallest principled addition: parity-aware diagonal propagation
- Likely general or task-shaped: `general`

### 63. `8698868d`
- What the task appears to require: decode small codebook examples and render matched larger tiles/blocks in a target layout
- What `aria` currently has that is relevant: panel/grid extraction, crop, simple packing
- First failed layer: `binding`
- Missing representation: codebook entries linked to target tiles
- Missing derivation: which exemplar matches which target block
- Missing canonical operation: codebook-driven tile rewrite
- Smallest principled addition: exemplar-to-target tile decode
- Likely general or task-shaped: `general-ish`

### 64. `88bcf3b4`
- What the task appears to require: extend sparse motifs along oriented paths or trunks from local seeds
- What `aria` currently has that is relevant: trace fragments and object detection
- First failed layer: `derivation`
- Missing representation: oriented seed endpoints and target trunk
- Missing derivation: direction and continuation policy per seed
- Missing canonical operation: sparse motif projection along a path
- Smallest principled addition: oriented seed projection with stop rules
- Likely general or task-shaped: `general`

### 65. `88e364bc`
- What the task appears to require: classify framed outline shapes with small colored cues and compose canonical outputs from them
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: outline shape class plus cue markers
- Missing derivation: which outline or cue determines the selected output
- Missing canonical operation: cue-conditioned outline selection/assembly
- Smallest principled addition: outline shape-key decode with local cues
- Likely general or task-shaped: `general-ish`

### 66. `89565ca0`
- What the task appears to require: extract several framed rectangles and repack/recompose them into a new canonical arrangement
- What `aria` currently has that is relevant: frame extraction and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: multiple framed objects with stable ordering and relations
- Missing derivation: destination arrangement and overlap/order policy
- Missing canonical operation: framed-object repack/assembly
- Smallest principled addition: ordered framed-object packing
- Likely general or task-shaped: `general`

### 67. `898e7135`
- What the task appears to require: map sparse symbols to a larger colored output panel with transformed iconic content
- What `aria` currently has that is relevant: object detection, crop, simple rewrite
- First failed layer: `binding`
- Missing representation: symbol classes tied to output panel semantics
- Missing derivation: which symbol controls panel color and internal icon
- Missing canonical operation: symbol-conditioned panel render
- Smallest principled addition: sparse symbol decode to panel template
- Likely general or task-shaped: `general-ish`

### 68. `8b7bacbf`
- What the task appears to require: route/complete colored paths through a graph-like scene with anchored nodes
- What `aria` currently has that is relevant: trace infrastructure only
- First failed layer: `execution`
- Missing representation: graph nodes/edges and path targets
- Missing derivation: which route is chosen and where colors transfer
- Missing canonical operation: graph path completion/routing
- Smallest principled addition: node-target path routing
- Likely general or task-shaped: `general`

### 69. `8b9c3697`
- What the task appears to require: carve or fill a connected route between marked regions inside a larger background field
- What `aria` currently has that is relevant: crop, limited trace, region extraction
- First failed layer: `execution`
- Missing representation: marked source/target regions plus traversable corridor
- Missing derivation: exact route from marker to target shape
- Missing canonical operation: region-to-region corridor fill
- Smallest principled addition: constrained corridor routing/fill
- Likely general or task-shaped: `general`

### 70. `8e5c0c38`
- What the task appears to require: match sparse symbol groups and emit a simplified canonical output retaining only the relationally important pieces
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `binding`
- Missing representation: relation between multiple sparse symbol types
- Missing derivation: which symbols are preserved, paired, or discarded
- Missing canonical operation: relation-based sparse symbol selection
- Smallest principled addition: sparse relational select-and-render
- Likely general or task-shaped: `general-ish`

### 71. `8f215267`
- What the task appears to require: rewrite labeled horizontal boxes using nearby symbol/color counts as controls
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: box roles plus local control symbols
- Missing derivation: how controls map to box interior content
- Missing canonical operation: control-conditioned box rewrite
- Smallest principled addition: box template rewrite from adjacent controls
- Likely general or task-shaped: `general`

### 72. `8f3a5a89`
- What the task appears to require: infer a bounding frame or enclosing transform from sparse corner/edge hints
- What `aria` currently has that is relevant: frame detection and crop
- First failed layer: `derivation`
- Missing representation: partial boundary hints as a single latent frame
- Missing derivation: full enclosure implied by sparse seeds
- Missing canonical operation: inferred frame construction
- Smallest principled addition: sparse-to-frame completion
- Likely general or task-shaped: `general`

### 73. `9385bd28`
- What the task appears to require: match small partial motifs to larger reference panels and rewrite selected regions accordingly
- What `aria` currently has that is relevant: crop and panel extraction
- First failed layer: `binding`
- Missing representation: partial motif to reference panel correspondence
- Missing derivation: which reference governs each target rewrite
- Missing canonical operation: partial-to-reference transfer
- Smallest principled addition: partial motif matching with target transfer
- Likely general or task-shaped: `general-ish`

### 74. `97d7923e`
- What the task appears to require: summarize grouped vertical bars into a smaller canonical chart/sequence
- What `aria` currently has that is relevant: `OBJECT_REPACK` and crop
- First failed layer: `output`
- Missing representation: grouped bars with semantic grouping/order
- Missing derivation: summary statistic or retained subset per group
- Missing canonical operation: grouped bar-summary render
- Smallest principled addition: bar-group summarize and repack
- Likely general or task-shaped: `general-ish`

### 75. `981571dc`
- What the task appears to require: repair a globally symmetric pattern with masked/damaged regions
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`
- First failed layer: `execution`
- Missing representation: masked damage windows, not just one damage color
- Missing derivation: symmetry source selection when multiple candidate mirrors exist
- Missing canonical operation: mask-guided global symmetry reconstruction
- Smallest principled addition: extend symmetry repair to explicit damage masks
- Likely general or task-shaped: `general`

### 76. `9aaea919`
- What the task appears to require: transplant exemplar motifs from one part of the scene into matching target regions elsewhere
- What `aria` currently has that is relevant: crop, object detection, simple rewrite
- First failed layer: `binding`
- Missing representation: exemplar motifs and target holes/slots
- Missing derivation: which exemplar maps to which target region
- Missing canonical operation: exemplar motif transfer
- Smallest principled addition: motif-keyed transplant into matched regions
- Likely general or task-shaped: `general-ish`

### 77. `9bbf930d`
- What the task appears to require: transform striped/banded panels by inserting routed internal paths or overlays
- What `aria` currently has that is relevant: panel extraction and `PANEL_BOOLEAN`
- First failed layer: `execution`
- Missing representation: stripe bands plus overlaid routed path
- Missing derivation: route shape through the band structure
- Missing canonical operation: band-local path overlay
- Smallest principled addition: stripe/band routing overlay
- Likely general or task-shaped: `general-ish`

### 78. `a251c730`
- What the task appears to require: extract large bordered regions from patterned backgrounds and repack selected interiors into compact outputs
- What `aria` currently has that is relevant: frame extraction, crop, `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: multiple large framed regions with interior anomalies as ordered parts
- Missing derivation: which subregion is canonical output and how it is cropped/repacked
- Missing canonical operation: framed-interior extract and compact pack
- Smallest principled addition: large-frame interior selection plus repack
- Likely general or task-shaped: `general`

### 79. `a25697e4`
- What the task appears to require: compose a few sparse colored vectors/fragments into a canonical tiny output
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `execution`
- Missing representation: sparse fragments as vector-like directed parts
- Missing derivation: how parts combine or cancel into the output motif
- Missing canonical operation: sparse fragment composition
- Smallest principled addition: directed fragment compose-render
- Likely general or task-shaped: `general-ish`

### 80. `a32d8b75`
- What the task appears to require: use a side legend/control strip to rewrite a large textured field into a compact normalized result
- What `aria` currently has that is relevant: panel extraction, frame extraction, crop
- First failed layer: `binding`
- Missing representation: legend strip with typed entries and a separate query/workspace region
- Missing derivation: how legend entries drive the rewritten output texture
- Missing canonical operation: legend-strip decode and structured render
- Smallest principled addition: strip-legend decode over a target region
- Likely general or task-shaped: `general`

## Tasks 81–100

### 81. `a395ee82`
- What the task appears to require: repack a few sparse motifs into a canonical aligned arrangement based on their relative positions/colors
- What `aria` currently has that is relevant: object detection, `OBJECT_REPACK`, crop
- First failed layer: `output`
- Missing representation: sparse motifs as ordered parts of one composite arrangement
- Missing derivation: how relative positions determine the packed layout
- Missing canonical operation: sparse-part canonical assembly
- Smallest principled addition: relation-aware object pack/assemble
- Likely general or task-shaped: `general-ish`

### 82. `a47bf94d`
- What the task appears to require: route a rectilinear network through fixed junctions while placing colored control symbols at the correct graph positions
- What `aria` currently has that is relevant: crop and basic tracing
- First failed layer: `binding`
- Missing representation: graph nodes/junction slots plus control-symbol roles
- Missing derivation: which symbol belongs at which junction of the route
- Missing canonical operation: graph-slot placement over a routed skeleton
- Smallest principled addition: routed host-slot placement
- Likely general or task-shaped: `general`

### 83. `a6f40cea`
- What the task appears to require: resolve layered overlapping frames/boxes into a simpler canonical output using border priority and internal content
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `binding`
- Missing representation: overlapping hosts with z-order / priority roles
- Missing derivation: which overlap relation determines the output summary
- Missing canonical operation: layered host/slot composition
- Smallest principled addition: overlap-priority host-slot render
- Likely general or task-shaped: `general-ish`

### 84. `aa4ec2a5`
- What the task appears to require: detect blue objects and wrap/rewrite them into outlined framed versions on a new background
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `execution`
- Missing representation: object bbox plus canonical frame template
- Missing derivation: which objects get framed and how much padding/border
- Missing canonical operation: object-to-framed-template render
- Smallest principled addition: framed-object template render
- Likely general or task-shaped: `general`

### 85. `abc82100`
- What the task appears to require: decode border-coded control panels and place the resulting symbols into a sparse output scene
- What `aria` currently has that is relevant: panel extraction and crop
- First failed layer: `binding`
- Missing representation: control strips/border codes as typed roles
- Missing derivation: mapping border code to output symbol identity and placement
- Missing canonical operation: border-code decode and place
- Smallest principled addition: strip/border legend decode
- Likely general or task-shaped: `general-ish`

### 86. `b0039139`
- What the task appears to require: extract a small stencil from one region and tile/render it periodically with overlap/border-trim rules
- What `aria` currently has that is relevant: panel extraction and crop
- First failed layer: `binding`
- Missing representation: typed control bands for stencil, separator, and fill/background
- Missing derivation: how the extracted stencil is trimmed, overlapped, and repeated
- Missing canonical operation: periodic stencil render with overlap policy
- Smallest principled addition: periodic stencil decode/render
- Likely general or task-shaped: `task-shaped leaning`

### 87. `b10624e5`
- What the task appears to require: combine information across partition cells/quadrants to rewrite selected cells with transferred motifs
- What `aria` currently has that is relevant: partition/panel extraction and `PANEL_BOOLEAN`
- First failed layer: `binding`
- Missing representation: typed cell roles across the partition grid
- Missing derivation: which cell donates content to which other cell
- Missing canonical operation: partition-cell analogical rewrite
- Smallest principled addition: cell-grid registration and transfer
- Likely general or task-shaped: `general`

### 88. `b5ca7ac4`
- What the task appears to require: select framed colored blocks and rearrange or normalize them into a cleaner canonical multi-block output
- What `aria` currently has that is relevant: frame extraction and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: multiple framed objects with class/ordering roles
- Missing derivation: which framed blocks are retained and how they are laid out
- Missing canonical operation: framed-block repack
- Smallest principled addition: frame-aware object repack
- Likely general or task-shaped: `general-ish`

### 89. `b6f77b65`
- What the task appears to require: assemble several nested frame-like motifs into one canonical stacked composition
- What `aria` currently has that is relevant: frame extraction and crop
- First failed layer: `output`
- Missing representation: nested frame motifs as ordered assembly parts
- Missing derivation: the order and target alignment of the stacked output
- Missing canonical operation: nested-frame assembly render
- Smallest principled addition: multi-object stack/assemble
- Likely general or task-shaped: `general-ish`

### 90. `b99e7126`
- What the task appears to require: read repeated grid tiles, locate the anomalous or marked subset, and summarize/rewrite them in a compact result
- What `aria` currently has that is relevant: partition extraction and `PANEL_BOOLEAN`
- First failed layer: `binding`
- Missing representation: repeated cell grid with anomaly/selection semantics
- Missing derivation: which repeated tiles are special and how they project to the output
- Missing canonical operation: repeated-grid anomaly summary
- Smallest principled addition: tile-grid select-and-summarize
- Likely general or task-shaped: `general`

### 91. `b9e38dc0`
- What the task appears to require: grow or complete branched contours into filled canonical plant/funnel-like shapes
- What `aria` currently has that is relevant: tracing and simple fills only
- First failed layer: `execution`
- Missing representation: branch skeleton plus enclosed/fillable region
- Missing derivation: growth direction and fill boundary
- Missing canonical operation: contour growth plus interior fill
- Smallest principled addition: branch-guided shape completion
- Likely general or task-shaped: `general-ish`

### 92. `bf45cf4b`
- What the task appears to require: map small local motifs to a larger structured pattern through a codebook-like analogy
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `binding`
- Missing representation: exemplar motifs and target pattern slots
- Missing derivation: which small motif controls which part of the large pattern
- Missing canonical operation: motif analogy transfer
- Smallest principled addition: local codebook analogy render
- Likely general or task-shaped: `general-ish`

### 93. `c4d067a0`
- What the task appears to require: convert sparse color-count cues into grouped block layouts on a new canvas
- What `aria` currently has that is relevant: `OBJECT_REPACK` and crop
- First failed layer: `output`
- Missing representation: counted symbol groups as target block counts/positions
- Missing derivation: how many blocks of each color go to each cluster
- Missing canonical operation: count-to-block summary render
- Smallest principled addition: counted block assembly
- Likely general or task-shaped: `general`

### 94. `c7f57c3e`
- What the task appears to require: reassemble several small multicolor motifs into a larger canonical composite motif
- What `aria` currently has that is relevant: object detection and crop
- First failed layer: `output`
- Missing representation: component motifs as parts of one larger template
- Missing derivation: relative placement of each part in the canonical composite
- Missing canonical operation: multi-part motif assembly
- Smallest principled addition: canonical composite render from parts
- Likely general or task-shaped: `general-ish`

### 95. `cb2d8a2c`
- What the task appears to require: extend orthogonal corridors/bars from sparse seed lines and markers
- What `aria` currently has that is relevant: trace primitives only
- First failed layer: `execution`
- Missing representation: seed bars and target corridor axes
- Missing derivation: extension direction and stop conditions
- Missing canonical operation: orthogonal corridor extension/fill
- Smallest principled addition: axis-constrained corridor grow
- Likely general or task-shaped: `general`

### 96. `cbebaa4b`
- What the task appears to require: complete a routed orthogonal path network across several local examples
- What `aria` currently has that is relevant: tracing and crop
- First failed layer: `execution`
- Missing representation: partial path skeleton with divider constraints
- Missing derivation: route completion policy
- Missing canonical operation: orthogonal path completion
- Smallest principled addition: constrained path-network completion
- Likely general or task-shaped: `general`

### 97. `d35bdbdc`
- What the task appears to require: fill and regularize a large Y-like contour into a canonical colored region
- What `aria` currently has that is relevant: basic region fill and symmetry fragments
- First failed layer: `execution`
- Missing representation: contour arms plus filled interior target region
- Missing derivation: which contour is the boundary and which color fills where
- Missing canonical operation: contour-to-filled-shape render
- Smallest principled addition: contour closure plus region fill
- Likely general or task-shaped: `general-ish`

### 98. `d59b0160`
- What the task appears to require: clean noisy/damaged interior regions by propagating the dominant enclosing structure and removing intrusions
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR` and crop
- First failed layer: `execution`
- Missing representation: enclosing region and interior noise mask
- Missing derivation: which pixels are noise vs structure-preserving exceptions
- Missing canonical operation: bounded region cleanup/fill
- Smallest principled addition: enclosed-region denoise/majority repair
- Likely general or task-shaped: `general`

### 99. `d8e07eb2`
- What the task appears to require: decode multi-panel stripe/legend scenes and rewrite them into compact structured outputs
- What `aria` currently has that is relevant: panel extraction and `PANEL_BOOLEAN`
- First failed layer: `binding`
- Missing representation: typed panel roles for legend, query, and answer
- Missing derivation: how stripe/legend panels map to target output symbols
- Missing canonical operation: multi-panel legend decode
- Smallest principled addition: panel-role legend rewrite
- Likely general or task-shaped: `general`

### 100. `da515329`
- What the task appears to require: iteratively fill a spiral/cross path from a simple seed
- What `aria` currently has that is relevant: tracing only
- First failed layer: `execution`
- Missing representation: iterative fill state and turning rule
- Missing derivation: spiral step order and when to turn
- Missing canonical operation: spiral/cross iterative fill
- Smallest principled addition: stateful spiral fill primitive
- Likely general or task-shaped: `general`

## Tasks 101–117

### 101. `db0c5428`
- What the task appears to require: use one complete prototype motif and stamp completed copies at sparse anchor positions
- What `aria` currently has that is relevant: object detection, crop, basic transfer ideas
- First failed layer: `binding`
- Missing representation: prototype-vs-anchor roles plus anchor-local substitution rules
- Missing derivation: how the prototype aligns and whether anchor color changes interior cells
- Missing canonical operation: anchored template completion
- Smallest principled addition: prototype-guided template transfer to marked anchors
- Likely general or task-shaped: `general-ish`

### 102. `db695cfb`
- What the task appears to require: sparse diagonal interaction between aligned seeds/components with emitted diagonal effects
- What `aria` currently has that is relevant: trace/ray infrastructure and object detection
- First failed layer: `derivation`
- Missing representation: diagonally aligned seed pairs and their interaction frame
- Missing derivation: exact direction rule and how pair geometry changes the emitted diagonals
- Missing canonical operation: sparse diagonal interaction or pair-conditioned ray emission
- Smallest principled addition: diagonal pair interaction op with derived direction
- Likely general or task-shaped: `general`

### 103. `dbff022c`
- What the task appears to require: rewrite multicolor local motifs under a consistent structural transform while preserving the canvas
- What `aria` currently has that is relevant: object detection, crop, local rewrite fragments
- First failed layer: `execution`
- Missing representation: local motif classes independent of connected-component boundaries
- Missing derivation: which motif class maps to which transformed local arrangement
- Missing canonical operation: motif-conditioned local structural rewrite
- Smallest principled addition: small-pattern match and rewrite over a fixed canvas
- Likely general or task-shaped: `general-ish`

### 104. `dd6b8c4b`
- What the task appears to require: read simple geometric slash/corner arrangements and keep or remove parts according to a small directional rule
- What `aria` currently has that is relevant: crop and basic line reasoning
- First failed layer: `binding`
- Missing representation: diagonal/corner roles inside the tiny pattern
- Missing derivation: which slash or corner is the controlling feature for the rewrite
- Missing canonical operation: small geometric pattern selection/rewrite
- Smallest principled addition: tiny motif decode with orientation-based rewrite
- Likely general or task-shaped: `general-ish`

### 105. `de809cff`
- What the task appears to require: infer larger rectilinear/enclosing structure from sparse corner-like seeds and add the missing highlight cells
- What `aria` currently has that is relevant: trace fragments and simple frame detection
- First failed layer: `execution`
- Missing representation: sparse seeds as partial corners of a larger host structure
- Missing derivation: which enclosure/bridge is implied by the observed corner geometry
- Missing canonical operation: sparse corner-to-rectangle/bridge completion
- Smallest principled addition: inferred rectilinear enclosure render
- Likely general or task-shaped: `general`

### 106. `dfadab01`
- What the task appears to require: normalize nested colored structures by role, dropping some colors and recoloring others according to structural position
- What `aria` currently has that is relevant: object detection, crop, simple recolor
- First failed layer: `binding`
- Missing representation: nested part roles inside the multicolor structures
- Missing derivation: which roles survive, collapse, or change color in the output
- Missing canonical operation: role-conditioned structural recolor/simplify
- Smallest principled addition: hierarchical part-role rewrite
- Likely general or task-shaped: `general-ish`

### 107. `e12f9a14`
- What the task appears to require: complete or regularize symmetric cross/frame-like patterns while preserving large-scale layout
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`, crop, limited frame logic
- First failed layer: `execution`
- Missing representation: explicit symmetry scaffold and repeated arm/frame structure
- Missing derivation: how the local seed shapes extend across the larger symmetric host
- Missing canonical operation: scaffold-guided symmetry completion
- Smallest principled addition: structured symmetry completion beyond single-color damage repair
- Likely general or task-shaped: `general`

### 108. `e3721c99`
- What the task appears to require: remove nuisance/noise color from large structured scenes while keeping the meaningful colored geometry intact
- What `aria` currently has that is relevant: `SYMMETRY_REPAIR`, crop, region cleanup fragments
- First failed layer: `execution`
- Missing representation: nuisance mask vs structural colors inside a large canvas
- Missing derivation: which cells of the removable color are true noise and which are structural
- Missing canonical operation: bounded region denoise or intrusion removal
- Smallest principled addition: structure-preserving noise-color cleanup
- Likely general or task-shaped: `general`

### 109. `e376de54`
- What the task appears to require: transform striped/diagonal bordered shapes according to a consistent directional rule while preserving shape identity
- What `aria` currently has that is relevant: crop, trace, simple symmetry
- First failed layer: `derivation`
- Missing representation: stripe direction and shape orientation as first-class state
- Missing derivation: how the stripe orientation determines the output rewrite
- Missing canonical operation: direction-conditioned stripe/shape rewrite
- Smallest principled addition: oriented stripe-structure transform
- Likely general or task-shaped: `general-ish`

### 110. `e8686506`
- What the task appears to require: summarize detected component roles into a compact canonical output grid/strip
- What `aria` currently has that is relevant: object detection and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: component roles beyond simple size/order summaries
- Missing derivation: how roles determine row/column placement in the compact result
- Missing canonical operation: role-summary render
- Smallest principled addition: component-role summary pack
- Likely general or task-shaped: `general`

### 111. `e87109e9`
- What the task appears to require: read a large structured scene with margins/bands/panels and strip out the nonessential shell to preserve only the meaningful internal pattern
- What `aria` currently has that is relevant: panel extraction, frame extraction, crop
- First failed layer: `binding`
- Missing representation: shell/band roles vs interior-signal roles
- Missing derivation: which outer bands are discarded and how the interior is rebased
- Missing canonical operation: structured shell removal plus canonical crop
- Smallest principled addition: band/frame role selection and rebasing
- Likely general or task-shaped: `general-ish`

### 112. `edb79dae`
- What the task appears to require: extract several structured subregions from a large scene and compose a smaller canonical answer from the selected ones
- What `aria` currently has that is relevant: frame extraction, crop, `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: candidate subregions with stable selection and placement roles
- Missing derivation: which extracted regions appear in the answer and how they are arranged
- Missing canonical operation: selected-subregion extract and composite pack
- Smallest principled addition: multi-region select-and-assemble
- Likely general or task-shaped: `general`

### 113. `eee78d87`
- What the task appears to require: decode a tiny input tile into a much larger framed macro-pattern using a chosen fill mode
- What `aria` currently has that is relevant: basic tiling and crop only
- First failed layer: `execution`
- Missing representation: the small tile as a mode/program specification rather than literal content
- Missing derivation: which expansion/fill rule the input encodes
- Missing canonical operation: macro tile fill render
- Smallest principled addition: tile-to-macro expansion with mode selection
- Likely general or task-shaped: `general-ish`

### 114. `f560132c`
- What the task appears to require: select a subset of sparse objects from a large scene and normalize them into a compact square output
- What `aria` currently has that is relevant: object detection, crop, `OBJECT_REPACK`
- First failed layer: `binding`
- Missing representation: which objects are signal vs distractor and how they relate
- Missing derivation: selection rule and compact destination arrangement
- Missing canonical operation: relation-based object select-and-pack
- Smallest principled addition: filtered object pack by relational predicate
- Likely general or task-shaped: `general`

### 115. `f931b4a8`
- What the task appears to require: convert tiny colored arrangements into compact size-varying summary outputs driven by component structure rather than raw pixel counts
- What `aria` currently has that is relevant: object detection and `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: component structure/features that determine both output size and layout
- Missing derivation: which features map to which compact template
- Missing canonical operation: feature-conditioned compact summary render
- Smallest principled addition: component-feature summary templates
- Likely general or task-shaped: `general-ish`

### 116. `faa9f03d`
- What the task appears to require: remove or rewrite one color class in a repeated border/stripe scene according to a local structural rule
- What `aria` currently has that is relevant: simple recolor and crop
- First failed layer: `execution`
- Missing representation: local neighborhood/stripe rule that identifies the removable color
- Missing derivation: which occurrences of the disappearing color are structural exceptions vs deletions
- Missing canonical operation: local conditional delete/rewrite on repeated patterns
- Smallest principled addition: pattern-conditioned local rewrite
- Likely general or task-shaped: `general`

### 117. `fc7cae8d`
- What the task appears to require: extract, normalize, and repack salient interior structures from large framed scenes into smaller outputs with varying size
- What `aria` currently has that is relevant: frame extraction, crop, `OBJECT_REPACK`
- First failed layer: `output`
- Missing representation: interior structures with stable salience and packing order
- Missing derivation: which interior substructure becomes the answer and how it is compacted
- Missing canonical operation: salient interior extract-and-pack
- Smallest principled addition: frame-interior selection plus canonical repack
- Likely general or task-shaped: `general`

## Cross-cutting Notes

Across the full unsolved ledger:
- coarse perception is often good enough to expose panels, frames, objects, separators, and some large regions
- the dominant first breaks are usually `binding`, `execution`, and `output`
- repeated missing pieces include:
  - typed control/query/answer roles
  - prototype/anchor registration
  - legend and codebook decoding
  - route/grow/fill dynamics
  - structured extract-and-pack beyond the current `OBJECT_REPACK`
  - mask-guided or fold-style symmetry reconstruction
