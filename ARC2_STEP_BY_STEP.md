# ARC-2 Step-By-Step System

## Purpose

The current `aria` stack is still biased toward grid-to-grid transforms.
That is the wrong center of gravity for ARC-2.

This note uses the first 10 public ARC-2 training tasks as the seed corpus
for a new system whose default representation is:

- explicit output specification before rendering
- scene entities
- roles
- correspondences
- typed actions over entities
- render to pixels only at the end

Scope here:

- first 10 public-v2 training tasks in lexical order
- identify the human step program for each
- extract recurring motifs
- define the initial typed IR and execution loop

Task list:

1. `00576224`
2. `007bbfb7`
3. `009d5c81`
4. `00d62c1b`
5. `00dbd492`
6. `017c7c7b`
7. `025d127b`
8. `03560426`
9. `045e512c`
10. `0520fde7`

## First 10 Tasks As Scene Programs

| Task | Human step program | Needed entities / relations | Existing family name sanity check |
|---|---|---|---|
| `00576224` | Extract the 2x2 template. Infer a 3x3 tile layout. Place identity copies on rows 0 and 2 of the tile grid, and left-right flipped copies on row 1. Render the 6x6 canvas. | `Template`, `TileLayout`, `Transform(I/FLR)`, `Canvas` | `tile_transform` |
| `007bbfb7` | Treat each non-zero input cell as a marker. For each marker, stamp the full 3x3 input template into the corresponding 3x3 slot of the output meta-grid. Background cells produce empty slots. | `Template`, `MarkerCells`, `SlotGrid`, `Stamp` | `self_template_stamp` |
| `009d5c81` | Parse two objects: a large target shape and a small reference glyph. Normalize the small glyph into a shape key. Look up the output color from that key. Recolor the large object with that color and erase the reference glyph. | `TargetObject`, `ReferenceObject`, `ShapeKey`, `ColorLookup` | `reference_transfer` |
| `00d62c1b` | Detect color-3 boundaries. Find enclosed interior regions under 8-connectivity. Fill only the enclosed holes with color 4. Leave open areas unchanged. | `Boundary`, `InteriorRegion`, `EnclosedBy` | `enclosed_region_fill` |
| `00dbd492` | Detect separate outlined regions. For each closed outline, fill its interior with a color determined by the outline's local structure / box type. Preserve outline pixels and holes. | `OutlinedRegion`, `InteriorRegion`, `RegionType`, `RegionColorMap` | `fill_enclosed_outline` |
| `017c7c7b` | Treat the 6x3 input as a vertically periodic stripe pattern. Recolor foreground `1 -> 2`. Extend the pattern downward to height 9 by continuing the same period. | `Panel`, `VerticalPeriod`, `ColorMap`, `ExtendAxis` | inference fallback |
| `025d127b` | Detect each slanted / sheared component. Straighten it into a canonical axis-aligned form while keeping color and size. Re-embed it in the same grid. | `Object`, `ShearAxis`, `CanonicalForm`, `Reembed` | `ShearStraightenMacro` |
| `03560426` | Detect disconnected blocks. Sort them in chain order from the bottom-left source arrangement. Sequentially place them from the top-left so each next block touches the previous one at a corner. | `ObjectSet`, `OrderedChain`, `CornerAttachment` | `ordered_chain_layout` |
| `045e512c` | Detect prototype objects and directional marker cues. Infer one or more propagation directions per prototype. Clone the prototype repeatedly along its permitted axis/trajectory while preserving its local template. | `Prototype`, `DirectionCue`, `CloneSeries`, `PlacementPath` | `directional_clone_layout` |
| `0520fde7` | Split the input into two 3x3 panels using the separator column. Compute the cellwise intersection of foreground occupancy between left and right panels. Emit a 3x3 output with color 2 at intersecting foreground cells. | `Separator`, `PanelPair`, `BooleanCombine(intersect)` | `separator_grid_boolean` |

## Recurring Motifs In The First 10

These ten tasks are already enough to reject a pixel-transform-first ontology.

### 1. Template expansion

- `00576224`
- `007bbfb7`
- `045e512c`

Common structure:

- detect a prototype template
- infer a placement layout
- stamp transformed copies into slots

### 2. Reference-driven rewrite

- `009d5c81`

Common structure:

- parse target object vs reference object
- extract a key from the reference
- use the key to choose an action on the target

### 3. Region semantics

- `00d62c1b`
- `00dbd492`

Common structure:

- boundaries matter more than raw pixels
- fill behavior depends on closedness and local region type

### 4. Layout / array / order reasoning

- `03560426`
- `045e512c`

Common structure:

- infer ordering or directional slots
- place objects according to that inferred structure

### 5. Panel algebra

- `0520fde7`

Common structure:

- split scene into panels
- run typed boolean / relational logic between panels

### 6. Canonicalization before action

- `017c7c7b`
- `025d127b`

Common structure:

- normalize or extend a structured object/pattern
- then render the canonicalized result

## Initial ARC-2 IR

The new system should reason over scene structures first.
Pixels are only the terminal rendering target.

Before any scene action, the program must commit to:

1. output grid size
2. output background

Those are not implementation details. They are the first two semantic
decisions in the program.

### Output specification

- `OutputGridSpec`
  - `shape`
  - `background`

### Core entities

- `Scene`
- `Panel`
- `Object`
- `Template`
- `Boundary`
- `InteriorRegion`
- `Separator`
- `Slot`
- `SlotGrid`
- `OrderedChain`
- `CloneSeries`
- `ReferenceObject`
- `LegendMap`
- `ShapeKey`
- `CanonicalForm`

### Core relations

- `contains`
- `adjacent_to`
- `encloses`
- `corresponds_to`
- `indexed_by`
- `ordered_before`
- `copies_from`
- `uses_key`
- `aligned_with`
- `extends_along`

### Core actions

- `infer_output_size`
- `infer_output_background`
- `initialize_output_scene`
- `extract_template`
- `infer_tile_layout`
- `stamp_template`
- `extract_reference_key`
- `lookup_color`
- `recolor_object`
- `detect_closed_boundaries`
- `fill_enclosed_regions`
- `split_by_separator`
- `boolean_combine_panels`
- `canonicalize_object`
- `extend_periodic_pattern`
- `order_objects_into_chain`
- `place_chain`
- `clone_along_path`
- `render_scene`

## New Execution Loop

The system should default to this loop:

1. `infer_output_size`
   - same as input, crop, upscale, tile-grid, panel-derived, chain-derived
2. `infer_output_background`
   - preserve input background, adopt separator/background color, explicit new bg
3. `initialize_output_scene`
   - create an empty output scene with committed shape + background
4. `parse_scene`
   - detect panels, objects, boundaries, separators, repeated slots
5. `assign_roles`
   - which entities are templates, references, legends, targets, boundaries
6. `infer_structure`
   - arrays, chains, correspondences, panel pairings, region types
7. `synthesize_step_program`
   - a short typed plan over scene entities
8. `execute_scene_program`
   - mutate scene structures on top of the initialized output scene
9. `render_scene`
   - convert the final scene state back to a grid
10. `verify_exact`

## What The New System Should Stop Doing By Default

Do not start same-size ARC-2 tasks by ranking pixel/grid transforms.

For ARC-2, the default question should be:

- what is the output size?
- what is the output background?
- what are the entities?
- what roles do they play?
- what is the correspondence?
- what action is being applied to those entities?

Only after that should execution produce pixels.

## First Implementation Slice

Implement the smallest typed scene-program slice that directly covers several
of the first 10 tasks:

1. `Template + SlotGrid + Stamp`
   - covers `00576224`, `007bbfb7`
2. `ReferenceObject + ShapeKey + RecolorTarget`
   - covers `009d5c81`
3. `Boundary + InteriorRegion + Fill`
   - covers `00d62c1b`, `00dbd492`
4. `Separator + PanelPair + BooleanCombine`
   - covers `0520fde7`
5. `OrderedChain + PlaceChain`
   - covers `03560426`

`045e512c` and `025d127b` should come after the first slice:

- `045e512c` needs richer directional path inference
- `025d127b` needs canonicalization of sheared objects

## Practical Consequence

The new ARC-2 system should be a scene-program system, not a transform
catalog with better ranking.

That means:

- old lane machinery becomes fallback / baseline / verifier support
- new work starts from typed scene IR
- task solving becomes short step programs over entities and relations
