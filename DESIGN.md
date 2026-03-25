# ARIA: Abstract Reasoning via Inductive Abstraction

## System design v3

---

## Thesis

ARIA is a step machine. It extracts a typed state graph from the input, then a
learned proposer emits a multi-step program of typed bindings that can analyze the
input, compute output dimensions, construct new grids, and read cross-demo context
— all within a total, deterministic, exactly-verifiable runtime. Verified programs
are decomposed and their reusable sub-sequences are admitted to a growing library,
expanding the machine's vocabulary over time.

---

## What this is

Five components, one loop:

1. Object-centric state graph extraction (deterministic, no learning)
2. Typed step-machine runtime (fixed registry, currently 126 ops)
3. Learned proposer (fine-tuned ~4B code model emitting typed step programs)
4. Exact verification with three modes (stateless, leave-one-out, sequential)
5. Growing abstraction library (verified sub-programs admitted via MDL gate)

The proposer is the only neural component. Everything else is symbolic,
deterministic, and inspectable.

## What this is not

- Not an ensemble of independent solvers that vote at the end
- Not an LLM wrapper with a refinement harness
- Not a neural transduction model that predicts pixels directly
- Not a brute-force search over arbitrary code
- No "fallback to Python." Every solution is a typed, verifiable, inspectable
  program in the step language. If the runtime can't express it, the system
  fails honestly rather than escaping to untyped code.

---

## Architecture

```
Grid Pairs ──→ [1. State Graph Extraction] ──→ State Graphs + Deltas
                                                      │
                                      ┌───────────────┘
                                      ▼
                              [2. Learned Proposer]  ←── Library index
                                      │
                                      ▼
                          [Parser + Runtime Checks]
                                      │  parse / exec error → re-propose
                                      ▼
                              [3. Runtime Execution]
                                      │
                                      ▼
                              [4. Exact Verification]
                                     / \
                               pass /   \ fail (with structured error)
                                   /     \
                                  ▼       └──→ re-propose (max 4 rounds)
                            Submit output
                                  │
                                  ▼
                        [5. Abstraction Mining]
                                  │
                                  ▼
                        [Library Admission Gate]
```

---

## 1. State graph extraction

### Purpose

Convert every raw ARC grid (2D array of integers 0-9) into a typed, structured
representation. This is deterministic — no learning, no model. It is a compiler
front-end.

### Algorithms

All classical, all deterministic:

- Connected-component labeling (4-connected and 8-connected, both computed)
- Background color detection (most frequent color; border color heuristic)
- Shape classification (bounding box aspect ratio + mask pattern matching)
- Per-object symmetry detection (rotational, reflective)
- Global symmetry detection (tiling, periodicity, axis symmetries)
- Pairwise spatial relation computation (adjacency, containment, alignment)
- Multi-resolution parsing: each grid is parsed at pixel level, connected-component
  level, and sub-grid level (if tiling is detected). The proposer sees all parses.

### Output types

```
ObjectNode {
  id         : Int
  color      : Color(0..9)
  mask       : BitGrid              -- exact pixel mask within bbox
  bbox       : (x, y, w, h)
  shape      : Rect | Line | L | T | Cross | Dot | Irregular
  symmetry   : Set(Rot90 | Rot180 | ReflH | ReflV | ReflD)
  size       : Int                  -- pixel count
}

RelationEdge {
  src, dst   : ObjectNode.id
  spatial    : Set(Above | Below | Left | Right | Diagonal)
  topo       : Set(Adjacent | Contains | Overlaps | Disjoint)
  align      : Set(AlignH | AlignV | AlignDiag)
  match      : Set(SameColor | SameShape | SameSize | Mirror)
}

GridContext {
  dims       : (rows, cols)
  bg_color   : Color
  is_tiled   : Maybe(tile_rows, tile_cols)
  symmetry   : Set(GlobalRot | GlobalRefl | Periodic)
  palette    : Set(Color)
  obj_count  : Int
}

Delta {                             -- computed per demo pair
  added      : List(ObjectNode)
  removed    : List(ObjectNode.id)
  modified   : List(ObjectNode.id, field, old_val, new_val)
  dims_changed : Maybe(old_dims, new_dims)
}
```

### Design decisions

Immutable. Nothing downstream writes to the state graph. It is the fixed input to
all downstream components.

Multi-parse. Some grids don't have obvious objects (full-grid patterns, gradients).
The extractor produces multiple parses at different granularities. The proposer
sees all of them and selects which level of abstraction to reason at.

The delta is the primary signal. The proposer's job is: "explain the delta as a
program." What changed between input and output, and what sequence of typed steps
would produce that change?

Graceful fallback for objectless grids. When connected-component labeling finds
only one component (the entire grid), the extractor falls back to sub-grid
decomposition, row/column analysis, and per-cell property computation. The state
graph always has content; it's just at a different granularity.

---

## 2. Typed step-machine runtime

### Purpose

Define the implemented ARIA DSL and runtime in `aria.runtime`. The language is
step-based, deterministic, side-effect free, and backed by a fixed op registry.
The current implementation has a typed AST, a static checker module, and an
executor/verifier path that runs programs against ARC demos.

### Program model

A program is a sequence of steps plus the name of the binding to yield. The
implemented AST is:

```
Program {
  steps   : tuple[Step, ...]
  output  : str
}

Step =
  | Bind(name: str, typ: Type, expr: Expr)
  | Assert(pred: Expr)

Expr =
  | Ref(name: str)
  | Literal(value: Any, typ: Type)
  | Call(op: str, args: tuple[Expr, ...])
  | Lambda(param: str, param_type: Type, body: Expr)
```

Execution in `aria.runtime.executor.execute` is:

```
env = { "input": input_grid }
if ctx is not None:
  env["ctx"] = ctx

for step in program.steps:
  match step:
    case Bind(name, _, expr):
      env[name] = eval_expr(expr, env)
    case Assert(pred):
      if not eval_expr(pred, env):
        FAIL

return env[program.output]
```

Key implementation notes:
- Declared types live on `Bind`, but the executor does not re-check them at runtime.
- Static checking is handled by `aria.runtime.type_system.type_check`.
- The executor supports calling bound callable values and partial application of ops.
- The AST and parser both support `Lambda`. Multi-arg lambda syntax is lowered to nested single-arg lambdas internally.

### Surface syntax

The current proposer parser in `aria.proposer.parser` accepts:

```text
bind <name> : <TYPE> = <expr>
bind <name> = <expr>
assert <expr>
yield <name>

|x: TYPE| <expr>
|x: TYPE, y: TYPE, z: TYPE| <expr>
```

Current parsing behavior:
- `bind name = expr` is allowed; the parser marks the binding as untyped and `type_check` infers its type from the expression.
- Integer literals, booleans, enum literals, tuple literals, and `{src: dst}` color-map literals are supported.
- Infix `+`, `-`, and `*` are lowered to `add`, `sub`, and `mul`.
- Lambda syntax is supported directly in expressions. Multi-parameter lambdas are parsed as nested curried lambdas.
- Tuple literals with dynamic members are lowered to `make_tuple(...)`.
- `program_to_text` serializes the same AST using `let ...` and `-> output`.

### Why multi-step matters

Many ARC tasks cannot be expressed as a single expression `Grid -> Grid`:

(A) Computed output dimensions. The output grid's size depends on analysis of
    the input (e.g., count objects, then create an N×1 grid). The output grid
    doesn't exist until an intermediate computation produces the dimensions.

(B) Input encodes instructions + data. A single input grid contains both a
    "rule region" (e.g., a color mapping table) and a "data region." The program
    must decompose the input into semantic zones, interpret one zone as
    instructions, then apply those instructions to another zone.

(C) Iterative construction. The rule is "extend this pattern one level" or
    "continue this growth." Building the output requires detecting the current
    state, computing the next iteration, and placing it, each step depending on
    results from prior steps.

(D) Cell-level and neighborhood reasoning. Some tasks are easiest to express as
    local transforms, neighborhood counts, pattern replacement, or symmetry
    completion rather than object-only rewrites.

(E) Cross-demo reasoning. The transformation depends on information that can
    only be recovered by reading across multiple demo pairs: shared codebooks,
    progression patterns, disambiguation of predicates.

### Type system

The implemented `Type` enum is:

```
GRID
OBJECT
OBJECT_SET
OBJECT_LIST
COLOR
INT
BOOL
DIMS
DIR
AXIS
SHAPE
PROPERTY
COLOR_MAP
INT_LIST
REGION
ZONE
ZONE_LIST
PAIR
TASK_CTX
SORT_DIR
SIZE_RANK
ZONE_ROLE
PREDICATE
OBJ_TRANSFORM
GRID_TRANSFORM
CALLABLE
```

Important implementation details:
- `INT` and `COLOR` are assignment-compatible in the static checker.
- `CALLABLE` accepts `PREDICATE`, `OBJ_TRANSFORM`, and `GRID_TRANSFORM`.
- Multi-arg callable syntax is represented internally as nested `Lambda` nodes and applied curried at runtime.
- `DIMS` is used both for `(rows, cols)` and for position tuples passed to ops like `place_at`, `embed`, and `flood_fill`.
- `REGION` is currently overloaded: some ops expect rectangular `(x, y, w, h)` tuples, while others return sets of absolute `(row, col)` cells.
- `OBJECT_LIST` is not parameterized; it is used for ordered object lists and some list-shaped grouped values.
- `PAIR` is a generic tuple slot, not a first-class parameterized pair type in the checker.

### Operation catalog (current implementation)

Source of truth: the runtime registry loaded by `import aria.runtime`. The
current core registry has 126 ops.

#### Selection / object query

```
find_objects(grid: GRID) -> OBJECT_SET
by_color(color: COLOR) -> PREDICATE
by_shape(shape: SHAPE) -> PREDICATE
by_size_rank(rank: INT, objects: OBJECT_SET) -> OBJECT
where(pred: PREDICATE, objects: OBJECT_SET) -> OBJECT_SET
excluding(to_remove: OBJECT_SET, from_set: OBJECT_SET) -> OBJECT_SET
related_to(obj: OBJECT, rel_type: INT) -> OBJECT_SET
background_obj(grid: GRID) -> OBJECT
singleton(objects: OBJECT_SET) -> OBJECT
nth(idx: INT, obj_list: OBJECT_SET) -> OBJECT
```

#### Object transforms

```
translate(dir: DIR, amount: INT, obj: OBJECT) -> OBJECT
rotate(degrees: INT, obj: OBJECT) -> OBJECT
reflect(axis: AXIS, obj: OBJECT) -> OBJECT
resize_obj(factor: INT, obj: OBJECT) -> OBJECT
recolor(color: COLOR, obj: OBJECT) -> OBJECT
gravity(dir: DIR, obj: OBJECT) -> OBJECT
extend(dir: DIR, amount: INT, obj: OBJECT) -> OBJECT
```

#### Grid construction / composition

```
new_grid(dims: DIMS, color: COLOR) -> GRID
from_object(obj: OBJECT) -> GRID
crop(grid: GRID, region: REGION) -> GRID
place_at(obj: OBJECT, pos: DIMS, grid: GRID) -> GRID
fill_region(region: REGION, color: COLOR, grid: GRID) -> GRID
embed(small: GRID, large: GRID, pos: DIMS) -> GRID
stack_h(a: GRID, b: GRID) -> GRID
stack_v(a: GRID, b: GRID) -> GRID
overlay(top: GRID, bottom: GRID) -> GRID
apply_color_map(mapping: COLOR_MAP, grid: GRID) -> GRID
fill_cells(grid: GRID, values: INT_LIST) -> GRID
```

#### Grid transforms

```
rotate_grid(degrees: INT, grid: GRID) -> GRID
reflect_grid(axis: AXIS, grid: GRID) -> GRID
transpose_grid(grid: GRID) -> GRID
tile_grid(grid: GRID, rows: INT, cols: INT) -> GRID
upscale_grid(grid: GRID, factor: INT) -> GRID
fill_enclosed(grid: GRID, fill_color: COLOR) -> GRID
```

#### Dimensions

```
dims_of(grid: GRID) -> DIMS
dims_make(r: INT, c: INT) -> DIMS
scale_dims(dims: DIMS, factor: INT) -> DIMS
obj_dims(obj: OBJECT) -> DIMS
rows_of(dims: DIMS) -> INT
cols_of(dims: DIMS) -> INT
```

#### Analysis / decomposition

```
find_zones(grid: GRID) -> ZONE_LIST
zone_by_role(zones: ZONE_LIST, role: ZONE_ROLE) -> ZONE
zone_to_grid(zone: ZONE) -> GRID
extract_map(zone: ZONE) -> COLOR_MAP
count(objects: OBJECT_SET) -> INT
length(obj_list: OBJECT_LIST) -> INT
group_by(prop: PROPERTY, objects: OBJECT_SET) -> OBJECT_LIST
sort_by(prop: PROPERTY, direction: SORT_DIR, objects: OBJECT_SET) -> OBJECT_LIST
unique_colors(objects: OBJECT_SET) -> INT_LIST
max_val(ints: INT_LIST) -> INT
min_val(ints: INT_LIST) -> INT
```

#### Arithmetic / logic / accessors / list utilities

```
make_tuple() -> PAIR
add(a: INT, b: INT) -> INT
sub(a: INT, b: INT) -> INT
mul(a: INT, b: INT) -> INT
div(a: INT, b: INT) -> INT
mod(a: INT, b: INT) -> INT
isqrt(n: INT) -> INT
abs(n: INT) -> INT
eq(a: INT, b: INT) -> BOOL
neq(a: INT, b: INT) -> BOOL
lt(a: INT, b: INT) -> BOOL
gt(a: INT, b: INT) -> BOOL
lte(a: INT, b: INT) -> BOOL
gte(a: INT, b: INT) -> BOOL
and(a: BOOL, b: BOOL) -> BOOL
or(a: BOOL, b: BOOL) -> BOOL
not(a: BOOL) -> BOOL
get_color(obj: OBJECT) -> COLOR
get_size(obj: OBJECT) -> INT
get_shape(obj: OBJECT) -> SHAPE
get_pos_x(obj: OBJECT) -> INT
get_pos_y(obj: OBJECT) -> INT
get_width(obj: OBJECT) -> INT
get_height(obj: OBJECT) -> INT
sum_list(ints: INT_LIST) -> INT
range_list(n: INT) -> INT_LIST
reverse_list(lst: INT_LIST) -> INT_LIST
index_of(val: INT, lst: INT_LIST) -> INT
contains(val: INT, lst: INT_LIST) -> BOOL
at(idx: INT, lst: INT_LIST) -> INT
```

#### Higher-order / control flow

```
compose(f: CALLABLE, g: CALLABLE) -> CALLABLE
map_obj(f: OBJ_TRANSFORM, objects: OBJECT_SET) -> OBJECT_SET
map_list(f: CALLABLE, lst: OBJECT_LIST) -> OBJECT_LIST
fold(f: CALLABLE, init: GRID, lst: OBJECT_LIST) -> GRID
if_then_else(cond: BOOL, a: GRID, b: GRID) -> GRID
repeat_apply(n: INT, f: GRID_TRANSFORM, grid: GRID) -> GRID
for_each_place(objects: OBJECT_SET, pos_fn: CALLABLE, grid: GRID) -> GRID
```

#### Topological

```
flood_fill(grid: GRID, pos: DIMS, color: COLOR) -> GRID
boundary(obj: OBJECT) -> REGION
interior(obj: OBJECT) -> REGION
connect(obj1: OBJECT, obj2: OBJECT) -> REGION
hull(obj: OBJECT) -> OBJECT
connected_components(grid: GRID) -> OBJECT_SET
```

#### Cell-level

```
cell_map(grid: GRID, fn: CALLABLE) -> GRID
neighbor_map(grid: GRID, fn: CALLABLE) -> GRID
neighbor_map_8(grid: GRID, fn: CALLABLE) -> GRID
conditional_fill(grid: GRID, color: COLOR, target: COLOR) -> GRID
fill_where_neighbor_count(grid: GRID, neighbor_color: COLOR, min_count: INT, fill_color: COLOR) -> GRID
fill_between(grid: GRID, color: COLOR, fill_color: COLOR) -> GRID
propagate(grid: GRID, source_color: COLOR, fill_color: COLOR, bg_color: COLOR) -> GRID
```

#### Pattern matching / symmetry / boolean grid ops

```
find_pattern(grid: GRID, pattern: GRID) -> INT_LIST
replace_pattern(grid: GRID, pattern: GRID, replacement: GRID) -> GRID
complete_symmetry_h(grid: GRID) -> GRID
complete_symmetry_v(grid: GRID) -> GRID
grid_and(a: GRID, b: GRID) -> GRID
grid_or(a: GRID, b: GRID) -> GRID
grid_xor(a: GRID, b: GRID) -> GRID
grid_diff(a: GRID, b: GRID) -> GRID
```

#### Row / column / color statistics

```
most_common_color(grid: GRID) -> COLOR
count_color(grid: GRID, color: COLOR) -> INT
get_row(grid: GRID, idx: INT) -> GRID
get_col(grid: GRID, idx: INT) -> GRID
set_row(grid: GRID, idx: INT, row: GRID) -> GRID
set_col(grid: GRID, idx: INT, col: GRID) -> GRID
sort_rows(grid: GRID) -> GRID
sort_cols(grid: GRID) -> GRID
unique_rows(grid: GRID) -> GRID
unique_cols(grid: GRID) -> GRID
```

#### Cross-demo context

```
demo_count(ctx: TASK_CTX) -> INT
demo_at(ctx: TASK_CTX, idx: INT) -> PAIR
infer_step(ctx: TASK_CTX) -> GRID_TRANSFORM
infer_map(ctx: TASK_CTX, prop_in: INT, prop_out: INT) -> COLOR_MAP
disambiguate(ctx: TASK_CTX, predicates: OBJECT_LIST) -> PREDICATE
infer_iteration(ctx: TASK_CTX, grid: GRID) -> INT
predict_dims(ctx: TASK_CTX, grid: GRID) -> DIMS
```

### Constraints

- No arbitrary code execution. Calls are resolved through the op registry or a bound callable in the environment.
- No mutation of previously bound names. Execution only extends `env`.
- No recursion in the current text syntax. Iteration happens through bounded ops like `repeat_apply`, `map_obj`, `map_list`, and `fold`.
- Execution is pure from the program's perspective: ops return new values rather than mutating prior bindings.
- Static type checking is now part of the proposer loop before execution. The current flow is parse -> type_check -> execute -> verify.
- Some runtime representations are intentionally coarse today (`REGION`, `PAIR`, `OBJECT_LIST`, partial application, callable slots). The registry is the truth for what the executor will accept.
- Library abstractions are admitted as new named operations with the same registry-based calling convention. They extend the catalog, not the AST.

---

## 3. Learned proposer

### Purpose

The only neural component. Given the state graph, deltas, and library index,
emit candidate programs in the step language. Its contract is narrow:

- Input: serialized state graph + deltas + available operations (core + library)
- Output: a sequence of typed Bind/Assert/Yield steps in the runtime's syntax
- NOT Python. NOT natural language. A small typed language that can be checked
  in milliseconds.

### Architecture

A ~4B parameter code-generation model (e.g., fine-tuned Qwen or CodeLlama).
Not a frontier model. The proposer doesn't need to know about the world — it
needs to be good at emitting 3-12 lines of typed code against the current DSL
surface and op registry.

### Input format

The proposer receives a structured prompt containing:

```
[STATE_GRAPH]
  objects: [{id:0, color:3, shape:Rect, bbox:(1,1,3,3), size:9}, ...]
  relations: [{src:0, dst:1, spatial:{Right}, topo:{Adjacent}}, ...]
  context: {dims:(10,10), bg_color:0, obj_count:4}

[DELTAS]
  demo_0: {added:[], removed:[], modified:[{id:0, field:color, 3→7}]}
  demo_1: {added:[{id:5, ...}], removed:[{id:2}], modified:[]}
  ...

[LIBRARY]
  core: [find_objects, by_color, where, translate, rotate, ...]
  learned: [mirror_and_swap, fill_interior, complete_band, ...]

[PRIOR_ATTEMPTS]  (rounds 2+)
  attempt_1: {program: "...", failed_demo: 2, error: "dims mismatch (3,3) vs (4,4)"}
  ...
```

### Output format

The proposer currently emits `bind` / `assert` / `yield` text and the parser
turns that into the AST above. A minimal valid program looks like:

```
bind objects : OBJECT_SET = find_objects(input)
bind largest : OBJECT = by_size_rank(0, objects)
bind result : GRID = from_object(largest)
yield result
```

In the current implementation:
- Type annotations on `bind` are optional in parser input.
- Dict literals like `{1: 3, 2: 7}` parse as `COLOR_MAP`.
- Arithmetic syntax `a + b`, `a - b`, `a * b` lowers to `add`, `sub`, `mul`.
- Lambda syntax is available directly, for example `|obj: OBJECT| get_color(obj)` or `|row: INT, col: INT, val: INT| add(val, 1)`.
- The serializer used in corpus/reporting prints `let ...` and `-> output` instead of `bind ...` and `yield ...`.

### Type checking

`aria.runtime.type_system.type_check` exists and walks the program against the
same op registry the executor uses. It:

1. Resolves each op signature from the registry
2. Infers expression types from the current environment
3. Checks compatibility, including `INT`/`COLOR` interchangeability and callable subtypes
4. Verifies the final output binding exists

Current implementation note: the proposer harness now rejects candidates through
`type_check` before execution. The loop is parse -> type_check -> execute ->
verify, and all three failure modes are fed back to re-proposal.

### Generation strategy

The proposer generates k=32 candidate programs per round using temperature
sampling. Diversity matters more than precision at this stage — verification
will filter. Programs are generated in parallel for efficiency.

### Training

Training data: (state_graph, deltas, correct_program) triples from:

1. ARC training tasks. For each task, search for correct multi-step programs by
   enumerating step sequences up to length 12, using the runtime as a verifier.
   Keep all distinct correct programs per task (there are often multiple).

2. Synthetic tasks. Sample random step sequences of length 3-10 from the
   operation catalog. Execute them on random input grids to produce input/output
   pairs. This gives unlimited training data where the correct program is known
   by construction.

3. Difficulty calibration. Weight training toward programs of length 5-10
   (the hardest to propose). Filter synthetic tasks to require at least 2
   non-trivial operations (not just identity or single recolor).

Fine-tuning objective: standard next-token prediction on the program text,
conditioned on the state graph + delta representation.

### Error-conditioned re-proposal

On verification failure, the proposer receives structured feedback:

```
[PRIOR_ATTEMPTS]
  attempt_1:
    program: "bind objects = ... yield result"
    verification_mode: stateless
    failed_demo: 2
    error_type: wrong_output
    diff: {
      expected_dims: (4, 4)
      actual_dims: (3, 3)
      pixel_diff_count: 7
      pixel_diff_summary: "row 3 entirely wrong, cols 2-3 wrong in rows 1-2"
    }
    step_trace: [
      {step: "bind objects", value: "[obj0, obj1, obj2]", ok: true},
      {step: "bind n", value: "3", ok: true},
      {step: "bind out_dims", value: "(3,3)", SUSPECT: "expected (4,4)"}
    ]
```

The step trace is critical. It tells the proposer exactly where the computation
diverged from the expected result, so it can fix specific steps rather than
rewriting from scratch.

---

## 4. Exact verification

### Purpose

The one hard gate. Binary pass/fail, no soft scoring, no partial credit.
Every candidate program is executed against the demo pairs. If the output
doesn't match pixel-for-pixel, the program is rejected.

### Three verification modes

Programs fall into categories based on whether and how they read cross-demo
context. The type checker can determine the mode by static analysis: if the
program contains any `ctx`-reading operations, it's context-dependent.

#### Mode A: Stateless

The program does not read TaskContext. Each demo pair is verified independently.

```
verify_stateless(P, demos):
  for (in_grid, out_grid) in demos:
    if execute(P, in_grid, empty_ctx) != out_grid: FAIL
  PASS
```

Cheapest mode. No context leakage risk. Estimated ~40% of ARC tasks.

#### Mode B: Context-reading (leave-one-out)

The program reads TaskContext (for codebook extraction, disambiguation, etc.).
When verifying demo i, the context contains all demos EXCEPT demo i.

```
verify_loo(P, demos):
  for i in 0..len(demos):
    ctx_i = { demos: demos[:i] + demos[i+1:] }
    if execute(P, demos[i].in, ctx_i) != demos[i].out: FAIL
  PASS
```

Prevents the program from trivially looking up the answer. If the program
produces correct output on held-out pairs, it has genuinely learned the rule.
Estimated ~40% of ARC tasks.

#### Mode C: Sequential / progression

Demos are ordered steps in a sequence, not independent samples of the same rule.
The program sees all prior demos as history.

```
verify_sequential(P, demos):
  for i in 1..len(demos):
    history = demos[:i]
    if execute(P, demos[i].in, { demos: history }) != demos[i].out: FAIL
  PASS
```

The program must predict each step given only prior history. Handles progression
tasks, growth patterns, and sequence-continuation problems.
Estimated ~20% of ARC tasks.

#### Mode selection

The type checker determines the mode by static analysis:
- Program has no `ctx`-reading operations → Mode A
- Program reads `ctx` and task appears unordered → Mode B
- Program reads `ctx` with sequential operations (infer_step, infer_iteration) → Mode C

As a safety net, if a program passes ANY applicable mode, it is accepted.
This handles edge cases where the task structure is ambiguous.

### Verification invariants

Across all three modes:
- Deterministic: same program + same inputs = same output, always
- Pixel-perfect: every cell in the output grid must match exactly
- No lookup cheating: the program never sees the expected output for the pair
  it's being tested on
- Millisecond-scale: programs are small (3-12 steps), execution is fast

### At test time

When applying a verified program to the test input:
- Mode A: execute(P, test_input, empty_ctx)
- Mode B: execute(P, test_input, { demos: all_demos })
- Mode C: execute(P, test_input, { demos: all_demos })

The test input is novel — the program has never seen it. If the program truly
captured the rule (rather than memorizing the demos), it will produce the
correct output.

---

## 5. Growing abstraction library

### Purpose

Verified programs are decomposed, and reusable sub-sequences are admitted
into the library as new named operations, extending the runtime's vocabulary.
This is the mechanism by which ARIA builds up domain knowledge over time.

### How it works

#### Step 1: Decompose verified programs

Every verified program is analyzed for sub-sequences of steps that form a
self-contained "recipe" — a subsequence where:
- The inputs are all bound before the subsequence starts
- The outputs are used after the subsequence ends
- The subsequence can be parameterized over the specific values used

Example:
```
-- Verified program:
bind objects  = find_objects(input)
bind filtered = where(by_color(3), objects)        ─┐
bind reflected = map_obj(reflect(HORIZONTAL), filtered)  │ candidate
bind recolored = map_obj(recolor(7), reflected)    ─┘   subsequence
bind result   = for_each_place(recolored, ..., input)
yield result

-- Extracted candidate:
mirror_and_recolor(axis: AXIS, old_c: COLOR, new_c: COLOR, grid: GRID) =
  bind filtered  = where(by_color(old_c), find_objects(grid))
  bind reflected = map_obj(reflect(axis), filtered)
  bind result    = map_obj(recolor(new_c), reflected)
  return result
```

#### Step 2: Score by MDL (Minimum Description Length)

Each candidate is scored by: does naming this sub-sequence make the total
description of all verified programs shorter?

```
MDL_improvement = description_length(all_programs, old_library)
               - description_length(all_programs, old_library + candidate)
```

If MDL improves (the total corpus gets shorter when we name this pattern),
the candidate passes the compression test.

#### Step 3: Type the new operation

The admitted abstraction gets a typed signature:

```
mirror_and_recolor : AXIS → COLOR → COLOR → GRID → GRID
```

It becomes a first-class citizen in the runtime. The proposer can use it in
future programs exactly like a core operation. The type checker validates
calls to it like any other operation.

### Admission gate (all must hold)

1. The sub-sequence appeared in at least 2 independently verified programs.
   (At test time, this threshold drops to 1 for temporary task-specific
   abstractions.)

2. MDL of the total program corpus improves when the abstraction is named.

3. The abstraction type-checks as a standalone operation.

4. The abstraction is not equivalent to an existing library entry (checked
   by running both on a diverse set of test inputs and comparing outputs).

### Library lifecycle

The library starts with 126 core operations in the current implementation
(the runtime's built-in registry).

After processing the ARC training set (offline, during training), it grows
to approximately 80-120 entries. These are the Level 1 abstractions — learned
compositions that compress the training set.

At test time, task-specific candidates can be admitted temporarily for the
duration of a single task (requiring only 1 verified use, not 2). These
are Level 2 abstractions — they extend the proposer's vocabulary for the
current problem without polluting the permanent library.

After each evaluation session, Level 2 abstractions that were used successfully
across multiple tasks are promoted to Level 1 (permanent). This is how the
library grows over time.

### Multi-step abstractions

Because programs are now step sequences (not expressions), admitted abstractions
can themselves be multi-step. This is important for ARC-AGI-2 tasks that
require complex compositions.

Example of a multi-step library entry:

```
extract_and_apply_codebook : Grid → Grid
  bind zones    = find_zones(input)
  bind key      = zone_by_role(zones, RULE)
  bind data     = zone_by_role(zones, DATA)
  bind mapping  = extract_map(key)
  bind result   = apply_color_map(mapping, zone_to_grid(data))
  return result
```

This captures the entire "input encodes instructions + data" pattern as a
single reusable operation. When the proposer encounters a similar task, it
can emit one step instead of five.

---

## 6. Refinement loop

### Purpose

When the first batch of proposals fails verification, retry with better signal.
Not an ensemble. Not a meta-controller. Just structured retry.

### Protocol

```
Round 1: Propose k=32 candidates. Type-check all. Execute and verify survivors.
         If any pass → done.

Round 2: Error-conditioned re-proposal. Proposer sees prior attempts + their
         failure details (which demo failed, what the diff was, step trace
         showing where computation diverged). Propose 32 more.

Round 3: Partial-match seeded re-proposal. Take programs from rounds 1-2 that
         passed N-1 of N demos. Proposer sees these as "almost right, fix the
         edge case." Propose 32 more.

Round 4: Expanded library re-proposal. Compose new candidate abstractions on
         the fly from existing library entries. Add them to the proposer's
         available operations. Propose 32 more with the expanded vocabulary.
```

Maximum 4 rounds. Fixed budget, not an open-ended search.

### Cost profile

- Round 1 success (easy tasks): ~$0.10/task
- Round 2 success (medium tasks): ~$0.50/task
- Round 3-4 success (hard tasks): ~$2-3/task
- All rounds fail: ~$3/task (wasted, but bounded)
- Average target: ~$1/task

### Adaptive compute

A lightweight difficulty estimator (based on state graph features: object count,
delta complexity, whether tiling is detected, number of distinct colors) predicts
how many rounds will likely be needed. Easy tasks get k=16 in round 1 (cheaper).
Hard tasks get k=48 in round 1 (more diversity upfront).

This is not a meta-controller. It's just a knob on the sampling budget.

---

## 7. Concrete task examples

### Example A: Computed output dimensions

Task: Input is a 12×12 grid with 4 colored objects. Output is a 2×2 grid, each
cell colored by the corresponding object's color, sorted by size.

```
bind objects  = find_objects(input)                 -- ObjectSet, 4 objects
bind count    = count(objects)                      -- Int = 4
bind sorted   = sort_by(SIZE, DESC, objects)        -- ObjectList
bind colors   = map_list(get_color(), sorted)       -- IntList = [3, 7, 1, 5]
bind side     = isqrt(count)                        -- Int = 2
bind out_dims = dims_make(side, side)               -- Dims = (2, 2)
bind canvas   = new_grid(out_dims, 0)               -- Grid 2×2, all black
bind result   = fill_cells(canvas, colors)          -- Grid 2×2, colored
yield result
```

8 steps. Output grid doesn't exist until step 7. Dimensions depend on step 2.
Verification: stateless (Mode A). No cross-demo context needed.

### Example B: Input encodes instructions + data

Task: Input contains a small 3×2 "key" region (top-left, showing which colors
map to which) and a larger "target" region (rest of grid). Apply the key mapping
to the target.

```
bind zones     = find_zones(input)                  -- ZoneList
bind key_zone  = zone_by_role(zones, RULE)          -- Zone (the 3×2 key)
bind data_zone = zone_by_role(zones, DATA)          -- Zone (the target)
bind mapping   = extract_map(key_zone)              -- ColorMap {3→7, 5→1}
bind data_grid = zone_to_grid(data_zone)            -- Grid
bind result    = apply_color_map(mapping, data_grid) -- Grid
yield result
```

6 steps. Steps 2-3 decompose the input semantically. Step 4 interprets one
zone as a lookup table.
Verification: stateless (Mode A).

### Example C: Progression / sequence continuation

Task: Demo 1 shows a line of length 1. Demo 2 shows length 2. Demo 3 shows
length 3. Test asks for the next step.

```
bind step_rule = infer_step(ctx)                    -- GridTransform
bind n         = infer_iteration(ctx, input)        -- Int = next step count
bind result    = repeat_apply(n, step_rule, input)
yield result
```

3 steps. The rule and iteration count are both inferred from cross-demo context.
Verification: sequential (Mode C). Each demo is verified as a step in the
sequence, with only prior demos as context.

### Example D: Cross-demo codebook

Task: Each demo pair shows a different color being mapped. No single pair
contains the full mapping. The test input uses all colors.

```
bind mapping  = infer_map(ctx, 0, 0)                -- ColorMap from all demos
bind result   = apply_color_map(mapping, input)
yield result
```

2 steps. The mapping is extracted by reading across all demo pairs.
Verification: leave-one-out (Mode B). When verifying demo i, the mapping is
inferred from all other demos. If the inferred partial mapping still correctly
predicts demo i's output, the program is valid.

### Example E: Cell-level connect-the-dots

Task: Input contains endpoints of a colored line. Output fills the cells
between matching endpoints horizontally and vertically.

```
bind result = fill_between(input, 3, 3)
yield result
```

2 steps. This is the main reason the implementation grew beyond the original
object-centric catalog: some ARC tasks are better expressed as direct cell-level
transforms than as object rewrites.
Verification: stateless (Mode A).

---

## 8. ARC-3 extension path

ARC-AGI-3 will introduce interactive, temporal, and agentic tasks. The system
needs to reason about state sequences and actions, not just static transforms.

### What changes

For ARC-1/2, a program is: given state, produce output.
For ARC-3, a program is: given state, choose an action. Repeatedly.

The step machine extends naturally:

```
-- New step types for ARC-3:
Step =
  | Bind(...)                      -- same as before
  | Assert(...)                    -- same as before
  | Observe(name, sensor_expr)     -- read from environment
  | Act(action_expr)               -- emit action to environment
  | Loop(condition, body_steps)    -- bounded loop over timesteps

-- A policy is a program with Observe/Act steps:
bind state     = observe(current_grid)
bind objects   = find_objects(state)
bind target    = by_size_rank(0, objects)
bind direction = direction_toward(target, goal_position)
act move(direction)
```

The state graph still works — it's computed per timestep. The type system
adds `Action` and `Observation` types. The library still works — reusable
sub-policies are admitted the same way. Verification extends to trace-level:
does the policy produce the correct action sequence when run on the demo
environment?

### What stays the same

- State graph extraction (per timestep)
- Typed operations (spatial transforms, analysis, construction)
- The proposer (now emitting policies instead of single-shot programs)
- Exact verification (on traces instead of single pairs)
- Abstraction library (sub-policies instead of sub-programs)

This is a genuine extension, not a redesign. The step machine already supports
sequential computation — ARC-3 just adds a feedback loop with the environment.

---

## 9. Training plan

### Phase 1: Program corpus construction (2 weeks)

For each ARC-1/2 training task (1400 total):
- Run the state graph extractor
- Search for correct multi-step programs by enumeration up to length 12,
  using the runtime as a verifier
- Keep all distinct correct programs per task
- Expect: 800-1000 tasks have at least one discoverable program

For synthetic tasks (unlimited):
- Sample random step sequences of length 3-10
- Execute on random input grids to produce pairs
- Filter for non-trivial tasks (at least 2 meaningful operations)
- Target: 200,000 synthetic tasks

### Phase 2: Proposer training (3 weeks)

Fine-tune a ~4B code model on (state_graph, deltas, program) triples.
Multi-task objective: next-token prediction on program text, conditioned
on structured input.

Curriculum:
- Week 1: Synthetic tasks only (high volume, clean labels)
- Week 2: Mix in real ARC tasks (lower volume, harder)
- Week 3: Hard-example mining — oversample tasks where the proposer fails,
  undersample tasks it solves in round 1

### Phase 3: Library bootstrapping (1 week)

Run the trained proposer on the full ARC training set. Collect all verified
programs. Run the abstraction mining + MDL admission pipeline. Produce the
Level 1 library (target: 80-120 entries).

Re-train the proposer with the library available (1 additional fine-tuning
pass). This teaches it to use library entries when they're applicable.

### Phase 4: Evaluation and tuning (2 weeks)

- ARC-AGI-1 public eval (400 tasks)
- ARC-AGI-2 public eval (120 tasks)
- Ablation studies: proposer without library, library without refinement,
  single-round only, expression-only (no multi-step)
- Cost profiling: measure actual $/task distribution

Total: ~8 weeks. Not 16. The system is simpler now.

---

## 10. Risks and honest unknowns

### Zone decomposition is hard

`find_zones` and `zone_by_role` are doing real semantic work — deciding which
part of the input is "instructions" and which is "data." This is not a solved
problem. The deterministic heuristics (spatial separation, color density,
sub-grid regularity) will handle many cases but not all. This is the component
most likely to need iteration.

Mitigation: The proposer can also learn to decompose inputs by emitting
explicit crop/select steps. Zone decomposition is a convenience, not a
requirement. If the heuristic fails, the proposer can work around it.

### The proposer may not be good enough at 4B

Emitting 5-12 lines of typed code with correct step ordering and type
agreement is hard for a small model. NVARC showed that 4B models can
solve ARC tasks, but they were doing direct transduction, not program
generation.

Mitigation: parse-time, execution-time, and verification failures all produce
cheap feedback, so bad proposals are still inexpensive to filter. If 4B proves
insufficient, 7-8B is
the next step, still well within efficient inference budgets.

### The operation catalog may be incomplete

Even with 126 registered ops, the catalog may still be incomplete. Some tasks may require operations
we haven't anticipated.

Mitigation: The abstraction library partially addresses this — composed
operations extend the vocabulary. But there may be genuinely missing
primitives. The system should track "tasks where no program was found"
and analyze them for missing capabilities. Growing the core catalog is
an explicit maintenance task.

### Cross-demo reasoning adds verification complexity

Leave-one-out verification is correct but expensive when there are 5 demo
pairs (5 executions instead of 1). Sequential verification has subtle
ordering dependencies.

Mitigation: Cross-demo tasks are a minority (~40% LOO, ~20% sequential).
The type checker routes to the cheapest applicable mode. And even 5×
execution is still fast for small programs.

### Some tasks have no clean program

A small fraction of ARC tasks may resist clean decomposition into typed
steps. The rule may be "vaguely aesthetic" or depend on spatial intuitions
that don't decompose into the operation catalog.

This is an honest limitation. ARIA will fail on these tasks. That's
preferable to escaping into untyped code and losing verifiability.

---

## 11. What success looks like

### Quantitative targets (grounded, not aspirational)

- ARC-AGI-1 public eval: 55-65%
- ARC-AGI-2 public eval: 30-40%
- Average cost per task: <$2
- Median program length: 5-8 steps
- Library size after training: 80-120 entries
- Round 1 solve rate: 40-50% (the rest need refinement)

These are lower than the v1 targets because they're honest. 55% on ARC-1
with a clean, inspectable, typed system would be a meaningful result —
it would demonstrate that structured program induction (not brute-force
search, not LLM prompting) can compete with much more expensive approaches.

### Qualitative targets

- Every solution is human-readable. You can look at a 7-step program and
  understand exactly what the system did and why.
- Every solution is verified. No "probably right" — either it passes
  pixel-perfect verification on all demos, or it's rejected.
- The library tells a story. After training, the 80-120 learned abstractions
  are a compressed vocabulary of "ARC reasoning patterns." Analyzing the
  library is itself a scientific contribution.
- The system degrades gracefully. Tasks it can't solve are cleanly identified
  (no program found), not silently wrong.

---

## Summary

```
ARIA v3:

  State graph    → immutable typed structure, deterministic extraction
  Step machine   → multi-step programs with computed dims, zones, cross-demo
  Proposer       → fine-tuned ~4B code model, emits typed step programs
  Verification   → three modes (stateless / LOO / sequential), exact, binary
  Library        → MDL-gated admission of verified multi-step sub-programs

  One machine. One loop. One thesis.
```
