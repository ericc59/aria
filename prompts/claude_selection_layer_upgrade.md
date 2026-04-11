# Claude Prompt: Selection Layer Upgrade For `aria/search`

You are working in:

`/Users/ericc59/Dev/aria`

Read these first:

- `/Users/ericc59/Dev/aria/docs/AGI.md`
- `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`

Current evidence from refreshed eval runs

`v1-train`:
- solved: 29/400
- biggest failure clusters:
  - selection: 198
  - dims_change: 127

`v2-eval`:
- solved: 11/120
- biggest failure clusters:
  - selection: 57
  - dims_change: 35

Important conclusion:
The next major bottleneck is NOT another random primitive.
The next major bottleneck is:
SELECTION

This is stable across both datasets.

Goal for this pass

Build the next layer of selection capability for `aria/search`.

This should be:
- fact-based
- explicit
- exact
- reusable
- aligned with the roadmap’s direction:
  - low-level primitives below
  - learned patterns/macros above
  - no benchmark-family runtime ontology

What this pass is NOT

Do NOT:
- add task-id logic
- add benchmark-family labels
- add a giant new AST op
- add fuzzy classifiers
- build a monolithic solver bucket
- solve one task with a hidden decoder and call it “selection”

What to build

Build a stronger derive-time selection substrate based on explicit per-object / per-region / per-panel facts and small rule induction over those facts.

That means:
- richer fact extraction
- richer selector predicates
- small exact rule search for selecting changed targets

The problem we need to solve is:
`aria/search` is still weak at expressing and discovering rules like:
- select the only object with property X
- select all objects that satisfy relation R to object Y
- select all regions whose fact profile matches output changes
- select panels/objects by counts, overlaps, adjacency, containment, anchor status, role, etc.

Architecture target

This pass should strengthen:
- perception/facts
- derive-time selectors
- rule induction over selectors

It should NOT hardcode:
- one benchmark pattern
- one task-specific partition
- one named “family”

Concrete implementation plan

PART A: Define richer selection facts

Add or extend a fact layer that exposes reusable selector-relevant attributes for objects/regions/panels.

Expected homes:
- `/Users/ericc59/Dev/aria/aria/search/motif.py`
- `/Users/ericc59/Dev/aria/aria/search/binding.py`
- `/Users/ericc59/Dev/aria/aria/search/registration.py`
- or a new file like:
  - `/Users/ericc59/Dev/aria/aria/search/selection_facts.py`

At minimum, I want facts like:
- color
- size
- bbox dims
- area
- aspect-ish signals
- edge touching / corner touching
- containment / enclosed-in
- adjacency / near-separator / near-frame
- anchor presence
- role hints (if available from binding)
- overlap with source/target masks
- count-based context:
  - unique color in scene
  - unique shape in panel
  - largest / smallest in group
  - only object touching boundary
- per-panel facts where relevant

Keep facts explicit and typed.
Do NOT add raw hashes as the primary mechanism.

PART B: Define richer selector predicates

Add a selector predicate layer that can express small exact conditions over these facts.

Examples of acceptable selector predicates:
- `color == c`
- `size == n`
- `touches_boundary`
- `inside_frame`
- `adjacent_to_separator`
- `unique_color_in_scene`
- `largest_of_color`
- `has_anchor`
- `role == TARGET`
- conjunctions of 1–3 such predicates

This should remain small and interpretable.

Expected home:
- likely extend the existing selection/binding/search machinery rather than inventing a separate giant DSL

PART C: Add exact selector rule induction

Add a derive-time procedure that:
1. computes changed objects/regions from demos
2. computes candidate fact predicates
3. searches small conjunctions/disjunctions over those predicates
4. finds a selector rule that exactly picks the changed targets across demos
5. lowers that into the existing search representation cleanly

This is the critical new capability.

The rule search must be:
- exact
- small
- interpretable
- bounded

Do NOT implement:
- generic decision trees
- black-box ML selection
- fuzzy clustering

PART D: Integrate into canonical derive

Wire this into the canonical search derive path.

Expected home:
- `/Users/ericc59/Dev/aria/aria/search/derive.py`

I want:
- richer selection candidates to be generated before the system gives up into `selection` failure
- still exact verification across demos
- no task-local hacks

PART E: Evaluate on one or two real selection failures

Pick 1–2 tasks from the current unsolved set that are selection-dominated and likely to benefit from this layer.

You choose them, but they should be:
- genuinely selection failures
- not blocked primarily on output construction
- not requiring a brand-new execution primitive

The point is to validate the selection layer itself, not to cheat a task through.

Success means:
- the new selector/rule layer explains the target selection more cleanly than before
- ideally at least one selection-dominated task now solves canonically

PART F: Tests

Add focused tests.

I want:
1. unit tests for the new fact extraction / predicate logic
2. unit tests for the selector-rule induction
3. at least one task-level regression on a real selection-dominated task

Suggested files:
- `/Users/ericc59/Dev/aria/tests/test_search_selection_facts.py`
- `/Users/ericc59/Dev/aria/tests/test_search_selection_rules.py`
- maybe update an existing task regression file if more appropriate

PART G: Verification commands

Run and report at least:

1.
    python -m py_compile \
      /Users/ericc59/Dev/aria/aria/search/derive.py \
      /Users/ericc59/Dev/aria/aria/search/binding.py \
      /Users/ericc59/Dev/aria/aria/search/motif.py

Plus any new/changed modules you add.

2.
    pytest -q \
      /Users/ericc59/Dev/aria/tests/test_search_selection_facts.py \
      /Users/ericc59/Dev/aria/tests/test_search_selection_rules.py

3.
Run the chosen real tasks through:
    python /Users/ericc59/Dev/aria/scripts/eval_arc2.py \
      --dataset v1-train \
      --task TASK_ID \
      --time-budget 5

and/or `v2-eval` if appropriate.

PART H: Changelog

Update:
- `/Users/ericc59/Dev/aria/changelog.md`

Keep it concise.
Add only a short note about:
- richer selection facts
- selector-rule induction
- which real task(s) it improved, if any

PART I: Commit

When done and verified, commit the changes.

Success criterion

This pass is successful if:
1. `aria/search` can express richer exact selection conditions than before
2. those conditions are based on explicit facts, not benchmark labels
3. selector-rule induction is exact and bounded
4. no task-id logic or hidden family solver was added
5. ideally at least one real selection-dominated task improves or solves

Deliverable format

At the end, report:
1. what fact layer you added
2. what selector predicates you added
3. what rule induction you added
4. what files changed
5. what real task(s) improved
6. why this is aligned with:
   - `/Users/ericc59/Dev/aria/docs/AGI.md`
   - `/Users/ericc59/Dev/aria/docs/raw/aria_learning_roadmap.md`

Reminder

The point is not to add a “selection family” solver.
The point is to strengthen the explicit fact -> selector -> exact-rule path inside canonical `aria/search`.
