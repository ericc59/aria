Read `/Users/ericc59/Dev/aria/docs/raw/arc2_v2_eval_gap_ledger.md` and use it as the only source of truth.

Important:
- Do NOT redo the per-task analysis.
- Do NOT use solver family labels.
- Do NOT infer from memory or prior ARC lore.
- If something is not in the ledger, do not invent it.
- Be skeptical about grouping. Merge only when the first real missing thing and the smallest principled addition are actually the same.

Workflow

1. Build a first-pass synthesis from the ledger.
2. Then use the Codex CLI as an adversarial reviewer on your first pass.
   - If you have shell access and `codex` is available, invoke the local Codex CLI using the normal command for this environment.
   - Give Codex the ledger path and your first-pass synthesis.
   - Ask Codex to review for:
     - over-grouping
     - false merges
     - weak priorities
     - recommendations not actually supported by the ledger
     - places where a “gap” is really multiple different missing additions
   - Treat Codex as a reviewer only, not as a new source of task analysis.
   - If Codex CLI is unavailable, say so explicitly and continue without it.
   - if Codex CLI is unavailable, still write the file and clearly mark the missing review step instead of stopping.
3. Revise your synthesis after the review.
4. Report what Codex challenged and whether you accepted or rejected each challenge.

Task

Synthesize the completed ARC-2 `v2-eval` unsolved-task ledger into a rigorous gap analysis for `aria/search`.

Output exactly these sections:

1. Ranked Gap Table

For each recurring gap, provide:
- `gap_id`
- `kind` — one of: `representation`, `binding`, `derivation`, `execution`, `output`
- `short_description`
- `first_break_frequency`
- `task_ids`
- `closest_existing_aria_substrate`
- `smallest_principled_addition`
- `canonical_or_task_shaped`
- `priority` — `high`, `medium`, or `low`

Rules for the table:
- group only where the first real missing thing is the same
- if two gaps look similar but require different minimal additions, keep them separate
- include task IDs for every row
- keep rows concrete and non-hand-wavy

2. Top 5 Canonical Additions

Rank the 5 most promising missing additions by:
- recurrence
- architectural cleanliness
- likely value for ARC-2
- likelihood of fitting canonical `aria/search`

For each one, give:
- `name`
- `why_it_recurs`
- `why_it_is_or_is_not_clean`
- `which_existing_ops_or_substrates_it_builds_on`
- `task_ids`

3. Extensibility Check

Assess which current canonical ops/substrates seem closest to useful extension:
- `PANEL_BOOLEAN`
- `PANEL_REPAIR`
- `SYMMETRY_REPAIR`
- `OBJECT_REPACK`
- crop/frame/panel extraction
- tracing / propagation infrastructure

For each:
- `what_it_already_covers`
- `what_extension_the_ledger_supports`
- `what_extension_the_ledger_does_not_support`

4. Anti-Targets

List the gaps that still look too bespoke or too decode-specific to pursue now.

For each:
- `gap_id`
- `why_it_looks_bespoke`
- `task_ids`

5. Recommendation

Give exactly:
- the single best next canonical addition to pursue
- the single best second choice
- why those two beat the alternatives
- which 3–6 ARC-2 task IDs should be used as representative tasks for each

6. Codex Review Delta

Summarize the Codex CLI review:
- `codex_available` — yes/no
- `main_challenges_codex_raised`
- `which_challenges_you_accepted`
- `which_challenges_you_rejected`
- `how_the_final_output_changed_after_review`

Hard constraints

- Start from the ledger, not from buckets.
- Do not collapse different execution primitives into one “family”.
- Do not recommend anything unless the ledger supports it.
- If the ledger does not support a clean merge, say so.
- Be explicit about uncertainty.
- Prefer gaps framed as missing latent state, relation, or transition semantics over surface-pattern categories.
- Penalize proposals whose smallest addition is primarily a decoder for benchmark-specific visual syntax.
- Prefer additions that compose with existing canonical substrate (`PANEL_*`, `SYMMETRY_REPAIR`, `OBJECT_REPACK`, crop/frame extraction, tracing) over isolated new ops.
- If a proposed addition is mostly a task-shaped renderer or codebook, classify it as an anti-target unless the ledger shows a deeper shared state/transition abstraction.
- Optimize for generalized symbolic dynamics, not family recovery.

Output style

- Use tables where useful.
- Be concise but specific.
- Quote task IDs everywhere relevant.
- Optimize for decision-making, not prose.
