Use `/Users/ericc59/Dev/aria/docs/raw/arc2_v2_eval_gap_ledger.md` as the source of truth.

Important:
- Do NOT redo the task analyses.
- Do NOT start from families or solver labels.
- Build only from the completed ledger in that file.

Goal

Convert the completed task-by-task ledger into a ranked map of the missing pieces in `aria/search`, based on repeated first-breaks and repeated missing representations/ops.

For each recurring gap, produce:
- `gap_id` — short canonical name
- `kind` — one of:
  - representation
  - derivation
  - execution
  - output
  - binding
- `short description`
- `first-break frequency`
- `task_ids`
- `existing aria substrate that is closest`
- `smallest principled addition`
- `looks canonical or task-shaped?`
- `priority` — high / medium / low

Rules

- Base the table only on the completed ledger.
- Group only where the missing thing is genuinely the same.
- Prefer “missing latent state / relation / operation” over thematic labels.
- Keep each row tight and concrete.
- If two gaps are similar but not the same, keep them separate.
- If a gap is really just a special case of another, merge it only if the smallest principled addition is the same.

Then add a final section:

1. top 5 missing canonical additions by recurrence
2. which ones seem most ARC-2-specific
3. which ones could likely help both ARC-2 and ARC-1
4. which current canonical ops are closest to being extensible
5. which gaps still look too bespoke to pursue

Output format

1. Ranked gap table
2. Then a short synthesis

Do not start with broad buckets.
Start from the actual repeated first-breaks in the ledger.
