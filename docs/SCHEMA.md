# Docs Wiki Schema

## Structure

```
docs/
  raw/          # Immutable source documents (PDFs, articles, data)
  wiki/         # LLM-maintained synthesized pages (markdown)
    index.md    # Content catalog — every page listed with summary
    log.md      # Chronological record of ingests, queries, lints
  SCHEMA.md     # This file — conventions and workflows
```

## Conventions

- **Raw sources are immutable.** Never modify files in `raw/`. They are the source of truth.
- **Wiki pages are LLM-maintained.** The LLM creates, updates, and cross-references all pages in `wiki/`. The human reads and directs.
- **Topic-organized, not source-organized.** Don't create one wiki page per source. Synthesize across sources by topic. One page on "refinement loops" is better than three pages summarizing three papers.
- **Index stays current.** Update `wiki/index.md` on every ingest or new page creation. It's the LLM's entry point for finding relevant pages.
- **Log is append-only.** Add an entry to `wiki/log.md` for every ingest, significant query result filed as a page, or lint pass. Format: `## [YYYY-MM-DD] verb | description`.

## Workflows

### Ingest a new source
1. Place the source in `docs/raw/`
2. Read the source
3. Discuss key takeaways with the user
4. Update existing wiki pages where the new source adds or contradicts information
5. Create new wiki pages if the source covers topics not yet in the wiki
6. Add the source to the registry in `index.md`
7. Update `index.md` page listings if pages were created or significantly changed
8. Append to `log.md`

### Answer a question / analysis
1. Read `wiki/index.md` to find relevant pages
2. Read relevant wiki pages
3. Synthesize an answer
4. If the answer is substantial and reusable, file it as a new wiki page
5. Update `index.md` and `log.md` if a page was created

### Lint
1. Check for contradictions between pages
2. Check for stale claims superseded by newer sources
3. Check for orphan pages not linked from index
4. Check for important concepts mentioned but lacking their own page
5. Suggest new sources to investigate
6. Update `log.md`

## Page Guidelines

- Keep pages focused. One topic per page.
- Use `[[page_name]]` style links for cross-references within wiki pages.
- Include a "Relevance to Aria" section on pages covering external research/approaches.
- Prefer tables for structured comparisons.
- No fluff. Every sentence should carry information.
