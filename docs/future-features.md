# Future Features

Ideas and enhancements to consider. Not prioritized — add as they come up.

---

## Search

- **Unified keyword filter**: When `metadata_filters` includes `keywords`, auto-search both frontmatter `keywords` AND enriched `enr_keywords` fields. Currently only checks the exact column name specified, so filtering `{"keywords": "ganesh"}` misses docs where "ganesh" appears only in `enr_keywords` (or vice versa). Same idea could apply to `tags` vs `enr_suggested_tags`.

- **Per-document dedup in results**: Option to collapse multiple chunks from the same doc into a single result (show best chunk, note total matching chunks). Currently the same PDF can occupy 3+ result slots with different page chunks.
