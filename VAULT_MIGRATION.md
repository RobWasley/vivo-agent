# Vault Migration Summary

## Overview
Vivo's conversation/session/memory persistence is Markdown-first: every
record is a vault document with JSON-compatible frontmatter, written
atomically. Pre-vault JSON files are imported automatically at startup, so
an upgrade never silently amnesiates the store.

## File Layout
```
data/vault/
  sessions/
    index.md            # Session index: active pointer + sessions JSON in frontmatter
    <session-id>.md     # One document per session (turns as JSON array in frontmatter)
  memories/
    MEMORY.md           # Curated core-memory index (rebuilt from core facts)
    facts/<id>.md       # One document per memory fact (text/core/date in frontmatter)
    dreams/<date>-<id>.md  # One document per dream pass (summary, new_facts, stats, ...)
  vault.log             # Activity log
```

## Frontmatter Rules (`app/vault.py`)
- Values that are lists or dicts are written as JSON and parsed back to the
  same Python type (a naive space-joined form shredded items containing
  spaces on read-back — regression-tested in `tests/test_vault.py`).
- Scalars, booleans and nulls use plain YAML-ish literals.

## Session Document Format
```markdown
---
id: 20260927-210836
created_at: 2026-09-27T21:08:36.680417+00:00
updated_at: 2026-09-27T21:08:36Z
title: 20260927-210836
summary: user greeted the assistant
compactions: 0
turns: [["hello", "hi there"], ["bye", "goodbye"]]
---

## Summary

user greeted the assistant

## Turns

### Turn 1

**You**: hello

**Vivo**: hi there
```

## Migration (two paths, same target layout)

### 1. Automatic, at startup (runtime)
- `app/memory.py`: `MemoryStore` imports `data/memory.json` (facts) and
  `data/memory.md` (core index); `DreamStore` imports `data/dreams.json`.
  Per-record idempotent (dedup by id / `<date>-<id>.md` name); legacy files
  are deleted after import. If only the facts survived (empty legacy
  index), the core index is rebuilt from the core facts.
- `app/conversation.py`: `SessionStore` imports `data/conversation.json`
  and `data/sessions.json` + `data/sessions/<id>.json`, then **adopts
  orphan session documents** in `data/vault/sessions/` that are missing
  from the index (their `turns` frontmatter may be a JSON array or an int).

### 2. Standalone script (`app/migration.py`)
Manual one-shot conversion (`migrate(data_dir, vault_root)`), guarded by a
`data/.migrated` marker; `rollback()` removes the marker. Its output mirrors
the runtime frontmatter layout exactly (verified by
`test_full_migration_documents_readable_by_runtime_stores`), so either path
produces a store the runtime can read back.

## Test Results
- Full suite: **386 passed** (in-container, `pytest tests/ -q`).
- `tests/test_vault.py`: 54 · `tests/test_migration.py`: 23 ·
  `tests/test_memory.py`: 21 · `tests/test_sessions.py`: 31 ·
  `tests/test_conversation.py`: 21 · `tests/test_graph.py`: 8.
- Environment-dependent: `test_weather_live` hits the real open-meteo API
  and `test_pipeline.py::test_utterance_reply` is timing-sensitive; both can
  fail on flaky runs unrelated to vault changes.
