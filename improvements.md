# Context & Memory Improvements

This document specifies improvements to conversation context management,
memory dreaming, and system observability. Each item includes backend API
changes, WebSocket events, and the UI changes needed to support it.

---

## 1. Make Compaction Visible to the User

### Problem
Compaction is completely silent — no UI notification, no transcript entry,
no WebSocket event. The user never knows their conversation history was
summarised.

### Backend

#### 1a. Broadcast a `compact` WebSocket event

Add a broadcast function in `pipeline.py` (same pattern as `broadcast_wake`,
`broadcast_dream_state`):

```python
_compact_callbacks: list[Callable[[], None]] = []

def on_compact(cb: Callable[[], None]) -> None:
    _compact_callbacks.append(cb)

def broadcast_compact(summary: str, turns_before: int, turns_after: int) -> None:
    msg = {
        "type": "compact",
        "summary": summary,
        "turns_before": turns_before,
        "turns_after": turns_after,
    }
    for cb in _compact_callbacks:
        try:
            cb(msg)
        except Exception:
            pass
```

Call `broadcast_compact(...)` from `Conversation.compact()` after the
summary is applied and saved (after `self._save()` and before
`self._compacting = False`).

#### 1b. Add a system entry to the transcript

The transcript panel already supports `entry-hint`-style entries for system
messages (see the "Tap mic, then talk" placeholder). Compaction should
produce a visible, expandable system entry:

```
— sys    Context refreshed
  Earlier conversation summarised (5 turns -> 47-char summary, kept 10)
  [expand]
```

In `app.js`, when the `"compact"` WebSocket message is received:
- Create an `entry-hint` with class `compact-entry`
- Show a collapsed summary line
- An expand toggle reveals the actual summary text and statistics
- The entry is scrollable and stays in the transcript (doesn't auto-scroll
  away like agent replies)

### UI

- New nav button or context menu action to view recent compaction history
  (kept separately, not in conversation transcript)
- Compact entries are timestamped

---

## 2. LLM-Based Dreaming

### Problem
The current `dream()` mechanism is a deterministic scoring function: it
ranks facts by keyword heuristics and promotes top-3 to core memory. No LLM
is involved, so dreams can't surfice insights, connections, or emergent
patterns.

### New Architecture

Replace `MemoryStore.consolidate()` with a two-phase LLM-assisted process:

#### 2a. Phase 1 — Score & Select (existing, unchanged)

The keyword-scoring heuristic remains as a lightweight filter to produce a
candidate set of ~15-20 facts. This avoids sending the entire memory archive
to the LLM on every dream pass.

```python
def _dream_candidates(self) -> list[dict]:
    """Return top-N facts by the scoring heuristic."""
    items = [item["text"] for item in self._facts()
             if not self._is_transient(item["text"])]
    scored = sorted(items, key=lambda x: self._score(x), reverse=True)
    return scored[:20]
```

#### 2b. Phase 2 — Dream Pass (new, LLM call)

Send the candidates + core memory index to the LLM with a dream persona,
requesting:
- **Connections** between facts (e.g. "User mentioned Project A and
  Project B in the same conversation")
- **Actionable insights** (e.g. "User has set 3 reminders for next week —
  suggest a weekly review task")
- **New candidate facts** not yet stored (e.g. a preference that was
  stated implicitly but never observed)
- **Pruning suggestions** (facts that are outdated or contradicted)

```python
DREAM_PROMPT = """
You are consolidating a local memory archive. Here are the current facts:

### Core memory (always active)
{core_memory}

### Archive candidates (ranked by relevance)
{candidates}

Return a JSON object with these fields:
- summary: short one-line summary of what was consolidated (for logging)
- new_facts: list of strings — facts worth adding to the archive that
  aren't already present
- connections: list of strings — observations linking facts together
- pruned_ids: list of fact IDs — facts that should be marked as non-core
  or archived

Keep it concise. Prefer observations over restating facts."""
```

#### 2c. Apply results

- Add any `new_facts` to the archive
- Un-check `core` for any `pruned_ids`
- Store `connections` as a new entity type: a `Dream` record (see §3)

### WebSocket Events

On a dream pass (manual or auto), broadcast:

```json
{
    "type": "dream",
    "phase": "consolidating",
    "active": true
}
// ...
{
    "type": "dream",
    "phase": "complete",
    "summary": "Consolidated 12 facts, found 3 connections",
    "new_facts": 1,
    "connections": 3,
    "pruned": 2
}
```

### UI Changes

- The existing "Dream" button remains (triggers immediate pass)
- The hint message format changes from `"dreamed: - User prefers short answers"`
  to `"dreamed: 12 facts consolidated, 3 connections, 1 new fact"`
- Clicking the hint expands to show the connections and new facts

---

## 3. Dreams Page — Dream Log

### Overview
A new `skills-dialog`-style modal (`dreams-dialog`) that shows a scrollable
log of all dream passes, with expandable entries showing connections, new
facts, and pruned items. Modeled on the existing Skills / Memory / Reminders
dialogs.

### HTML Structure

```html
<dialog id="dreams" class="skills-dialog">
    <div class="settings-head">
        <h2>Dreams</h2>
        <button class="ghost" id="btn-dreams-x" title="Close">&times;</button>
    </div>
    <div class="skills-body">
        <aside class="skills-list" aria-label="Dream log">
            <div class="skills-list-head">
                <span>Dream log</span>
                <button class="ghost tiny" id="btn-dream-now" title="Run a dream pass now">Dream now</button>
            </div>
            <div id="dreams-list-items" class="skills-list-items"></div>
        </aside>
        <div class="skills-editor" id="dream-editor" hidden>
            <label>
                <span>Pass</span>
                <div id="dream-pass-info"></div>
            </label>
            <label>
                <span>Connections</span>
                <div id="dream-connections"></div>
            </label>
            <label>
                <span>New facts suggested</span>
                <div id="dream-new-facts"></div>
            </label>
            <label>
                <span>Pruned</span>
                <div id="dream-pruned"></div>
            </label>
            <label class="dream-actions">
                <button id="btn-dream-add-facts" class="ghost" disabled>Add facts to memory</button>
            </label>
        </div>
    </div>
    <div class="settings-foot">
        <span class="settings-msg" id="dreams-msg"></span>
        <span class="settings-spacer"></span>
        <button class="ghost" id="btn-dreams-close">Close</button>
    </div>
</dialog>
```

### Nav Button

In `index.html`, add to the nav button group (after the existing "Dream"
button):

```html
<button class="nav-btn" id="btn-dreams" title="View dream log">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ...>
        <path d="M12 2a7 7 0 0 0 5 12 5 5 0 0 1-5 5 5 5 0 0 1-5-5 7 7 0 0 0 5-12z"/>
    </svg>
    <span class="nav-label">Dreams</span>
</button>
```

### Backend API

#### 3a. Dream store

A new `DreamStore` class (similar to `MemoryStore`) persisted as JSON:

```python
# data/dreams.json
[
    {
        "id": "abc123",
        "ts": 1697000000.0,
        "phase": "complete",
        "summary": "Consolidated 12 facts, found 3 connections",
        "connections": ["User has 3 project-related facts in memory"],
        "new_facts": ["User prefers JSON config over TOML"],
        "pruned_ids": ["fact-001", "fact-042"]
    }
]
```

```python
class DreamStore:
    def __init__(self, path: str = "data/dreams.json"):
        self.path = Path(path)
        # load / create
    
    def add(self, dream: dict) -> str:
        """Save a dream pass result, return id."""
    
    def list(self) -> list[dict]:
        """Return all dream entries, newest first."""
    
    def clear(self) -> None:
        """Clear all entries."""
```

#### 3b. API endpoints (in `main.py`)

```python
@app.get("/api/dreams")
async def list_dreams():
    """List all dream pass entries."""
    store = app.state.engines.dream_store
    return {"dreams": store.list()}

@app.delete("/api/dreams")
async def clear_dreams():
    """Clear all dream log entries."""
    app.state.engines.dream_store.clear()
    return {"ok": True}
```

The `DreamScheduler.trigger()` method would store the result in the
`DreamStore` in addition to consolidating memory.

### UI (app.js)

Element cache additions:

```js
dreamsDlg: $("dreams"),
dreamsList: $("dreams-list-items"),
dreamEditor: $("dream-editor"),
dreamPassInfo: $("dream-pass-info"),
dreamConnections: $("dream-connections"),
dreamNewFacts: $("dream-new-facts"),
dreamPruned: $("dream-pruned"),
btnDreamNow: $("btn-dream-now"),
btnDreamAddFacts: $("btn-dream-add-facts"),
btnDreamsClose: $("btn-dreams-close"),
btnDreamsX: $("btn-dreams-x"),
```

Functions:

```js
const openDreams = async () => {
    await loadDreams();
    el.dreamsDlg.showModal();
};

const loadDreams = async () => {
    const res = await fetch("/api/dreams");
    const { dreams } = await res.json();
    el.dreamsList.innerHTML = dreams.map(d => `
        <div class="dream-entry" data-id="${d.id}">
            <span class="d">${new Date(d.ts * 1000).toLocaleString()}</span>
            <span class="t">${escapeHtml(d.summary)}</span>
            <span class="meta">${d.connections?.length || 0} connections, ${d.new_facts?.length || 0} new</span>
        </div>
    `).join("");
    // Add click handlers to expand entries
};

const showDreamDetail = (id) => {
    const dream = dreams.find(d => d.id === id);
    el.dreamPassInfo.textContent = dream.summary;
    el.dreamConnections.innerHTML = (dream.connections || []).map(c => `<p>${c}</p>`).join("");
    el.dreamNewFacts.innerHTML = (dream.new_facts || []).map(f => `<p>${f}</p>`).join("");
    el.dreamPruned.innerHTML = (dream.pruned_ids || []).map(i => `<span>${i}</span>`).join("");
    el.dreamEditor.hidden = false;
};
```

---

## 4. System Logging Page

### Overview
A new page/modal that shows a live, scrollable log of system events:
actions, errors, warnings, reminders firing, config changes, session
lifecycle, etc. This is **not** the conversation transcript — it's the
system's operational log.

### Architecture

#### 4a. WebSocket log stream

Instead of (or in addition to) Python `logging`, add a `LogStore` that
writes structured log entries to a file and broadcasts them over WebSocket.

```python
# app/logging_store.py
import json
import time
from pathlib import Path

class LogStore:
    MAX_ENTRIES = 1000  # keep a rolling buffer
    PATH = "data/log.json"

    def __init__(self):
        self.entries: list[dict] = []
        self._callbacks: list[Callable[[dict], None]] = []
        self._load()

    def log(self, level: str, message: str, **extra) -> dict:
        entry = {
            "ts": time.time(),
            "level": level,
            "message": message,
            **extra,
        }
        self.entries.append(entry)
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries = self.entries[-self.MAX_ENTRIES:]
        self._save()
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception:
                pass
        return entry

    def list(self) -> list[dict]:
        return list(self.entries)

    def clear(self) -> None:
        self.entries = []
        self._save()
```

#### 4b. Structured event types

Define a set of event types that get logged with contextual metadata:

| Type | Level | Context | Example |
|------|-------|---------|---------|
| `config_change` | info | `key`, `old`, `new` | `compact_after_chars: 12000 -> 65000` |
| `session_create` | info | `session_id`, `name` | Session created |
| `session_switch` | info | `from`, `to` | Switched session |
| `session_delete` | info | `session_id` | Session deleted |
| `dream_start` | info | `mode: manual|auto` | Dream pass started |
| `dream_complete` | info | `summary`, `connections`, `new_facts` | Dream finished |
| `compaction` | info | `turns_before`, `turns_after`, `summary_len` | Context compacted |
| `reminder_fire` | info | `reminder_id`, `text` | Reminder fired |
| `reminder_miss` | warning | `reminder_id`, `error` | Reminder callback failed |
| `skill_update` | info | `name`, `action: create|update|delete` | Skill saved |
| `memory_add` | info | `fact_id`, `core` | Fact added |
| `memory_update` | info | `fact_id` | Fact updated |
| `voice_upload` | info | `slug`, `duration` | Voice uploaded |
| `stt_bench` | info | `model`, `compute` | STT benchmark completed |
| `tts_bench` | info | `model`, `threads` | TTS benchmark completed |
| `pipeline_error` | error | `error`, `stage` | Pipeline failed at STT stage |
| `wake_success` | info | `mode: phrase|text` | Woken by wake phrase |
| `wake_expire` | info | `idle_seconds` | Wake session expired |

#### 4c. Bridge to Python logging

A `LogHandler` that bridges `logging` calls to the `LogStore`:

```python
import logging

class VivoLogHandler(logging.Handler):
    def __init__(self, store: LogStore):
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord):
        # Map Python log levels to our levels
        level_map = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "critical",
        }
        self.store.log(
            level=level_map.get(record.levelno, "info"),
            message=self.format(record),
            logger=record.name,
        )
```

#### 4d. API endpoints (in `main.py`)

```python
@app.get("/api/logs")
async def get_logs():
    """Get log entries, optionally filtered by level."""
    store = app.state.engines.log_store
    level = request.query_params.get("level")
    entries = store.list()
    if level:
        entries = [e for e in entries if e["level"] == level]
    return {"logs": entries}

@app.delete("/api/logs")
async def clear_logs():
    """Clear all log entries."""
    app.state.engines.log_store.clear()
    return {"ok": True}
```

### UI — Log Page

#### 5a. HTML

```html
<dialog id="logs" class="skills-dialog">
    <div class="settings-head">
        <h2>System Log</h2>
        <button class="ghost" id="btn-logs-x" title="Close">&times;</button>
    </div>
    <div class="skills-body">
        <div class="logs-controls">
            <select id="log-level-filter" title="Filter by level">
                <option value="all">All levels</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
            </select>
            <button class="ghost tiny" id="btn-log-clear" title="Clear log">Clear</button>
            <button class="ghost tiny" id="btn-log-export" title="Export as JSON">Export</button>
        </div>
        <div id="logs-list" class="logs-list"></div>
    </div>
    <div class="settings-foot">
        <span class="settings-msg" id="logs-msg"></span>
        <span class="settings-spacer"></span>
        <button class="ghost" id="btn-logs-close">Close</button>
    </div>
</dialog>
```

#### 5b. Nav Button

```html
<button class="nav-btn" id="btn-logs" title="System log">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ...>
        <path d="M12 2H2v20h20V2H12zM4 4h6v3H4V4zm0 6h16v3H4v-3zm0 6h10v3H4v-3z"/>
    </svg>
    <span class="nav-label">Log</span>
</button>
```

#### 5c. WebSocket Real-Time Updates

Log entries are broadcast to all connected clients in real-time:

```python
def broadcast_log_entry(entry: dict) -> None:
    for cb in _log_callbacks:
        try:
            cb(entry)
        except Exception:
            pass
```

```json
{
    "type": "log_entry",
    "entry": {
        "ts": 1697000000.0,
        "level": "info",
        "message": "Context compacted: 15 turns -> 82-char summary, kept 10",
        "source": "vivo.conversation"
    }
}
```

#### 5d. UI (app.js)

```js
const logLevels = {
    debug: { color: "#6b7280", label: "dbg" },
    info: { color: "#38bdf8", label: "info" },
    warning: { color: "#f59e0b", label: "warn" },
    error: { color: "#ef4444", label: "err" },
    critical: { color: "#dc2626", label: "crit" },
};

let logFilter = "all";
let logEntries = [];

const openLogs = async () => {
    await loadLogs();
    el.logsDlg.showModal();
};

const loadLogs = async () => {
    const level = logFilter === "all" ? "" : `?level=${logFilter}`;
    const res = await fetch(`/api/logs${level}`);
    const { logs } = await res.json();
    logEntries = logs;
    renderLogs();
};

const renderLogs = () => {
    el.logsList.innerHTML = logEntries.map(e => {
        const l = logLevels[e.level] || logLevels.info;
        const time = new Date(e.ts * 1000).toLocaleTimeString();
        return `
            <div class="log-entry" data-level="${e.level}">
                <span class="log-time">${time}</span>
                <span class="log-level" style="color:${l.color}">${l.label}</span>
                <span class="log-source">${escapeHtml(e.source || "")}</span>
                <span class="log-msg">${escapeHtml(e.message)}</span>
            </div>
        `;
    }).join("");
    el.logsList.scrollTop = el.logsList.scrollHeight;
};

// Real-time WebSocket handling
case "log_entry":
    logEntries.push(m.entry);
    if (logFilter === "all" || m.entry.level === logFilter) {
        renderLogs();
    }
    break;
```

#### 5e. CSS

```css
.logs-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}

.logs-list {
    flex: 1;
    overflow-y: auto;
    font-family: "IBM Plex Mono", monospace;
    font-size: 13px;
    line-height: 1.6;
}

.log-entry {
    display: flex;
    gap: 8px;
    padding: 4px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}

.log-entry:hover {
    background: rgba(255,255,255,0.02);
}

.log-time { color: var(--text-muted); min-width: 70px; }
.log-level { min-width: 40px; font-weight: 600; }
.log-source { color: var(--text-muted); min-width: 120px; }
.log-msg { flex: 1; }
```

---

## 5. Token-Aware Compaction Thresholds

### Problem
`compact_after_chars / 4` is a rough heuristic. Different texts (multilingual,
code snippets, long vs. short responses) have very different token densities.
A char-based threshold means compaction triggers at unpredictable token
boundaries.

### Approach

Use the LLM's tokenizer to count tokens rather than characters. Two options:

#### Option A — Use the LLM client's tokenizer (preferred)

If the LLM client exposes a tokenizer (e.g., `tiktoken` for OpenAI models,
or the tokenizer from the model's HuggingFace config), use it directly:

```python
from tiktoken import get_encoding

ENCODING = get_encoding("cl100k_base")  # matches gpt-4o, gpt-3.5-turbo

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))
```

Then change `Conversation.needs_compaction()`:

```python
def needs_compaction(self) -> bool:
    return (
        self.size_tokens() > self.compact_after_tokens
        and len(self.turns) > self.keep_recent_turns
    )

def size_tokens(self) -> int:
    total = count_tokens(self.summary or "")
    for user_text, assistant_text in self.turns:
        total += count_tokens(user_text) + count_tokens(assistant_text)
    return total
```

#### Option B — Estimate tokens from chars (fallback)

For models without an exposed tokenizer, keep the char-based heuristic
but make it configurable and document the approximation:

```python
# vivo.toml
[memory]
compact_after_chars = 65000       # approximate 16k tokens (÷4 heuristic)
```

### Config Schema Changes

```python
# config_schema.py
"compact_after_tokens": {
    "type": "integer",
    "min": 1000,
    "max": 128000,
    "default": 16000,
    "unit": "tokens",
    "help": "Trigger compaction when conversation history (summary + turns) exceeds this many tokens.",
    "hidden": True,  # hidden by default; advanced users can enable
}
```

---

## 6. Compaction Guardrails

### Problem
After many compaction rounds, progressive summarization can degrade —
information is lost as each summary becomes input to the next summarization.

### Solutions

#### 6a. Hard summary length limit

The existing `SUMMARY_MAX_TOKENS = 300` in `agent.py` already caps summary
length, but it's not enforced strictly. Make it a hard limit:

```python
# agent.py
def summarize(self, messages: List[dict]) -> str:
    response = self._client.chat(..., max_tokens=200)  # conservative budget
    return response.choices[0].message.content[:500]  # hard char cap
```

#### 6b. Periodic summary reset

Every N compactions, generate a "fresh" summary from the full history
(starting over) rather than doing another progressive step. This is more
expensive (more tokens) but prevents degradation:

```python
RESET_EVERY = 10  # compactions

def compact(self, summarize: Summarizer) -> bool:
    ...
    reset = self._compaction_count % RESET_EVERY == 0
    if reset:
        # Summarize ALL old turns (no previous summary)
        messages = [(u, a) for u, a in old]
    else:
        # Progressive: prepend previous summary
        messages = [f"Previous summary: {old_summary}"] + list(old)
    ...
```

---

## Summary of Changes by Area

### Backend (`app/`)

| File | Change |
|------|--------|
| `conversation.py` | `broadcast_compact()`, token-aware `size_tokens()`, compaction counter |
| `pipeline.py` | `broadcast_compact()`, `broadcast_log_entry()`, `on_compact()` |
| `memory.py` | `DreamStore` class, `dream_candidates()`, LLM dream pass |
| `logging_store.py` | NEW — `LogStore` class with file persistence |
| `agent.py` | Enforce `max_tokens` strictly, add dream persona method |
| `main.py` | `/api/dreams`, `/api/logs` endpoints |
| `config.py` | `compact_after_tokens` setting (option A) |
| `config_schema.py` | New schema keys for token thresholds, logging |

### Frontend (`static/`)

| File | Change |
|------|--------|
| `index.html` | New dialogs: `dreams`, `logs`; nav buttons: `btn-dreams`, `btn-logs` |
| `app.js` | Element caches, open/close handlers, WebSocket event handlers, render functions |
| `style.css` | Styles for dream entries, log entries, compact expand/collapse |

### WebSocket Messages

| Type | Direction | When |
|------|-----------|------|
| `"compact"` | server -> client | After compaction completes |
| `"dream"` | server -> client | Dream phase changes (starting/complete) |
| `"log_entry"` | server -> client | Every system log entry |

---

## Implementation Order (Recommended)

1. **§4 System Logging Page** — foundational, makes everything else observable
2. **§1 Make Compaction Visible** — quick win, low risk, high user value
3. **§5 Token-Aware Thresholds** — backend-only improvement
4. **§3 Dreams Page** — new UI, depends on dream changes
5. **§2 LLM-Based Dreaming** — replaces current scoring, more complex
6. **§6 Compaction Guardrails** — polish, requires compaction to be visible first