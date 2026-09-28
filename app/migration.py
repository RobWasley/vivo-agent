"""Legacy JSON-to-Markdown migration for Vivo.

Converts (matching the runtime vault layout exactly):
  - data/memory.json  (facts list)  -> data/vault/memories/facts/<id>.md
  - data/memory.md    (core index)  -> data/vault/memories/MEMORY.md
  - data/dreams.json  (dream logs)  -> data/vault/memories/dreams/<date>-<id>.md
  - data/sessions.json + data/sessions/<id>.json -> data/vault/sessions/<id>.md
    (plus data/vault/sessions/index.md with the active pointer)

The migration is idempotent: if the vault already exists, it skips without
overwriting. A .migrated marker file is created to record completion.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.vault import Vault, Document

log = logging.getLogger("vivo.migration")

MIGRATION_MARKER = "data/.migrated"
LEGACY_MEMORY_JSON = "data/memory.json"
LEGACY_MEMORY_MD = "data/memory.md"
LEGACY_DREAMS_JSON = "data/dreams.json"
LEGACY_SESSIONS_JSON = "data/sessions.json"
LEGACY_SESSIONS_DIR = "data/sessions"
LEGACY_CONVERSATION_JSON = "data/conversation.json"


def _read_json(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning default on any error."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    """Write data as JSON to path."""
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_text(path: str) -> str:
    """Read a text file, returning empty string on any error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_text(path: str, text: str) -> None:
    """Write text to a file."""
    Path(path).write_text(text, encoding="utf-8")


def _human_date(timestamp: float | int | str) -> str:
    """Convert a timestamp to YYYY-MM-DD."""
    try:
        ts = float(timestamp)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _year_month(timestamp: float | int | str) -> tuple[str, str]:
    """Convert a timestamp to (year, YYYY-MM) for directory structure."""
    try:
        ts = float(timestamp)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y"), dt.strftime("%Y-%m")
    except (ValueError, TypeError, OSError):
        return datetime.now(timezone.utc).strftime("%Y"), datetime.now(timezone.utc).strftime("%Y-%m")


def _epoch(value) -> float:
    """Coerce an index timestamp to a float epoch (now if unparseable)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return time.time()


def migrate(data_dir: str = "data", vault_root: str | None = None) -> dict[str, int]:
    """Run the full migration from legacy JSON to Markdown vault.

    Returns:
        Dict with counts: {memories, dreams, conversations, total}
    """
    counts = {"memories": 0, "dreams": 0, "conversations": 0, "total": 0}
    marker_path = os.path.join(data_dir, ".migrated")

    # If already migrated, skip
    if os.path.exists(marker_path):
        log.info("Migration already completed (%s exists), skipping.", marker_path)
        return counts

    vault = Vault(root=vault_root or "data/vault")

    # Phase 1: Migrate memories
    counts.update(_migrate_memories(vault, data_dir))

    # Phase 2: Migrate dreams
    counts.update(_migrate_dreams(vault, data_dir))

    # Phase 3: Migrate sessions/conversations
    counts.update(_migrate_sessions(vault, data_dir))

    # Create marker
    os.makedirs(data_dir, exist_ok=True)
    Path(marker_path).touch()
    counts["total"] = counts["memories"] + counts["dreams"] + counts["conversations"]
    log.info(
        "Migration complete: %d memories, %d dreams, %d conversations (%d total)",
        counts["memories"],
        counts["dreams"],
        counts["conversations"],
        counts["total"],
    )
    return counts


def _migrate_memories(vault: Vault, data_dir: str = "data") -> dict[str, int]:
    """Migrate memory.json and memory.md to vault documents."""
    counts = {"memories": 0}
    memory_json = os.path.join(data_dir, "memory.json")
    memory_md = os.path.join(data_dir, "memory.md")

    # Migrate memory.json (facts list)
    if not os.path.exists(memory_json):
        return counts

    facts = _read_json(memory_json, [])
    if not isinstance(facts, list):
        return counts

    # Collect core facts for MEMORY.md
    core_facts: list[dict] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("id", "")
        if not fact_id:
            continue

        text = str(fact.get("text", ""))
        date_str = str(fact.get("date", ""))
        is_core = bool(fact.get("core", False))

        if not text:
            continue

        # Create individual fact document (frontmatter mirrors the runtime
        # MemoryStore layout, which reads every field from the frontmatter)
        doc = vault.write(
            "memories", "facts", f"{fact_id}.md",
            title=text[:80] if text else "Memory",
            frontmatter={
                "id": fact_id,
                "text": text,
                "date": date_str,
                "core": is_core,
            },
            content=text,
        )
        counts["memories"] += 1

        if is_core:
            core_facts.append({"id": fact_id, "text": text, "date": date_str})

    # Create MEMORY.md with core facts
    if os.path.exists(memory_md) or core_facts:
        content_lines = ["# Memory"]
        if core_facts:
            groups: dict[str, list[str]] = {}
            for item in core_facts:
                groups.setdefault(item["date"], []).append(item["text"])
            for fact_date, items in sorted(groups.items()):
                content_lines.append(f"\n## {fact_date}\n\n")
                content_lines.extend(f"- {item}" for item in items)

        vault.write(
            "memories", "MEMORY.md",
            title="Core Memory",
            content="\n".join(content_lines),
        )
        counts["memories"] += 1  # Count MEMORY.md

    return counts


def _migrate_dreams(vault: Vault, data_dir: str = "data") -> dict[str, int]:
    """Migrate dreams.json to vault documents."""
    counts = {"dreams": 0}
    dreams_json = os.path.join(data_dir, "dreams.json")

    if not os.path.exists(dreams_json):
        return counts

    dreams = _read_json(dreams_json, [])
    if not isinstance(dreams, list):
        return counts

    for dream in dreams:
        if not isinstance(dream, dict):
            continue

        dream_id = dream.get("id", "")
        if not dream_id:
            dream_id = str(int(dream.get("ts", time.time())))

        ts = dream.get("ts", time.time())
        summary = str(dream.get("summary", ""))
        llm = bool(dream.get("llm", False))
        added = dream.get("new_facts", [])
        connections = dream.get("connections", [])
        pruned = dream.get("pruned", [])
        stats = dream.get("stats", {})
        if not isinstance(stats, dict):
            stats = {}

        # Build content body
        body_lines = []
        if summary:
            body_lines.append(f"## Summary\n\n{summary}\n")
        if added:
            body_lines.append("## New Facts\n\n")
            for fact in added:
                body_lines.append(f"- {fact}")
            body_lines.append("")
        if connections:
            body_lines.append("## Connections\n\n")
            for conn in connections:
                body_lines.append(f"- {conn}")
            body_lines.append("")
        if pruned:
            body_lines.append("## Pruned\n\n")
            for pid in pruned:
                body_lines.append(f"- {pid}")
            body_lines.append("")

        date_str = _human_date(ts)
        doc = vault.write(
            "memories", "dreams", f"{date_str}-{dream_id}.md",
            title=f"Dream {date_str}",
            # Frontmatter mirrors the runtime DreamStore layout, which reads
            # every field (including stats) from the frontmatter.
            frontmatter={
                "id": dream_id,
                "ts": str(ts),
                "llm": llm,
                "summary": summary,
                "new_facts": added,
                "connections": connections,
                "pruned": pruned,
                "stats": stats,
            },
            content="\n".join(body_lines),
        )
        counts["dreams"] += 1

    return counts


def _migrate_sessions(vault: Vault, data_dir: str = "data") -> dict[str, int]:
    """Migrate sessions.json, session files and conversation.json into the
    runtime vault layout: sessions/<id>.md plus sessions/index.md."""
    counts = {"conversations": 0}
    sessions_json = os.path.join(data_dir, "sessions.json")
    sessions_dir = os.path.join(data_dir, "sessions")
    legacy_conversation = os.path.join(data_dir, "conversation.json")

    index: dict[str, Any] = {"active": None, "sessions": {}}

    # Migrate legacy single conversation.json
    if os.path.exists(legacy_conversation):
        raw = _read_json(legacy_conversation)
        if isinstance(raw, dict) and (raw.get("turns") or raw.get("summary")):
            session_id = f"legacy-{int(time.time())}"
            _write_conversation(
                vault,
                session_id=session_id,
                summary=raw.get("summary", ""),
                turns=raw.get("turns", []),
                title="Legacy Conversation",
                compactions=int(raw.get("compactions") or 0),
            )
            now = time.time()
            index["active"] = session_id
            index["sessions"][session_id] = {
                "name": "",
                "created": now,
                "last_used": now,
                "turns": len(raw.get("turns") or []),
            }
            counts["conversations"] += 1

    # Migrate the sessions.json index
    if os.path.exists(sessions_json):
        sessions_raw = _read_json(sessions_json, {})
        if isinstance(sessions_raw, dict):
            # Handle both formats: {"active": ..., "sessions": {...}} or just {...}
            if "sessions" in sessions_raw and isinstance(sessions_raw["sessions"], dict):
                sessions = sessions_raw["sessions"]
                active = sessions_raw.get("active")
            else:
                sessions = sessions_raw
                active = None
            if isinstance(active, str):
                index["active"] = active
            for session_id, entry in sessions.items():
                if not isinstance(entry, dict):
                    continue
                session_path = os.path.join(sessions_dir, f"{session_id}.json")
                if not os.path.exists(session_path):
                    continue
                raw = _read_json(session_path)
                if not isinstance(raw, dict):
                    continue
                turns = raw.get("turns", [])
                summary = raw.get("summary", "")
                if not turns and not summary:
                    continue
                _write_conversation(
                    vault,
                    session_id=session_id,
                    summary=summary,
                    turns=turns,
                    title=str(entry.get("name") or "") or f"Conversation {session_id[:8]}",
                    compactions=int(raw.get("compactions") or 0),
                )
                index["sessions"][session_id] = {
                    "name": str(entry.get("name") or ""),
                    "created": _epoch(entry.get("created")),
                    "last_used": _epoch(entry.get("last_used")),
                    "turns": len(turns),
                }
                counts["conversations"] += 1

    # Write the runtime session index if the vault does not have one yet
    if counts["conversations"] and not vault.read("sessions", "index.md"):
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body_lines = ["# Session Index"]
        for sid, entry in index["sessions"].items():
            body_lines.append(f"\n## {sid}")
            body_lines.append(f"\n- Name: {entry.get('name', '')}")
            body_lines.append(f"- Created: {entry.get('created', 0)}")
            body_lines.append(f"- Last used: {entry.get('last_used', 0)}")
            body_lines.append(f"- Turns: {entry.get('turns', 0)}")
        vault.write(
            "sessions", "index.md",
            title="Session Index",
            frontmatter={
                "id": "index",
                "updated_at": now_iso,
                "active": index["active"],
                "sessions": json.dumps(index["sessions"]),
            },
            content="\n".join(body_lines),
        )

    return counts


def _write_conversation(
    vault: Vault,
    session_id: str,
    summary: str,
    turns: list,
    title: str,
    compactions: int = 0,
) -> None:
    """Write a single conversation document in the runtime layout
    (sessions/<id>.md). Turns go in the frontmatter as a JSON array,
    mirroring Conversation._save, so the SessionStore loads them back."""
    clean_turns = [
        [str(t[0]), str(t[1])]
        for t in turns
        if isinstance(t, (list, tuple)) and len(t) == 2
    ]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    body_lines = []
    if summary:
        body_lines.append(f"## Summary\n\n{summary}\n")
    if compactions > 0:
        body_lines.append(f"## Metadata\n\nCompactions: {compactions}\n")
    if clean_turns:
        body_lines.append("## Turns\n\n")
        for i, (user_text, assistant_text) in enumerate(clean_turns, 1):
            body_lines.append(f"### Turn {i}\n\n")
            body_lines.append(f"**You**: {user_text}\n\n")
            body_lines.append(f"**Vivo**: {assistant_text}\n\n")

    vault.write(
        "sessions", f"{session_id}.md",
        title=title or session_id,
        frontmatter={
            "id": session_id,
            "updated_at": now_iso,
            "summary": summary or "",
            "compactions": compactions,
            "turns": json.dumps(clean_turns),
        },
        content="\n".join(body_lines),
    )


def rollback(data_dir: str = "data") -> None:
    """Roll back migration by removing the marker file.

    Does NOT delete vault documents. Callers must manually remove
    data/vault/ if they want a full rollback.
    """
    marker = Path(os.path.join(data_dir, ".migrated"))
    if marker.exists():
        marker.unlink()
        log.info("Migration marker removed. Vault documents preserved.")
    else:
        log.info("No migration marker found; nothing to roll back.")
