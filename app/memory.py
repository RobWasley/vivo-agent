from __future__ import annotations

import logging
import re
import time
from datetime import date
import json
from pathlib import Path
from threading import RLock
from typing import Iterable
from uuid import uuid4

log = logging.getLogger("vivo.memory")

DEFAULT_MEMORY_PATH = "data/memory.md"
DEFAULT_DREAMS_PATH = "data/dreams.json"
MEMORY_LOCK = RLock()


class MemoryStore:
    """Local archive with a small curated core-memory Markdown index."""

    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self.path = Path(path)
        self.facts_path = self.path.with_suffix(".json")
        with MEMORY_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text("# Memory\n\n", encoding="utf-8")
            if not self.facts_path.exists():
                self._save_facts(self._legacy_facts())
                self._write_index()

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def observe(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        with MEMORY_LOCK:
            facts = self._facts()
            if any(fact.lower() == item["text"].lower() for item in facts):
                return
            facts.append(self._new_fact(fact, core=True))
            self._save_facts(facts)
            self._write_index(facts)

    def list(self) -> list[dict[str, str]]:
        return self._facts()

    def add(self, fact: str, core: bool = False) -> dict[str, str]:
        fact = fact.strip()
        if not fact:
            raise ValueError("memory fact is required")
        if len(fact) > 500:
            raise ValueError("memory fact must be at most 500 characters")
        with MEMORY_LOCK:
            facts = self._facts()
            if any(fact.lower() == item["text"].lower() for item in facts):
                raise ValueError("memory fact already exists")
            item = self._new_fact(fact, core=core)
            facts.append(item)
            self._save_facts(facts)
            self._write_index(facts)
            return item

    def update(self, fact_id: str, text: str) -> dict[str, str]:
        text = text.strip()
        if not text:
            raise ValueError("memory fact is required")
        if len(text) > 500:
            raise ValueError("memory fact must be at most 500 characters")
        with MEMORY_LOCK:
            facts = self._facts()
            for item in facts:
                if item["id"] == fact_id:
                    item["text"] = text
                    self._save_facts(facts)
                    self._write_index(facts)
                    return item
            raise KeyError(fact_id)

    def delete(self, fact_id: str) -> None:
        with MEMORY_LOCK:
            facts = self._facts()
            remaining = [item for item in facts if item["id"] != fact_id]
            if len(remaining) == len(facts):
                raise KeyError(fact_id)
            self._save_facts(remaining)
            self._write_index(remaining)

    def set_core(self, fact_id: str, core: bool) -> dict[str, str]:
        with MEMORY_LOCK:
            facts = self._facts()
            for item in facts:
                if item["id"] == fact_id:
                    item["core"] = bool(core)
                    self._save_facts(facts)
                    self._write_index(facts)
                    return item
            raise KeyError(fact_id)

    def core_summary(self) -> str:
        facts = [item["text"] for item in self._facts() if item["core"]]
        return "\n".join(f"- {fact}" for fact in facts) or "No important memory yet."

    def archive_text(self) -> str:
        facts = self._facts()
        if not facts:
            return "(no saved memory)"
        return "\n".join(
            f"- [{item['date']}] {item['text']}" + (" (core)" if item["core"] else "")
            for item in facts
        )

    def _score(self, candidate: str) -> tuple[int, int, str]:
        text = candidate.lower()
        score = 0
        if re.search(r"prefer|likes|prefers|favorite|important|remember|needs|always", text):
            score += 3
        if re.search(r"project|repo|vivo|agent|memory|reminder|time|weather|location|timezone|units", text):
            score += 2
        if len(text.split()) <= 12:
            score += 1
        count = sum(1 for token in re.findall(r"[a-z0-9]+", text) if len(token) > 3)
        return score, count, candidate

    def dream(self, candidates: Iterable[str] | None = None) -> str:
        items = list(candidates) if candidates is not None else [item["text"] for item in self._facts()]
        items = [
            item.strip() for item in items
            if item and item.strip() and not self._is_transient(item)
        ]
        if not items:
            return "No important memory yet."

        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        scored = sorted(deduped, key=lambda item: self._score(item), reverse=True)
        keep = scored[:3]
        summary = "\n".join(f"- {item}" for item in keep)
        return summary

    def consolidate(self) -> str:
        """Promote high-value archive facts into the small core-memory index."""
        with MEMORY_LOCK:
            summary = self.dream()
            selected = set() if summary == "No important memory yet." else {
                line[2:] for line in summary.splitlines()
            }
            facts = self._facts()
            for item in facts:
                item["core"] = item["text"] in selected
            self._save_facts(facts)
            self._write_index(facts)
            return summary

    def _dream_candidates(self, limit: int = 20) -> list[dict[str, str]]:
        """Facts worth handing to an LLM dream pass: core facts first (they
        must survive), then the most recent archive facts."""
        facts = self._facts()
        core = [item for item in facts if item["core"]]
        archive = sorted(
            (item for item in facts if not item["core"]),
            key=lambda item: item["date"],
            reverse=True,
        )
        return (core + archive)[:limit]

    def apply_dream(self, result: dict) -> dict:
        """Apply a structured dream pass: keep new facts, drop pruned ones.
        Core facts are never pruned. Returns stats about what changed."""
        stats = {"added": 0, "skipped": 0, "pruned": 0}
        with MEMORY_LOCK:
            facts = self._facts()
            for raw in result.get("new_facts") or []:
                text = str(raw).strip()
                if not text or len(text) > 500:
                    stats["skipped"] += 1
                    continue
                if any(text.lower() == item["text"].lower() for item in facts):
                    stats["skipped"] += 1
                    continue
                facts.append(self._new_fact(text, core=False))
                stats["added"] += 1
            core_ids = {item["id"] for item in facts if item["core"]}
            for raw_id in result.get("pruned") or []:
                fact_id = str(raw_id)
                if fact_id in core_ids:
                    stats["skipped"] += 1
                    continue
                if any(item["id"] == fact_id for item in facts):
                    facts = [item for item in facts if item["id"] != fact_id]
                    stats["pruned"] += 1
            self._save_facts(facts)
            self._write_index(facts)
        return stats

    @staticmethod
    def _is_transient(candidate: str) -> bool:
        text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", candidate.strip())
        return text.lower().startswith(("reminder fired:", "dream update:"))

    def _legacy_facts(self) -> list[dict[str, str]]:
        text = self.read()
        facts = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("- "):
                value = line[2:].strip()
                if not self._is_transient(value):
                    facts.append(self._new_fact(
                        re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", value), core=True
                    ))
        return facts

    def _facts(self) -> list[dict[str, str]]:
        try:
            raw = json.loads(self.facts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        return [
            {
                "id": str(item["id"]),
                "text": str(item["text"]),
                "date": str(item["date"]),
                "core": bool(item.get("core", False)),
            }
            for item in raw
            if isinstance(item, dict) and all(key in item for key in ("id", "text", "date"))
        ]

    @staticmethod
    def _new_fact(text: str, core: bool = False) -> dict[str, str]:
        return {"id": uuid4().hex, "text": text, "date": date.today().isoformat(), "core": core}

    def _save_facts(self, facts: list[dict[str, str]]) -> None:
        temporary_path = self.facts_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.facts_path)

    def _write_index(self, facts: list[dict[str, str]] | None = None) -> None:
        facts = facts if facts is not None else self._facts()
        facts = [item for item in facts if item["core"]]
        content = "# Memory\n"
        if facts:
            groups: dict[str, list[str]] = {}
            for item in facts:
                groups.setdefault(item["date"], []).append(item["text"])
            for fact_date, items in groups.items():
                content += f"\n## {fact_date}\n\n"
                content += "\n".join(f"- {item}" for item in items) + "\n"
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(self.path)


class DreamStore:
    """History of dream passes: one JSON record per pass.

    Records are appended newest-last and listed that way; a corrupt file is
    treated as empty, like the rest of vivo's persistence.
    """

    def __init__(self, path: str = DEFAULT_DREAMS_PATH):
        self.path = Path(path)
        with MEMORY_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._save([])

    def list(self) -> list[dict]:
        return self._dreams()

    def get(self, dream_id: str) -> dict | None:
        for dream in self.list():
            if dream.get("id") == dream_id:
                return dream
        return None

    def add(self, dream: dict) -> dict:
        record = dict(dream)
        record.setdefault("id", uuid4().hex)
        record.setdefault("ts", time.time())
        with MEMORY_LOCK:
            dreams = self._dreams()
            dreams.append(record)
            self._save(dreams)
        return record

    def clear(self) -> None:
        with MEMORY_LOCK:
            self._save([])

    def _dreams(self) -> list[dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        return [
            dream
            for dream in raw
            if isinstance(dream, dict) and isinstance(dream.get("ts"), (int, float))
        ]

    def _save(self, dreams: list[dict]) -> None:
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(dreams, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.path)


class DreamScheduler:
    """Background timer for periodic memory consolidation.

    Each pass asks the LLM to distil the archive into a summary, new facts,
    cross-fact connections and facts to prune (`Agent.dream_pass`). If no
    agent is wired up, or the LLM pass fails, the deterministic
    high-value-fact selection (`MemoryStore.consolidate`) runs instead, so
    dreaming keeps working with a cold or broken model. Every pass is
    recorded in a `DreamStore` and reported through `on_state` (called with
    ``(active, phase)``) and `on_complete` (called with the record).
    """

    def __init__(
        self,
        store: MemoryStore,
        interval_seconds: float = 3600.0,
        on_state=None,
        agent=None,
        dreams: DreamStore | None = None,
        on_complete=None,
    ):
        self.store = store
        self.interval_seconds = interval_seconds
        self.agent = agent
        self.dreams = dreams
        self._stop = False
        self._thread = None
        self._state = False
        self._state_cb = on_state
        self._complete_cb = on_complete

    def _set_state(self, active: bool, phase: str = "") -> None:
        if self._state == active and not phase:
            return
        self._state = active
        if self._state_cb is not None:
            self._state_cb(active, phase)

    def trigger(self) -> dict:
        """Run one dream pass immediately and return its record.

        The pass is intentionally synchronous so a manual trigger can return
        its result to the caller (HTTP or UI) without a side-channel wait.
        """
        self._set_state(True, "consolidating")
        try:
            record = self._run_pass()
        except Exception:
            log.exception("dream pass failed; keeping previous memory")
            record = self._failed_record()
        finally:
            self._set_state(False)
        if self._complete_cb is not None:
            try:
                self._complete_cb(record)
            except Exception:
                log.exception("dream completion callback failed")
        return record

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = __import__("threading").Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop:
            time.sleep(self.interval_seconds)
            if self._stop:
                return
            self.trigger()

    # -- passes -------------------------------------------------------------
    def _run_pass(self) -> dict:
        if self.agent is not None:
            try:
                return self._llm_pass()
            except Exception:
                log.exception(
                    "LLM dream pass failed; using deterministic consolidation"
                )
        return self._deterministic_pass()

    def _llm_pass(self) -> dict:
        candidates = self.store._dream_candidates()
        result = self.agent.dream_pass(self.store.archive_text(), candidates)
        stats = self.store.apply_dream(result)
        record = {
            "id": uuid4().hex,
            "ts": time.time(),
            "llm": True,
            "summary": result.get("summary") or "",
            "new_facts": result.get("new_facts") or [],
            "connections": result.get("connections") or [],
            "pruned": result.get("pruned") or [],
            "stats": stats,
        }
        if self.dreams is not None:
            record = self.dreams.add(record)
        return record

    def _deterministic_pass(self) -> dict:
        summary = self.store.consolidate()
        promoted = sum(1 for item in self.store.list() if item["core"])
        record = {
            "id": uuid4().hex,
            "ts": time.time(),
            "llm": False,
            "summary": summary,
            "new_facts": [],
            "connections": [],
            "pruned": [],
            "stats": {"promoted": promoted},
        }
        if self.dreams is not None:
            record = self.dreams.add(record)
        return record

    def _failed_record(self) -> dict:
        return {
            "id": uuid4().hex,
            "ts": time.time(),
            "llm": False,
            "summary": "No important memory yet.",
            "new_facts": [],
            "connections": [],
            "pruned": [],
            "stats": {},
        }
