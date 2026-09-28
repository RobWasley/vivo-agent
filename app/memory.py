from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Iterable
from uuid import uuid4

from app.reminders import next_due, utcnow
from app.vault import Vault

log = logging.getLogger("vivo.memory")

MEMORY_LOCK = RLock()


def _as_list(value) -> list:
    """Normalise a frontmatter value that should be a list (tolerating
    legacy docs where it was stored as a JSON string or a scalar)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _as_dict(value) -> dict:
    """Normalise a frontmatter value that should be a dict (tolerating
    legacy docs where it was stored as a string)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _norm_text(text: str) -> str:
    """Normalise a fact for duplicate detection: trimmed, casefolded, and
    with whitespace runs collapsed to single spaces."""
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


class MemoryStore:
    """Local archive with a small curated core-memory Markdown index."""

    def __init__(self, path: str | None = None, vault_root: str | None = None):
        if path is not None:
            vault_root = str(Path(path).parent / "vault")
        self.vault = Vault(root=vault_root or "data/vault")
        self._legacy_md = Path(path) if path is not None else Path("data") / "memory.md"
        self._legacy_facts = self._legacy_md.with_suffix(".json")
        MEMORY_LOCK.acquire()
        try:
            # Ensure MEMORY.md exists
            if not self.vault.read("memories", "MEMORY.md"):
                self.vault.write(
                    "memories", "MEMORY.md",
                    title="Core Memory",
                    content="# Memory\n",
                )
            self._migrate_legacy()
        finally:
            MEMORY_LOCK.release()

    def read(self) -> str:
        """Return the core memory index (MEMORY.md)."""
        doc = self.vault.read("memories", "MEMORY.md")
        return doc.content if doc else "# Memory\n"

    def observe(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        with MEMORY_LOCK:
            facts = self._facts()
            if _norm_text(fact) in {_norm_text(item["text"]) for item in facts}:
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
            if _norm_text(fact) in {_norm_text(item["text"]) for item in facts}:
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

    def dedupe(self) -> int:
        """Collapse exact (normalised) duplicate facts, keeping the best copy:
        a core fact beats an archive copy, then the earliest date wins.
        Called once at engine startup; safe to re-run (idempotent).
        Returns how many fact documents were removed."""
        with MEMORY_LOCK:
            facts = self._facts()
            winners: dict[str, dict[str, str]] = {}
            order: list[str] = []
            for item in facts:
                key = _norm_text(item["text"]) or f"::{item['id']}"
                if key not in winners:
                    winners[key] = item
                    order.append(key)
                    continue
                # rank: core first, then earlier date (lower tuple wins)
                if self._dedupe_rank(item) < self._dedupe_rank(winners[key]):
                    winners[key] = item
            keep = {item["id"] for item in winners.values()}
            removed = len(facts) - len(keep)
            if removed:
                self._save_facts([item for item in facts if item["id"] in keep])
                self._write_index()
                log.info("deduped %d duplicate memory facts", removed)
            return removed

    @staticmethod
    def _dedupe_rank(item: dict[str, str]) -> tuple:
        return (not item["core"], item["date"] or "")

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
        """Apply a structured dream pass: keep new facts, drop pruned ones,
        and merge near-duplicate clusters into their clearest phrasing. Core
        facts are never pruned outright, but a merge may drop a duplicate
        copy of one as long as the kept fact survives. Returns stats about
        what changed."""
        stats = {"added": 0, "skipped": 0, "pruned": 0, "merged": 0}
        with MEMORY_LOCK:
            facts = self._facts()
            for raw in result.get("new_facts") or []:
                text = str(raw).strip()
                if not text or len(text) > 500:
                    stats["skipped"] += 1
                    continue
                if _norm_text(text) in {_norm_text(item["text"]) for item in facts}:
                    stats["skipped"] += 1
                    continue
                facts.append(self._new_fact(text, core=False))
                stats["added"] += 1
            by_id = {item["id"]: item for item in facts}
            prune_ids = set()
            for raw_id in result.get("pruned") or []:
                fact_id = str(raw_id)
                item = by_id.get(fact_id)
                if item is None:
                    continue
                if item["core"]:
                    stats["skipped"] += 1
                    continue
                prune_ids.add(fact_id)
            merge_ids = set()
            for group in result.get("merge") or []:
                if not isinstance(group, dict):
                    continue
                keep = str(group.get("keep") or "").strip()
                if not keep or keep not in by_id or keep in prune_ids:
                    continue
                for raw_id in group.get("drop") or []:
                    fact_id = str(raw_id)
                    if (
                        fact_id in by_id
                        and fact_id != keep
                        and fact_id not in prune_ids | merge_ids
                    ):
                        merge_ids.add(fact_id)
            stats["pruned"] = len(prune_ids)
            stats["merged"] = len(merge_ids)
            drop_ids = prune_ids | merge_ids
            if drop_ids:
                facts = [item for item in facts if item["id"] not in drop_ids]
            self._save_facts(facts)
            self._write_index(facts)
        return stats

    @staticmethod
    def _is_transient(candidate: str) -> bool:
        text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", candidate.strip())
        return text.lower().startswith(("reminder fired:", "dream update:"))

    # -- legacy (pre-vault) migration ---------------------------------
    def _migrate_legacy(self) -> None:
        """Import pre-vault memory data once, then remove the legacy files.

        memory.json held the fact archive and memory.md the curated core
        index; both are rebuilt from the vault on every access now, so
        without this step an upgrade would silently amnesiate the store.
        """
        try:
            self._migrate_legacy_facts()
            self._migrate_legacy_index()
        except Exception:  # noqa: BLE001 - migration must not break startup
            log.exception("legacy memory migration failed")

    def _migrate_legacy_facts(self) -> None:
        if not self._legacy_facts.exists():
            return
        try:
            raw = json.loads(self._legacy_facts.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("unreadable legacy facts file %s", self._legacy_facts)
            return
        if not isinstance(raw, list):
            return
        facts = [
            {
                "id": str(item["id"]),
                "text": str(item["text"]),
                "date": str(item.get("date", "")),
                "core": bool(item.get("core", False)),
            }
            for item in raw
            if isinstance(item, dict) and item.get("id") and item.get("text")
        ]
        if facts:
            existing = self._facts()
            have = {item["id"] for item in existing}
            have_text = {_norm_text(item["text"]) for item in existing}
            imported = 0
            skipped = 0
            for fact in facts:
                if fact["id"] in have:
                    continue
                # Legacy files can hold the same text under different ids;
                # dedupe by normalised text as well as by id.
                if _norm_text(fact["text"]) in have_text:
                    skipped += 1
                    continue
                have_text.add(_norm_text(fact["text"]))
                self.vault.write(
                    "memories", "facts", f"{fact['id']}.md",
                    title=fact["text"][:80] or "Memory",
                    frontmatter={
                        "id": fact["id"],
                        "text": fact["text"],
                        "date": fact["date"],
                        "core": fact["core"],
                    },
                    content=fact["text"],
                )
                imported += 1
            if imported:
                log.info("migrated %d legacy memory facts to vault", imported)
            if skipped:
                log.info("skipped %d duplicate legacy memory facts", skipped)
        try:
            self._legacy_facts.unlink()
        except OSError:
            pass

    def _migrate_legacy_index(self) -> None:
        if not self._legacy_md.exists():
            return
        try:
            legacy = self._legacy_md.read_text(encoding="utf-8").strip()
        except OSError:
            return
        doc = self.vault.read("memories", "MEMORY.md")
        current = (doc.content if doc else "").strip()
        if legacy and (not current or current == "# Memory"):
            self.vault.write(
                "memories", "MEMORY.md",
                title="Core Memory",
                content=legacy,
            )
            log.info("migrated legacy memory index to vault")
        elif (
            not legacy
            and (not current or current == "# Memory")
            and any(item["core"] for item in self._facts())
        ):
            # Legacy index file was empty, but core facts exist (e.g. only
            # memory.json survived): rebuild the index from them.
            self._write_index(self._facts())
            log.info("rebuilt memory index from core facts")
        try:
            self._legacy_md.unlink()
        except OSError:
            pass

    def _facts(self) -> list[dict[str, str]]:
        """Read all fact documents from the vault and return as list of dicts."""
        docs = self.vault.list_docs("memories/facts")
        facts = []
        for doc in docs:
            if doc.frontmatter:
                facts.append({
                    "id": str(doc.frontmatter.get("id", "")),
                    "text": str(doc.frontmatter.get("text", "")),
                    "date": str(doc.frontmatter.get("date", "")),
                    "core": bool(doc.frontmatter.get("core", False)),
                })
        return facts

    def _save_facts(self, facts: list[dict[str, str]]) -> None:
        """Write all facts to their individual vault documents."""
        # Read existing docs to track which are still present
        existing_docs = self.vault.list_docs("memories/facts")
        existing_ids = {doc.frontmatter.get("id") for doc in existing_docs
                        if doc and doc.frontmatter}

        # Remove deleted facts
        current_ids = {f["id"] for f in facts}
        for fact_id in (existing_ids - current_ids):
            if fact_id:
                try:
                    self.vault.delete("memories", "facts", f"{fact_id}.md")
                except Exception:
                    pass

        # Write/update all facts
        for fact in facts:
            fact_id = fact["id"]
            self.vault.write(
                "memories", "facts", f"{fact_id}.md",
                title=fact["text"][:80] if fact["text"] else "Memory",
                frontmatter={
                    "id": fact_id,
                    "text": fact["text"],
                    "date": fact["date"],
                    "core": fact.get("core", False),
                },
                content=fact["text"],
            )

    @staticmethod
    def _new_fact(text: str, core: bool = False) -> dict[str, str]:
        return {"id": uuid4().hex, "text": text, "date": date.today().isoformat(), "core": core}

    def _write_index(self, facts: list[dict[str, str]] | None = None) -> None:
        facts = facts if facts is not None else self._facts()
        core_facts = [item for item in facts if item["core"]]
        content_lines = ["# Memory"]
        if core_facts:
            groups: dict[str, list[str]] = {}
            for item in core_facts:
                groups.setdefault(item["date"], []).append(item["text"])
            for fact_date, items in groups.items():
                content_lines.append(f"\n## {fact_date}\n\n")
                content_lines.extend(f"- {item}" for item in items)
        self.vault.write(
            "memories", "MEMORY.md",
            title="Core Memory",
            content="\n".join(content_lines),
        )


class DreamStore:
    """History of dream passes: one markdown document per pass.

    Uses the vault's memories/dreams/ directory for storage.
    """

    def __init__(self, path: str | None = None, vault_root: str | None = None):
        if path is not None:
            vault_root = str(Path(path).parent / "vault")
        self.vault = Vault(root=vault_root or "data/vault")
        self._legacy = Path(path) if path is not None else Path("data") / "dreams.json"
        MEMORY_LOCK.acquire()
        try:
            self._migrate_legacy()
        finally:
            MEMORY_LOCK.release()

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
            self._save(record)
        return record

    def clear(self) -> None:
        with MEMORY_LOCK:
            docs = self.vault.list_docs("memories/dreams")
            for doc in docs:
                try:
                    filename = Path(doc.path).name
                    self.vault.delete("memories", "dreams", filename)
                except Exception:
                    pass

    # -- legacy (pre-vault) migration ---------------------------------
    def _migrate_legacy(self) -> None:
        """Import the pre-vault dreams.json archive once, then remove it.

        Per-record dedupe (by date-id filename) makes the import safe even
        when the vault already holds newer runtime dreams."""
        if not self._legacy.exists():
            return
        try:
            raw = json.loads(self._legacy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("unreadable legacy dreams file %s", self._legacy)
            return
        if not isinstance(raw, list) or not raw:
            return
        migrated = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record.setdefault("id", uuid4().hex)
            record.setdefault("ts", time.time())
            if self.vault.read("memories", "dreams", self._dream_name(record)):
                continue  # already present
            self._save(record)
            migrated += 1
        if migrated:
            log.info("migrated %d legacy dream records to vault", migrated)
        try:
            self._legacy.unlink()
        except OSError:
            pass

    def _dreams(self) -> list[dict]:
        docs = self.vault.list_docs("memories/dreams")
        dreams = []
        for doc in docs:
            if doc and doc.frontmatter:
                ts = doc.frontmatter.get("ts")
                if ts is not None and isinstance(ts, (int, float, str)):
                    try:
                        ts_num = float(ts)
                    except (ValueError, TypeError):
                        continue
                    dreams.append({
                        "id": str(doc.frontmatter.get("id", "")),
                        "ts": ts_num,
                        "llm": bool(doc.frontmatter.get("llm", False)),
                        "summary": str(doc.frontmatter.get("summary", "")),
                        "new_facts": _as_list(doc.frontmatter.get("new_facts")),
                        "connections": _as_list(doc.frontmatter.get("connections")),
                        "pruned": _as_list(doc.frontmatter.get("pruned")),
                        "merge": _as_list(doc.frontmatter.get("merge")),
                        "stats": _as_dict(doc.frontmatter.get("stats")),
                    })
        dreams.sort(key=lambda d: d["ts"])
        return dreams

    @staticmethod
    def _dream_name(dream: dict) -> str:
        ts = dream.get("ts", time.time())
        try:
            ts_num = float(ts)
        except (TypeError, ValueError):
            ts_num = time.time()
        date_str = time.strftime("%Y-%m-%d", time.gmtime(ts_num))
        return f"{date_str}-{dream['id']}.md"

    def _save(self, dream: dict) -> None:
        ts = dream.get("ts", time.time())
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = time.time()
        date_str = time.strftime("%Y-%m-%d", time.gmtime(ts))
        dream_id = dream["id"]
        summary = dream.get("summary", "")
        llm = dream.get("llm", False)
        new_facts = dream.get("new_facts", [])
        connections = dream.get("connections", [])
        pruned = dream.get("pruned", [])
        merge = dream.get("merge", [])
        stats = dream.get("stats", {})

        body_lines = []
        if summary:
            body_lines.append(f"## Summary\n\n{summary}\n")
        if new_facts:
            body_lines.append("## New Facts\n\n")
            body_lines.extend(f"- {f}" for f in new_facts)
            body_lines.append("")
        if connections:
            body_lines.append("## Connections\n\n")
            body_lines.extend(f"- {c}" for c in connections)
            body_lines.append("")
        if pruned:
            body_lines.append("## Pruned\n\n")
            body_lines.extend(f"- {p}" for p in pruned)
            body_lines.append("")
        if merge:
            body_lines.append("## Merged\n\n")
            for group in merge:
                if not isinstance(group, dict):
                    continue
                keep = str(group.get("keep", ""))
                dropped = ", ".join(str(d) for d in group.get("drop") or [])
                body_lines.append(f"- kept {keep}, dropped {dropped}")
            body_lines.append("")
        if stats:
            body_lines.append("## Stats\n\n")
            for k, v in stats.items():
                body_lines.append(f"- {k}: {v}")
            body_lines.append("")

        self.vault.write(
            "memories", "dreams", self._dream_name(dream),
            title=f"Dream {date_str}",
            frontmatter={
                "id": dream_id,
                "ts": str(ts),
                "llm": llm,
                "summary": summary,
                "new_facts": new_facts,
                "connections": connections,
                "pruned": pruned,
                "merge": merge,
                "stats": stats,
            },
            content="\n".join(body_lines),
        )


class DreamScheduler:
    """Background scheduler for memory consolidation.

    Each pass asks the LLM to distil the archive into a summary, new facts,
    cross-fact connections, facts to prune and near-duplicate merges
    (`Agent.dream_pass`). If no agent is wired up, or the LLM pass fails, the
    deterministic high-value-fact selection (`MemoryStore.consolidate`) runs
    instead, so dreaming keeps working with a cold or broken model. Meaningful
    passes (ones that changed memory or produced connections) are recorded in
    a `DreamStore`; empty re-summaries are skipped so the history does not
    bloat. Every pass is reported through `on_state` (called with
    ``(active, phase)``) and `on_complete` (called with the record).

    When `dream_time` (HH:MM) is set the pass runs once a night at that
    wall-clock time in the user's time zone; otherwise it falls back to a
    free-running `interval_seconds` cadence. An empty time and a non-positive
    interval disable dreaming (the loop idles and re-checks on config change).
    """

    def __init__(
        self,
        store: MemoryStore,
        interval_seconds: float = 3600.0,
        dream_time: str | None = None,
        on_state=None,
        agent=None,
        dreams: DreamStore | None = None,
        on_complete=None,
    ):
        self.store = store
        self.interval_seconds = interval_seconds
        self.dream_time = dream_time or ""
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
            delay = self._next_delay()
            if delay is None:
                # Dreaming is disabled (no daily time, non-positive interval):
                # idle and re-check so a config edit can re-enable it.
                self._sleep(30.0)
                continue
            self._sleep(delay)
            if self._stop:
                return
            self.trigger()

    def _next_delay(self) -> float | None:
        """Seconds until the next scheduled pass, or None while disabled.

        A set `dream_time` wins (the next daily wall-clock occurrence in the
        user's zone); otherwise a positive `interval_seconds` is used.
        """
        if self.dream_time and self.dream_time.strip():
            try:
                due = next_due("daily", time=self.dream_time.strip())
            except ValueError as e:
                log.warning("invalid dream_time %r (%s); using the interval", self.dream_time, e)
            else:
                return max(0.0, (due - utcnow()).total_seconds())
        if self.interval_seconds > 0:
            return float(self.interval_seconds)
        return None

    def _sleep(self, seconds: float) -> None:
        """Sleep for `seconds`, waking on stop so shutdown stays prompt even
        across a long (e.g. nightly) wait."""
        end = time.monotonic() + max(0.0, seconds)
        while not self._stop:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 30.0))

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

    @staticmethod
    def _is_meaningful(record: dict, stats: dict) -> bool:
        """Whether a pass is worth recording in the dream history: it changed
        memory, or produced connections. A pass that only re-summarised adds
        nothing but noise."""
        return bool(
            stats.get("added") or stats.get("pruned") or stats.get("merged")
            or record.get("connections")
        )

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
            "merge": result.get("merge") or [],
            "stats": stats,
        }
        if self.dreams is not None and self._is_meaningful(record, stats):
            record = self.dreams.add(record)
        return record

    def _deterministic_pass(self) -> dict:
        before = {item["id"] for item in self.store.list() if item["core"]}
        summary = self.store.consolidate()
        after = {item["id"] for item in self.store.list() if item["core"]}
        record = {
            "id": uuid4().hex,
            "ts": time.time(),
            "llm": False,
            "summary": summary,
            "new_facts": [],
            "connections": [],
            "pruned": [],
            "merge": [],
            "stats": {"promoted": len(after)},
        }
        if self.dreams is not None and before != after:
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
            "merge": [],
            "stats": {},
        }
