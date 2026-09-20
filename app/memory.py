from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable


DEFAULT_MEMORY_PATH = "data/memory.md"


class MemoryStore:
    """Simple local memory file with a lightweight 'dream' summariser."""

    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Memory\n\n", encoding="utf-8")

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def observe(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        existing = self.read()
        if fact.lower() in existing.lower():
            return
        with self.path.open("a", encoding="utf-8") as f:
            if not existing.endswith("\n"):
                f.write("\n")
            f.write(f"- [{date.today().isoformat()}] {fact}\n")

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
        items = list(candidates) if candidates is not None else self._read_facts()
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
        """Keep only high-value facts in a small, dated on-disk index."""
        summary = self.dream()
        facts = [] if summary == "No important memory yet." else [
            line[2:] for line in summary.splitlines()
        ]

        content = "# Memory\n"
        if facts:
            content += f"\n## {date.today().isoformat()}\n\n"
            content += "\n".join(f"- {fact}" for fact in facts) + "\n"
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(self.path)
        return summary

    @staticmethod
    def _is_transient(candidate: str) -> bool:
        text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", candidate.strip())
        return text.lower().startswith(("reminder fired:", "dream update:"))

    def _read_facts(self) -> list[str]:
        text = self.read()
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("- "):
                lines.append(line[2:].strip())
        return lines


class DreamScheduler:
    """Background timer for periodic memory consolidation."""

    def __init__(
        self,
        store: MemoryStore,
        interval_seconds: float = 3600.0,
        on_state=None,
    ):
        self.store = store
        self.interval_seconds = interval_seconds
        self._stop = False
        self._thread = None
        self._state = False
        self._state_cb = on_state

    def _set_state(self, active: bool) -> None:
        if self._state == active:
            return
        self._state = active
        if self._state_cb is not None:
            self._state_cb(active)

    def trigger(self) -> str:
        """Run one dream pass immediately, reporting the summary and state.

        The pass is intentionally synchronous so a manual trigger can return its
        result to the caller (HTTP or UI) without a side-channel wait.
        """
        self._set_state(True)
        try:
            return self.store.consolidate()
        finally:
            self._set_state(False)

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
            import time as _time
            _time.sleep(self.interval_seconds)
            if self._stop:
                return
            self.trigger()
