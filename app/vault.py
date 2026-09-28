from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("vivo.vault")


FRONTMATTER_DELIMITER = "---"
VAULT_ROOT_DEFAULT = "data/vault"


class Document:
    """A single vault document with frontmatter and markdown body."""

    def __init__(
        self,
        id: str,
        path: str,
        title: str = "",
        created_at: str = "",
        updated_at: str = "",
        frontmatter: dict[str, Any] | None = None,
        content: str = "",
        tags: list[str] | None = None,
    ):
        self.id = id
        self.path = path
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
        self.frontmatter = frontmatter or {}
        self.content = content
        self.tags = tags or []

    @property
    def summary(self) -> str:
        return self.content[:200].strip() if self.content else ""


class VaultError(Exception):
    """Base exception for vault operations."""


class PathTraversalError(VaultError):
    """Attempted path traversal outside the vault."""


class FileLockedError(VaultError):
    """Operation attempted on a locked path."""


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and markdown body from text.

    Returns (frontmatter_dict, body_string).
    """
    frontmatter: dict[str, Any] = {}
    body = text

    stripped = text.lstrip("\n\r")
    if not stripped.startswith(FRONTMATTER_DELIMITER):
        return frontmatter, text

    parts = stripped.split(FRONTMATTER_DELIMITER, 2)
    if len(parts) < 3:
        return frontmatter, text

    fm_text = parts[1].strip()
    body = parts[2].lstrip("\n\r") if len(parts) > 2 else ""

    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.lower() in ("true",):
            value = True
        elif value.lower() in ("false",):
            value = False
        elif value.lower() in ("null", "none"):
            value = None
        elif value.isdigit():
            value = int(value)
        elif value.startswith("["):
            # Try to parse as JSON array first (handles nested arrays)
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                try:
                    inner = value[1:-1].strip()
                    if "," in inner:
                        value = [
                            v.strip().strip('"').strip("'")
                            for v in inner.split(",")
                            if v.strip()
                        ]
                    elif inner:
                        value = [
                            v.strip().strip('"').strip("'")
                            for v in inner.split()
                            if v.strip()
                        ]
                    else:
                        value = []
                except Exception:
                    value = []
        elif value.startswith("{"):
            # JSON object (e.g. a stats dict); keep the raw string if it
            # does not parse.
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        frontmatter[key] = value

    return frontmatter, body


def _format_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Format frontmatter dict into YAML-like text."""
    lines = []
    for key, value in frontmatter.items():
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        elif value is None:
            val_str = "null"
        elif isinstance(value, (list, dict)):
            # JSON round-trips exactly; the naive space-joined form shredded
            # list items containing spaces on read-back.
            val_str = json.dumps(value, ensure_ascii=False)
        else:
            val_str = str(value)
        lines.append(f"{key}: {val_str}")
    return "\n".join(lines)


class Vault:
    """Markdown-first document vault with frontmatter, search, and atomic writes."""

    def __init__(self, root: str | Path | None = None, log_path: str | None = None):
        self.root = Path(root or os.environ.get("MEMORY_VAULT", VAULT_ROOT_DEFAULT))
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(
            log_path
            or os.environ.get("MEMORY_VAULT_LOG", str(self.root / "vault.log"))
        )
        self._lock: set[str] = set()

    # ── Path safety ──────────────────────────────────────────────

    def _resolve_safe(self, *parts: str) -> Path:
        """Resolve a path inside the vault, rejecting traversal attacks."""
        base = self.root
        target = base.joinpath(*parts)
        resolved = Path(os.path.realpath(target))
        vault_real = Path(os.path.realpath(base))
        if not str(resolved).startswith(str(vault_real)):
            raise PathTraversalError(
                f"Path traversal blocked: {resolved} outside {vault_real}"
            )
        return resolved

    # ── Atomic writes ────────────────────────────────────────────

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write content atomically using rename-from-tempfile."""
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ── Document lifecycle ───────────────────────────────────────

    def read(self, *path_parts: str) -> Document | None:
        """Read a vault document by path parts.

        Args:
            *path_parts: Path components relative to vault root,
                        e.g. "conversations", "2025", "2025-06", "abc-123.md"
        Returns:
            Document if found, None otherwise.
        """
        target = self._resolve_safe(*path_parts)
        if not target.is_file():
            return None

        text = target.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)

        doc_id = fm.get("id", target.stem)
        return Document(
            id=str(doc_id),
            path=str(target),
            title=fm.get("title", doc_id),
            created_at=fm.get("created_at", ""),
            updated_at=fm.get("updated_at", ""),
            frontmatter=fm,
            content=body.strip(),
            tags=fm.get("tags", []),
        )

    def write(
        self,
        *path_parts: str,
        title: str = "",
        frontmatter: dict[str, Any] | None = None,
        content: str = "",
    ) -> Document:
        """Write a new vault document atomically.

        Args:
            *path_parts: Path components for the document.
            title: Display title.
            frontmatter: YAML frontmatter dict.
            content: Markdown body.
        Returns:
            The written Document.
        """
        if path_parts and path_parts[-1].startswith("."):
            raise VaultError("Documents cannot start with a dot")

        target = self._resolve_safe(*path_parts)
        now = datetime.now(timezone.utc).isoformat()
        fm = {
            "id": str(frontmatter.get("id", target.stem)) if frontmatter else str(target.stem),
            "created_at": now,
            "updated_at": now,
        }
        if title:
            fm["title"] = title
        if frontmatter:
            fm.update(frontmatter)

        content_lines = [FRONTMATTER_DELIMITER]
        content_lines.append(_format_frontmatter(fm))
        content_lines.append(FRONTMATTER_DELIMITER)
        if content.strip():
            content_lines.append("")
            content_lines.append(content.rstrip())

        doc = Document(
            id=fm["id"],
            path=str(target),
            title=title,
            created_at=fm["created_at"],
            updated_at=fm["updated_at"],
            frontmatter=fm,
            content=content,
        )
        self._atomic_write(target, "\n".join(content_lines))
        self._log("write", doc_id=fm["id"], path=str(target))
        return doc

    def update(
        self,
        *path_parts: str,
        title: str | None = None,
        frontmatter: dict[str, Any] | None = None,
        content: str | None = None,
        append: bool = False,
    ) -> Document:
        """Update an existing document atomically.

        Args:
            *path_parts: Path to the document.
            title: New title (optional).
            frontmatter: Fields to merge/overwrite.
            content: New content (or append if True).
            append: Whether to append to existing content.
        Returns:
            The updated Document.
        """
        existing = self.read(*path_parts)
        if not existing:
            raise VaultError(f"Document not found: {'/'.join(path_parts)}")

        now = datetime.now(timezone.utc).isoformat()
        fm = dict(existing.frontmatter)
        fm["updated_at"] = now

        if title is not None:
            fm["title"] = title
        if content is not None:
            if append:
                new_content = existing.content + "\n" + content
            else:
                new_content = content
        else:
            new_content = existing.content

        if frontmatter:
            fm.update(frontmatter)

        if title is not None:
            fm["title"] = title

        content_lines = [FRONTMATTER_DELIMITER]
        content_lines.append(_format_frontmatter(fm))
        content_lines.append(FRONTMATTER_DELIMITER)
        if new_content.strip():
            content_lines.append("")
            content_lines.append(new_content.rstrip())

        target = self._resolve_safe(*path_parts)
        self._atomic_write(target, "\n".join(content_lines))
        self._log("update", doc_id=fm["id"], path=str(target))

        return Document(
            id=fm["id"],
            path=str(target),
            title=fm.get("title", ""),
            created_at=fm.get("created_at", ""),
            updated_at=fm["updated_at"],
            frontmatter=fm,
            content=new_content,
        )

    def delete(self, *path_parts: str) -> bool:
        """Delete a vault document.

        Args:
            *path_parts: Path to the document.
        Returns:
            True if deleted, False if not found.
        """
        target = self._resolve_safe(*path_parts)
        if not target.is_file():
            return False

        doc = self.read(*path_parts)
        target.unlink()
        self._log("delete", doc_id=doc.id if doc else "unknown", path=str(target))

        # Clean up empty parent directories up to the vault root
        parent = target.parent
        while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

        return True

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 20,
        path_prefix: str | None = None,
        frontmatter_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Full-text search across vault documents.

        Args:
            query: Search terms (space-separated, case-insensitive).
            limit: Maximum results.
            path_prefix: Restrict to documents under this path.
            frontmatter_filter: Filter by frontmatter key=value pairs.
        Returns:
            List of matching Documents sorted by relevance.
        """
        terms = [t.lower() for t in query.split() if t]
        results: list[tuple[int, Document]] = []

        def _score_match(doc: Document) -> int:
            """Simple relevance scoring."""
            score = 0
            text_to_search = f"{doc.title.lower()} {doc.content.lower()}"
            for term in terms:
                if term in text_to_search:
                    score += 1
            return score

        def _matches_filter(doc: Document) -> bool:
            """Check frontmatter filter conditions."""
            if not frontmatter_filter:
                return True
            for key, value in frontmatter_filter.items():
                if doc.frontmatter.get(key) != value:
                    return False
            return True

        for root, dirs, files in os.walk(self.root):
            # Skip vault log directory
            if "vault.log" in root:
                continue

            root_path = Path(root)
            rel = str(root_path.relative_to(self.root))

            if path_prefix and not rel.startswith(path_prefix):
                continue

            for filename in files:
                if not filename.endswith(".md"):
                    continue
                if filename.startswith("."):
                    continue
                if filename == "vault.log":
                    continue

                try:
                    target = root_path / filename
                    doc_text = target.read_text(encoding="utf-8", errors="replace")
                    fm, body = _parse_frontmatter(doc_text)

                    if not _matches_filter(Document(
                        id=fm.get("id", target.stem),
                        path=str(target),
                        frontmatter=fm,
                        content=body,
                    )):
                        continue

                    if not terms:
                        score = 1
                    else:
                        score = _score_match(Document(
                            id=fm.get("id", target.stem),
                            path=str(target),
                            title=fm.get("title", target.stem),
                            frontmatter=fm,
                            content=body,
                        ))

                    if score > 0:
                        doc = Document(
                            id=fm.get("id", target.stem),
                            path=str(target),
                            title=fm.get("title", target.stem),
                            created_at=fm.get("created_at", ""),
                            updated_at=fm.get("updated_at", ""),
                            frontmatter=fm,
                            content=body,
                            tags=fm.get("tags", []),
                        )
                        results.append((score, doc))
                except Exception:
                    continue

        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:limit]]

    def list_docs(
        self,
        path: str | None = None,
        recursive: bool = True,
    ) -> list[Document]:
        """List all vault documents.

        Args:
            path: Subdirectory to list from.
            recursive: Whether to list recursively.
        Returns:
            List of Documents.
        """
        base = self.root
        if path:
            base = self._resolve_safe(path)
            if not base.is_dir():
                return []

        docs = []
        if recursive:
            for root, dirs, files in os.walk(base):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    if filename.startswith("."):
                        continue

                    try:
                        target = Path(root) / filename
                        doc_text = target.read_text(encoding="utf-8", errors="replace")
                        fm, body = _parse_frontmatter(doc_text)

                        doc = Document(
                            id=fm.get("id", target.stem),
                            path=str(target),
                            title=fm.get("title", target.stem),
                            created_at=fm.get("created_at", ""),
                            updated_at=fm.get("updated_at", ""),
                            frontmatter=fm,
                            content=body,
                            tags=fm.get("tags", []),
                        )
                        docs.append(doc)
                    except Exception:
                        continue
        else:
            for item in base.iterdir():
                if item.is_file() and item.name.endswith(".md") and not item.name.startswith("."):
                    try:
                        doc_text = item.read_text(encoding="utf-8", errors="replace")
                        fm, body = _parse_frontmatter(doc_text)

                        doc = Document(
                            id=fm.get("id", item.stem),
                            path=str(item),
                            title=fm.get("title", item.stem),
                            created_at=fm.get("created_at", ""),
                            updated_at=fm.get("updated_at", ""),
                            frontmatter=fm,
                            content=body,
                            tags=fm.get("tags", []),
                        )
                        docs.append(doc)
                    except Exception:
                        continue

        docs.sort(key=lambda d: d.updated_at or "", reverse=True)
        return docs

    # ── Locking ──────────────────────────────────────────────────

    def lock(self, *path_parts: str) -> None:
        """Lock a path to prevent concurrent modification."""
        path_key = "/".join(path_parts)
        self._lock.add(path_key)

    def unlock(self, *path_parts: str) -> None:
        """Unlock a previously locked path."""
        path_key = "/".join(path_parts)
        self._lock.discard(path_key)

    def is_locked(self, *path_parts: str) -> bool:
        """Check if a path is locked."""
        return "/".join(path_parts) in self._lock

    # ── Logging ──────────────────────────────────────────────────

    def _log(self, action: str, doc_id: str = "", path: str = "") -> None:
        """Append a vault operation log entry."""
        ts = datetime.now(timezone.utc).isoformat()
        log_line = f"{ts} | {action} | doc={doc_id} | path={path}\n"
        log.debug("Vault: %s", log_line.strip())
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass
