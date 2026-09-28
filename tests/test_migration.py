from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.migration import (
    migrate,
    _migrate_memories,
    _migrate_dreams,
    _migrate_sessions,
    _human_date,
    _year_month,
    _read_json,
    _write_text,
    _write_conversation,
    rollback,
)
from app.memory import DreamStore, MemoryStore
from app.vault import Vault


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create a temporary data and vault directory."""
    data_dir = str(tmp_path / "data")
    vault_dir = str(tmp_path / "vault")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir, vault_dir


@pytest.fixture
def legacy_files(tmp_dirs):
    """Create sample legacy JSON files."""
    data_dir, vault_dir = tmp_dirs

    # Create memory.json
    facts = [
        {"id": "fact001", "text": "Rob lives in Gloucester, UK", "date": "2025-06-15", "core": True},
        {"id": "fact002", "text": "Rob prefers Python", "date": "2025-06-16", "core": False},
    ]
    Path(data_dir, "memory.json").write_text(json.dumps(facts))

    # Create memory.md
    Path(data_dir, "memory.md").write_text("# Memory\n\n## 2025-06-15\n\n- Rob lives in Gloucester, UK")

    # Create dreams.json
    Path(data_dir, "dreams.json").write_text(
        json.dumps([{
            "id": "dream001",
            "ts": 1718505600.0,
            "llm": True,
            "summary": "User is based in UK",
            "new_facts": ["Rob uses Linux"],
            "connections": [],
            "pruned": [],
        }])
    )

    # Create sessions directory and files
    sessions_dir = Path(data_dir) / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # sessions.json
    Path(data_dir, "sessions.json").write_text(
        json.dumps({
            "active": "sess001",
            "sessions": {
                "sess001": {
                    "name": "Test Session",
                    "created": 1718505600.0,
                    "last_used": 1718600000.0,
                    "turns": 5,
                }
            }
        })
    )

    # Session file
    (sessions_dir / "sess001.json").write_text(
        json.dumps({
            "summary": "Test conversation summary",
            "turns": [
                ["Hello", "Hi there!"],
                ["How are you?", "I'm doing well, thanks!"],
            ],
            "compactions": 0,
        })
    )

    return data_dir, vault_dir


class TestHumanDate:
    """Test timestamp to date conversion utilities."""

    def test_timestamp_to_date(self):
        # 1718505600 = 2024-06-16 00:00:00 UTC
        date_str = _human_date(1718505600.0)
        assert date_str == "2024-06-16"

    def test_string_timestamp(self):
        date_str = _human_date("1718505600.0")
        assert date_str == "2024-06-16"

    def test_invalid_timestamp(self):
        date_str = _human_date("invalid")
        assert len(date_str) == 10  # YYYY-MM-DD format


class TestYearMonth:
    """Test timestamp to year/month conversion."""

    def test_timestamp_to_year_month(self):
        # 1718505600 = 2024-06-16 00:00:00 UTC
        year, ym = _year_month(1718505600.0)
        assert year == "2024"
        assert ym == "2024-06"

    def test_invalid_timestamp(self):
        year, ym = _year_month("invalid")
        assert len(year) == 4
        assert len(ym) == 7  # YYYY-MM format


class TestReadJson:
    """Test JSON file reading utility."""

    def test_read_valid_json(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')
        result = _read_json(str(test_file))
        assert result == {"key": "value"}

    def test_read_invalid_json(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text("{invalid")
        result = _read_json(str(test_file), default={})
        assert result == {}

    def test_read_missing_file(self):
        result = _read_json("/nonexistent/path.json", default="fallback")
        assert result == "fallback"


class TestWriteConversation:
    """Test conversation document writing."""

    def test_write_conversation_with_summary(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        _write_conversation(
            vault,
            session_id="test-1",
            summary="Test summary",
            turns=[["Hello", "Hi!"]],
            title="Test",
        )
        doc = vault.read("sessions", "test-1.md")
        assert doc is not None
        assert doc.frontmatter["id"] == "test-1"
        assert "Test summary" in doc.content
        assert "Hello" in doc.content
        assert "Hi!" in doc.content
        # Turns live in the frontmatter (runtime layout), not just the body
        turns = doc.frontmatter["turns"]
        assert turns == [["Hello", "Hi!"]]

    def test_write_conversation_without_summary(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        _write_conversation(
            vault,
            session_id="test-2",
            summary="",
            turns=[["What's up?", "Not much!"]],
            title="Simple",
        )
        doc = vault.read("sessions", "test-2.md")
        assert doc is not None
        assert "What's up?" in doc.content
        assert doc.frontmatter["turns"] == [["What's up?", "Not much!"]]


class TestMigrateMemories:
    """Test memory migration."""

    def test_migrate_no_files(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        counts = _migrate_memories(vault, data_dir)
        assert counts["memories"] == 0

    def test_migrate_with_facts(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        os.makedirs(data_dir, exist_ok=True)

        facts = [
            {"id": "f1", "text": "Rob likes Python", "date": "2024-06-15", "core": True},
            {"id": "f2", "text": "Rob uses Linux", "date": "2024-06-16", "core": False},
        ]
        Path(data_dir, "memory.json").write_text(json.dumps(facts))
        Path(data_dir, "memory.md").write_text("# Memory\n\n- Core fact")

        counts = _migrate_memories(vault, data_dir)
        assert counts["memories"] == 3  # 2 facts + MEMORY.md

        # Verify fact documents
        doc1 = vault.read("memories", "facts", "f1.md")
        assert doc1 is not None
        assert doc1.content == "Rob likes Python"
        # the runtime MemoryStore reads text from the frontmatter
        assert doc1.frontmatter["text"] == "Rob likes Python"
        assert doc1.frontmatter["core"] is True

        doc2 = vault.read("memories", "facts", "f2.md")
        assert doc2 is not None
        assert doc2.content == "Rob uses Linux"
        assert doc2.frontmatter["core"] is False

        # Verify MEMORY.md
        memory_doc = vault.read("memories", "MEMORY.md")
        assert memory_doc is not None
        assert "Rob likes Python" in memory_doc.content


class TestMigrateDreams:
    """Test dream migration."""

    def test_migrate_no_file(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        counts = _migrate_dreams(vault, data_dir)
        assert counts["dreams"] == 0

    def test_migrate_with_dreams(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        os.makedirs(data_dir, exist_ok=True)

        dreams = [{
            "id": "d1",
            "ts": 1718505600.0,
            "llm": True,
            "summary": "Test dream",
            "new_facts": ["Fact A"],
            "connections": ["C1"],
            "pruned": ["p1"],
        }]
        Path(data_dir, "dreams.json").write_text(json.dumps(dreams))

        counts = _migrate_dreams(vault, data_dir)
        assert counts["dreams"] == 1

        doc = vault.read("memories", "dreams", "2024-06-16-d1.md")
        assert doc is not None
        assert "Test dream" in doc.content
        assert doc.frontmatter["llm"] is True
        # the runtime DreamStore reads all of these from the frontmatter
        assert doc.frontmatter["summary"] == "Test dream"
        assert doc.frontmatter["new_facts"] == ["Fact A"]
        assert doc.frontmatter["connections"] == ["C1"]
        assert doc.frontmatter["pruned"] == ["p1"]
        assert doc.frontmatter["stats"] == {}


class TestMigrateSessions:
    """Test session migration."""

    def test_migrate_no_sessions(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        counts = _migrate_sessions(vault, data_dir)
        assert counts["conversations"] == 0

    def test_migrate_legacy_conversation(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        os.makedirs(data_dir, exist_ok=True)

        Path(data_dir, "conversation.json").write_text(
            json.dumps({"summary": "Legacy summary", "turns": [["Hi", "Hello"]]})
        )

        counts = _migrate_sessions(vault, data_dir)
        assert counts["conversations"] == 1

        # Find the migrated conversation (skip the index document)
        docs = [d for d in vault.list_docs("sessions") if d.id != "index"]
        assert len(docs) == 1
        assert "Legacy summary" in docs[0].content
        assert docs[0].frontmatter["turns"] == [["Hi", "Hello"]]

        # A runtime session index was written
        index = vault.read("sessions", "index.md")
        assert index is not None
        assert index.frontmatter["active"] in index.frontmatter["sessions"]

    def test_migrate_from_sessions_index(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        os.makedirs(data_dir, exist_ok=True)

        sessions_dir = Path(data_dir) / "sessions"
        sessions_dir.mkdir(exist_ok=True)

        Path(data_dir, "sessions.json").write_text(
            json.dumps({
                "active": "s1",
                "sessions": {
                    "s1": {
                        "name": "My Session",
                        "created": 1718505600.0,
                        "last_used": 1718600000.0,
                        "turns": 2,
                    }
                }
            })
        )

        (sessions_dir / "s1.json").write_text(
            json.dumps({
                "summary": "Session summary",
                "turns": [["Q", "A"]],
                "compactions": 1,
            })
        )

        counts = _migrate_sessions(vault, data_dir)
        assert counts["conversations"] == 1

        doc = vault.read("sessions", "s1.md")
        assert doc is not None
        assert doc.frontmatter["id"] == "s1"
        assert "Session summary" in doc.content
        assert "Q" in doc.content
        assert "A" in doc.content
        assert doc.frontmatter["turns"] == [["Q", "A"]]
        assert doc.frontmatter["compactions"] == 1

        # The runtime index carries the session's metadata
        index = vault.read("sessions", "index.md")
        assert index is not None
        assert index.frontmatter["active"] == "s1"
        entry = index.frontmatter["sessions"]["s1"]
        assert entry["name"] == "My Session"
        assert entry["created"] == 1718505600.0
        assert entry["last_used"] == 1718600000.0
        assert entry["turns"] == 1


class TestFullMigration:
    """Test end-to-end migration with all legacy files."""

    def test_full_migration(self, legacy_files):
        data_dir, vault_dir = legacy_files
        vault = Vault(root=vault_dir)

        counts = migrate(data_dir=data_dir, vault_root=vault_dir)

        assert counts["memories"] == 3  # 2 facts + MEMORY.md
        assert counts["dreams"] == 1
        assert counts["conversations"] == 1
        assert counts["total"] == 5

        # Verify vault documents exist
        memory = vault.read("memories", "MEMORY.md")
        assert memory is not None

        facts = vault.list_docs("memories/facts")
        assert len(facts) == 2

        dreams = vault.list_docs("memories/dreams")
        assert len(dreams) == 1

        convos = [d for d in vault.list_docs("sessions") if d.id != "index"]
        assert len(convos) == 1

    def test_full_migration_documents_readable_by_runtime_stores(
        self, legacy_files, monkeypatch
    ):
        """The standalone script's output must be consumable by the runtime
        stores (same frontmatter layout), or an upgrade amnesiates the data."""
        data_dir, vault_dir = legacy_files
        migrate(data_dir=data_dir, vault_root=vault_dir)

        # The stores' legacy paths are CWD-relative; point them at the tmp tree
        monkeypatch.chdir(Path(data_dir).parent)

        store = MemoryStore(vault_root=vault_dir)
        facts = {item["id"]: item for item in store.list()}
        assert facts["fact001"]["text"] == "Rob lives in Gloucester, UK"
        assert facts["fact001"]["core"] is True
        assert facts["fact002"]["text"] == "Rob prefers Python"
        assert "Rob lives in Gloucester, UK" in store.core_summary()

        dreams = DreamStore(vault_root=vault_dir)
        records = {d["id"]: d for d in dreams.list()}
        assert records["dream001"]["summary"] == "User is based in UK"
        assert records["dream001"]["new_facts"] == ["Rob uses Linux"]
        assert records["dream001"]["llm"] is True

    def test_migration_idempotent(self, legacy_files):
        """Migration should be idempotent - running twice doesn't duplicate."""
        data_dir, vault_dir = legacy_files
        vault = Vault(root=vault_dir)

        # First run
        migrate(data_dir=data_dir, vault_root=vault_dir)

        # Remove marker to simulate re-run (normally this wouldn't happen)
        marker_path = os.path.join(data_dir, ".migrated")
        Path(marker_path).unlink()

        # Second run with same legacy files (but vault already has the docs)
        migrate(data_dir=data_dir, vault_root=vault_dir)

        # Counts should not have doubled
        facts = vault.list_docs("memories/facts")
        assert len(facts) == 2

    def test_migration_skips_if_already_migrated(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        vault = Vault(root=vault_dir)
        marker_path = os.path.join(data_dir, ".migrated")
        os.makedirs(data_dir, exist_ok=True)
        Path(marker_path).touch()

        counts = migrate(data_dir=data_dir, vault_root=vault_dir)
        assert counts["total"] == 0


class TestRollback:
    """Test migration rollback."""

    def test_rollback_removes_marker(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        marker_path = os.path.join(data_dir, ".migrated")
        os.makedirs(data_dir, exist_ok=True)
        Path(marker_path).touch()

        rollback(data_dir)
        assert not os.path.exists(marker_path)

    def test_rollback_no_marker(self, tmp_dirs):
        data_dir, vault_dir = tmp_dirs
        marker_path = os.path.join(data_dir, ".migrated")
        if os.path.exists(marker_path):
            os.unlink(marker_path)

        rollback(data_dir)
        assert not os.path.exists(marker_path)
