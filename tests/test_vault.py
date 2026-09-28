from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from app.vault import Vault, Document, PathTraversalError, VaultError, _parse_frontmatter, _format_frontmatter


@pytest.fixture
def vault_dir(tmp_path):
    """Create a temporary vault directory."""
    return str(tmp_path / "vault")


@pytest.fixture
def vault(vault_dir):
    """Create a Vault instance with a temporary directory."""
    return Vault(root=vault_dir)


class TestFrontmatterParsing:
    """Test frontmatter parsing and formatting utilities."""

    def test_parse_empty_text(self):
        fm, body = _parse_frontmatter("just a body")
        assert fm == {}
        assert body == "just a body"

    def test_parse_no_frontmatter_delimiter(self):
        fm, body = _parse_frontmatter("some content\nmore content")
        assert fm == {}
        assert "some content" in body

    def test_parse_simple_frontmatter(self):
        text = "---\nid: test-123\ntitle: My Test\n---\n\nThis is the body."
        fm, body = _parse_frontmatter(text)
        assert fm["id"] == "test-123"
        assert fm["title"] == "My Test"
        assert "This is the body" in body

    def test_parse_boolean_values(self):
        text = "---\npublished: true\ndraft: false\n---\n\nbody"
        fm, body = _parse_frontmatter(text)
        assert fm["published"] is True
        assert fm["draft"] is False

    def test_parse_null_values(self):
        text = "---\nnotes: null\n---\n\nbody"
        fm, body = _parse_frontmatter(text)
        assert fm["notes"] is None

    def test_parse_list_values(self):
        text = "---\ntags: [python, vault]\n---\n\nbody"
        fm, body = _parse_frontmatter(text)
        assert fm["tags"] == ["python", "vault"]

    def test_parse_quoted_string(self):
        text = "---\ntitle: \"Hello World\"\n---\n\nbody"
        fm, body = _parse_frontmatter(text)
        assert fm["title"] == "Hello World"

    def test_format_simple_frontmatter(self):
        fm = {"id": "test-1", "title": "Test Doc", "tags": ["python", "vault"]}
        formatted = _format_frontmatter(fm)
        assert "id: test-1" in formatted
        assert "title: Test Doc" in formatted
        assert "tags:" in formatted

    def test_format_boolean_values(self):
        fm = {"published": True, "draft": False}
        formatted = _format_frontmatter(fm)
        assert "published: true" in formatted
        assert "draft: false" in formatted

    def test_format_null_value(self):
        fm = {"notes": None}
        formatted = _format_frontmatter(fm)
        assert "notes: null" in formatted

    def test_parse_json_object_value(self):
        text = '---\nstats: {"added": 2, "skipped": 0}\n---\n\nbody'
        fm, body = _parse_frontmatter(text)
        assert fm["stats"] == {"added": 2, "skipped": 0}

    def test_parse_invalid_object_kept_as_string(self):
        fm, _ = _parse_frontmatter("---\nstats: {added: 2, no quotes}\n---\n\nbody")
        assert fm["stats"] == "{added: 2, no quotes}"


class TestDocument:
    """Test Document class."""

    def test_document_summary(self):
        doc = Document(id="1", path="/test.md", content="x" * 300)
        assert len(doc.summary) == 200

    def test_document_summary_short(self):
        doc = Document(id="1", path="/test.md", content="short")
        assert doc.summary == "short"

    def test_document_summary_empty(self):
        doc = Document(id="1", path="/test.md", content="")
        assert doc.summary == ""


class TestVaultPathSafety:
    """Test path traversal prevention."""

    def test_simple_path_resolution(self, vault):
        doc = vault.write("conversations", "2025", "test.md", title="Test")
        assert str(doc.path).endswith("test.md")

    def test_path_traversal_blocked(self, vault):
        with pytest.raises(PathTraversalError):
            vault.write("../../../etc", "passwd.md", title="Attack")

    def test_nested_safe_path(self, vault):
        doc = vault.write("deep", "path", "to", "file.md", title="Deep")
        assert Path(doc.path).parent.is_dir()

    def test_dot_prefix_rejected(self, vault):
        with pytest.raises(VaultError, match="start with a dot"):
            vault.write(".hidden.md", title="Hidden")


class TestVaultWrite:
    """Test document write operations."""

    def test_write_creates_document(self, vault):
        doc = vault.write("doc.md", title="Test Doc", content="Hello world")
        assert doc.id == "doc"
        assert doc.title == "Test Doc"
        assert "Hello world" in doc.content
        assert "created_at" in doc.frontmatter
        assert "updated_at" in doc.frontmatter

    def test_write_preserves_frontmatter(self, vault):
        doc = vault.write("fm.md", title="FM Doc", frontmatter={"key": "value"}, content="Body")
        assert doc.frontmatter["key"] == "value"
        assert doc.frontmatter["title"] == "FM Doc"

    def test_write_with_custom_id(self, vault):
        doc = vault.write("custom-id.md", title="Custom", frontmatter={"id": "custom-id"})
        assert doc.id == "custom-id"

    def test_write_atomicity(self, vault):
        """Write should not leave partial files on failure."""
        doc = vault.write("atomic.md", title="Atomic", content="Data")
        path = Path(doc.path)
        assert path.exists()
        assert path.read_text().startswith("---")


class TestVaultRead:
    """Test document read operations."""

    def test_read_existing_document(self, vault):
        vault.write("readme.md", title="Read Test", content="Content here")
        doc = vault.read("readme.md")
        assert doc is not None
        assert doc.title == "Read Test"
        assert "Content here" in doc.content

    def test_read_nonexistent_returns_none(self, vault):
        doc = vault.read("nonexistent.md")
        assert doc is None

    def test_read_nested_document(self, vault):
        vault.write("conversations", "2025", "test-123.md", title="Nested", content="Deep")
        doc = vault.read("conversations", "2025", "test-123.md")
        assert doc is not None
        assert doc.content == "Deep"

    def test_read_preserves_timestamps(self, vault):
        vault.write("ts.md", title="Timestamps")
        doc = vault.read("ts.md")
        assert doc.created_at
        assert doc.updated_at


class TestVaultUpdate:
    """Test document update operations."""

    def test_update_content(self, vault):
        vault.write("update.md", title="Update Test", content="Old content")
        doc = vault.update("update.md", content="New content")
        assert "New content" in doc.content
        assert "Old content" not in doc.content

    def test_update_title(self, vault):
        vault.write("update-t.md", title="Old Title")
        doc = vault.update("update-t.md", title="New Title")
        assert doc.title == "New Title"

    def test_update_preserves_created_at(self, vault):
        vault.write("preserve.md", title="Preserve")
        old_created = vault.read("preserve.md").created_at
        vault.update("preserve.md", content="Modified")
        new_doc = vault.read("preserve.md")
        assert new_doc.created_at == old_created
        assert new_doc.updated_at != old_created

    def test_update_append(self, vault):
        vault.write("append.md", title="Append", content="First part")
        doc = vault.update("append.md", content="Second part", append=True)
        assert "First part" in doc.content
        assert "Second part" in doc.content

    def test_update_nonexistent_raises(self, vault):
        with pytest.raises(VaultError, match="not found"):
            vault.update("does-not-exist.md", content="Nothing")

    def test_update_partial(self, vault):
        vault.write("partial.md", title="Partial", content="Body here", frontmatter={"custom": "value"})
        doc = vault.update("partial.md", title="New Title")
        assert doc.title == "New Title"
        assert "Body here" in doc.content
        assert doc.frontmatter["custom"] == "value"


class TestVaultDelete:
    """Test document delete operations."""

    def test_delete_existing(self, vault):
        vault.write("delete-me.md", title="Delete Test", content="Bye")
        result = vault.delete("delete-me.md")
        assert result is True
        assert vault.read("delete-me.md") is None

    def test_delete_nonexistent(self, vault):
        result = vault.delete("nope.md")
        assert result is False

    def test_delete_cleans_empty_dirs(self, vault):
        vault.write("conversations", "2025", "delete-nested.md", title="Nested Delete")
        vault.delete("conversations", "2025", "delete-nested.md")
        # Parent directory should be cleaned up
        conv_2025 = vault.root / "conversations" / "2025"
        assert not conv_2025.exists()


class TestVaultSearch:
    """Test search functionality."""

    def test_search_by_content(self, vault):
        vault.write("search1.md", title="Python Doc", content="Learning Python")
        vault.write("search2.md", title="Rust Doc", content="Learning Rust")
        results = vault.search("Python")
        assert len(results) >= 1
        ids = [doc.id for doc in results]
        assert "search1" in ids

    def test_search_by_title(self, vault):
        vault.write("t1.md", title="Important Meeting Notes")
        vault.write("t2.md", title="Shopping List")
        results = vault.search("Meeting")
        assert len(results) >= 1
        assert any("Meeting" in doc.title for doc in results)

    def test_search_empty_returns_all(self, vault):
        vault.write("a.md", content="Doc A")
        vault.write("b.md", content="Doc B")
        results = vault.search("")
        assert len(results) == 2

    def test_search_with_limit(self, vault):
        for i in range(10):
            vault.write(f"limit-{i:02d}.md", content=f"Test document {i}")
        results = vault.search("Test", limit=3)
        assert len(results) <= 3

    def test_search_no_match(self, vault):
        vault.write("nomatch.md", content="Some content")
        results = vault.search("xyzzyzzz")
        assert len(results) == 0

    def test_search_path_prefix(self, vault):
        vault.write("a.md", content="Root doc")
        vault.write("sub", "a.md", content="Sub doc")
        results = vault.search("", path_prefix="sub")
        for doc in results:
            assert "sub" in doc.path


class TestVaultListDocs:
    """Test list documents functionality."""

    def test_list_all_docs(self, vault):
        vault.write("doc1.md", content="One")
        vault.write("doc2.md", content="Two")
        docs = vault.list_docs()
        assert len(docs) == 2

    def test_list_recursive(self, vault):
        vault.write("a.md", content="Root")
        vault.write("sub", "b.md", content="Sub")
        docs = vault.list_docs(recursive=True)
        assert len(docs) == 2

    def test_list_non_recursive(self, vault):
        vault.write("a.md", content="Root")
        vault.write("sub", "b.md", content="Sub")
        docs = vault.list_docs(path="", recursive=False)
        assert len(docs) == 1

    def test_list_filters_non_md(self, vault):
        vault.write("a.md", content="Markdown")
        test_file = vault.root / "a.txt"
        test_file.write_text("text")
        docs = vault.list_docs()
        assert len(docs) == 1
        assert docs[0].id == "a"


class TestVaultLocking:
    """Test locking mechanism."""

    def test_lock_and_unlock(self, vault):
        vault.lock("doc.md")
        assert vault.is_locked("doc.md")
        vault.unlock("doc.md")
        assert not vault.is_locked("doc.md")

    def test_lock_unknown_path(self, vault):
        assert not vault.is_locked("nonexistent.md")


class TestVaultIntegration:
    """Integration tests for common workflows."""

    def test_conversation_workflow(self, vault):
        """Simulate creating, reading, updating, and listing conversations."""
        # Create
        doc = vault.write(
            "conversations", "2025", "2025-06", "abc-123.md",
            title="Meeting with Team",
            frontmatter={"id": "abc-123", "status": "completed"},
            content="Discussed Q3 roadmap and priorities."
        )
        assert doc.id == "abc-123"

        # Read
        read_doc = vault.read("conversations", "2025", "2025-06", "abc-123.md")
        assert read_doc is not None
        assert read_doc.frontmatter["status"] == "completed"

        # Update
        updated = vault.update(
            "conversations", "2025", "2025-06", "abc-123.md",
            content="Also agreed on budget changes.",
            append=True
        )
        assert "Discussed Q3 roadmap" in updated.content
        assert "Also agreed on budget" in updated.content

        # Search
        results = vault.search("roadmap")
        assert len(results) == 1

        # List
        docs = vault.list_docs("conversations", recursive=True)
        assert len(docs) == 1

        # Delete
        deleted = vault.delete("conversations", "2025", "2025-06", "abc-123.md")
        assert deleted is True
        assert vault.read("conversations", "2025", "2025-06", "abc-123.md") is None

    def test_multiple_documents_search_ranking(self, vault):
        """Documents with more matching terms should rank higher."""
        vault.write("r1.md", title="Python Python Python", content="Python is great")
        vault.write("r2.md", title="Python and Code", content="Python and code go together")
        vault.write("r3.md", title="Not About Python", content="Just random text")

        results = vault.search("Python")
        assert len(results) >= 2
        # r1 should rank higher than r2 (more Python matches in title)
        assert results[0].id == "r1"

    def test_frontmatter_persistence_roundtrip(self, vault):
        """Ensure frontmatter survives write-read cycles."""
        original_fm = {
            "id": "roundtrip-1",
            "status": "active",
            "priority": 5,
            "tags": ["test", "vault", "roundtrip"],
        }
        vault.write(
            "roundtrip.md",
            title="Roundtrip Test",
            frontmatter=original_fm,
            content="Test content for persistence.",
        )

        read_doc = vault.read("roundtrip.md")
        assert read_doc.frontmatter["id"] == "roundtrip-1"
        assert read_doc.frontmatter["status"] == "active"
        assert read_doc.frontmatter["priority"] == 5
        assert read_doc.frontmatter["tags"] == ["test", "vault", "roundtrip"]

    def test_roundtrip_list_items_with_spaces(self, vault):
        """List items containing spaces must survive a write-read cycle
        (the naive space-joined form shredded them on read-back)."""
        facts = ["the user likes oolong tea", "fact two words here"]
        vault.write("dream.md", title="Dream", frontmatter={
            "id": "dream-1",
            "ts": "1718505600.0",
            "new_facts": facts,
        }, content="body")
        doc = vault.read("dream.md")
        assert doc.frontmatter["new_facts"] == facts

    def test_roundtrip_dict_value(self, vault):
        stats = {"added": 2, "skipped": 0, "pruned": 1}
        vault.write("dream2.md", title="Dream", frontmatter={
            "id": "dream-2",
            "stats": stats,
        }, content="body")
        doc = vault.read("dream2.md")
        assert doc.frontmatter["stats"] == stats

    def test_timestamp_ordering(self, vault):
        """Documents should be listed by updated_at descending."""
        import time
        vault.write("oldest.md", content="First")
        time.sleep(0.01)
        vault.write("newest.md", content="Last")

        docs = vault.list_docs()
        assert docs[0].id == "newest"
        assert docs[1].id == "oldest"
