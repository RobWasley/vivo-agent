from app import config, tools
from app.memory import MemoryStore


def test_dream_compacts_store_without_recursive_updates(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    store.observe("Reminder fired: water the plants")
    store.observe("Rob prefers metric units")
    store.observe("The weather station is in the garden")

    summary = store.consolidate()

    assert "Rob prefers metric units" in summary
    assert "Reminder fired" not in summary
    assert "Dream update:" not in store.read()
    assert store.read().count("Rob prefers metric units") == 1
    assert "## " in store.read()


def test_memory_tools_use_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    assert tools.execute("remember", {"fact": "Rob prefers short answers"}) == "memory saved"
    assert "Rob prefers short answers" in tools.execute("read_memory", {})