from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, main, tools
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
    assert len(store.list()) == 3
    assert sum(item["core"] for item in store.list()) == 2


def test_core_memory_is_a_curated_subset_of_the_archive(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    detailed = store.add("A project note for later")
    core = store.add("Rob prefers concise answers", core=True)

    assert "concise answers" in store.core_summary()
    assert "project note" not in store.core_summary()
    assert detailed["core"] is False and core["core"] is True

    store.set_core(detailed["id"], True)
    assert "project note" in store.read()
    assert "(core)" in store.archive_text()


def test_memory_tools_use_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    assert tools.execute("remember", {"fact": "Rob prefers short answers"}) == "memory saved"
    assert "Rob prefers short answers" in tools.execute("read_memory", {})


def test_memory_api_manages_archive_and_core(tmp_path):
    app = FastAPI()
    app.add_api_route("/api/memory", main.get_memory, methods=["GET"])
    app.add_api_route("/api/memory", main.add_memory, methods=["POST"])
    app.add_api_route("/api/memory/{fact_id}", main.update_memory, methods=["PUT"])
    app.add_api_route("/api/memory/{fact_id}", main.delete_memory, methods=["DELETE"])
    app.state.engines = type("Engines", (), {"memory": MemoryStore(str(tmp_path / "memory.md"))})()

    with TestClient(app) as client:
        created = client.post("/api/memory", json={"text": "Project details", "core": False})
        assert created.status_code == 200
        fact = created.json()["fact"]
        assert client.get("/api/memory").json()["core_count"] == 0

        updated = client.put(f"/api/memory/{fact['id']}", json={"text": "Project details", "core": True})
        assert updated.status_code == 200 and updated.json()["fact"]["core"] is True
        assert client.delete(f"/api/memory/{fact['id']}").status_code == 200
        assert client.get("/api/memory").json()["facts"] == []