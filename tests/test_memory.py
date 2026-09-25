from fastapi import FastAPI
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor

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


def test_parallel_memory_writes_preserve_every_fact(tmp_path):
    memory_path = str(tmp_path / "memory.md")
    facts = [f"Fact {index}" for index in range(8)]

    with ThreadPoolExecutor(max_workers=len(facts)) as pool:
        list(pool.map(lambda fact: MemoryStore(memory_path).observe(fact), facts))

    assert {item["text"] for item in MemoryStore(memory_path).list()} == set(facts)


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


# -- LLM dreaming: candidates, application, history (T031) --------------------


def test_dream_candidates_list_core_facts_first(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    store.add("archive one")
    core = store.add("core fact", core=True)
    store.add("archive two")

    candidates = store._dream_candidates()

    assert candidates[0]["id"] == core["id"]
    assert {item["id"] for item in candidates} == {
        item["id"] for item in store.list()
    }


def test_apply_dream_adds_skips_and_prunes_only_non_core(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    core = store.add("Rob prefers concise answers", core=True)
    store.add("The weather station is in the garden")
    drop = store.add("Reminder fired: water the plants")

    stats = store.apply_dream({
        "summary": "S",
        "new_facts": [
            "Rob prefers concise answers",  # duplicate
            "New durable fact",
            "x" * 600,  # too long
            "   ",  # blank
        ],
        "connections": [],
        "pruned": [drop["id"], core["id"], "no-such-id"],
    })

    items = store.list()
    texts = [item["text"] for item in items]
    ids = [item["id"] for item in items]
    assert "New durable fact" in texts
    assert "Reminder fired: water the plants" not in texts
    assert "The weather station is in the garden" in texts
    assert core["id"] in ids  # core facts are never pruned
    assert stats == {"added": 1, "skipped": 4, "pruned": 1}


def test_apply_dream_empty_result_is_a_noop(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    store.add("a fact")

    assert store.apply_dream({}) == {"added": 0, "skipped": 0, "pruned": 0}
    assert len(store.list()) == 1


def test_dream_store_appends_lists_and_clears(tmp_path):
    from app.memory import DreamStore

    path = str(tmp_path / "dreams.json")
    dreams = DreamStore(path=path)
    assert dreams.list() == []
    first = dreams.add({"summary": "s1", "llm": False})
    second = dreams.add({"summary": "s2", "llm": True})
    assert first["id"] and second["id"] != first["id"]
    assert [d["summary"] for d in dreams.list()] == ["s1", "s2"]
    assert dreams.get(second["id"])["summary"] == "s2"

    dreams.clear()
    assert DreamStore(path=path).list() == []


def test_dream_store_corrupt_file_starts_empty(tmp_path):
    from app.memory import DreamStore

    p = tmp_path / "dreams.json"
    p.write_text("{not json", encoding="utf-8")
    dreams = DreamStore(path=str(p))
    assert dreams.list() == []
    record = dreams.add({"summary": "ok"})
    assert dreams.get(record["id"]) is not None


def test_dream_scheduler_llm_pass_applies_and_records(tmp_path):
    from app.memory import DreamScheduler, DreamStore, MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    dreams = DreamStore(path=str(tmp_path / "dreams.json"))
    core = store.add("core fact", core=True)
    archive = store.add("archive fact")

    class FakeAgent:
        def dream_pass(self, archive_text, candidates):
            assert "core fact" in archive_text
            return {
                "summary": "LLM summary",
                "new_facts": ["fresh fact"],
                "connections": ["core fact <-> archive fact"],
                "pruned": [archive["id"], core["id"]],
            }

    completed = []
    scheduler = DreamScheduler(
        store, interval_seconds=999, agent=FakeAgent(), dreams=dreams,
        on_complete=completed.append,
    )
    record = scheduler.trigger()

    assert record["llm"] is True
    assert record["summary"] == "LLM summary"
    assert record["stats"]["pruned"] == 1  # core fact survived
    assert [d["id"] for d in dreams.list()] == [record["id"]]
    assert completed == [record]
    texts = [item["text"] for item in store.list()]
    assert "fresh fact" in texts and "archive fact" not in texts


def test_dream_scheduler_falls_back_when_llm_raises(tmp_path):
    from app.memory import DreamScheduler, MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    store.observe("Rob prefers concise answers")

    class BrokenAgent:
        def dream_pass(self, archive_text, candidates):
            raise RuntimeError("model down")

    record = DreamScheduler(store, interval_seconds=999, agent=BrokenAgent()).trigger()

    assert record["llm"] is False
    assert "concise answers" in record["summary"]