import json

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


# -- legacy (pre-vault) data migration ----------------------------------------


def test_legacy_memory_json_is_migrated_into_the_vault(tmp_path):
    legacy_facts = tmp_path / "memory.json"
    legacy_facts.write_text(json.dumps([
        {"id": "abc", "text": "Rob lives in Gloucester", "date": "2025-06-15", "core": True},
        {"id": "def", "text": "Rob prefers Python", "date": "2025-06-16", "core": False},
    ]))
    (tmp_path / "memory.md").write_text(
        "# Memory\n\n## 2025-06-15\n\n- Rob lives in Gloucester"
    )

    store = MemoryStore(str(tmp_path / "memory.md"))

    assert {item["id"] for item in store.list()} == {"abc", "def"}
    assert "Rob lives in Gloucester" in store.core_summary()
    assert "Rob prefers Python" not in store.core_summary()
    assert "Rob lives in Gloucester" in store.read()
    assert not legacy_facts.exists()
    assert not (tmp_path / "memory.md").exists()

    # A reappeared legacy file must not duplicate already-imported facts
    legacy_facts.write_text(json.dumps([
        {"id": "abc", "text": "Rob lives in Gloucester", "date": "2025-06-15", "core": True},
        {"id": "ghi", "text": "A newer legacy fact", "date": "2025-06-20", "core": False},
    ]))
    store2 = MemoryStore(str(tmp_path / "memory.md"))
    assert {item["id"] for item in store2.list()} == {"abc", "def", "ghi"}


def test_legacy_index_rebuilt_when_only_facts_survived(tmp_path):
    (tmp_path / "memory.json").write_text(json.dumps([
        {"id": "abc", "text": "Rob lives in Gloucester", "date": "2025-06-15", "core": True},
    ]))
    (tmp_path / "memory.md").write_text("")

    store = MemoryStore(str(tmp_path / "memory.md"))

    assert "Rob lives in Gloucester" in store.core_summary()
    assert "Rob lives in Gloucester" in store.read()
    assert not (tmp_path / "memory.json").exists()


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
    assert stats == {"added": 1, "skipped": 4, "pruned": 1, "merged": 0}


def test_apply_dream_empty_result_is_a_noop(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    store.add("a fact")

    assert store.apply_dream({}) == {
        "added": 0, "skipped": 0, "pruned": 0, "merged": 0,
    }
    assert len(store.list()) == 1


def test_apply_dream_merges_near_duplicates(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    core_keep = store.add("Rob prefers concise answers", core=True)
    core_dup = store.add("Rob prefers concise answers.", core=True)
    archive_a = store.add("The weather station is in the garden")
    archive_b = store.add("weather station lives in the garden")
    keeper = store.add("keeper fact")
    pruned_away = store.add("stale fact")

    stats = store.apply_dream({
        "new_facts": [],
        "connections": [],
        "pruned": [pruned_away["id"], core_keep["id"]],
        "merge": [
            {"keep": core_keep["id"], "drop": [core_dup["id"]]},
            {"keep": archive_a["id"], "drop": [archive_b["id"]]},
            {"keep": "no-such-id", "drop": [keeper["id"]]},
            {"keep": keeper["id"], "drop": [pruned_away["id"]]},
        ],
    })

    ids = {item["id"] for item in store.list()}
    assert core_keep["id"] in ids  # a merge may drop a core fact's copy...
    assert core_dup["id"] not in ids  # ...as long as the kept one survives
    assert archive_a["id"] in ids and archive_b["id"] not in ids
    assert keeper["id"] in ids and pruned_away["id"] not in ids
    assert stats == {"added": 0, "skipped": 1, "pruned": 1, "merged": 2}


def test_apply_dream_merge_is_ignored_when_the_keep_is_pruned(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    keep = store.add("keeper fact")
    drop = store.add("duplicate copy")

    stats = store.apply_dream({
        "pruned": [keep["id"]],
        "merge": [{"keep": keep["id"], "drop": [drop["id"]]}],
    })

    # the merged copy must survive: it is the only copy left
    assert [item["id"] for item in store.list()] == [drop["id"]]
    assert stats == {"added": 0, "skipped": 0, "pruned": 1, "merged": 0}


def test_dedupe_collapses_normalized_duplicates(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.md"))
    for fact_id, text, fact_date, core in [
        ("f1", "The user likes oolong tea", "2026-01-05", False),
        ("f2", "the user   LIKES oolong  tea", "2026-01-02", False),
        ("f3", "Rob prefers metric units", "2026-01-01", True),
        ("f4", "roB  PREFERS metric units", "2026-02-01", False),
    ]:
        store.vault.write(
            "memories", "facts", f"{fact_id}.md",
            title=text[:80],
            frontmatter={
                "id": fact_id, "text": text, "date": fact_date, "core": core,
            },
            content=text,
        )

    removed = store.dedupe()

    assert removed == 2
    items = {item["id"]: item for item in store.list()}
    # earliest date wins among archive copies; core beats a later archive copy
    assert set(items) == {"f2", "f3"}
    assert store.vault.read("memories", "facts", "f1.md") is None
    assert store.vault.read("memories", "facts", "f4.md") is None


def test_legacy_migration_dedupes_by_text(tmp_path):
    legacy = tmp_path / "memory.json"
    legacy.write_text(json.dumps([
        {"id": "a1", "text": "Rob has a garden station ticket",
         "date": "2026-01-01", "core": True},
        {"id": "a2", "text": "rob has   a garden station ticket",
         "date": "2026-01-02", "core": False},
    ]), encoding="utf-8")

    store = MemoryStore(str(tmp_path / "memory.md"))

    items = store.list()
    assert len(items) == 1
    assert items[0]["id"] == "a1" and items[0]["core"] is True
    assert not legacy.exists()


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


def test_legacy_dreams_json_is_migrated_into_the_vault(tmp_path):
    from app.memory import DreamStore

    legacy = tmp_path / "dreams.json"
    legacy.write_text(json.dumps([
        {
            "id": "d1", "ts": 1718505600.0, "llm": True,
            "summary": "User is based in UK",
            "new_facts": ["Rob uses Linux", "the user likes oolong tea"],
            "connections": ["gloucester <-> garden station"],
            "pruned": [],
            "stats": {"added": 2, "skipped": 0, "pruned": 0},
        },
        {
            "id": "d2", "ts": 1718592000.0, "llm": False,
            "summary": "second pass", "new_facts": [],
            "connections": [], "pruned": [], "stats": {"promoted": 1},
        },
    ]))

    dreams = DreamStore(path=str(legacy))

    assert [d["id"] for d in dreams.list()] == ["d1", "d2"]
    first = dreams.get("d1")
    assert first["summary"] == "User is based in UK"
    # Multi-word list items and the stats dict must round-trip intact
    assert first["new_facts"] == ["Rob uses Linux", "the user likes oolong tea"]
    assert first["connections"] == ["gloucester <-> garden station"]
    assert first["stats"] == {"added": 2, "skipped": 0, "pruned": 0}
    assert not legacy.exists()

    # A second store over the same vault does not re-import or duplicate
    assert [d["id"] for d in DreamStore(path=str(legacy)).list()] == ["d1", "d2"]


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


def test_noise_dream_pass_is_not_recorded(tmp_path):
    from app.memory import DreamScheduler, DreamStore, MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    store.add("a stable fact")
    dreams = DreamStore(path=str(tmp_path / "dreams.json"))

    class QuietAgent:
        def dream_pass(self, archive_text, candidates):
            return {
                "summary": "nothing new to say",
                "new_facts": ["a stable fact"],  # exact duplicate -> skipped
                "connections": [],
                "pruned": [],
            }

    completed = []
    scheduler = DreamScheduler(
        store, interval_seconds=999, agent=QuietAgent(), dreams=dreams,
        on_complete=completed.append,
    )
    record = scheduler.trigger()

    assert record["summary"] == "nothing new to say"
    assert dreams.list() == []  # no change, no connections -> not recorded
    assert completed == [record]  # but still reported to the UI


def test_merge_only_dream_pass_is_recorded(tmp_path):
    from app.memory import DreamScheduler, DreamStore, MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    keep = store.add("The assistant is named vivo.")
    drop = store.add("vivo is the name of the assistant.")
    dreams = DreamStore(path=str(tmp_path / "dreams.json"))

    class MergingAgent:
        def dream_pass(self, archive_text, candidates):
            return {
                "summary": "tidied near-duplicates",
                "new_facts": [],
                "connections": [],
                "pruned": [],
                "merge": [{"keep": keep["id"], "drop": [drop["id"]]}],
            }

    scheduler = DreamScheduler(
        store, interval_seconds=999, agent=MergingAgent(), dreams=dreams,
    )
    record = scheduler.trigger()

    assert record["stats"]["merged"] == 1
    assert [item["id"] for item in store.list()] == [keep["id"]]
    assert [d["id"] for d in dreams.list()] == [record["id"]]
    # the merge group round-trips through the dream store
    assert dreams.get(record["id"])["merge"] == [
        {"keep": keep["id"], "drop": [drop["id"]]},
    ]


# -- scheduling (daily time vs interval fallback) -----------------------------


def test_dream_scheduler_daily_time_drives_next_delay(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    import app.memory as mem
    from app.memory import DreamScheduler, MemoryStore

    now = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mem, "utcnow", lambda: now)
    monkeypatch.setattr(mem, "next_due", lambda repeat, time=None: now + timedelta(hours=5))

    scheduler = DreamScheduler(MemoryStore(str(tmp_path / "m.md")),
                               interval_seconds=9999, dream_time="03:00")
    assert scheduler._next_delay() == 5 * 3600


def test_dream_scheduler_daily_time_uses_user_zone(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    import app.config as config
    import app.memory as mem
    import app.reminders as reminders
    from app.memory import DreamScheduler, MemoryStore

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    # 09:00 UTC == 10:00 in London (BST, +1); the 12:00 dream is 2h away.
    now = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mem, "utcnow", lambda: now)
    monkeypatch.setattr(reminders, "utcnow", lambda: now)

    scheduler = DreamScheduler(MemoryStore(str(tmp_path / "m.md")),
                               interval_seconds=0, dream_time="12:00")
    delay = scheduler._next_delay()
    due_local = (now + timedelta(seconds=delay)).astimezone(ZoneInfo("Europe/London"))
    assert (due_local.hour, due_local.minute) == (12, 0)
    assert delay == 2 * 3600


def test_dream_scheduler_interval_fallback_when_no_time(tmp_path):
    from app.memory import DreamScheduler, MemoryStore

    store = MemoryStore(str(tmp_path / "m.md"))
    assert DreamScheduler(store, interval_seconds=3600, dream_time="")._next_delay() == 3600.0
    assert DreamScheduler(store, interval_seconds=3600, dream_time=None)._next_delay() == 3600.0


def test_dream_scheduler_invalid_time_falls_back_to_interval(tmp_path):
    from app.memory import DreamScheduler, MemoryStore

    # "25:99" is not a valid HH:MM -> next_due raises -> the interval is used.
    scheduler = DreamScheduler(MemoryStore(str(tmp_path / "m.md")),
                               interval_seconds=1200, dream_time="25:99")
    assert scheduler._next_delay() == 1200.0


def test_dream_scheduler_disabled_when_no_time_and_no_interval(tmp_path):
    from app.memory import DreamScheduler, MemoryStore

    store = MemoryStore(str(tmp_path / "m.md"))
    assert DreamScheduler(store, interval_seconds=0, dream_time="")._next_delay() is None
    assert DreamScheduler(store, interval_seconds=0, dream_time=None)._next_delay() is None