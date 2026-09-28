"""T044: /api/graph — the vault as an Obsidian-style node graph.

build_graph assembles one node per vault document (core anchor, facts,
dreams, sessions) with explicit edges (core anchor, dream -> produced
facts, session -> facts discussed) plus weak theme edges between nodes
that share salient words.
"""

from __future__ import annotations

from app.conversation import SessionStore
from app.graph import build_graph
from app.memory import DreamStore, MemoryStore


def make_vault(tmp_path):
    memory = MemoryStore(str(tmp_path / "memory.md"))
    dreams = DreamStore(path=str(tmp_path / "dreams.json"))
    sessions = SessionStore(data_dir=str(tmp_path))
    return memory, dreams, sessions


def test_core_anchor_links_only_core_facts(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    core_fact = memory.add("Rob lives in Gloucester", core=True)
    archive_fact = memory.add("Rob likes oolong tea")
    graph = build_graph(memory, dreams, sessions)
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert by_id["core"]["type"] == "core"
    assert by_id[core_fact["id"]]["core"] is True
    assert by_id[archive_fact["id"]]["core"] is False

    core_links = [l for l in graph["links"] if l["kind"] == "core"]
    assert [(l["source"], l["target"]) for l in core_links] == [("core", core_fact["id"])]


def test_dream_links_to_the_facts_it_produced(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    fact = memory.add("Rob uses Linux on his desktop")
    dream = dreams.add({
        "id": "d1",
        "ts": 1718505600.0,
        "llm": True,
        "summary": "consolidated desktop habits",
        "new_facts": ["Rob uses Linux on his desktop", "unrelated observation"],
    })
    graph = build_graph(memory, dreams, sessions)

    fact_links = [
        l for l in graph["links"]
        if l["kind"] == "fact" and {l["source"], l["target"]} == {dream["id"], fact["id"]}
    ]
    assert len(fact_links) == 1

    # substring match works in both directions
    memory.add("The desktop machine runs Linux")
    dream2 = dreams.add({
        "id": "d2",
        "ts": 1718592000.0,
        "llm": False,
        "summary": "second pass",
        "new_facts": ["the desktop machine runs"],
    })
    graph = build_graph(memory, dreams, sessions)
    second = _fact_id(graph, "The desktop machine runs Linux")
    pair = [
        l for l in graph["links"]
        if l["kind"] == "fact" and {l["source"], l["target"]} == {dream2["id"], second}
    ]
    assert len(pair) == 1


def test_session_links_to_facts_it_discussed(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    fact = memory.add("garden station ticket")
    sid = sessions.active_id
    sessions.rename(sid, "Garden talk")
    sessions.conversation_for(sid).add_turn(
        "Do you remember my garden station ticket?", "Yes, you have a garden station ticket."
    )
    graph = build_graph(memory, dreams, sessions)

    talk = [
        l for l in graph["links"]
        if l["kind"] == "talk" and {l["source"], l["target"]} == {sid, fact["id"]}
    ]
    assert len(talk) == 1
    session_node = next(n for n in graph["nodes"] if n["id"] == sid)
    assert session_node["label"] == "Garden talk"
    assert session_node["sub"] == "1 turns"


def test_theme_edges_need_two_salient_shared_words(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    memory.add("the zebra quasar is visible tonight")
    fact_b = memory.add("quasar zebra observations from the yard")
    graph = build_graph(memory, dreams, sessions)

    themes = [l for l in graph["links"] if l["kind"] == "theme"]
    assert len(themes) == 1
    assert themes[0]["weight"] == 2
    assert "quasar" in themes[0]["label"] and "zebra" in themes[0]["label"]

    # shared words that are all stopwords never become an edge
    memory.add("the and about that was a thing")
    memory.add("the about and of that thing")
    graph = build_graph(memory, dreams, sessions)
    pairs = [
        (l["source"], l["target"])
        for l in graph["links"]
        if l["kind"] == "theme" and _fact_id(graph, "the and about that was a thing") in (l["source"], l["target"])
    ]
    assert pairs == []


def _fact_id(graph, text):
    return next(n["id"] for n in graph["nodes"] if n["type"] == "fact" and n["label"] == text)


def test_theme_edges_capped_per_node_and_deduped(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    memory.add("zebra quasar hub fact")
    for i in range(9):
        memory.add(f"zebra quasar spoke number {i} alpha")
    graph = build_graph(memory, dreams, sessions)

    hub = _fact_id(graph, "zebra quasar hub fact")
    hub_themes = [
        l for l in graph["links"]
        if l["kind"] == "theme" and hub in (l["source"], l["target"])
    ]
    assert len(hub_themes) <= 6

    seen = set()
    for link in graph["links"]:
        key = (link["source"], link["target"]) if link["source"] < link["target"] else (link["target"], link["source"])
        assert key not in seen, f"duplicate link pair {key}"
        seen.add(key)


def test_explicit_link_wins_over_theme_for_same_pair(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    fact = memory.add("zebra quasar lives in the garden")
    dreams.add({
        "id": "d1",
        "ts": 1718505600.0,
        "llm": True,
        "summary": "zebra quasar in the garden",
        "new_facts": ["zebra quasar lives in the garden"],
    })
    graph = build_graph(memory, dreams, sessions)

    pair_links = [
        l for l in graph["links"]
        if {l["source"], l["target"]} == {fact["id"], "d1"}
    ]
    assert [l["kind"] for l in pair_links] == ["fact"]


def test_degree_counts_and_empty_vault(tmp_path):
    memory, dreams, sessions = make_vault(tmp_path)
    graph = build_graph(memory, dreams, sessions)

    assert [n for n in graph["nodes"] if n["type"] == "core"] == [
        {"id": "core", "type": "core", "label": "MEMORY", "sub": "core memory index", "degree": 0}
    ]
    for node in graph["nodes"]:
        touches = sum(
            1 for l in graph["links"] if node["id"] in (l["source"], l["target"])
        )
        assert node["degree"] == touches

    # only the core anchor and the one auto-created empty session
    assert {n["type"] for n in graph["nodes"]} == {"core", "session"}
    assert graph["links"] == []


def test_api_graph_returns_nodes_and_links(tmp_path):
    from fastapi.testclient import TestClient

    from app import main

    memory, dreams, sessions = make_vault(tmp_path)
    memory.add("zebra quasar fact one")
    memory.add("zebra quasar fact two")
    main.app.state.engines = type("Engines", (), {
        "memory": memory, "dreams": dreams, "sessions": sessions,
    })()

    # plain TestClient (no context manager): skips main.app's lifespan boot
    client = TestClient(main.app)
    res = client.get("/api/graph")
    assert res.status_code == 200
    body = res.json()
    assert body["nodes"] and body["links"]
    assert {n["type"] for n in body["nodes"]} == {"core", "fact", "session"}
