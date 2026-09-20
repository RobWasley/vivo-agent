from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, main, tools
from app.pipeline import skills_context
from app.skills import SkillStore


def test_skill_store_create_read_list_update_and_delete(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))

    assert store.save("daily-summary", "Summarise the day.", "# Daily summary\n\nBe concise.") == "created"
    assert store.list() == [{"name": "daily-summary", "description": "Summarise the day."}]
    assert "# Daily summary" in store.read("daily-summary")
    assert store.save("daily-summary", "Summarise the user's day.", "Use two sentences.") == "updated"

    store.delete("daily-summary")
    assert store.list() == []


def test_skill_tools_are_scoped_to_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    result = tools.execute("save_skill", {
        "name": "gloucester-weather",
        "description": "Give a Gloucester weather briefing.",
        "instructions": "Use the weather tool without a location.",
    })
    assert result == "skill created: gloucester-weather"
    assert "gloucester-weather" in tools.execute("list_skills", {})
    assert "weather tool" in tools.execute("read_skill", {"name": "gloucester-weather"})
    assert tools.execute("delete_skill", {"name": "gloucester-weather"}) == "skill deleted: gloucester-weather"


def test_skill_store_rejects_paths_and_bad_metadata(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))

    try:
        store.save("../outside", "Description", "Instructions")
    except ValueError as exc:
        assert "lowercase" in str(exc)
    else:
        raise AssertionError("invalid skill name was accepted")


def test_skills_context_is_a_compact_index(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.save("daily-summary", "Summarise the day.", "Use two sentences.")

    context = skills_context(store)

    assert "daily-summary: Summarise the day." in context
    assert "Use two sentences." not in context
    assert "read_skill" in context


def test_skills_api_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    app = FastAPI()
    app.add_api_route("/api/skills", main.list_skills, methods=["GET"])
    app.add_api_route("/api/skills/{name}", main.get_skill, methods=["GET"])
    app.add_api_route("/api/skills/{name}", main.save_skill, methods=["PUT"])
    app.add_api_route("/api/skills/{name}", main.delete_skill, methods=["DELETE"])
    payload = {
        "name": "daily-summary",
        "description": "Summarise the day.",
        "instructions": "Use two sentences.",
    }

    with TestClient(app) as client:
        saved = client.put("/api/skills/daily-summary", json=payload)
        assert saved.status_code == 200 and saved.json()["action"] == "created"
        assert client.get("/api/skills").json()["skills"] == [{
            "name": "daily-summary", "description": "Summarise the day.",
        }]
        assert client.get("/api/skills/daily-summary").json()["instructions"] == "Use two sentences."
        assert client.delete("/api/skills/daily-summary").status_code == 200
        assert client.get("/api/skills/daily-summary").status_code == 404