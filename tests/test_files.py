from app import config
from app import tools


def test_write_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = tools.write_file("notes/plan.txt", "line1\nline2")
    assert out.startswith("wrote 11 chars"), out
    assert tools.read_file("notes/plan.txt") == "line1\nline2"


def test_write_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert tools.write_file("../evil.txt", "x").startswith("error")
    assert tools.write_file("/etc/evil.txt", "x").startswith("error")


def test_list_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    out = tools.list_dir()
    assert "a.py" in out and "b/" in out
    assert ".git" not in out and "node_modules" not in out


def test_list_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert tools.list_dir("nope").startswith("error")


def test_dispatch_new_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = tools.execute("write_file", {"path": "x.txt", "content": "hi"})
    assert out.startswith("wrote")
    assert tools.execute("list_dir", {}) == "x.txt"
    assert tools.execute("exec", {"command": "echo via-dispatch"}).startswith("via-dispatch")


def test_dispatch_global_cap(monkeypatch):
    import app.tools as t

    monkeypatch.setattr(t, "get_time", lambda: "x" * 500)
    monkeypatch.setattr(t, "MAX_RESULT_CHARS", 100)
    out = t.execute("get_time", {})
    assert "chars truncated" in out and len(out) < 250
