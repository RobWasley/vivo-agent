from app import config, tools


def test_get_time():
    out = tools.get_time()
    assert "It is" in out and ":" in out


def test_weather():
    out = tools.weather(52.52, 13.405)  # Berlin
    assert "degrees" in out and "wind" in out


def test_read_file_ok(tmp_path, monkeypatch):
    (tmp_path / "note.txt").write_text("hello sandbox")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert tools.read_file("note.txt") == "hello sandbox"


def test_read_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert tools.read_file("nope.txt").startswith("error")


def test_read_file_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert tools.read_file("../etc/passwd").startswith("error")
    assert tools.read_file("/etc/passwd").startswith("error")


def test_unknown_tool():
    assert tools.execute("nope", {}).startswith("error")
