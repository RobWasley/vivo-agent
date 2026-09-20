from app import config, tools
from app.tools import ToolRegistry

_CURRENT = {
    "temperature_2m": 21.4,
    "apparent_temperature": 19.9,
    "weather_code": 2,
    "wind_speed_10m": 12.5,
    "relative_humidity_2m": 60.2,
}


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _fake_httpx(geo_results):
    """Stands in for httpx.get: geocoding by URL, forecast otherwise."""
    calls = []

    def get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if "geocoding" in url:
            return _Resp({"results": geo_results})
        return _Resp({"current": _CURRENT})

    return get, calls


def test_get_time():
    out = tools.get_time()
    assert "It is" in out and ":" in out and " in " in out


def test_get_time_user_timezone(monkeypatch):
    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    assert "Europe/London" in tools.get_time()


def test_get_time_bad_timezone_falls_back(monkeypatch):
    monkeypatch.setattr(config, "USER_TIMEZONE", "Not/AZone")
    out = tools.get_time()
    assert "It is" in out and "Not/AZone" not in out


def test_weather_live():
    out = tools.weather("Berlin")
    assert "degrees" in out and "wind" in out


def test_weather_default_location_metric(monkeypatch):
    get, calls = _fake_httpx(geo_results=[{"latitude": 51.45, "longitude": -2.59}])
    monkeypatch.setattr(tools.httpx, "get", get)
    monkeypatch.setattr(config, "USER_UNITS", "metric")
    monkeypatch.setattr(config, "USER_LOCATION", "Bristol, UK")
    out = tools.weather()
    assert "degrees Celsius" in out and "kilometers per hour" in out
    # the profile location is geocoded, then the forecast runs
    assert calls[0][1]["name"] == "Bristol, UK"
    assert "temperature_unit" not in calls[1][1]


def test_weather_named_location_imperial(monkeypatch):
    get, calls = _fake_httpx(geo_results=[{"latitude": 40.71, "longitude": -74.0}])
    monkeypatch.setattr(tools.httpx, "get", get)
    monkeypatch.setattr(config, "USER_UNITS", "imperial")
    out = tools.weather("New York")
    assert "degrees Fahrenheit" in out and "miles per hour" in out
    assert calls[1][1]["temperature_unit"] == "fahrenheit"
    assert calls[1][1]["wind_speed_unit"] == "mph"


def test_weather_no_location_anywhere(monkeypatch):
    def no_http(*a, **k):
        raise AssertionError("no HTTP call expected")

    monkeypatch.setattr(tools.httpx, "get", no_http)
    monkeypatch.setattr(config, "USER_LOCATION", "")
    out = tools.weather()
    assert out.startswith("error") and "no location" in out


def test_weather_unknown_place(monkeypatch):
    get, _ = _fake_httpx(geo_results=[])
    monkeypatch.setattr(tools.httpx, "get", get)
    out = tools.weather("Xyzzyville")
    assert out.startswith("error") and "could not find" in out


def test_geocode_is_cached(monkeypatch):
    get, calls = _fake_httpx(geo_results=[{"latitude": 1.0, "longitude": 2.0}])
    monkeypatch.setattr(tools.httpx, "get", get)
    tools.weather("Reykjavik")
    tools.weather("Reykjavik")
    assert sum(1 for url, _ in calls if "geocoding" in url) == 1


def test_read_file_ok(tmp_path, monkeypatch):
    (tmp_path / "note.txt").write_text("hello sandbox")
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert tools.read_file("note.txt") == "hello sandbox"


def test_read_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert tools.read_file("nope.txt").startswith("error")


def test_read_file_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert tools.read_file("../etc/passwd").startswith("error")
    assert tools.read_file("/etc/passwd").startswith("error")


def test_unknown_tool():
    assert tools.execute("nope", {}).startswith("error")


def test_conversation_tools_are_advertised_to_the_agent():
    names = {tool["function"]["name"] for tool in tools.TOOLS}
    assert {
        "list_conversations", "create_conversation", "switch_conversation",
        "rename_conversation", "delete_conversation",
    } <= names


def test_tool_registry_validates_and_executes():
    registry = ToolRegistry()
    registry.register(
        tools.ToolDefinition(
            name="echo",
            description="echo a value",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            func=lambda value: f"echo:{value}",
        )
    )

    assert registry.get("echo") is not None
    result = registry.execute("echo", {"value": "hello"})
    assert result == "echo:hello"

    bad = registry.execute("echo", {})
    assert bad.startswith("error")


def test_tool_registry_exposes_openai_schema_and_known_tools():
    registry = ToolRegistry()
    registry.register(
        tools.ToolDefinition(
            name="ping",
            description="ping",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: text,
        )
    )

    defs = registry.get_definitions()
    assert defs[0]["function"]["name"] == "ping"
    assert "ping" in registry.tool_names
