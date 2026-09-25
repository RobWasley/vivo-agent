import logging

from app.logging_store import LogStore, VivoLogHandler


def test_add_and_list_order_and_fields():
    seen = []
    store = LogStore(on_entry=seen.append)
    entry = store.add("INFO", "hello", event="session_create", session_id="s1")
    store.add("warning", "loud thing")

    assert entry["level"] == "INFO"
    assert entry["event"] == "session_create"
    assert entry["session_id"] == "s1"
    rows = store.list()
    assert [r["message"] for r in rows] == ["hello", "loud thing"]
    assert rows[1]["level"] == "WARNING"
    # the WS callback saw every entry as it arrived
    assert [e["message"] for e in seen] == ["hello", "loud thing"]


def test_list_filters_by_exact_level():
    store = LogStore()
    store.add("INFO", "a")
    store.add("ERROR", "b")
    store.add("INFO", "c")
    assert [e["message"] for e in store.list(level="error")] == ["b"]
    assert [e["message"] for e in store.list(level="info")] == ["a", "c"]


def test_ring_buffer_keeps_last_n_and_persists(tmp_path):
    path = str(tmp_path / "log.json")
    store = LogStore(path=path, max_entries=3)
    for i in range(5):
        store.add("INFO", f"m{i}")
    store.flush()
    assert [e["message"] for e in store.list()] == ["m2", "m3", "m4"]

    reloaded = LogStore(path=path, max_entries=3)
    assert [e["message"] for e in reloaded.list()] == ["m2", "m3", "m4"]


def test_corrupt_file_starts_empty(tmp_path):
    p = tmp_path / "log.json"
    p.write_text("{not json", encoding="utf-8")
    store = LogStore(path=str(p))
    assert store.list() == []
    store.add("INFO", "still works")
    store.flush()


def test_clear_persists(tmp_path):
    path = str(tmp_path / "log.json")
    store = LogStore(path=path, max_entries=10)
    store.add("INFO", "x")
    store.clear()
    store.flush()
    assert store.list() == []
    assert LogStore(path=path).list() == []


def test_message_truncated_and_newlines_normalised():
    store = LogStore()
    entry = store.add("INFO", "a" * 900 + "\r\nb")
    assert entry["message"] == "a" * 500


def test_on_entry_exception_is_swallowed():
    def boom(_entry):
        raise RuntimeError("console gone")

    store = LogStore(on_entry=boom)
    entry = store.add("INFO", "ok")
    assert entry["message"] == "ok"


def test_handler_routes_vivo_logs_with_event():
    store = LogStore()
    handler = VivoLogHandler(store)
    logger = logging.getLogger("vivo.test_logging")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.info("reminder fired", extra={"event": "reminder_fire"})
        logger.error("boom")
    finally:
        logger.removeHandler(handler)
    rows = store.list()
    assert rows[0]["level"] == "INFO"
    assert rows[0]["event"] == "reminder_fire"
    assert rows[1]["level"] == "ERROR"
    assert "event" not in rows[1]
