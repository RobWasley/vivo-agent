from app.browser_stream import _fps_message


def test_browser_fps_message_is_bounded():
    assert _fps_message('{"type":"browser_fps","maxFps":99}') == (
        '{"type": "config", "maxFps": 30}'
    )
    assert _fps_message('{"type":"browser_fps","maxFps":0}') == (
        '{"type": "config", "maxFps": 1}'
    )


def test_browser_fps_message_ignores_other_messages():
    assert _fps_message('{"type":"input_mouse"}') is None
    assert _fps_message("not json") is None