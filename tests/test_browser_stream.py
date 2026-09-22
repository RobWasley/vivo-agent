from app.browser_stream import _fps_message, _input_message


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


def test_browser_input_messages_are_normalized():
    assert _input_message(
        '{"type":"input_mouse","x":12.6,"y":8.2,"eventType":"mousePressed",'
        '"button":"left","modifiers":8}'
    ) == (
        '{"type": "input_mouse", "x": 13, "y": 8, "eventType": "mousePressed", '
        '"button": "left", "modifiers": 8, "clickCount": 1}'
    )
    assert _input_message(
        '{"type":"input_keyboard","key":"a","eventType":"keyDown",'
        '"code":"KeyA","text":"a","modifiers":2}'
    ) == (
        '{"type": "input_keyboard", "key": "a", "eventType": "keyDown", '
        '"code": "KeyA", "text": "a", "modifiers": 2}'
    )
    assert _input_message(
        '{"type":"input_mouse","x":12,"y":8,"eventType":"mouseWheel",'
        '"deltaX":0,"deltaY":100,"modifiers":0}'
    ) == (
        '{"type": "input_mouse", "x": 12, "y": 8, "eventType": "mouseWheel", '
        '"button": "none", "modifiers": 0, "clickCount": 0, '
        '"deltaX": 0, "deltaY": 100}'
    )


def test_browser_input_messages_reject_invalid_payloads():
    assert _input_message('{"type":"input_mouse","x":-1,"y":0,"eventType":"mouseMoved"}') is None
    assert _input_message('{"type":"input_keyboard","key":"a","eventType":"keyPress"}') is None
    assert _input_message('{"type":"input_mouse","x":0,"y":0,"eventType":"mouseWheel","deltaY":"100"}') is None
    assert _input_message('{"type":"input_touch"}') is None