import httpx

from app import config
from app import web


class FakeDDGS:
    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results=5):
        return [
            {
                "title": "Example <b>Domain</b>",
                "href": "https://example.com",
                "body": "This domain is <i>for</i> examples.",
            },
            {"title": "Second", "href": "https://example.org", "body": "a snippet"},
        ]


class EmptyDDGS:
    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results=5):
        return []


class BoomDDGS:
    def __init__(self, timeout=None):
        raise RuntimeError("net down")


def test_web_search_format(monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    out = web.web_search("test query")
    assert out.startswith("Results for: test query")
    assert "1. Example Domain" in out, out  # tags cleaned
    assert "https://example.com" in out
    assert "2. Second" in out


def test_web_search_count_clamped(monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    out = web.web_search("q", count=99)
    assert out.count("\n   http") == 2, "count above 5 was not clamped"


def test_web_search_no_results(monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", EmptyDDGS)
    assert web.web_search("zzz").startswith("No results")


def test_web_search_error(monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", BoomDDGS)
    out = web.web_search("zzz")
    assert out.startswith("error: web search failed")


def test_web_fetch_bad_url():
    assert web.web_fetch("ftp://example.com").startswith("error")
    assert web.web_fetch("").startswith("error")


def test_web_fetch_via_reader(monkeypatch):
    body = "# Title\n\nSome markdown body. " * 30

    def fake_get(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            return httpx.Response(200, text=body, request=httpx.Request("GET", url))
        return httpx.Response(500, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    out = web.web_fetch("https://example.com/page")
    assert out.startswith(web.UNTRUSTED_BANNER)
    assert "Title" in out


def test_web_fetch_direct_html_fallback(monkeypatch):
    def fake_get(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            raise httpx.ConnectError("no reader")
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            html=(
                "<html><body><nav>menu</nav><h1>Head</h1>"
                "<p>Para one.</p><p>Para two.</p>"
                "<script>bad()</script></body></html>"
            ),
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    out = web.web_fetch("https://example.com")
    assert out.startswith(web.UNTRUSTED_BANNER)
    assert "Head" in out and "Para one." in out
    assert "menu" not in out and "bad()" not in out


def test_web_fetch_http_error(monkeypatch):
    def fake_get(url, **kw):
        raise httpx.ConnectError("no reader")

    def fake_get2(url, **kw):
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert web.web_fetch("https://example.com").startswith("error: fetch failed")
    monkeypatch.setattr(httpx, "get", fake_get2)
    assert "HTTP 404" in web.web_fetch("https://example.com")


def test_web_fetch_retries_transient_server_error(monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return httpx.Response(503, request=httpx.Request("GET", url))
        return httpx.Response(
            200, text="retry succeeded " * 10, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    assert web.web_fetch("https://example.com").startswith(web.UNTRUSTED_BANNER)
    assert len(calls) == 2


def test_web_fetch_truncation(monkeypatch):
    def fake_get(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            return httpx.Response(200, text="x" * 5000, request=httpx.Request("GET", url))
        return httpx.Response(500, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(config, "FETCH_MAX_CHARS", 500)
    out = web.web_fetch("https://example.com")
    assert "truncated at" in out and len(out) < 700


def test_web_search_live():
    out = web.web_search("llama.cpp", count=3)
    assert out.startswith("Results for") or out.startswith("No results"), out[:200]
    if out.startswith("Results"):
        assert "http" in out
