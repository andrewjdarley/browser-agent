"""Tests for the network_list_types/network_list/network_inspect actions -
passive response capture (_on_network_response/_capture_network_body) and
the three query actions built on top of the captured log."""

import asyncio
from pathlib import Path

import pytest

from browser_use_demo.tools.base import ToolError
from browser_use_demo.tools.browser import (
    DOM_MUTATING_ACTIONS,
    MAX_NETWORK_BODY_CHARS,
    MAX_NETWORK_LIST_RESULTS,
    MAX_NETWORK_LOG_ENTRIES,
    BrowserTool,
)


class FakeRequest:
    def __init__(self, method="GET", resource_type="xhr"):
        self.method = method
        self.resource_type = resource_type


class FakeResponse:
    def __init__(
        self,
        url,
        status=200,
        content_type="application/json",
        body='{"ok": true}',
        method="GET",
        resource_type="xhr",
        text_error=None,
    ):
        self.url = url
        self.status = status
        self.headers = {"content-type": content_type} if content_type else {}
        self.request = FakeRequest(method=method, resource_type=resource_type)
        self._body = body
        self._text_error = text_error

    async def text(self):
        if self._text_error:
            raise self._text_error
        return self._body


def make_tool(tmp_path: Path) -> BrowserTool:
    return BrowserTool(run_dir=tmp_path)


class TestOnNetworkResponse:
    # These three use FakeResponse's default JSON content-type, which makes
    # _on_network_response schedule a body-capture task via asyncio.create_task -
    # that needs a genuinely running event loop (as it always has in production,
    # since Playwright only ever invokes this handler from inside one), so these
    # are real async tests, not just sync calls wrapped in an async def. Without
    # a running loop, create_task raises (caught by _on_network_response's own
    # try/except) but leaves an orphaned, never-awaited coroutine behind that
    # Python's GC then warns about - the asyncio.sleep(0) lets that task
    # actually finish rather than get abandoned when the test ends.
    @pytest.mark.asyncio
    async def test_captures_metadata_synchronously(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/api/items", status=201, method="POST"))
        await asyncio.sleep(0)

        assert len(tool._network_log) == 1
        entry = tool._network_log[0]
        assert entry["id"] == "n1"
        assert entry["url"] == "https://example.com/api/items"
        assert entry["status"] == 201
        assert entry["method"] == "POST"
        assert entry["resource_type"] == "xhr"
        assert entry["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_ids_increment_across_calls(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/a"))
        tool._on_network_response(FakeResponse("https://example.com/b"))
        await asyncio.sleep(0)
        assert [e["id"] for e in tool._network_log] == ["n1", "n2"]

    def test_never_raises_on_malformed_response(self, tmp_path):
        tool = make_tool(tmp_path)

        class Broken:
            @property
            def url(self):
                raise RuntimeError("boom")

        tool._on_network_response(Broken())  # must not raise
        assert len(tool._network_log) == 0

    @pytest.mark.asyncio
    async def test_log_evicts_oldest_beyond_cap(self, tmp_path):
        tool = make_tool(tmp_path)
        for i in range(MAX_NETWORK_LOG_ENTRIES + 5):
            tool._on_network_response(FakeResponse(f"https://example.com/{i}"))
        await asyncio.sleep(0)
        assert len(tool._network_log) == MAX_NETWORK_LOG_ENTRIES
        # Oldest 5 evicted, so the log starts at n6
        assert tool._network_log[0]["id"] == "n6"


class TestCaptureNetworkBody:
    @pytest.mark.asyncio
    async def test_captures_body_for_json_content_type(self, tmp_path):
        tool = make_tool(tmp_path)
        response = FakeResponse("https://example.com/api", content_type="application/json", body='{"x": 1}')
        tool._on_network_response(response)

        # _on_network_response schedules the body-capture task; let the event loop run it.
        await asyncio.sleep(0)
        entry = tool._network_log[0]
        assert entry["body"] == '{"x": 1}'
        assert entry["body_truncated"] is False

    @pytest.mark.asyncio
    async def test_skips_body_for_binary_content_type(self, tmp_path):
        tool = make_tool(tmp_path)
        response = FakeResponse("https://example.com/logo.png", content_type="image/png", body="unused")
        tool._on_network_response(response)

        await asyncio.sleep(0)
        assert tool._network_log[0]["body"] is None

    @pytest.mark.asyncio
    async def test_truncates_long_body(self, tmp_path):
        tool = make_tool(tmp_path)
        entry = {
            "id": "n1",
            "url": "https://example.com",
            "method": "GET",
            "status": 200,
            "resource_type": "xhr",
            "content_type": "application/json",
            "body": None,
            "body_truncated": False,
        }
        long_body = "x" * (MAX_NETWORK_BODY_CHARS + 500)
        response = FakeResponse("https://example.com", body=long_body)
        await tool._capture_network_body(entry, response)

        assert len(entry["body"]) == MAX_NETWORK_BODY_CHARS
        assert entry["body_truncated"] is True

    @pytest.mark.asyncio
    async def test_text_error_leaves_body_none(self, tmp_path):
        tool = make_tool(tmp_path)
        entry = {
            "id": "n1",
            "url": "https://example.com",
            "method": "GET",
            "status": 200,
            "resource_type": "xhr",
            "content_type": "application/json",
            "body": None,
            "body_truncated": False,
        }
        response = FakeResponse("https://example.com", text_error=RuntimeError("body already consumed"))
        await tool._capture_network_body(entry, response)  # must not raise
        assert entry["body"] is None


class TestNetworkListTypes:
    @pytest.mark.asyncio
    async def test_empty_log_message(self, tmp_path):
        tool = make_tool(tmp_path)
        result = await tool._network_list_types()
        assert "No network activity captured" in result.output

    @pytest.mark.asyncio
    async def test_groups_by_type_and_host(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://a.com/x", resource_type="xhr"))
        tool._on_network_response(FakeResponse("https://a.com/y", resource_type="xhr"))
        tool._on_network_response(FakeResponse("https://b.com/z.png", resource_type="image"))

        result = await tool._network_list_types()
        assert "3 response(s) captured" in result.output
        assert "xhr: 2" in result.output
        assert "image: 1" in result.output
        assert "a.com: 2" in result.output
        assert "b.com: 1" in result.output


class TestNetworkList:
    @pytest.mark.asyncio
    async def test_empty_log_message(self, tmp_path):
        tool = make_tool(tmp_path)
        result = await tool._network_list(None)
        assert "No network activity captured" in result.output

    @pytest.mark.asyncio
    async def test_filters_by_url_substring(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/api/users"))
        tool._on_network_response(FakeResponse("https://example.com/static/app.js", content_type="text/javascript"))

        result = await tool._network_list("users")
        assert "1 matching response(s)" in result.output
        assert "/api/users" in result.output
        assert "app.js" not in result.output

    @pytest.mark.asyncio
    async def test_filters_by_status_code(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/ok", status=200))
        tool._on_network_response(FakeResponse("https://example.com/missing", status=404))

        result = await tool._network_list("404")
        assert "1 matching response(s)" in result.output
        assert "/missing" in result.output

    @pytest.mark.asyncio
    async def test_no_match_message(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/a"))
        result = await tool._network_list("nothing-like-this")
        assert "No captured responses matched" in result.output

    @pytest.mark.asyncio
    async def test_caps_results_and_notes_remainder(self, tmp_path):
        tool = make_tool(tmp_path)
        for i in range(MAX_NETWORK_LIST_RESULTS + 10):
            tool._on_network_response(FakeResponse(f"https://example.com/{i}"))

        result = await tool._network_list(None)
        assert f"showing {MAX_NETWORK_LIST_RESULTS} most recent" in result.output
        assert "10 more not shown" in result.output
        # Most recent entries shown, not the oldest
        assert "n1 |" not in result.output


class TestNetworkInspect:
    @pytest.mark.asyncio
    async def test_unknown_id_raises_tool_error(self, tmp_path):
        tool = make_tool(tmp_path)
        with pytest.raises(ToolError):
            await tool._network_inspect("n999")

    @pytest.mark.asyncio
    async def test_returns_captured_body(self, tmp_path):
        tool = make_tool(tmp_path)
        response = FakeResponse("https://example.com/api", body='{"hello": "world"}')
        tool._on_network_response(response)

        await asyncio.sleep(0)
        result = await tool._network_inspect("n1")
        assert "https://example.com/api" in result.output
        assert "Status: 200" in result.output
        assert '{"hello": "world"}' in result.output

    @pytest.mark.asyncio
    async def test_reports_body_not_captured_for_binary(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._on_network_response(FakeResponse("https://example.com/logo.png", content_type="image/png"))

        await asyncio.sleep(0)
        result = await tool._network_inspect("n1")
        assert "not captured" in result.output

    @pytest.mark.asyncio
    async def test_reports_truncation_note(self, tmp_path):
        tool = make_tool(tmp_path)
        entry = {
            "id": "n1",
            "url": "https://example.com",
            "method": "GET",
            "status": 200,
            "resource_type": "xhr",
            "content_type": "application/json",
            "body": "x" * MAX_NETWORK_BODY_CHARS,
            "body_truncated": True,
        }
        tool._network_log.append(entry)
        result = await tool._network_inspect("n1")
        assert "truncated" in result.output


class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_network_list_types_action_routes(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._initialized = True  # skip _ensure_browser's real launch path
        result = await tool._dispatch(action="network_list_types")
        assert "No network activity captured" in result.output

    @pytest.mark.asyncio
    async def test_network_list_action_routes_with_match(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._initialized = True
        tool._on_network_response(FakeResponse("https://example.com/api/users"))
        result = await tool._dispatch(action="network_list", text="users")
        assert "/api/users" in result.output

    @pytest.mark.asyncio
    async def test_network_inspect_requires_text(self, tmp_path):
        tool = make_tool(tmp_path)
        tool._initialized = True
        with pytest.raises(ToolError):
            await tool._dispatch(action="network_inspect")

    def test_network_actions_are_not_dom_mutating(self):
        assert "network_list_types" not in DOM_MUTATING_ACTIONS
        assert "network_list" not in DOM_MUTATING_ACTIONS
        assert "network_inspect" not in DOM_MUTATING_ACTIONS
