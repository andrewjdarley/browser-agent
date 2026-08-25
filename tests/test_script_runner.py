"""Tests for ScriptRunnerTool (#9's run_script): step validation, both
execution modes (raw-fetch "fast" path reusing batch_extract's mechanism,
and real-page "full" path with a fresh Page per item), per-item error
isolation, concurrency bounding, save_as behavior, and rate-limit retry -
all without a real Playwright browser or real network."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_use_demo.tools.file_output import FileOutputTool
from browser_use_demo.tools.script_runner import MAX_ATTEMPTS_PER_ITEM, ScriptRunnerTool


@pytest.fixture(autouse=True)
def fast_page_ready_wait():
    """wait_for_page_ready's settle delays (SCREENSHOT_SETTLE_DELAY_S /
    TEXT_READ_SETTLE_DELAY_S) live in browser.py's own asyncio, called from
    every "full path" step here (navigate/screenshot/get_page_text/
    execute_js all call it) - without this, every such test would pay ~1.5-
    2s of real sleep. None of these tests are testing that timing itself
    (browser.py's own TestScreenshotSettleDelay/TestTextReadSettleDelay
    already cover that), so make it instant across this whole file."""
    with patch("browser_use_demo.tools.browser.asyncio.sleep", new=AsyncMock()):
        yield


class FakeFetchResponse:
    """Response shape for browser_tool._context.request.get (the fast path)."""

    def __init__(self, status=200, body="<html></html>", headers=None):
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def text(self):
        return self._body


def make_fake_page(*, goto_status=200, evaluate_result=None, evaluate_side_effect=None):
    """A stand-in Page for the "full" (navigate) execution path."""
    page = MagicMock()

    async def default_evaluate(*args, **kwargs):
        if evaluate_side_effect is not None:
            return evaluate_side_effect(*args, **kwargs)
        if "document.readyState" in (args[0] if args else ""):
            return "complete"
        return evaluate_result if evaluate_result is not None else {"ok": True}

    page.evaluate = AsyncMock(side_effect=default_evaluate)

    async def goto(url, **kwargs):
        response = MagicMock()
        response.status = goto_status
        response.headers = {}
        return response

    page.goto = AsyncMock(side_effect=goto)
    page.screenshot = AsyncMock(side_effect=lambda path, **kw: Path(path).write_bytes(b"\x89PNG\r\n"))
    page.close = AsyncMock()
    return page


def make_browser_tool(tmp_path: Path, *, get_side_effect=None, evaluate_result=None, new_page_factory=None):
    """A stand-in for BrowserTool exposing just what ScriptRunnerTool touches:
    _ensure_browser(), _context.request.get()/_context.new_page(), _page.evaluate()
    (the shared page, used only by the fast/no-navigate path), run_dir/width/height."""
    bt = MagicMock()
    bt._ensure_browser = AsyncMock()
    bt.run_dir = tmp_path
    bt.width = 1920
    bt.height = 1080
    bt._context = MagicMock()
    if get_side_effect is not None:
        bt._context.request.get = AsyncMock(side_effect=get_side_effect)
    bt._page = MagicMock()
    bt._page.evaluate = AsyncMock(
        side_effect=evaluate_result if callable(evaluate_result) else (lambda *a, **k: {"ok": True})
    )
    if new_page_factory is not None:
        bt._context.new_page = AsyncMock(side_effect=new_page_factory)
    else:
        bt._context.new_page = AsyncMock(side_effect=lambda: make_fake_page())
    return bt


def make_tool(tmp_path, **kwargs):
    browser_tool = make_browser_tool(tmp_path, **kwargs)
    file_output_tool = FileOutputTool(run_dir=tmp_path)
    return ScriptRunnerTool(browser_tool=browser_tool, file_output_tool=file_output_tool), browser_tool


class TestValidation:
    @pytest.mark.asyncio
    async def test_rejects_empty_items(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = await tool(items=[], script=[{"action": "execute_js", "text": "1"}])
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_rejects_empty_script(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = await tool(items=["https://a.test"], script=[])
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_rejects_unsupported_action(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = await tool(items=["https://a.test"], script=[{"action": "left_click", "ref": "ref_1"}])
        assert result.error is not None
        assert "left_click" in result.error
        assert "dispatch_subagents" in result.error

    @pytest.mark.asyncio
    async def test_fast_path_rejects_non_execute_js_steps(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = await tool(
            items=["https://a.test"],
            script=[{"action": "execute_js", "text": "1"}, {"action": "wait", "duration": 1}],
        )
        assert result.error is not None
        assert "navigate" in result.error


class TestFastPath:
    """No navigate as the first step - raw fetch of the item as a URL,
    mirroring batch_extract's own mechanism exactly."""

    @pytest.mark.asyncio
    async def test_fetches_and_extracts(self, tmp_path):
        tool, browser_tool = make_tool(
            tmp_path,
            get_side_effect=lambda url: FakeFetchResponse(body="<title>hi</title>"),
            evaluate_result=lambda *a, **k: {"title": "hi"},
        )
        result = await tool(items=["https://a.test/x"], script=[{"action": "execute_js", "text": "(doc) => doc.title"}])
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 1
        assert parsed["results"][0]["steps"][0]["output"] == {"title": "hi"}
        browser_tool._context.request.get.assert_awaited_once_with("https://a.test/x")

    @pytest.mark.asyncio
    async def test_substitutes_item_in_js(self, tmp_path):
        seen_js = []

        async def evaluate(wrapper_js, args):
            seen_js.append(args[1])
            return None

        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeFetchResponse())
        browser_tool._page.evaluate = AsyncMock(side_effect=evaluate)
        await tool(
            items=["widget-42"],
            script=[{"action": "execute_js", "text": "(doc) => '{{item}}'"}],
        )
        assert seen_js == ["(doc) => 'widget-42'"]

    @pytest.mark.asyncio
    async def test_client_error_status_fails_without_retry(self, tmp_path):
        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeFetchResponse(status=404))
        result = await tool(items=["https://a.test/missing"], script=[{"action": "execute_js", "text": "1"}])
        parsed = json.loads(result.output)
        assert parsed["failed"] == 1
        assert "404" in parsed["results"][0]["error"]
        assert browser_tool._context.request.get.await_count == 1

    @pytest.mark.asyncio
    async def test_extraction_exception_isolated_per_item(self, tmp_path):
        async def evaluate(wrapper_js, args):
            raise RuntimeError("bad js")

        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeFetchResponse())
        browser_tool._page.evaluate = AsyncMock(side_effect=evaluate)
        result = await tool(items=["https://a.test"], script=[{"action": "execute_js", "text": "doc.bogus()"}])
        parsed = json.loads(result.output)
        assert "failed" in parsed["results"][0]["error"]


class TestFullPath:
    """First step is navigate - a fresh, real (mocked) Page per item."""

    @pytest.mark.asyncio
    async def test_navigate_and_get_page_text(self, tmp_path):
        text_result = {"title": "T", "url": "https://a.test", "source": "main", "text": "hello world"}
        tool, browser_tool = make_tool(
            tmp_path,
            new_page_factory=lambda: make_fake_page(evaluate_result=text_result),
        )
        result = await tool(
            items=["https://a.test"],
            script=[{"action": "navigate", "text": "{{item}}"}, {"action": "get_page_text"}],
        )
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 1
        steps = parsed["results"][0]["steps"]
        assert steps[0]["action"] == "navigate"
        assert "https://a.test" in steps[0]["output"]
        assert steps[1]["output"] == "hello world"

    @pytest.mark.asyncio
    async def test_screenshot_step_saves_file_and_omits_base64(self, tmp_path):
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: make_fake_page())
        result = await tool(
            items=["https://a.test"],
            script=[{"action": "navigate", "text": "{{item}}"}, {"action": "screenshot"}],
        )
        parsed = json.loads(result.output)
        steps = parsed["results"][0]["steps"]
        assert "Screenshot saved as" in steps[1]["output"]
        assert "base64" not in json.dumps(parsed).lower()
        # a real file was actually written
        saved_files = list(tmp_path.glob("screenshot_*.png"))
        assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_execute_js_step(self, tmp_path):
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: make_fake_page(evaluate_result={"n": 1}))
        result = await tool(
            items=["https://a.test"],
            script=[{"action": "navigate", "text": "{{item}}"}, {"action": "execute_js", "text": "document.title"}],
        )
        parsed = json.loads(result.output)
        assert parsed["results"][0]["steps"][1]["output"] == '{"n": 1}'

    @pytest.mark.asyncio
    async def test_wait_step(self, tmp_path):
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: make_fake_page())
        with patch("browser_use_demo.tools.script_runner.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await tool(
                items=["https://a.test"],
                script=[{"action": "navigate", "text": "{{item}}"}, {"action": "wait", "duration": 2}],
            )
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 1
        assert "2" in parsed["results"][0]["steps"][1]["output"]

    @pytest.mark.asyncio
    async def test_page_closed_after_item(self, tmp_path):
        page = make_fake_page()
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: page)
        await tool(items=["https://a.test"], script=[{"action": "navigate", "text": "{{item}}"}])
        page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_step_failure_isolated_with_partial_results(self, tmp_path):
        async def evaluate(*args, **kwargs):
            if args and "document.readyState" in args[0]:
                return "complete"  # let wait_for_page_ready's poll succeed
            raise RuntimeError("boom")  # the actual execute_js step fails

        page = make_fake_page()
        page.evaluate = AsyncMock(side_effect=evaluate)
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: page)
        result = await tool(
            items=["https://a.test"],
            script=[{"action": "navigate", "text": "{{item}}"}, {"action": "execute_js", "text": "boom()"}],
        )
        parsed = json.loads(result.output)
        assert parsed["failed"] == 1
        item_result = parsed["results"][0]
        assert len(item_result["steps"]) == 1  # navigate succeeded before execute_js failed
        assert "execute_js" in item_result["error"]

    @pytest.mark.asyncio
    async def test_never_touches_coordinators_own_page(self, tmp_path):
        """Concurrent items must each get their own Page, not fight over
        browser_tool._page - that's the entire reason dispatch_subagents
        uses independent browser instances instead of sharing one."""
        tool, browser_tool = make_tool(tmp_path, new_page_factory=lambda: make_fake_page())
        await tool(
            items=["https://a.test/1", "https://a.test/2"],
            script=[{"action": "navigate", "text": "{{item}}"}],
        )
        browser_tool._page.evaluate.assert_not_called()


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_pages_bounded_by_concurrency(self, tmp_path):
        max_concurrent_seen = 0
        current = 0

        def new_page_factory():
            nonlocal current
            current += 1
            return make_fake_page()

        # Simulate overlap by delaying inside navigate.
        async def slow_goto(url, **kwargs):
            nonlocal max_concurrent_seen, current
            max_concurrent_seen = max(max_concurrent_seen, current)
            await asyncio.sleep(0.02)
            response = MagicMock()
            response.status = 200
            response.headers = {}
            return response

        def new_page_with_slow_goto():
            page = make_fake_page()
            page.goto = AsyncMock(side_effect=slow_goto)

            async def close():
                nonlocal current
                current -= 1

            page.close = AsyncMock(side_effect=close)
            return page

        tool, _ = make_tool(tmp_path, new_page_factory=new_page_with_slow_goto)
        await tool(
            items=[f"https://a.test/{i}" for i in range(10)],
            script=[{"action": "navigate", "text": "{{item}}"}],
            concurrency=3,
        )
        assert max_concurrent_seen <= 3

    @pytest.mark.asyncio
    async def test_concurrency_above_hard_cap_is_clamped_not_crashing(self, tmp_path):
        tool, _ = make_tool(tmp_path, new_page_factory=lambda: make_fake_page())
        result = await tool(
            items=["https://a.test"], script=[{"action": "navigate", "text": "{{item}}"}], concurrency=9999
        )
        assert json.loads(result.output)["succeeded"] == 1


class TestSaveAs:
    @pytest.mark.asyncio
    async def test_save_as_writes_file_and_returns_summary(self, tmp_path):
        tool, _ = make_tool(
            tmp_path,
            get_side_effect=lambda url: FakeFetchResponse(),
            evaluate_result=lambda *a, **k: {"big": "x" * 100},
        )
        result = await tool(
            items=["https://a.test/1", "https://a.test/2"],
            script=[{"action": "execute_js", "text": "1"}],
            save_as="results.json",
        )
        assert "x" * 100 not in result.output
        assert "Saved file: results.json" in result.output
        saved = json.loads((tmp_path / "results.json").read_text())
        assert len(saved) == 2


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_fast_path_retries_on_429(self, tmp_path):
        calls = {"n": 0}

        def get(url):
            calls["n"] += 1
            if calls["n"] < 2:
                return FakeFetchResponse(status=429, headers={"retry-after": "0.01"})
            return FakeFetchResponse(body="ok")

        tool, _ = make_tool(tmp_path, get_side_effect=get)
        result = await tool(items=["https://a.test"], script=[{"action": "execute_js", "text": "1"}])
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 1
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_full_path_gives_up_after_max_attempts_on_persistent_429(self, tmp_path):
        tool, browser_tool = make_tool(tmp_path, new_page_factory=lambda: make_fake_page(goto_status=429))
        # The rate limiter's own backoff sleep lives in batch_extract.py's
        # asyncio (AdaptiveRateLimiter.wait), not script_runner's - that's
        # the one actually slowing this retry loop down.
        with patch("browser_use_demo.tools.batch_extract.asyncio.sleep", new=AsyncMock()):
            result = await tool(items=["https://a.test"], script=[{"action": "navigate", "text": "{{item}}"}])
        parsed = json.loads(result.output)
        assert parsed["failed"] == 1
        assert f"gave up after {MAX_ATTEMPTS_PER_ITEM} attempts" in parsed["results"][0]["error"]
