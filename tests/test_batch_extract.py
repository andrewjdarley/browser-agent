"""Tests for BatchExtractTool: concurrent fetch-based extraction across many
URLs, per-item error isolation, save_as behavior, and the AdaptiveRateLimiter
backoff logic - all without a real Playwright browser or real network."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_use_demo.tools.batch_extract import (
    MAX_ATTEMPTS_PER_URL,
    AdaptiveRateLimiter,
    BatchExtractTool,
)
from browser_use_demo.tools.file_output import FileOutputTool


class FakeResponse:
    def __init__(self, status=200, body="<html></html>", headers=None):
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def text(self):
        return self._body


def make_browser_tool(get_side_effect, evaluate_result=None):
    """A stand-in for BrowserTool exposing just what BatchExtractTool touches:
    _ensure_browser(), _context.request.get(), _page.evaluate()."""
    bt = MagicMock()
    bt._ensure_browser = AsyncMock()
    bt._context = MagicMock()
    bt._context.request.get = AsyncMock(side_effect=get_side_effect)
    bt._page = MagicMock()
    bt._page.evaluate = AsyncMock(
        side_effect=evaluate_result if callable(evaluate_result) else (lambda *a, **k: {"ok": True})
    )
    return bt


def make_tool(tmp_path, get_side_effect, evaluate_result=None):
    browser_tool = make_browser_tool(get_side_effect, evaluate_result)
    file_output_tool = FileOutputTool(run_dir=tmp_path)
    return BatchExtractTool(browser_tool=browser_tool, file_output_tool=file_output_tool), browser_tool


class TestEmptyUrls:
    @pytest.mark.asyncio
    async def test_rejects_empty_urls(self, tmp_path):
        tool, _ = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse())
        result = await tool(urls=[], extract_js="(doc) => doc.title")
        assert result.error is not None


class TestExtractionPath:
    @pytest.mark.asyncio
    async def test_uses_context_request_not_page_fetch(self, tmp_path):
        """The whole point of context.request over page-side fetch() is that
        it bypasses CORS - so the code must go through it, not page.evaluate
        with a JS fetch() call."""
        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse(body="<title>hi</title>"))
        await tool(urls=["https://example.com/a"], extract_js="(doc) => doc.title")

        browser_tool._context.request.get.assert_awaited_once_with("https://example.com/a")
        browser_tool._page.evaluate.assert_awaited_once()
        _wrapper_js, fetch_args = browser_tool._page.evaluate.call_args.args
        assert fetch_args == ["<title>hi</title>", "(doc) => doc.title"]

    @pytest.mark.asyncio
    async def test_successful_extraction_reported_per_url(self, tmp_path):
        tool, _ = make_tool(
            tmp_path,
            get_side_effect=lambda url: FakeResponse(body="<html></html>"),
            evaluate_result=lambda *a, **k: {"title": "Example"},
        )
        result = await tool(urls=["https://a.test", "https://b.test"], extract_js="(doc) => ({title: doc.title})")
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 2
        assert parsed["failed"] == 0
        assert all(r["data"] == {"title": "Example"} for r in parsed["results"])


class TestPerItemErrorIsolation:
    @pytest.mark.asyncio
    async def test_client_error_status_fails_immediately_no_retry(self, tmp_path):
        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse(status=404))
        result = await tool(urls=["https://a.test/missing"], extract_js="(doc) => doc.title")
        parsed = json.loads(result.output)
        assert parsed["failed"] == 1
        assert "404" in parsed["results"][0]["error"]
        # a 404 is a permanent client error - no point retrying it
        assert browser_tool._context.request.get.await_count == 1

    @pytest.mark.asyncio
    async def test_extraction_exception_is_isolated_per_item(self, tmp_path):
        async def evaluate(js, args):
            raise RuntimeError("bad extract_js")

        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse())
        browser_tool._page.evaluate = AsyncMock(side_effect=evaluate)
        result = await tool(urls=["https://a.test"], extract_js="(doc) => doc.bogus()")
        parsed = json.loads(result.output)
        assert "extraction failed" in parsed["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, tmp_path):
        def get(url):
            if "bad" in url:
                return FakeResponse(status=500)
            return FakeResponse(body="ok")

        tool, _ = make_tool(tmp_path, get_side_effect=get)
        result = await tool(
            urls=["https://a.test/good", "https://a.test/bad"], extract_js="(doc) => true"
        )
        parsed = json.loads(result.output)
        by_url = {r["url"]: r for r in parsed["results"]}
        assert "error" not in by_url["https://a.test/good"]
        assert "error" in by_url["https://a.test/bad"]


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_requests_bounded_by_concurrency(self, tmp_path):
        max_concurrent_seen = 0
        current = 0

        async def get(url):
            nonlocal max_concurrent_seen, current
            current += 1
            max_concurrent_seen = max(max_concurrent_seen, current)
            await asyncio.sleep(0.02)
            current -= 1
            return FakeResponse()

        tool, browser_tool = make_tool(tmp_path, get_side_effect=None)
        browser_tool._context.request.get = AsyncMock(side_effect=get)

        await tool(
            urls=[f"https://a.test/{i}" for i in range(10)],
            extract_js="(doc) => true",
            concurrency=3,
        )
        assert max_concurrent_seen <= 3

    @pytest.mark.asyncio
    async def test_concurrency_above_hard_cap_is_clamped_not_crashing(self, tmp_path):
        tool, _ = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse())
        result = await tool(urls=["https://a.test"], extract_js="(doc) => true", concurrency=9999)
        assert json.loads(result.output)["succeeded"] == 1


class TestSaveAs:
    @pytest.mark.asyncio
    async def test_save_as_writes_file_and_returns_summary_not_full_data(self, tmp_path):
        tool, _ = make_tool(
            tmp_path,
            get_side_effect=lambda url: FakeResponse(),
            evaluate_result=lambda *a, **k: {"big": "x" * 100},
        )
        result = await tool(
            urls=["https://a.test/1", "https://a.test/2"],
            extract_js="(doc) => true",
            save_as="results.json",
        )
        assert "x" * 100 not in result.output  # not dumping full data back into context
        assert "Saved file: results.json" in result.output
        saved = json.loads((tmp_path / "results.json").read_text())
        assert len(saved) == 2

    @pytest.mark.asyncio
    async def test_save_as_reports_failures_inline(self, tmp_path):
        tool, _ = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse(status=404))
        result = await tool(urls=["https://a.test/x"], extract_js="(doc) => true", save_as="out.json")
        assert "failed" in result.output.lower() or "1 failed" in result.output


class TestAdaptiveRateLimiter:
    @pytest.mark.asyncio
    async def test_starts_with_no_delay(self):
        limiter = AdaptiveRateLimiter()
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await limiter.wait()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limited_without_retry_after_grows_delay(self):
        limiter = AdaptiveRateLimiter()
        await limiter.on_rate_limited(retry_after=None)
        assert limiter._delay > 0

    @pytest.mark.asyncio
    async def test_rate_limited_respects_retry_after(self):
        limiter = AdaptiveRateLimiter(max_delay=30.0)
        await limiter.on_rate_limited(retry_after=5.0)
        assert limiter._delay >= 5.0

    @pytest.mark.asyncio
    async def test_delay_capped_at_max_delay(self):
        limiter = AdaptiveRateLimiter(max_delay=2.0)
        for _ in range(10):
            await limiter.on_rate_limited(retry_after=None)
        assert limiter._delay <= 2.0

    @pytest.mark.asyncio
    async def test_delay_decays_after_consecutive_successes(self):
        limiter = AdaptiveRateLimiter(max_delay=30.0, decrease_after=3)
        await limiter.on_rate_limited(retry_after=10.0)
        assert limiter._delay == 10.0
        for _ in range(3):
            await limiter.on_success()
        assert limiter._delay < 10.0

    @pytest.mark.asyncio
    async def test_success_is_a_noop_when_delay_already_zero(self):
        limiter = AdaptiveRateLimiter()
        for _ in range(20):
            await limiter.on_success()
        assert limiter._delay == 0.0


class TestRetryExhaustion:
    @pytest.mark.asyncio
    async def test_always_rate_limited_url_gives_up_after_max_attempts(self, tmp_path):
        tool, browser_tool = make_tool(tmp_path, get_side_effect=lambda url: FakeResponse(status=429))
        with patch("browser_use_demo.tools.batch_extract.asyncio.sleep", new=AsyncMock()):
            result = await tool(urls=["https://a.test/throttled"], extract_js="(doc) => true")
        parsed = json.loads(result.output)
        assert parsed["failed"] == 1
        assert f"gave up after {MAX_ATTEMPTS_PER_URL} attempts" in parsed["results"][0]["error"]
        assert browser_tool._context.request.get.await_count == MAX_ATTEMPTS_PER_URL

    @pytest.mark.asyncio
    async def test_recovers_after_transient_rate_limiting(self, tmp_path):
        calls = {"n": 0}

        def get(url):
            calls["n"] += 1
            if calls["n"] < 3:
                return FakeResponse(status=429, headers={"retry-after": "0.01"})
            return FakeResponse(body="ok")

        tool, browser_tool = make_tool(tmp_path, get_side_effect=get)
        result = await tool(urls=["https://a.test/flaky"], extract_js="(doc) => true")
        parsed = json.loads(result.output)
        assert parsed["succeeded"] == 1
        assert calls["n"] == 3
