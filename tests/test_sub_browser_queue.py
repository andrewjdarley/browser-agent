"""Tests for SubBrowserQueue (the queue_screenshots/queue_status/queue_clear/
queue_pause/queue_resume tools): step execution (navigate/scroll_to/
screenshot), concurrency bounded by max_fanout, pause/resume/clear
semantics, and per-item error isolation - all without a real Playwright
browser."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_use_demo.tools.sub_browser_queue import (
    SubBrowserQueue,
    build_queue_clear_tool_fn,
    build_queue_pause_tool_fn,
    build_queue_resume_tool_fn,
    build_queue_screenshots_tool_fn,
    build_queue_status_tool_fn,
)


@pytest.fixture(autouse=True)
def fast_page_ready_wait():
    """wait_for_page_ready's settle delay (SCREENSHOT_SETTLE_DELAY_S, 1.5s)
    would otherwise cost real wall-clock time on every navigate step here.
    Patches sub_browser_queue's own imported name specifically - NOT
    browser.py's module-level asyncio.sleep (the pattern test_script_runner.py
    uses) - because asyncio is a singleton module, so patching
    `browser_use_demo.tools.browser.asyncio.sleep` patches asyncio.sleep
    globally, for every module in the process. That silently breaks these
    tests in a different way: they need queue.add()'s fire-and-forget
    background task to actually run while a separate real-time polling loop
    (drain(), below) observes it - if asyncio.sleep is mocked globally,
    drain()'s own `await asyncio.sleep(...)` calls stop yielding real control
    back to the event loop, and the background task never gets scheduled at
    all (confirmed by hand: with the global patch, the scheduled task's
    coroutine literally never started running, even after 50+ "ticks")."""
    with patch("browser_use_demo.tools.sub_browser_queue.wait_for_page_ready", new=AsyncMock()):
        yield


def make_fake_page(*, goto_side_effect=None, scroll_side_effect=None):
    page = MagicMock()
    page.set_viewport_size = AsyncMock()

    async def default_goto(url, **kwargs):
        return MagicMock(status=200, headers={})

    page.goto = AsyncMock(side_effect=goto_side_effect or default_goto)

    async def default_evaluate(expression, *args, **kwargs):
        # capture_screenshot's full_page branch evaluates scrollHeight (needs
        # a number); wait_for_page_ready's own readyState poll is mocked
        # away entirely by the fast_page_ready_wait fixture, so that branch
        # is never actually hit here - this only needs to handle
        # scrollHeight to keep capture_screenshot's own comparison working.
        if "scrollHeight" in expression:
            return 800
        return "complete"

    page.evaluate = AsyncMock(side_effect=default_evaluate)

    locator = MagicMock()
    locator.scroll_into_view_if_needed = AsyncMock(side_effect=scroll_side_effect)
    text_locator = MagicMock()
    text_locator.first = locator
    page.get_by_text = MagicMock(return_value=text_locator)

    page.screenshot = AsyncMock(side_effect=lambda path, **kw: Path(path).write_bytes(b"\x89PNG\r\n" + b"0" * 100))
    page.close = AsyncMock()
    return page


def make_browser_tool(tmp_path: Path, page_factory=make_fake_page):
    bt = MagicMock()
    bt._ensure_browser = AsyncMock()
    bt.run_dir = tmp_path
    bt.width = 1920
    bt.height = 1080
    bt._context = MagicMock()
    bt._context.new_page = AsyncMock(side_effect=lambda: page_factory())
    return bt


async def drain(queue: SubBrowserQueue, timeout: float = 2.0):
    """Let scheduled item tasks (asyncio.create_task in _kick_off_more) run
    to completion - queue.add() only schedules them, it doesn't await them,
    matching the tool's real fire-and-forget contract. Uses real small
    sleeps (not bare asyncio.sleep(0)) so the loop gets enough wall-clock
    slices to fully drain a multi-await chain (goto -> wait_for_page_ready
    -> its internal wait_for/poll -> screenshot, several hops deep) -
    browser.py's own asyncio.sleep is mocked instant by the autouse fixture
    above, so real total time here is milliseconds, not the 1.5s it'd
    otherwise cost per navigate."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while queue._in_progress or queue._pending:
        if loop.time() > deadline:
            raise AssertionError("queue did not drain within timeout")
        await asyncio.sleep(0.01)


class TestAddAndExecution:
    @pytest.mark.asyncio
    async def test_add_returns_ids_immediately_without_waiting(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        ids = queue.add([[{"action": "navigate", "url": "example.com"}]])
        # add() is synchronous and returns before anything actually runs -
        # the fire-and-forget contract queue_screenshots' tool fn relies on.
        assert len(ids) == 1
        assert isinstance(ids[0], str) and ids[0]

    @pytest.mark.asyncio
    async def test_navigate_then_screenshot_produces_a_saved_file(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}, {"action": "screenshot"}]])
        await drain(queue)

        snap = queue.snapshot()
        assert len(snap["completed"]) == 1
        item = snap["completed"][0]
        assert item["status"] == "done"
        assert len(item["screenshots"]) == 1
        assert (tmp_path / item["screenshots"][0]).exists()

    @pytest.mark.asyncio
    async def test_navigate_prepends_https_when_no_scheme(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}]])
        await drain(queue)

        item = queue.snapshot()["completed"][0]
        assert "https://example.com" in item["log"][0]

    @pytest.mark.asyncio
    async def test_navigate_leaves_full_url_untouched(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "http://example.com/page"}]])
        await drain(queue)
        item = queue.snapshot()["completed"][0]
        assert "http://example.com/page" in item["log"][0]

    @pytest.mark.asyncio
    async def test_scroll_to_uses_get_by_text_with_target(self, tmp_path):
        captured = {}

        def page_factory():
            page = make_fake_page()
            original = page.get_by_text

            def spy(target, exact=False):
                captured["target"] = target
                return original(target, exact=exact)

            page.get_by_text = MagicMock(side_effect=spy)
            return page

        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path, page_factory), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}, {"action": "scroll_to", "target": "Pricing"}]])
        await drain(queue)

        assert captured["target"] == "Pricing"
        item = queue.snapshot()["completed"][0]
        assert item["status"] == "done"

    @pytest.mark.asyncio
    async def test_full_page_screenshot_flag_recorded_in_log(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}, {"action": "screenshot", "full_page": True}]])
        await drain(queue)
        item = queue.snapshot()["completed"][0]
        assert any("full page" in line for line in item["log"])

    @pytest.mark.asyncio
    async def test_multiple_screenshots_in_one_item_all_collected(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add(
            [
                [
                    {"action": "navigate", "url": "example.com"},
                    {"action": "screenshot"},
                    {"action": "scroll_to", "target": "Contact"},
                    {"action": "screenshot", "full_page": True},
                ]
            ]
        )
        await drain(queue)
        item = queue.snapshot()["completed"][0]
        assert item["status"] == "done"
        assert len(item["screenshots"]) == 2
        assert len(set(item["screenshots"])) == 2  # distinct filenames


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_unsupported_action_fails_just_that_item(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add(
            [
                [{"action": "click_something_unsupported"}],
                [{"action": "navigate", "url": "example.com"}, {"action": "screenshot"}],
            ]
        )
        await drain(queue)

        statuses = sorted(item["status"] for item in queue.snapshot()["completed"])
        assert statuses == ["done", "error"]

    @pytest.mark.asyncio
    async def test_navigate_failure_recorded_as_item_error(self, tmp_path):
        async def failing_goto(url, **kwargs):
            raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")

        bt = make_browser_tool(tmp_path, page_factory=lambda: make_fake_page(goto_side_effect=failing_goto))
        queue = SubBrowserQueue(browser_tool=bt, run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "nonexistent.invalid"}]])
        await drain(queue)

        item = queue.snapshot()["completed"][0]
        assert item["status"] == "error"
        assert "ERR_NAME_NOT_RESOLVED" in item["error"]

    @pytest.mark.asyncio
    async def test_page_open_failure_recorded_as_item_error(self, tmp_path):
        bt = make_browser_tool(tmp_path)
        bt._context.new_page = AsyncMock(side_effect=RuntimeError("browser crashed"))
        queue = SubBrowserQueue(browser_tool=bt, run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}]])
        await drain(queue)

        item = queue.snapshot()["completed"][0]
        assert item["status"] == "error"
        assert "browser crashed" in item["error"]


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_respects_max_fanout(self, tmp_path):
        gate = asyncio.Event()
        started = []

        async def blocking_goto(url, **kwargs):
            started.append(url)
            await gate.wait()
            return MagicMock(status=200, headers={})

        bt = make_browser_tool(tmp_path, page_factory=lambda: make_fake_page(goto_side_effect=blocking_goto))
        queue = SubBrowserQueue(browser_tool=bt, run_dir=tmp_path, max_fanout=2)
        queue.add([[{"action": "navigate", "url": f"example.com/{i}"}] for i in range(5)])

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 2.0
        while len(started) < 2 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # let it settle at exactly max_fanout, not just >=1
        snap = queue.snapshot()
        assert snap["in_progress_count"] == 2
        assert snap["pending_count"] == 3

        gate.set()
        await drain(queue)
        assert len(queue.snapshot()["completed"]) == 5

    @pytest.mark.asyncio
    async def test_finishing_an_item_immediately_starts_the_next_pending_one(self, tmp_path):
        queue = SubBrowserQueue(
            browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path, max_fanout=1
        )
        queue.add([[{"action": "navigate", "url": f"example.com/{i}"}] for i in range(3)])
        await drain(queue)
        assert len(queue.snapshot()["completed"]) == 3


class TestPauseResumeClear:
    @pytest.mark.asyncio
    async def test_pause_prevents_new_items_from_starting(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.pause()
        queue.add([[{"action": "navigate", "url": "example.com"}]])
        await asyncio.sleep(0.1)  # give it a real chance to (wrongly) start, if it were going to

        snap = queue.snapshot()
        assert snap["paused"] is True
        assert snap["pending_count"] == 1
        assert snap["in_progress_count"] == 0
        assert snap["completed"] == []

    @pytest.mark.asyncio
    async def test_resume_starts_queued_items(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.pause()
        queue.add([[{"action": "navigate", "url": "example.com"}]])
        queue.resume()
        await drain(queue)
        assert len(queue.snapshot()["completed"]) == 1

    def test_clear_drops_pending_and_completed(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.pause()
        queue.add([[{"action": "navigate", "url": "example.com"}]] * 3)
        queue._completed.append({"id": "old", "status": "done", "screenshots": [], "log": [], "error": None})

        counts = queue.clear()
        assert counts == {"pending_cleared": 3, "completed_cleared": 1}
        snap = queue.snapshot()
        assert snap["pending_count"] == 0
        assert snap["completed"] == []

    def test_clear_does_not_touch_in_progress(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue._in_progress["fake_id"] = {"id": "fake_id", "steps": []}
        queue.clear()
        assert queue.snapshot()["in_progress_count"] == 1


class TestSnapshotCap:
    def test_completed_history_capped(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        for i in range(250):
            queue._finish_item(f"id_{i}", {"id": f"id_{i}", "status": "done", "screenshots": [], "log": [], "error": None})
        assert len(queue.snapshot()["completed"]) == 200


class TestToolFunctions:
    """build_queue_*_tool_fn wraps each function in an SdkMcpTool (via the
    SDK's @tool decorator) - the actual callable lives at .handler, not the
    SdkMcpTool object itself."""

    @pytest.mark.asyncio
    async def test_queue_screenshots_tool_adds_and_returns_ids(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.pause()  # keep it from actually running for this smoke check
        fn = build_queue_screenshots_tool_fn(queue)
        result = await fn.handler({"items": [[{"action": "navigate", "url": "example.com"}]]})
        assert result["content"][0]["type"] == "text"
        assert "Queued 1 item" in result["content"][0]["text"]
        assert queue.snapshot()["pending_count"] == 1

    @pytest.mark.asyncio
    async def test_queue_screenshots_tool_rejects_empty_items(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        fn = build_queue_screenshots_tool_fn(queue)
        result = await fn.handler({"items": []})
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_queue_status_tool_reports_progress(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.add([[{"action": "navigate", "url": "example.com"}, {"action": "screenshot"}]])
        await drain(queue)

        fn = build_queue_status_tool_fn(queue)
        result = await fn.handler({})
        text = result["content"][0]["text"]
        assert "pending: 0" in text
        assert "1 of 1 completed" in text

    @pytest.mark.asyncio
    async def test_queue_clear_tool(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        queue.pause()
        queue.add([[{"action": "navigate", "url": "example.com"}]])
        fn = build_queue_clear_tool_fn(queue)
        result = await fn.handler({})
        assert "Cleared 1 pending" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_queue_pause_and_resume_tools(self, tmp_path):
        queue = SubBrowserQueue(browser_tool=make_browser_tool(tmp_path), run_dir=tmp_path)
        pause_fn = build_queue_pause_tool_fn(queue)
        resume_fn = build_queue_resume_tool_fn(queue)

        await pause_fn.handler({})
        assert queue.paused is True
        await resume_fn.handler({})
        assert queue.paused is False
