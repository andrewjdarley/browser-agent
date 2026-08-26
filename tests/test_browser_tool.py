"""Tests for the new BrowserTool behavior: full-page screenshots, the
height/timeout guards, back/forward navigation, and the page-ready wait."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_use_demo.tools.base import ToolError, ToolResult
from browser_use_demo.tools.browser import (
    DOM_DIFF_MAX_CHARS,
    DOM_MUTATING_ACTIONS,
    MAX_FULL_PAGE_HEIGHT_PX,
    MAX_SCREENSHOT_RAW_BYTES,
    BrowserTool,
)


def make_tool_with_mock_page(tmp_path: Path) -> tuple[BrowserTool, MagicMock]:
    tool = BrowserTool(run_dir=tmp_path)
    page = MagicMock()

    async def default_evaluate(expr):
        if "scrollHeight" in expr:
            return 100  # well under MAX_FULL_PAGE_HEIGHT_PX by default
        return "complete"  # document.readyState

    page.evaluate = AsyncMock(side_effect=default_evaluate)
    page.screenshot = AsyncMock(side_effect=lambda path, **kw: Path(path).write_bytes(b"\x89PNG\r\n"))
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    tool._page = page
    return tool, page


class TestFullPageScreenshot:
    @pytest.mark.asyncio
    async def test_passes_full_page_true_to_playwright(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._take_screenshot(full_page=True)
        assert page.screenshot.call_args.kwargs["full_page"] is True

    @pytest.mark.asyncio
    async def test_default_screenshot_is_viewport_only(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._take_screenshot()
        assert page.screenshot.call_args.kwargs["full_page"] is False

    @pytest.mark.asyncio
    async def test_writes_into_run_dir(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._take_screenshot()
        written_path = Path(page.screenshot.call_args.kwargs["path"])
        assert written_path.parent == tmp_path

    @pytest.mark.asyncio
    async def test_height_guard_raises_above_threshold(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        page.evaluate = AsyncMock(
            side_effect=lambda expr: MAX_FULL_PAGE_HEIGHT_PX + 1
            if "scrollHeight" in expr
            else "complete"
        )
        with pytest.raises(ToolError, match="too tall"):
            await tool._take_screenshot(full_page=True)

    @pytest.mark.asyncio
    async def test_height_guard_allows_below_threshold(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        page.evaluate = AsyncMock(
            side_effect=lambda expr: MAX_FULL_PAGE_HEIGHT_PX - 1
            if "scrollHeight" in expr
            else "complete"
        )
        result = await tool._take_screenshot(full_page=True)
        assert result.base64_image is not None


class TestOversizedScreenshotRetry:
    """A full_page capture that's too big gets retried as a clipped
    top-of-page capture instead of being sent whole (or just erroring) -
    regression coverage for a real crash where a large full-page screenshot
    exceeded the SDK transport's message buffer mid-session."""

    def _make_tool(self, tmp_path, *, full_page_size, clip_size=None, page_height=5000):
        """clip_size=None means the clipped retry isn't expected to be
        reached in that test; a value simulates its resulting file size."""
        tool = BrowserTool(run_dir=tmp_path)
        page = MagicMock()

        async def evaluate(expr):
            return page_height if "scrollHeight" in expr else "complete"

        def screenshot(path, **kwargs):
            if kwargs.get("full_page"):
                size = full_page_size
            elif "clip" in kwargs:
                size = clip_size if clip_size is not None else 100
            else:
                size = 100  # plain viewport fallback - always small
            Path(path).write_bytes(b"\x89" * size)

        page.evaluate = AsyncMock(side_effect=evaluate)
        page.screenshot = AsyncMock(side_effect=screenshot)
        tool._page = page
        return tool, page

    @pytest.mark.asyncio
    async def test_small_full_page_capture_is_sent_as_is(self, tmp_path):
        tool, page = self._make_tool(tmp_path, full_page_size=1000)
        result = await tool._take_screenshot(full_page=True)
        assert page.screenshot.await_count == 1
        assert result.output.startswith("Screenshot saved as ")
        assert "too large" not in result.output  # no truncation note

    @pytest.mark.asyncio
    async def test_oversized_capture_retries_with_clip(self, tmp_path):
        tool, page = self._make_tool(
            tmp_path,
            full_page_size=MAX_SCREENSHOT_RAW_BYTES + 1,
            clip_size=1000,
        )
        result = await tool._take_screenshot(full_page=True)

        assert page.screenshot.await_count == 2
        second_call_kwargs = page.screenshot.await_args_list[1].kwargs
        assert "clip" in second_call_kwargs
        assert second_call_kwargs["clip"]["y"] == 0
        assert "too large to send in one screenshot" in result.output
        assert result.base64_image is not None

    @pytest.mark.asyncio
    async def test_clip_height_clamped_to_actual_page_height(self, tmp_path):
        # Page is shorter than FULL_PAGE_FALLBACK_VIEWPORTS * viewport height
        tool, page = self._make_tool(
            tmp_path,
            full_page_size=MAX_SCREENSHOT_RAW_BYTES + 1,
            clip_size=1000,
            page_height=500,
        )
        await tool._take_screenshot(full_page=True)
        clip = page.screenshot.await_args_list[1].kwargs["clip"]
        assert clip["height"] == 500

    @pytest.mark.asyncio
    async def test_still_oversized_after_clip_falls_back_to_viewport(self, tmp_path):
        tool, page = self._make_tool(
            tmp_path,
            full_page_size=MAX_SCREENSHOT_RAW_BYTES + 1,
            clip_size=MAX_SCREENSHOT_RAW_BYTES + 1,
        )
        result = await tool._take_screenshot(full_page=True)

        assert page.screenshot.await_count == 3
        third_call_kwargs = page.screenshot.await_args_list[2].kwargs
        assert third_call_kwargs.get("full_page") is False
        assert "too large to capture in full even when clipped" in result.output
        assert result.base64_image is not None


class TestPageReadyWait:
    """settle_delay_s=0 in most of these - they're testing the readyState
    polling/timeout behavior specifically, not the settle delay (see
    TestScreenshotSettleDelay), and the default 2s delay would make these
    slow or (in the hard-timeout case) break the outer deadline being
    tested."""

    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_complete(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._wait_for_page_ready(timeout_s=1.0, poll_interval=0.05, settle_delay_s=0)
        # readyState evaluated at least once, no error
        page.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_gives_up_after_timeout_without_raising(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        page.evaluate = AsyncMock(return_value="loading")  # never becomes "complete"
        # Should not raise, should not hang beyond the timeout
        await tool._wait_for_page_ready(timeout_s=0.2, poll_interval=0.05, settle_delay_s=0)

    @pytest.mark.asyncio
    async def test_hard_timeout_even_if_evaluate_hangs(self, tmp_path):
        import asyncio

        tool, page = make_tool_with_mock_page(tmp_path)

        async def hang(_expr):
            await asyncio.sleep(10)

        page.evaluate = AsyncMock(side_effect=hang)
        # Must return well before the 10s hang would, proving the outer
        # asyncio.wait_for actually bounds a stuck page.evaluate() call.
        await asyncio.wait_for(
            tool._wait_for_page_ready(timeout_s=0.3, poll_interval=0.05, settle_delay_s=0),
            timeout=2.0,
        )


class TestScreenshotSettleDelay:
    """Regression coverage for a real failure: a screenshot taken right after
    navigate came back blank because document.readyState hit "complete"
    before the (client-rendered) page had actually painted its content. Fix
    is a flat hardcoded settle delay after the readyState poll - see
    SCREENSHOT_SETTLE_DELAY_S in browser.py."""

    @pytest.mark.asyncio
    async def test_wait_for_page_ready_sleeps_settle_delay_by_default(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._wait_for_page_ready(timeout_s=1.0, poll_interval=0.05)

        assert browser_module.SCREENSHOT_SETTLE_DELAY_S in sleep_calls

    @pytest.mark.asyncio
    async def test_screenshot_action_applies_settle_delay(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._take_screenshot()

        assert browser_module.SCREENSHOT_SETTLE_DELAY_S in sleep_calls

    @pytest.mark.asyncio
    async def test_settle_delay_can_be_disabled(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._wait_for_page_ready(timeout_s=1.0, poll_interval=0.05, settle_delay_s=0)

        assert browser_module.SCREENSHOT_SETTLE_DELAY_S not in sleep_calls


class TestTextReadSettleDelay:
    """Same readyState-isn't-enough problem as the screenshot case, but for
    get_page_text/read_page/execute_js - regression coverage for a real run
    where get_page_text caught a site's own transient "there was an error
    while loading" placeholder immediately after navigate. Shorter delay
    than screenshots (TEXT_READ_SETTLE_DELAY_S) since these don't need a
    full visual paint."""

    @pytest.mark.asyncio
    async def test_get_page_text_applies_settle_delay(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._get_page_text()

        assert browser_module.TEXT_READ_SETTLE_DELAY_S in sleep_calls

    @pytest.mark.asyncio
    async def test_read_page_applies_settle_delay(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._read_page()

        assert browser_module.TEXT_READ_SETTLE_DELAY_S in sleep_calls

    @pytest.mark.asyncio
    async def test_execute_js_applies_settle_delay(self, tmp_path, monkeypatch):
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._execute_js("1 + 1")

        assert browser_module.TEXT_READ_SETTLE_DELAY_S in sleep_calls

    @pytest.mark.asyncio
    async def test_dom_diff_internal_reads_are_not_delayed(self, tmp_path, monkeypatch):
        # _current_dom_text (used internally by find and the DOM-diff
        # auto-push) goes through _execute_js_from_file directly, not the
        # model-facing _execute_js action - it must NOT pick up this delay,
        # or every mutating action's diff computation would compound it.
        from browser_use_demo.tools import browser as browser_module

        tool, page = make_tool_with_mock_page(tmp_path)
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(browser_module.asyncio, "sleep", fake_sleep)
        await tool._current_dom_text("all")

        assert sleep_calls == []


class TestNavigateBackForward:
    @pytest.mark.asyncio
    async def test_back_calls_go_back_not_goto(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._navigate("back")
        page.go_back.assert_called_once()
        page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_forward_calls_go_forward_not_goto(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._navigate("forward")
        page.go_forward.assert_called_once()
        page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_regular_url_calls_goto(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        await tool._navigate("example.com")
        page.goto.assert_called_once()
        assert page.goto.call_args.args[0] == "https://example.com"
        page.go_back.assert_not_called()


class TestFindUsesConfiguredModel:
    """find (semantic element search) makes its own direct Anthropic API
    call, separate from the coordinator's SDK session - regression coverage
    for it previously being hardcoded to a specific old model with no way
    to change it. See model_config.FIND_MODEL."""

    @pytest.mark.asyncio
    async def test_find_calls_the_api_with_the_configured_find_model(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock as AM
        from unittest.mock import MagicMock as MM
        from unittest.mock import patch

        from browser_use_demo.model_config import FIND_MODEL

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        tool, page = make_tool_with_mock_page(tmp_path)
        page.evaluate = AM(return_value={"pageContent": '- link "[275]" [ref=ref_1] href="#x"'})

        mock_content = MM()
        mock_content.text = "FOUND: 1\nSHOWING: 1\n---\nref_1 | link | [275] | citation | matches"
        mock_response = MM()
        mock_response.content = [mock_content]
        mock_client = MM()
        mock_client.messages.create = AM(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            await tool._find("the 275th reference")

        assert mock_client.messages.create.call_args.kwargs["model"] == FIND_MODEL


def _make_element_info(**overrides):
    info = {
        "success": True,
        "coordinates": [60, 40],
        "elementInfo": "a.citation-link",
        "elementRef": "ref_1",
        "rect": {"left": 10, "top": 20, "right": 110, "bottom": 60, "width": 100, "height": 40},
        "attributes": {"type": "", "role": "", "ariaLabel": "", "text": "[275]"},
        "isVisible": True,
        "isInteractable": True,
    }
    info.update(overrides)
    return info


class TestOutline:
    """outline(ref) draws a bounding box around an element and screenshots
    it - a visual complement to find/execute_js for confirming a ref really
    points at the element it claims to. Named "outline", not "highlight",
    to avoid the model confusing this action with a task's own use of that
    word - see test_loop.py's highlight->outline rename regression guard."""

    @pytest.mark.asyncio
    async def test_draws_box_screenshots_and_removes_it(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        calls = []

        async def evaluate(expr, *args):
            calls.append((expr, args))
            if "getBoundingClientRect" in expr:
                return _make_element_info()
            if "scrollHeight" in expr:
                return 100
            if args:
                return None  # the box-drawing call: evaluate(js_fn, rect)
            if "getElementById" in expr:
                return None  # the cleanup call
            return "complete"  # readyState poll

        page.evaluate = AsyncMock(side_effect=evaluate)
        result = await tool._outline("ref_1")

        assert result.base64_image is not None
        assert "Outlined element with ref: ref_1" in result.output
        assert "a.citation-link" in result.output

        draw_calls = [c for c in calls if c[1]]
        assert len(draw_calls) == 1
        assert draw_calls[0][1][0]["left"] == 10  # the resolved rect was passed through

        remove_calls = [c for c in calls if "getElementById" in c[0] and not c[1]]
        assert len(remove_calls) == 1

    @pytest.mark.asyncio
    async def test_raises_when_ref_not_found(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)

        async def evaluate(expr, *args):
            if "getBoundingClientRect" in expr:
                return {
                    "success": False,
                    "message": 'No element found with reference: "ref_99".',
                }
            return "complete"

        page.evaluate = AsyncMock(side_effect=evaluate)
        with pytest.raises(ToolError, match="No element found"):
            await tool._outline("ref_99")

    @pytest.mark.asyncio
    async def test_removes_box_even_if_screenshot_fails(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        remove_called = []

        async def evaluate(expr, *args):
            if "getBoundingClientRect" in expr:
                return _make_element_info()
            if args:
                return None
            if "getElementById" in expr:
                remove_called.append(True)
                return None
            return "complete"

        page.evaluate = AsyncMock(side_effect=evaluate)
        page.screenshot = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(ToolError):
            await tool._outline("ref_1")

        assert remove_called == [True]


class TestDiffDom:
    """_diff_dom's pure line-diffing logic, independent of any page/mocking."""

    def test_no_baseline_yields_no_diff(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        assert tool._diff_dom(None, "- link \"A\" [ref=ref_1]") == ""

    def test_identical_trees_yield_no_diff(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tree = "- link \"A\" [ref=ref_1]\n- button \"B\" [ref=ref_2]"
        assert tool._diff_dom(tree, tree) == ""

    def test_changed_line_appears_as_remove_and_add(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        old = "- link \"A\" [ref=ref_1]\n- button \"B\" [ref=ref_2]"
        new = "- link \"A\" [ref=ref_1]\n- button \"C\" [ref=ref_2]"
        diff = tool._diff_dom(old, new)
        assert 'link "A"' not in diff  # unchanged line shouldn't show up at all
        # unified_diff prefixes with +/- on top of the tree's own "- " bullet
        assert '-- button "B" [ref=ref_2]' in diff
        assert '+- button "C" [ref=ref_2]' in diff

    def test_unchanged_lines_are_not_included(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        old = "- link \"A\" [ref=ref_1]\n- button \"B\" [ref=ref_2]"
        new = "- link \"A\" [ref=ref_1]\n- button \"B\" [ref=ref_2]\n- new \"C\" [ref=ref_3]"
        diff = tool._diff_dom(old, new)
        assert 'link "A"' not in diff
        assert 'new "C" [ref=ref_3]' in diff


class TestWrapDomNote:
    def test_short_content_is_not_truncated(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        note = tool._wrap_dom_note("Some label", "short content")
        assert "__PAGE_EXTRACTED__" in note
        assert "__FULL_CONTENT__" in note
        assert "short content" in note
        assert "truncated" not in note

    def test_long_content_gets_truncated(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        note = tool._wrap_dom_note("Some label", "x" * (DOM_DIFF_MAX_CHARS + 500))
        assert "truncated" in note
        # the actual embedded content is capped, even if the note has a bit more around it
        full_content_section = note.split("__FULL_CONTENT__\n", 1)[1]
        assert len(full_content_section) <= DOM_DIFF_MAX_CHARS + len("\n... (truncated)")


class TestAttachDomContext:
    """BrowserTool._attach_dom_context - the automatic DOM-push mechanism:
    navigate gets a full tree, other DOM-mutating actions get a diff against
    the last-seen tree, and non-mutating/read-only actions get neither. This
    is what lets the model see what changed without remembering to call
    read_page after every action - see loop.py's DOM-first prompt guidance."""

    @pytest.mark.asyncio
    async def test_navigate_attaches_full_tree_and_sets_snapshot(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._current_dom_text = AsyncMock(return_value="- link \"Home\" [ref=ref_1]")

        result = await tool._attach_dom_context(
            "navigate", ToolResult(output="Screenshot saved as x.png\n")
        )

        assert result.output.startswith("Screenshot saved as x.png")
        assert "__PAGE_EXTRACTED__" in result.output
        assert "page just loaded" in result.output
        assert 'link "Home" [ref=ref_1]' in result.output
        assert tool._last_dom_snapshot == "- link \"Home\" [ref=ref_1]"

    @pytest.mark.asyncio
    async def test_mutating_action_attaches_diff_against_last_snapshot(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._last_dom_snapshot = "- link \"A\" [ref=ref_1]"
        tool._current_dom_text = AsyncMock(
            return_value="- link \"A\" [ref=ref_1]\n- dialog \"Confirm\" [ref=ref_2]"
        )

        result = await tool._attach_dom_context(
            "left_click", ToolResult(output="Clicked element with ref: ref_1")
        )

        assert result.output.startswith("Clicked element with ref: ref_1")
        assert "__PAGE_EXTRACTED__" in result.output
        assert "DOM changes since your last action" in result.output
        assert 'dialog "Confirm" [ref=ref_2]' in result.output
        assert tool._last_dom_snapshot == "- link \"A\" [ref=ref_1]\n- dialog \"Confirm\" [ref=ref_2]"

    @pytest.mark.asyncio
    async def test_mutating_action_with_no_change_appends_nothing(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._last_dom_snapshot = "- link \"A\" [ref=ref_1]"
        tool._current_dom_text = AsyncMock(return_value="- link \"A\" [ref=ref_1]")

        original = ToolResult(output="Typed: hello")
        result = await tool._attach_dom_context("type", original)

        assert result.output == "Typed: hello"
        assert "__PAGE_EXTRACTED__" not in result.output

    @pytest.mark.asyncio
    async def test_mutating_action_with_no_prior_baseline_establishes_one(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        assert tool._last_dom_snapshot is None
        tool._current_dom_text = AsyncMock(return_value="- link \"A\" [ref=ref_1]")

        original = ToolResult(output="Scrolled down by 3 units")
        result = await tool._attach_dom_context("scroll", original)

        # nothing to diff against yet, so no note appended this time...
        assert result.output == "Scrolled down by 3 units"
        # ...but a baseline is now recorded for next time
        assert tool._last_dom_snapshot == "- link \"A\" [ref=ref_1]"

    @pytest.mark.asyncio
    async def test_read_page_syncs_snapshot_without_altering_result(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._current_dom_text = AsyncMock(return_value="- link \"A\" [ref=ref_1]")

        original = ToolResult(output="__PAGE_EXTRACTED__\nsummary\n__FULL_CONTENT__\ntree")
        result = await tool._attach_dom_context("read_page", original)

        assert result is original
        assert tool._last_dom_snapshot == "- link \"A\" [ref=ref_1]"

    @pytest.mark.asyncio
    async def test_non_mutating_action_is_left_untouched(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._current_dom_text = AsyncMock(return_value="- link \"A\" [ref=ref_1]")

        original = ToolResult(output="Screenshot saved as x.png\n")
        result = await tool._attach_dom_context("screenshot", original)

        assert result is original
        assert tool._last_dom_snapshot is None
        tool._current_dom_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_computing_dom_context_returns_original_result(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._last_dom_snapshot = "- link \"A\" [ref=ref_1]"
        tool._current_dom_text = AsyncMock(side_effect=RuntimeError("boom"))

        original = ToolResult(output="Clicked element with ref: ref_1")
        result = await tool._attach_dom_context("left_click", original)

        assert result is original

    @pytest.mark.asyncio
    async def test_no_page_returns_original_result(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        assert tool._page is None
        original = ToolResult(output="Clicked element with ref: ref_1")
        result = await tool._attach_dom_context("left_click", original)
        assert result is original

    @pytest.mark.asyncio
    async def test_error_result_is_left_untouched(self, tmp_path):
        tool = BrowserTool(run_dir=tmp_path)
        tool._page = MagicMock()
        tool._current_dom_text = AsyncMock(return_value="- link \"A\" [ref=ref_1]")

        original = ToolResult(error="something went wrong")
        result = await tool._attach_dom_context("left_click", original)

        assert result is original
        tool._current_dom_text.assert_not_called()

    def test_dom_mutating_actions_excludes_read_only_actions(self):
        # Sanity check the action classification itself - these should never
        # trigger a diff computation (screenshot/wait/find/outline/zoom/
        # get_page_text/read_page/navigate are all handled separately or are
        # read-only).
        assert "screenshot" not in DOM_MUTATING_ACTIONS
        assert "navigate" not in DOM_MUTATING_ACTIONS
        assert "read_page" not in DOM_MUTATING_ACTIONS
        assert "get_page_text" not in DOM_MUTATING_ACTIONS
        assert "wait" not in DOM_MUTATING_ACTIONS
        assert "find" not in DOM_MUTATING_ACTIONS
        assert "outline" not in DOM_MUTATING_ACTIONS
        assert "left_click" in DOM_MUTATING_ACTIONS
        assert "execute_js" in DOM_MUTATING_ACTIONS


class TestCallAttachesDomContext:
    """End-to-end through the public __call__ entry point (the actual path
    the MCP tool layer invokes - see agent_sdk_bridge.build_browser_tool_fn),
    confirming _dispatch and _attach_dom_context are actually wired together."""

    @pytest.mark.asyncio
    async def test_call_appends_dom_context_after_dispatch(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        tool._initialized = True  # skip _ensure_browser's real launch path
        tool._last_dom_snapshot = '- link "Old" [ref=ref_0]'  # give the diff something to compare against

        async def evaluate(expr, *args):
            # Order matters: browser_dom_script.js's own content also
            # contains "getBoundingClientRect", so the DOM-tree-wrapper
            # check must come first or it'd never be reached.
            if "__generateAccessibilityTree" in expr:
                return {"pageContent": '- link "A" [ref=ref_1]'}
            if "getBoundingClientRect" in expr:
                return {"success": True, "coordinates": [5, 5]}
            if "scrollHeight" in expr:
                return 100
            return "complete"

        page.evaluate = AsyncMock(side_effect=evaluate)
        page.mouse = MagicMock()
        page.mouse.move = AsyncMock()
        page.mouse.click = AsyncMock()

        result = await tool(action="left_click", ref="ref_1")

        assert result.output.startswith("Clicked element with ref: ref_1")
        assert "__PAGE_EXTRACTED__" in result.output

    @pytest.mark.asyncio
    async def test_call_propagates_dispatch_errors_without_computing_dom_context(self, tmp_path):
        tool, page = make_tool_with_mock_page(tmp_path)
        tool._initialized = True  # skip _ensure_browser's real launch path
        tool._current_dom_text = AsyncMock(side_effect=AssertionError("should not be called"))

        with pytest.raises(ToolError, match="Element reference is required"):
            await tool(action="scroll_to")  # missing required ref
