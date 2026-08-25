"""Real (non-mocked) regression tests for browser_dom_script.js's name
extraction. This file is exercised by read_page/find (via
window.__generateAccessibilityTree), and its correctness can't be caught by
the rest of the suite's mocked-Page tests, since those never run the actual
script against real DOM content."""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

DOM_SCRIPT = (
    Path(__file__).parent.parent
    / "browser_use_demo"
    / "browser_tool_utils"
    / "browser_dom_script.js"
).read_text()


async def _tree_for_html(html: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        await page.set_content(html)
        await page.add_script_tag(content=DOM_SCRIPT)
        tree = await page.evaluate("window.__generateAccessibilityTree('all')")
        await browser.close()
    return tree["pageContent"]


class TestNestedSpanTextExtraction:
    """Regression coverage for a real bug found via a live run: a link's
    visible text wrapped in nested <span>s (icon+label buttons, styled
    bracket/number badges, Wikipedia's citation links, etc. - a very common
    real-world pattern) used to come back completely unnamed in the
    accessibility tree, because name extraction only looked at direct
    text-node children of the <a>. That made "find item N" tasks impossible
    to do correctly through the DOM: read_page returned hundreds of
    nameless citation links, indistinguishable from each other."""

    @pytest.mark.asyncio
    async def test_link_text_wrapped_in_nested_spans_is_not_blank(self):
        html = (
            '<a href="#x"><span class="wrap"><span class="bracket">[</span>'
            "275<span class=\"bracket\">]</span></span></a>"
        )
        pageContent = await _tree_for_html(html)
        assert '"[275]"' in pageContent

    @pytest.mark.asyncio
    async def test_button_text_wrapped_in_nested_span_is_not_blank(self):
        html = '<button><span>Submit</span></button>'
        pageContent = await _tree_for_html(html)
        assert '"Submit"' in pageContent

    @pytest.mark.asyncio
    async def test_plain_unwrapped_link_text_still_works(self):
        """Not a regression - direct text-node content should still be
        picked up exactly as before."""
        html = '<a href="#x">Plain text</a>'
        pageContent = await _tree_for_html(html)
        assert '"Plain text"' in pageContent

    @pytest.mark.asyncio
    async def test_aria_label_still_takes_priority_over_visible_text(self):
        """Unchanged behavior: an explicit aria-label is still respected
        when present - this fix only changes what happens when there's no
        aria-label and the visible text is nested."""
        html = '<a href="#x" aria-label="Close dialog"><span>X</span></a>'
        pageContent = await _tree_for_html(html)
        assert '"Close dialog"' in pageContent
