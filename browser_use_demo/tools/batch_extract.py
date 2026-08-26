"""Scripted bulk data extraction across many URLs in one tool call.

Unlike dispatch_subagents (independent ClaudeSDKClient/LLM sessions per item -
real reasoning per item, but expensive and capped low), this tool does no
LLM work per item at all: it fetches each URL concurrently via the browser
context's own APIRequestContext (Playwright's `context.request`), which makes
the HTTP call directly rather than through the page's `fetch()` - so it isn't
subject to the browser's same-origin/CORS policy (that only restricts page
JS), and it automatically reuses the context's cookie jar, inheriting
whatever session the visible page is already logged into. Extraction runs via
the one shared BrowserTool page (a detached DOMParser document - it never
navigates or touches the visible viewport), so the VNC view is undisturbed.

Only sees server-rendered HTML - content that only appears after client-side
JS runs on a real navigation isn't visible to it. That's a deliberate scope
limit, not an oversight: for that case, dispatch_subagents or manual
navigation is still the right tool.
"""

import asyncio
import json
from typing import Any, Optional

from claude_agent_sdk import tool as sdk_tool

from ..agent_sdk_bridge import tool_result_to_sdk_content
from .base import ToolError, ToolResult
from .browser import BrowserTool
from .file_output import FileOutputTool

DEFAULT_CONCURRENCY = 10
MAX_CONCURRENCY_HARD_CAP = 30  # higher than dispatch_subagents' - these are cheap HTTP calls, not LLM sessions
MAX_ATTEMPTS_PER_URL = 4
REQUEST_TIMEOUT_S = 20.0

RATE_LIMIT_STATUS = {429, 503}
TRANSIENT_STATUS = {500, 502, 504}

# Runs in the page context against a detached (DOMParser) document, so it
# never touches the real page's location or viewport.
_EXTRACT_WRAPPER_JS = """([html, extractJs]) => {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const fn = new Function('doc', `return (${extractJs})(doc);`);
    return fn(doc);
}"""

BATCH_EXTRACT_INPUT_SCHEMA: dict = {
    "properties": {
        "urls": {
            "description": (
                "URLs to fetch and extract from. Each is fetched independently "
                "via an HTTP request using your current browser session's "
                "cookies - no navigation happens, so this works for any origin "
                "the current session is authenticated with, but only sees "
                "server-rendered HTML (no client-side JS execution)."
            ),
            "type": "array",
            "items": {"type": "string"},
        },
        "extract_js": {
            "description": (
                "A complete JS arrow function, e.g. \"(doc) => ({ title: "
                "doc.title, price: doc.querySelector('.price')?.textContent "
                "})\". Runs once per URL against a DOMParser-parsed version of "
                "that page's HTML. Must return a JSON-serializable value. Test "
                "it against one page with execute_js before running it across "
                "the whole batch."
            ),
            "type": "string",
        },
        "save_as": {
            "description": (
                "Optional filename to save the full JSON results to (e.g. "
                "'results.json') instead of returning them all in the tool "
                "response - use this for large batches so you don't have to "
                "hold hundreds of rows of data in context. Name this filename "
                "in your final message so the user can find it."
            ),
            "type": "string",
        },
        "concurrency": {
            "description": (
                f"Max simultaneous requests (default {DEFAULT_CONCURRENCY}, "
                f"hard-capped at {MAX_CONCURRENCY_HARD_CAP}). Automatically "
                "backs off if the target starts rate-limiting you, so this is "
                "a ceiling, not a guarantee."
            ),
            "type": "integer",
        },
    },
    "required": ["urls", "extract_js"],
    "type": "object",
}

BATCH_EXTRACT_DESCRIPTION = (
    "Extract the same structured data from many URLs at once, without "
    "visiting them one at a time. Fetches all URLs concurrently (reusing "
    "your current session's cookies, so authenticated/cross-origin URLs "
    "work) and runs your extract_js against each page's raw HTML. Much "
    "cheaper and faster than dispatch_subagents or manual navigation for "
    "pure data extraction at scale - reach for this whenever a task involves "
    "several similar items and the data you need is present in the page's "
    "initial HTML. Handles rate limiting automatically by backing off and "
    "retrying."
)


def _parse_retry_after(headers: dict) -> Optional[float]:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form, not the common numeric-seconds form - fall back to exponential backoff


class AdaptiveRateLimiter:
    """Shared across every URL in one batch_extract call. Starts at zero
    delay and backs off (AIMD: additive/exponential increase, multiplicative
    decrease) only when the target actually signals throttling, so the batch
    discovers a safe rate empirically instead of us guessing one up front."""

    def __init__(self, *, max_delay: float = 30.0, decrease_after: int = 10):
        self._delay = 0.0
        self._max_delay = max_delay
        self._decrease_after = decrease_after
        self._consecutive_successes = 0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._delay
        if delay > 0:
            await asyncio.sleep(delay)

    async def on_rate_limited(self, retry_after: Optional[float]) -> None:
        async with self._lock:
            self._consecutive_successes = 0
            if retry_after is not None:
                self._delay = min(max(self._delay, retry_after), self._max_delay)
            else:
                self._delay = min(max(self._delay * 2, 0.5), self._max_delay)

    async def on_success(self) -> None:
        async with self._lock:
            if self._delay <= 0:
                return
            self._consecutive_successes += 1
            if self._consecutive_successes >= self._decrease_after:
                self._delay = self._delay / 2 if self._delay > 0.05 else 0.0
                self._consecutive_successes = 0


class BatchExtractTool:
    """Fetches N URLs concurrently (via the shared BrowserTool's request
    context) and runs extract_js against each one's HTML."""

    name = "batch_extract"

    def __init__(self, *, browser_tool: BrowserTool, file_output_tool: FileOutputTool):
        self.browser_tool = browser_tool
        self.file_output_tool = file_output_tool

    async def __call__(
        self,
        *,
        urls: list[str],
        extract_js: str,
        save_as: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> ToolResult:
        if not urls:
            return ToolResult(error="urls must be a non-empty list")

        bounded_concurrency = max(1, min(concurrency or DEFAULT_CONCURRENCY, MAX_CONCURRENCY_HARD_CAP))

        # Ensures self.browser_tool._context/_page exist without performing a
        # visible action - same init path BrowserTool.__call__ itself uses.
        await self.browser_tool._ensure_browser()

        semaphore = asyncio.Semaphore(bounded_concurrency)
        rate_limiter = AdaptiveRateLimiter()

        raw_results = await asyncio.gather(
            *(self._fetch_one(url, extract_js, semaphore, rate_limiter) for url in urls),
            return_exceptions=True,
        )

        results: list[dict[str, Any]] = []
        succeeded = 0
        for url, raw in zip(urls, raw_results):
            if isinstance(raw, BaseException):
                results.append({"url": url, "error": str(raw)})
            else:
                results.append(raw)
                if "error" not in raw:
                    succeeded += 1
        failed = len(urls) - succeeded

        if save_as:
            save_result = await self.file_output_tool(
                filename=save_as, content=json.dumps(results, indent=2)
            )
            output = f"batch_extract: {succeeded}/{len(urls)} succeeded. {save_result.output}"
            errors = [r for r in results if "error" in r]
            if errors:
                output += f"\n{len(errors)} failed - first few: {json.dumps(errors[:5])}"
            return ToolResult(output=output)

        return ToolResult(
            output=json.dumps(
                {"results": results, "succeeded": succeeded, "failed": failed}, indent=2
            )
        )

    async def _fetch_one(
        self,
        url: str,
        extract_js: str,
        semaphore: asyncio.Semaphore,
        rate_limiter: AdaptiveRateLimiter,
    ) -> dict[str, Any]:
        async with semaphore:
            last_error = "unknown error"
            for _attempt in range(MAX_ATTEMPTS_PER_URL):
                await rate_limiter.wait()
                try:
                    response = await asyncio.wait_for(
                        self.browser_tool._context.request.get(url),
                        timeout=REQUEST_TIMEOUT_S,
                    )
                except Exception as e:
                    last_error = f"request failed: {e}"
                    continue

                if response.status in RATE_LIMIT_STATUS:
                    await rate_limiter.on_rate_limited(_parse_retry_after(response.headers))
                    last_error = f"rate limited (HTTP {response.status})"
                    continue

                if response.status in TRANSIENT_STATUS:
                    last_error = f"transient error (HTTP {response.status})"
                    continue

                if response.status >= 400:
                    return {"url": url, "error": f"HTTP {response.status}"}

                await rate_limiter.on_success()

                try:
                    html = await response.text()
                    data = await self.browser_tool._page.evaluate(
                        _EXTRACT_WRAPPER_JS, [html, extract_js]
                    )
                except Exception as e:
                    return {"url": url, "error": f"extraction failed: {e}"}

                return {"url": url, "data": data}

            return {
                "url": url,
                "error": f"gave up after {MAX_ATTEMPTS_PER_URL} attempts: {last_error}",
            }


def build_batch_extract_tool_fn(batch_extract_tool: BatchExtractTool):
    """Wrap a BatchExtractTool instance as an SDK-registrable @tool function,
    for the coordinator's own mcp_servers registration."""

    @sdk_tool("batch_extract", BATCH_EXTRACT_DESCRIPTION, BATCH_EXTRACT_INPUT_SCHEMA)
    async def batch_extract_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await batch_extract_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return batch_extract_fn
