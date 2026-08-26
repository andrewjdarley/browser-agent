"""A recipe runner over the browser tool's own action vocabulary: replays a
fixed sequence of steps once per item, with no LLM invoked per item.
Whether a script fetches (raw HTTP, fast) or navigates (a real rendered
page) falls out of which steps you put in it, not a separate flag or tool.

Step vocabulary is the "structured extraction/capture" subset of the
browser tool's own actions - navigate, screenshot, get_page_text,
execute_js, wait, scroll_to, network_list_types, network_list,
network_inspect. Deliberately excludes click/type/drag/hover/form_input/
key - items needing real per-item interaction/judgment belong in
dispatch_subagents instead.

scroll_to takes literal `text` (a substring match via
page.get_by_text(text, exact=False), same as sub_browser_queue.py's own
scroll_to), not a browser-tool ref - a ref from read_page/find is a WeakRef
into one page's JS heap, meaningless on a different item's page, and
there's no LLM in this loop to call read_page/find per item anyway.

network_list_types/network_list/network_inspect reuse the module-level
capture_network_response/*_text functions from browser.py: each full-mode
item registers its own page.on("response", ...) into a fresh per-item log,
only when the script actually uses one of these steps. Not available in
fast/fetch mode - there's no live page/response stream there.

Reuses AdaptiveRateLimiter and the raw-fetch extraction wrapper from
batch_extract.py directly - a script with no `navigate` step *is*
batch_extract's exact mechanism, reached through this tool's script shape.
"""

import asyncio
import json
from collections import deque
from itertools import count
from typing import Any, Optional

from claude_agent_sdk import tool as sdk_tool

from ..agent_sdk_bridge import tool_result_to_sdk_content
from .base import ToolError, ToolResult
from .batch_extract import (
    _EXTRACT_WRAPPER_JS,
    RATE_LIMIT_STATUS,
    REQUEST_TIMEOUT_S,
    TRANSIENT_STATUS,
    AdaptiveRateLimiter,
    _parse_retry_after,
)
from .browser import (
    BROWSER_TOOL_UTILS_DIR,
    MAX_NETWORK_LOG_ENTRIES,
    SCREENSHOT_SETTLE_DELAY_S,
    TEXT_READ_SETTLE_DELAY_S,
    BrowserTool,
    NetworkLogEntry,
    capture_network_response,
    capture_screenshot,
    network_inspect_text,
    network_list_text,
    network_list_types_text,
    wait_for_page_ready,
)
from .file_output import FileOutputTool

# Real pages are far heavier than batch_extract's bare HTTP requests (a full
# render + JS execution each), so both the default and the ceiling are much
# lower than batch_extract's - closer to dispatch_subagents' concurrency,
# which runs into the same per-page resource cost.
DEFAULT_CONCURRENCY = 5
MAX_CONCURRENCY_HARD_CAP = 10
MAX_ATTEMPTS_PER_ITEM = 3

STEP_ACTIONS = frozenset(
    {
        "navigate",
        "screenshot",
        "get_page_text",
        "execute_js",
        "wait",
        "scroll_to",
        "network_list_types",
        "network_list",
        "network_inspect",
    }
)

NETWORK_ACTIONS = frozenset({"network_list_types", "network_list", "network_inspect"})

_TEXT_SCRIPT = (BROWSER_TOOL_UTILS_DIR / "browser_text_script.js").read_text()


class _Retryable(Exception):
    """Raised internally to signal "back off and try this item again" -
    caught in _run_one_item's retry loop, never surfaced to the model."""

    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after


RUN_SCRIPT_INPUT_SCHEMA: dict = {
    "properties": {
        "items": {
            "description": (
                "One entry per iteration. Substituted for {{item}} anywhere it appears in a "
                "step's parameter values - typically URLs (when the script starts with "
                "`navigate`, or when it doesn't and each item IS the URL to fetch), but can be "
                "any string your script's steps use."
            ),
            "type": "array",
            "items": {"type": "string"},
        },
        "script": {
            "description": (
                "The fixed sequence of steps to run once per item, in order - same shape as "
                "the browser tool's own actions. Each step is an object with `action` (one of: "
                "navigate, screenshot, get_page_text, execute_js, wait, scroll_to, "
                "network_list_types, network_list, network_inspect) plus that action's usual "
                "parameter (text for navigate's URL, execute_js's code, scroll_to's CSS/"
                "Playwright-locator selector, network_list's filter, or network_inspect's entry "
                "id; full_page for screenshot; duration for wait). Use {{item}} in any string "
                "value to substitute the current item. If the FIRST step is `navigate`, each "
                "item gets its own real rendered page (JS runs, client-rendered content is "
                "visible) - required for screenshot, get_page_text, scroll_to, network_* steps, "
                "or any JS that depends on client-side rendering. scroll_to's `text` is literal "
                "text that appears on the page (a substring match, not a description - same "
                "semantics as queue_screenshots' scroll_to) - NOT a browser-tool ref (there's no "
                "read_page/find step here to produce one; the text match re-resolves "
                "independently per item's own page instead). network_list_types/network_list/"
                "network_inspect read that item's own "
                "captured response traffic since its page was created, same semantics as the "
                "browser tool's own network actions. If the script does NOT start with "
                "`navigate`, every step must be execute_js, and each item is fetched via raw "
                "HTTP instead (fast, but only sees server-rendered HTML - this is exactly "
                "batch_extract's mechanism, reached through this tool's script shape; scroll_to "
                "and network_* aren't available in this mode - no live page/response stream to "
                "act on). NOT supported: click/type/drag/hover/form_input/key - items needing "
                "real interaction or judgment belong in dispatch_subagents instead, not here."
            ),
            "type": "array",
            "items": {"type": "object"},
        },
        "save_as": {
            "description": (
                "Optional filename to save the full JSON results to (e.g. 'results.json') "
                "instead of returning them all in the tool response - use this for large "
                "batches, or any script step that takes a screenshot (the results already "
                "reference saved screenshot filenames, not embedded images). Name this filename "
                "in your final message so the user can find it."
            ),
            "type": "string",
        },
        "concurrency": {
            "description": (
                f"Max simultaneous items in flight (default {DEFAULT_CONCURRENCY}, hard-capped "
                f"at {MAX_CONCURRENCY_HARD_CAP} - real pages are heavier than batch_extract's "
                "bare requests, so this is deliberately lower). Automatically backs off if a "
                "navigate/fetch step gets rate-limited, so this is a ceiling, not a guarantee."
            ),
            "type": "integer",
        },
    },
    "required": ["items", "script"],
    "type": "object",
}

RUN_SCRIPT_DESCRIPTION = (
    "Run a fixed sequence of browser-tool-shaped steps once per item, without spinning up an "
    "LLM per item - write the recipe once (navigate/screenshot/get_page_text/execute_js/wait/"
    "scroll_to/network_list_types/network_list/network_inspect, same action names as the "
    "browser tool), it replays across all items concurrently and hands back one aggregated "
    "result. Subsumes batch_extract's shape too: a script with no navigate step fetches each "
    "item as a URL via raw HTTP instead of rendering it, same as batch_extract. Use this "
    "whenever a task involves several similar items and the same fixed steps would otherwise "
    "mean many manual tool calls - not for items that need real per-item interaction or "
    "judgment, that's still dispatch_subagents."
)


class ScriptRunnerTool:
    """Runs `script` once per item in `items`, over a pool of fresh Pages
    from the coordinator's own browser context (inherits cookies/login) when
    the script navigates, or via raw HTTP fetch (reusing batch_extract's
    exact mechanism) when it doesn't."""

    name = "run_script"

    def __init__(self, *, browser_tool: BrowserTool, file_output_tool: FileOutputTool):
        self.browser_tool = browser_tool
        self.file_output_tool = file_output_tool

    async def __call__(
        self,
        *,
        items: list[str],
        script: list[dict],
        save_as: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> ToolResult:
        if not items:
            return ToolResult(error="items must be a non-empty list")
        if not script:
            return ToolResult(error="script must be a non-empty list of steps")

        invalid_actions = sorted({s.get("action") for s in script} - STEP_ACTIONS)
        if invalid_actions:
            return ToolResult(
                error=(
                    f"Unsupported script action(s): {invalid_actions}. run_script only supports "
                    f"{sorted(STEP_ACTIONS)} - interaction actions (click/type/drag/hover/"
                    "form_input/key) aren't available here; use dispatch_subagents for items "
                    "that need real interaction or judgment."
                )
            )

        fast = script[0].get("action") != "navigate"
        if fast and any(s.get("action") != "execute_js" for s in script):
            return ToolResult(
                error=(
                    "A script that doesn't start with `navigate` runs against a raw HTTP fetch "
                    "of each item (treated as a URL) instead of a real page - in that mode every "
                    "step must be execute_js (screenshot/get_page_text/wait need a real rendered "
                    "page). Add a `navigate` step first if you need those."
                )
            )

        bounded_concurrency = max(1, min(concurrency or DEFAULT_CONCURRENCY, MAX_CONCURRENCY_HARD_CAP))
        await self.browser_tool._ensure_browser()

        semaphore = asyncio.Semaphore(bounded_concurrency)
        rate_limiter = AdaptiveRateLimiter()
        needs_network_capture = any(s.get("action") in NETWORK_ACTIONS for s in script)

        raw_results = await asyncio.gather(
            *(
                self._run_one_item(item, script, fast, needs_network_capture, semaphore, rate_limiter)
                for item in items
            ),
            return_exceptions=True,
        )

        results: list[dict[str, Any]] = []
        succeeded = 0
        for item, raw in zip(items, raw_results):
            if isinstance(raw, BaseException):
                results.append({"item": item, "error": str(raw)})
            else:
                results.append(raw)
                if "error" not in raw:
                    succeeded += 1
        failed = len(items) - succeeded

        if save_as:
            save_result = await self.file_output_tool(
                filename=save_as, content=json.dumps(results, indent=2)
            )
            output = f"run_script: {succeeded}/{len(items)} succeeded. {save_result.output}"
            errors = [r for r in results if "error" in r]
            if errors:
                output += f"\n{len(errors)} failed - first few: {json.dumps(errors[:5])}"
            return ToolResult(output=output)

        return ToolResult(
            output=json.dumps(
                {"results": results, "succeeded": succeeded, "failed": failed}, indent=2
            )
        )

    async def _run_one_item(
        self,
        item: str,
        script: list[dict],
        fast: bool,
        needs_network_capture: bool,
        semaphore: asyncio.Semaphore,
        rate_limiter: AdaptiveRateLimiter,
    ) -> dict[str, Any]:
        async with semaphore:
            last_error = "unknown error"
            for _attempt in range(MAX_ATTEMPTS_PER_ITEM):
                await rate_limiter.wait()
                try:
                    if fast:
                        result = await self._run_fast_item(item, script)
                    else:
                        result = await self._run_full_item(item, script, needs_network_capture)
                except _Retryable as e:
                    await rate_limiter.on_rate_limited(e.retry_after)
                    last_error = "rate limited or transient error"
                    continue
                await rate_limiter.on_success()
                return result

            return {
                "item": item,
                "error": f"gave up after {MAX_ATTEMPTS_PER_ITEM} attempts: {last_error}",
            }

    async def _run_fast_item(self, item: str, script: list[dict]) -> dict[str, Any]:
        """No navigate step: item IS the URL, fetched via raw HTTP - the
        exact mechanism batch_extract uses, reached through a script shape.
        Every step is execute_js (validated in __call__), run against the
        same fetched HTML each time - independent extractions from one
        snapshot, not a live page."""
        try:
            response = await asyncio.wait_for(
                self.browser_tool._context.request.get(item), timeout=REQUEST_TIMEOUT_S
            )
        except Exception as e:
            raise _Retryable() from e

        if response.status in RATE_LIMIT_STATUS:
            raise _Retryable(_parse_retry_after(response.headers))
        if response.status in TRANSIENT_STATUS:
            raise _Retryable()
        if response.status >= 400:
            return {"item": item, "error": f"HTTP {response.status}"}

        html = await response.text()
        step_results = []
        for step in script:
            js = _substitute(step.get("text", ""), item)
            try:
                data = await self.browser_tool._page.evaluate(_EXTRACT_WRAPPER_JS, [html, js])
            except Exception as e:
                return {
                    "item": item,
                    "steps": step_results,
                    "error": f"step {len(step_results) + 1} (execute_js) failed: {e}",
                }
            step_results.append({"action": "execute_js", "output": data})
        return {"item": item, "steps": step_results}

    async def _run_full_item(
        self, item: str, script: list[dict], needs_network_capture: bool
    ) -> dict[str, Any]:
        """First step is navigate: a real, JS-executing page, fresh per item
        (never the coordinator's own self._page - concurrent items must not
        share mutable page state, and the VNC view stays undisturbed)."""
        page = await self.browser_tool._context.new_page()
        # A fresh log/counter per item, only wired up when the script
        # actually has a network_* step - this item's page is the only
        # thing that ever writes to or reads from it, so there's no
        # cross-item ambiguity the way a browser-tool ref would have.
        network_log: "deque[NetworkLogEntry]" = deque(maxlen=MAX_NETWORK_LOG_ENTRIES)
        if needs_network_capture:
            counter = count(1)
            page.on("response", lambda response: capture_network_response(response, network_log, counter))
        try:
            step_results = []
            for step in script:
                action = step.get("action")
                params = {k: _substitute(v, item) for k, v in step.items() if k != "action"}
                try:
                    output = await self._run_full_step(page, action, params, network_log)
                except _Retryable:
                    raise
                except Exception as e:
                    return {
                        "item": item,
                        "steps": step_results,
                        "error": f"step {len(step_results) + 1} ({action}) failed: {e}",
                    }
                step_results.append({"action": action, "output": output})
            return {"item": item, "steps": step_results}
        finally:
            await page.close()

    async def _run_full_step(self, page, action: str, params: dict, network_log: "deque[NetworkLogEntry]") -> str:
        if action == "navigate":
            url = params.get("text")
            if not url:
                raise ToolError("navigate step requires 'text' (the URL)")
            if not url.startswith(("http://", "https://", "about:")):
                url = f"https://{url}"
            response = await page.goto(url, wait_until="domcontentloaded")
            if response is not None and response.status in RATE_LIMIT_STATUS:
                raise _Retryable(_parse_retry_after(response.headers))
            await wait_for_page_ready(page, settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            status = response.status if response is not None else "?"
            return f"Navigated to {url} (HTTP {status})"

        if action == "screenshot":
            full_page = bool(params.get("full_page", False))
            await wait_for_page_ready(page, settle_delay_s=SCREENSHOT_SETTLE_DELAY_S)
            result = await capture_screenshot(
                page,
                self.browser_tool.run_dir,
                width=self.browser_tool.width,
                height=self.browser_tool.height,
                full_page=full_page,
            )
            # Deliberately dropping base64_image here, not just the text -
            # inlining an image per step per item would blow up the
            # aggregated result for anything but a tiny item count. The
            # filename (in the output text below) is the deliverable;
            # save_as is how the model hands the actual files to the user.
            return result.output.strip()

        if action == "get_page_text":
            await wait_for_page_ready(page, settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            result = await page.evaluate(f"({_TEXT_SCRIPT})()")
            if isinstance(result, dict):
                return str(result.get("text", ""))
            return str(result)

        if action == "execute_js":
            code = params.get("text")
            if not code:
                raise ToolError("execute_js step requires 'text' (the JS code)")
            await wait_for_page_ready(page, settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            result = await page.evaluate(code)
            if result is None:
                return "undefined"
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)

        if action == "wait":
            duration = float(params.get("duration") or 1.0)
            await asyncio.sleep(duration)
            return f"waited {duration}s"

        if action == "scroll_to":
            target = params.get("text")
            if not target:
                raise ToolError(
                    "scroll_to step requires 'text' (literal text that appears on the page - "
                    "a substring match, not a description; same semantics as queue_screenshots' "
                    "scroll_to target)"
                )
            locator = page.get_by_text(target, exact=False).first
            await locator.scroll_into_view_if_needed(timeout=10000)
            await asyncio.sleep(0.3)
            return f"Scrolled to element matching {target!r}"

        if action == "network_list_types":
            return network_list_types_text(network_log)

        if action == "network_list":
            return network_list_text(network_log, params.get("text"))

        if action == "network_inspect":
            entry_id = params.get("text")
            if not entry_id:
                raise ToolError(
                    "network_inspect step requires 'text' (the entry id from a preceding "
                    "network_list step's output)"
                )
            return network_inspect_text(network_log, entry_id)

        raise ToolError(f"Unsupported script step action: {action!r}")


def _substitute(value: Any, item: str) -> Any:
    """Replace {{item}} with the current item in a string param value;
    non-string values pass through unchanged."""
    if isinstance(value, str):
        return value.replace("{{item}}", item)
    return value


def build_run_script_tool_fn(script_runner_tool: ScriptRunnerTool):
    """Wrap a ScriptRunnerTool instance as an SDK-registrable @tool function,
    for the coordinator's own mcp_servers registration."""

    @sdk_tool("run_script", RUN_SCRIPT_DESCRIPTION, RUN_SCRIPT_INPUT_SCHEMA)
    async def run_script_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await script_runner_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return run_script_fn
