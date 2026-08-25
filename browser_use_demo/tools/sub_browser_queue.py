"""A background screenshot-capture queue over the coordinator's own browser
context - "sub-browser sessions" in the sense of independent Pages (own
navigation/viewport, doesn't touch the coordinator's self._page), not
independent Browser processes. Deliberately reuses browser_tool._context
(same pattern batch_extract.py/script_runner.py already established) rather
than launching a separate browser: cheaper, and inherits whatever
cookies/login the visible session already has.

Shape: the agent drops instruction sequences ("visit X, scroll to Y,
screenshot" / "visit Z, full-page screenshot") into the queue via
queue_screenshots and gets item ids back immediately - it does not wait for
them. Items are picked up and run concurrently, up to max_fanout at a time,
each against its own fresh Page. Results (screenshots + a log) collect as
items finish, queryable via queue_status, and also rendered live in the
Streamlit sidebar panel (see streamlit.py's render_sub_browser_panel).

Concurrency model: everything runs on ONE asyncio event loop - the same one
Streamlit's session already drives (st.session_state.event_loop) - rather
than a separate thread with its own browser. A second thread would need its
own separate Playwright browser (Playwright's async Page/BrowserContext
objects are bound to the event loop that created them - they can't safely be
driven from a different thread), which would mean not sharing the
coordinator's session and paying for a second browser process. Instead,
_kick_off_more schedules item coroutines onto the current loop via
asyncio.create_task - real concurrency, since Streamlit's own agent-turn
processing already keeps that loop spinning through many await points - and
the sidebar panel's periodic st.fragment tick (see streamlit.py) calls
queue.pump() on every refresh so scheduled tasks keep advancing even when no
chat turn is active, not just while the agent happens to be mid-turn.
"""

import asyncio
import re
import uuid
from typing import Any, Optional

from claude_agent_sdk import tool as sdk_tool

from ..agent_sdk_bridge import tool_result_to_sdk_content
from .base import ToolResult
from .browser import BrowserTool, capture_screenshot, wait_for_page_ready

STEP_ACTIONS = frozenset({"navigate", "scroll_to", "screenshot"})

MAX_COMPLETED_HISTORY = 200
MAX_STATUS_RESULTS = 30

_SCREENSHOT_FILENAME_RE = re.compile(r"Screenshot saved as (\S+)")


class SubBrowserQueue:
    """Shared state + execution logic for the screenshot queue - one
    instance per Streamlit session (see streamlit.py's setup_state), kept
    across client reconnects the same way GuardrailPolicy is, so pending/
    completed items and the fanout setting survive a model/max_turns change.
    """

    def __init__(self, *, browser_tool: BrowserTool, run_dir, max_fanout: int = 8):
        self.browser_tool = browser_tool
        self.run_dir = run_dir
        self.max_fanout = max_fanout
        self.paused = False
        self._pending: list[dict[str, Any]] = []
        self._in_progress: dict[str, dict[str, Any]] = {}
        self._completed: list[dict[str, Any]] = []  # most-recent-first, capped

    def add(self, items: list[list[dict[str, Any]]]) -> list[str]:
        """Enqueue items (each a list of steps) and return their ids
        immediately - does not wait for anything to run."""
        ids = []
        for steps in items:
            item_id = uuid.uuid4().hex[:8]
            self._pending.append({"id": item_id, "steps": steps})
            ids.append(item_id)
        self._kick_off_more()
        return ids

    def clear(self) -> dict[str, int]:
        """Drops pending items and completed history. Already-in-progress
        items are left to finish naturally (a Page mid-navigation can't be
        safely torn down from here) rather than force-cancelled."""
        counts = {"pending_cleared": len(self._pending), "completed_cleared": len(self._completed)}
        self._pending.clear()
        self._completed.clear()
        return counts

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._kick_off_more()

    def snapshot(self) -> dict[str, Any]:
        return {
            "paused": self.paused,
            "max_fanout": self.max_fanout,
            "pending_count": len(self._pending),
            "in_progress_count": len(self._in_progress),
            "completed": list(self._completed),
        }

    def pump(self) -> None:
        """Start newly-startable items if there's headroom. A plain sync
        call - safe from add()/resume(), which already run inside an active
        event loop (a tool call, mid agent-turn). Does NOT by itself advance
        already-running item tasks; see pump_async for that."""
        self._kick_off_more()

    async def pump_async(self, budget_s: float = 0.3) -> None:
        """Start newly-startable items, then give the event loop budget_s of
        real time to actually advance already-scheduled item tasks - a bare
        pump() alone doesn't do this, since asyncio.create_task only
        SCHEDULES work, it doesn't run it. Needed because, unlike an agent
        turn (which keeps the loop spinning through many awaits for
        seconds), nothing else drives this queue's own event loop forward
        between chat turns - see the module docstring, and
        streamlit.py's render_sub_browser_panel, which is the one caller
        that actually needs this (run via loop.run_until_complete on the
        session's shared loop, on every fragment tick)."""
        self._kick_off_more()
        await asyncio.sleep(budget_s)
        self._kick_off_more()

    def _kick_off_more(self) -> None:
        if self.paused:
            return
        while self._pending and len(self._in_progress) < self.max_fanout:
            item = self._pending.pop(0)
            self._in_progress[item["id"]] = item
            asyncio.create_task(self._run_item(item))

    async def _run_item(self, item: dict[str, Any]) -> None:
        result: dict[str, Any] = {
            "id": item["id"],
            "steps": item["steps"],
            "status": "done",
            "screenshots": [],
            "log": [],
            "error": None,
        }
        try:
            await self.browser_tool._ensure_browser()
            page = await self.browser_tool._context.new_page()
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"could not open a page: {e}"
            self._finish_item(item["id"], result)
            return

        try:
            await page.set_viewport_size({"width": self.browser_tool.width, "height": self.browser_tool.height})
            for step in item["steps"]:
                await self._run_step(page, step, result)
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["log"].append(f"error: {e}")
        finally:
            await page.close()
            self._finish_item(item["id"], result)

    def _finish_item(self, item_id: str, result: dict[str, Any]) -> None:
        self._in_progress.pop(item_id, None)
        self._completed.insert(0, result)
        del self._completed[MAX_COMPLETED_HISTORY:]
        self._kick_off_more()

    async def _run_step(self, page, step: dict[str, Any], result: dict[str, Any]) -> None:
        action = step.get("action")
        if action == "navigate":
            url = step.get("url") or ""
            if not re.match(r"^\w+://", url):
                url = f"https://{url}"
            await page.goto(url, wait_until="load", timeout=30000)
            await wait_for_page_ready(page)
            result["log"].append(f"navigate -> {url}")
        elif action == "scroll_to":
            target = step.get("target") or ""
            locator = page.get_by_text(target, exact=False).first
            await locator.scroll_into_view_if_needed(timeout=10000)
            await asyncio.sleep(0.3)
            result["log"].append(f"scroll_to -> {target!r}")
        elif action == "screenshot":
            full_page = bool(step.get("full_page", False))
            shot = await capture_screenshot(
                page,
                self.run_dir,
                width=self.browser_tool.width,
                height=self.browser_tool.height,
                full_page=full_page,
            )
            match = _SCREENSHOT_FILENAME_RE.search(shot.output or "")
            filename = match.group(1) if match else None
            if filename:
                result["screenshots"].append(filename)
            result["log"].append(f"screenshot -> {filename}" + (" (full page)" if full_page else ""))
        else:
            raise ValueError(f"unsupported step action {action!r} - must be one of {sorted(STEP_ACTIONS)}")


# --- MCP tool wiring: five focused tools sharing one SubBrowserQueue instance ---

QUEUE_SCREENSHOTS_INPUT_SCHEMA: dict = {
    "properties": {
        "items": {
            "description": (
                "One entry per queue item, each a list of steps run in order against that "
                "item's own fresh page (independent of your current page - your session's "
                "cookies/login carry over, but navigating here never moves your own page). "
                "Step `action` is one of: `navigate` (needs `url`), `scroll_to` (needs "
                "`target` - text to find on the page and scroll into view), `screenshot` "
                "(optional `full_page`, default false). Example: "
                '[{"action": "navigate", "url": "example.com"}, '
                '{"action": "scroll_to", "target": "Pricing"}, {"action": "screenshot"}]'
            ),
            "type": "array",
            "items": {"type": "array", "items": {"type": "object"}},
        },
    },
    "required": ["items"],
    "type": "object",
}

QUEUE_SCREENSHOTS_DESCRIPTION = (
    "Add one or more instruction sequences to the sub-browser screenshot queue for visual "
    "reconnaissance across many pages/sites at once - returns immediately with the queued "
    "item ids, it does NOT wait for them to finish. Items run concurrently, up to the fanout "
    "limit set in the sidebar (default 8), each against its own independent page. Results "
    "(screenshots + a log) appear live in the sidebar panel as items complete, and via "
    "queue_status. Use this instead of navigating/screenshotting many pages yourself one at "
    "a time; it's not for interaction or reasoning (clicking, forms) - just navigate, scroll "
    "to something, and capture."
)

QUEUE_STATUS_DESCRIPTION = (
    f"Check the sub-browser screenshot queue's progress: how many items are pending/running, "
    f"and the {MAX_STATUS_RESULTS} most recently completed items (status, saved screenshot "
    "filenames, and a step-by-step log; the error if one failed)."
)

QUEUE_CLEAR_DESCRIPTION = (
    "Drop all pending (not-yet-started) items and completed history from the sub-browser "
    "screenshot queue. Items already in progress are left to finish, not force-cancelled."
)

QUEUE_PAUSE_DESCRIPTION = (
    "Pause the sub-browser screenshot queue - no new pending items will start (already "
    "in-progress ones finish normally). Use queue_resume to continue."
)

QUEUE_RESUME_DESCRIPTION = "Resume a paused sub-browser screenshot queue."


def build_queue_screenshots_tool_fn(queue: SubBrowserQueue):
    @sdk_tool("queue_screenshots", QUEUE_SCREENSHOTS_DESCRIPTION, QUEUE_SCREENSHOTS_INPUT_SCHEMA)
    async def queue_screenshots_fn(args: dict[str, Any]) -> dict[str, Any]:
        items = args.get("items") or []
        if not items:
            return tool_result_to_sdk_content(ToolResult(error="items must be a non-empty list"))
        ids = queue.add(items)
        return tool_result_to_sdk_content(
            ToolResult(output=f"Queued {len(ids)} item(s): {', '.join(ids)}")
        )

    return queue_screenshots_fn


def build_queue_status_tool_fn(queue: SubBrowserQueue):
    @sdk_tool("queue_status", QUEUE_STATUS_DESCRIPTION, {"properties": {}, "type": "object"})
    async def queue_status_fn(args: dict[str, Any]) -> dict[str, Any]:
        snap = queue.snapshot()
        completed = snap["completed"][:MAX_STATUS_RESULTS]
        lines = [
            f"paused: {snap['paused']}",
            f"pending: {snap['pending_count']}, in progress: {snap['in_progress_count']}, "
            f"max fanout: {snap['max_fanout']}",
            f"showing {len(completed)} of {len(snap['completed'])} completed item(s):",
        ]
        for item in completed:
            lines.append(
                f"- {item['id']} [{item['status']}] screenshots={item['screenshots']}"
                + (f" error={item['error']!r}" if item.get("error") else "")
            )
        return tool_result_to_sdk_content(ToolResult(output="\n".join(lines)))

    return queue_status_fn


def build_queue_clear_tool_fn(queue: SubBrowserQueue):
    @sdk_tool("queue_clear", QUEUE_CLEAR_DESCRIPTION, {"properties": {}, "type": "object"})
    async def queue_clear_fn(args: dict[str, Any]) -> dict[str, Any]:
        counts = queue.clear()
        return tool_result_to_sdk_content(
            ToolResult(
                output=f"Cleared {counts['pending_cleared']} pending and "
                f"{counts['completed_cleared']} completed item(s)."
            )
        )

    return queue_clear_fn


def build_queue_pause_tool_fn(queue: SubBrowserQueue):
    @sdk_tool("queue_pause", QUEUE_PAUSE_DESCRIPTION, {"properties": {}, "type": "object"})
    async def queue_pause_fn(args: dict[str, Any]) -> dict[str, Any]:
        queue.pause()
        return tool_result_to_sdk_content(ToolResult(output="Queue paused."))

    return queue_pause_fn


def build_queue_resume_tool_fn(queue: SubBrowserQueue):
    @sdk_tool("queue_resume", QUEUE_RESUME_DESCRIPTION, {"properties": {}, "type": "object"})
    async def queue_resume_fn(args: dict[str, Any]) -> dict[str, Any]:
        queue.resume()
        return tool_result_to_sdk_content(ToolResult(output="Queue resumed."))

    return queue_resume_fn
