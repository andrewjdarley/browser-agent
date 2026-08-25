# Modifications Copyright (c) 2025 Anthropic, PBC
# Modified from original Microsoft Playwright source
# Original Microsoft Playwright source licensed under Apache License 2.0
# See CHANGELOG.md for details

"""Browser automation tool using Playwright for web interaction."""

import asyncio
import base64
import difflib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict
from uuid import uuid4

from playwright.async_api import Browser, BrowserContext, Page

from ..display_constants import BROWSER_HEIGHT, BROWSER_WIDTH, DISPLAY_NUM
from ..model_config import FIND_MODEL
from .base import ToolError, ToolResult
from .coordinate_scaling import CoordinateScaler

# Simple logging for debugging - removed, using print directly


# Custom browser tool input schema
BROWSER_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "properties": {
        "action": {
            "description": 'The action to perform. The available actions are:\n* `navigate`: Navigate to a URL or use "back"/"forward" for browser history navigation. Automatically includes a screenshot of the loaded page.\n* `screenshot`: Take a screenshot of the current browser viewport.\n* `left_click`: Click the left mouse button at the specified coordinate or element reference.\n* `right_click`: Click the right mouse button at the specified coordinate or element reference.\n* `middle_click`: Click the middle mouse button at the specified coordinate or element reference.\n* `double_click`: Double-click the left mouse button at the specified coordinate or element reference.\n* `triple_click`: Triple-click the left mouse button at the specified coordinate or element reference.\n* `hover`: Move the mouse cursor to the specified coordinate or element reference without clicking. Useful for revealing tooltips, dropdown menus, or triggering hover states.\n* `left_click_drag`: Click and drag from start_coordinate to coordinate.\n* `left_mouse_down`: Press and hold the left mouse button at the specified coordinate.\n* `left_mouse_up`: Release the left mouse button at the specified coordinate.\n* `scroll`: Scroll the page in a specified direction.\n* `scroll_to`: Scroll to bring an element into view.\n* `type`: Type text at the current cursor position.\n* `key`: Press a key or key combination (supports standard keys and modifiers).\n* `hold_key`: Hold down a key or key combination for a specified duration.\n* `read_page`: Get the DOM tree structure, optionally filtered for interactive elements.\n* `find`: Semantically search the page for elements matching a description and return their refs (does not mark anything visually itself - use `outline` for that).\n* `outline`: Draw a bounding box around one element (by ref) and screenshot it, so it is visibly identifiable in the image - use right before treating an element as evidence for a specific/checkable claim, to visually confirm you have the right one.\n* `get_page_text`: Get all text content from the page.\n* `wait`: Wait for a specified duration in seconds.\n* `form_input`: Set the value of a form input element.\n* `zoom`: Take a zoomed screenshot of a specific region.\n* `execute_js`: Execute JavaScript code in the page context. Returns the result of the last expression.',
            "enum": [
                "navigate",
                "screenshot",
                "left_click",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "hover",
                "left_click_drag",
                "left_mouse_down",
                "left_mouse_up",
                "scroll",
                "scroll_to",
                "type",
                "key",
                "hold_key",
                "read_page",
                "find",
                "outline",
                "get_page_text",
                "wait",
                "form_input",
                "zoom",
                "execute_js",
            ],
            "type": "string",
        },
        "text": {
            "description": 'Required for: `navigate` (URL or "back"/"forward"), `type` (text to type), `key` (key combination), `hold_key` (key to hold), `find` (text to search), `execute_js` (valid JavaScript code ONLY - no explanatory text, just the code). Optional for `read_page` (filter type: "interactive"), click actions (modifier keys to hold during click).',
            "type": "string",
        },
        "ref": {
            "description": "Element reference string for targeting specific DOM elements. Required for `scroll_to`, `form_input`, and `outline`. Optional for click actions and `hover` as an alternative to coordinates.",
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required for mouse actions when `ref` is not provided: `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click`, `hover`, `left_mouse_down`, `left_mouse_up`, `scroll`. Also serves as the end coordinate for `left_click_drag`.",
            "type": "array",
            "items": {"type": "integer"},
        },
        "start_coordinate": {
            "description": "(x, y): The starting x and y coordinates for drag operations. Required only for `left_click_drag`.",
            "type": "array",
            "items": {"type": "integer"},
        },
        "scroll_direction": {
            "description": "The direction to scroll. Required for `scroll` action.",
            "enum": ["up", "down", "left", "right"],
            "type": "string",
        },
        "scroll_amount": {
            "description": "The number of scroll units (similar to mouse wheel clicks). Required for `scroll` action.",
            "type": "integer",
        },
        "duration": {
            "description": "Duration in seconds. Required for `hold_key` and `wait` actions. For `wait`, must be between 0 and 100 seconds.",
            "type": "number",
        },
        "value": {
            "description": "The value to set for a form input element. Required for `form_input` action. Can be string, number, or boolean depending on the input type.",
            "type": ["string", "number", "boolean"],
        },
        "region": {
            "description": "(x1, y1, x2, y2): Defines a rectangular region for the `zoom` action. Coordinates specify top-left (x1, y1) and bottom-right (x2, y2) corners.",
            "type": "array",
            "items": {"type": "integer"},
        },
        "full_page": {
            "description": 'For `screenshot` only. If true, captures the entire scrollable page in one image instead of just the current viewport. Prefer this over manually scrolling and re-screenshotting when you need to see a full page or verify off-screen content. Note: sticky/fixed-position headers may appear repeated at each "fold" in the stitched image (a browser rendering limitation). Fails with an error on extremely tall pages (e.g. infinite-scroll feeds) — fall back to `scroll`/`get_page_text` for those.',
            "type": "boolean",
        },
    },
    "required": ["action"],
    "type": "object",
}

BROWSER_TOOL_DESCRIPTION = """A browser automation tool for web interaction. Use this tool to navigate websites, interact with elements, and extract content.

Key actions:
- navigate: Go to a URL (automatically includes a screenshot)
- screenshot: Take a visual screenshot (pass full_page: true to capture the entire page in one shot instead of scrolling repeatedly)
- read_page: Get DOM structure with element references
- get_page_text: Extract all text content
- left_click, right_click, double_click: Click elements
- hover: Move cursor without clicking (for tooltips, dropdowns)
- type: Enter text at cursor
- scroll: Scroll the page
- form_input: Fill form fields
- outline: Draw a bounding box around an element (by ref) so it's visible in the screenshot
- execute_js: Run JavaScript in page context"""


OUTPUT_DIR = Path("/tmp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Roughly 14 stacked 1080p viewports. Above this, full-page capture is more
# likely to hang or produce a misleadingly huge image (e.g. infinite-scroll
# feeds) than to be useful.
MAX_FULL_PAGE_HEIGHT_PX = 15000
FULL_PAGE_SCREENSHOT_TIMEOUT_S = 15

# A full_page capture that ends up bigger than this (raw PNG bytes) doesn't
# get sent whole - a huge image isn't more useful to the model than a
# reasonably-sized one anyway, and cramming it through the IPC channel to
# the CLI subprocess risks exceeding its message buffer and killing the
# whole session, not just this one tool call. Retry with a clipped capture
# of just the top of the page instead - for most real pages (docs, PR pages,
# articles) the header/summary/most relevant content is there, and anything
# further down is what scroll/get_page_text are for.
MAX_SCREENSHOT_RAW_BYTES = 3 * 1024 * 1024  # 3 MiB raw PNG (~4 MiB base64)
FULL_PAGE_FALLBACK_VIEWPORTS = 4  # height of the top-of-page retry capture

# document.readyState hitting "complete" means the HTML/JS has loaded - it says
# nothing about whether a client-rendered page (React/Next.js apps, lazy images,
# fade-in animations) has actually painted its content yet. Confirmed against a
# real run where a screenshot taken immediately after navigate came back visibly
# blank on a readyState-complete page. Flat hardcoded wait for now, not adaptive -
# revisit with a real signal (network-idle, poll for non-empty content) if this
# proves insufficient for some sites.
SCREENSHOT_SETTLE_DELAY_S = 2.0

# Same readyState-isn't-enough problem as above, but for text/DOM reads
# (get_page_text, read_page, execute_js) rather than screenshots - confirmed
# against real runs hitting a site's own transient "there was an error while
# loading" placeholder text immediately after navigate. A shorter delay than
# screenshots: these reads don't need a full visual paint, just enough time
# for the initial client-side render to settle.
TEXT_READ_SETTLE_DELAY_S = 1.5

# Directory containing browser tool utility files (JS scripts)
BROWSER_TOOL_UTILS_DIR = Path(__file__).parent.parent / "browser_tool_utils"


class BrowserOptions(TypedDict):
    display_width_px: int
    display_height_px: int


Actions = Literal[
    "navigate",
    "screenshot",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "hover",
    "left_click_drag",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "scroll_to",
    "type",
    "key",
    "hold_key",
    "read_page",
    "find",
    "outline",
    "get_page_text",
    "wait",
    "form_input",
    "zoom",
    "execute_js",
]

# Actions that can plausibly change the page's DOM. Their results get an
# automatic diff against the last-known DOM tree appended (see
# BrowserTool._attach_dom_context) - navigate gets a full fresh tree instead
# (a real page load, not a diff-worthy change), and read-only actions
# (screenshot, wait, get_page_text, find, outline, zoom) get neither, since
# they don't move the model any further from what it already knows.
DOM_MUTATING_ACTIONS = frozenset(
    {
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "hover",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "type",
        "key",
        "hold_key",
        "scroll",
        "scroll_to",
        "form_input",
        "execute_js",
    }
)

# Cap on how much DOM-diff text gets appended to one action's result - a
# large diff (e.g. a client-routed SPA navigation that isn't a real
# `navigate` call) is still useful context, but shouldn't be unbounded.
DOM_DIFF_MAX_CHARS = 6000


class BrowserTool:
    """
    A browser automation tool using Playwright for web interaction.

    Key actions for extracting content:
    - read_page: Extract structured DOM tree with element references (USE THIS for analyzing page structure)
    - get_page_text: Extract all text content from the page (USE THIS for reading articles/posts)
    - screenshot: Take a visual screenshot (only for visual confirmation, not for reading content)

    Navigation actions:
    - navigate: Go to a URL
    - find: Search for elements on the page

    Interaction actions:
    - left_click, right_click, double_click: Click elements
    - type: Enter text
    - scroll: Scroll the page
    """

    name: Literal["browser"] = "browser"

    # Instance-level browser connection (recreated per request)
    _browser: Optional[Browser] = None
    _context: Optional[BrowserContext] = None
    _page: Optional[Page] = None
    _playwright = None

    def __init__(self, run_dir: Optional[Path] = None):
        """Initialize the browser tool with standard viewport dimensions.

        Args:
            run_dir: Directory to write screenshots into. Defaults to the
                flat OUTPUT_DIR root (used by tests/non-Streamlit callers).
        """
        super().__init__()
        # Use constants for display configuration
        self.width = BROWSER_WIDTH
        self.height = BROWSER_HEIGHT
        self.run_dir = run_dir if run_dir is not None else OUTPUT_DIR
        self._initialized = False
        self._event_loop = None  # Track which event loop we're initialized in
        self.cdp_url = None  # Initialize CDP URL attribute for cleanup method
        # Last full DOM tree seen (via navigate or read_page) - the baseline
        # that DOM_MUTATING_ACTIONS get diffed against. See _attach_dom_context.
        self._last_dom_snapshot: Optional[str] = None

    @property
    def options(self) -> BrowserOptions:
        """Return browser display options."""
        # Note: This implementation uses fixed 1920x1080 dimensions with empirical
        # coordinate correction. For the recommended approach using client-side
        # downscaling, see the "Handle coordinate scaling" section in the computer
        # use documentation.
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
        }

    async def _ensure_browser(self) -> None:
        """Launch browser and ensure page is ready."""
        # NOTE: We intentionally DON'T reset the browser if the event loop changes
        # The browser should persist across conversation turns
        # Commenting out event loop check that was causing browser resets:
        # try:
        #     current_loop = asyncio.get_running_loop()
        #     if self._initialized and hasattr(self, "_event_loop"):
        #         if self._event_loop != current_loop:
        #             self._initialized = False
        #             self._browser = None
        #             self._context = None
        #             self._page = None
        #             self._playwright = None
        # except RuntimeError:
        #     pass

        if self._initialized:
            print(
                f"[Browser] Reusing existing browser instance",
                file=sys.stderr,
                flush=True,
            )
            if self._page:
                current_url = self._page.url
                print(
                    f"[Browser] Current page URL: {current_url}",
                    file=sys.stderr,
                    flush=True,
                )

        if not self._initialized:
            print(
                f"[Browser] Initializing browser for first time",
                file=sys.stderr,
                flush=True,
            )
            if self._playwright is None:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()

            if self._browser is None:
                viewport_width = self.width
                viewport_height = self.height

                is_docker = os.path.exists("/.dockerenv")

                launch_args = [
                    "--start-maximized",
                    f"--window-size={viewport_width},{viewport_height}",
                    "--window-position=0,0",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-gpu-sandbox",
                    "--disable-software-rasterizer",
                ]

                if is_docker:
                    launch_args.extend([
                        f"--display=:{DISPLAY_NUM}",
                        "--disable-infobars",
                        "--disable-session-crashed-bubble",
                        "--no-first-run",
                        "--disable-features=TranslateUI",
                        "--disable-component-extensions-with-background-pages",
                    ])

                print(
                    f"[Browser] Launching browser with viewport {viewport_width}x{viewport_height}",
                    file=sys.stderr,
                    flush=True,
                )

                self._browser = await self._playwright.chromium.launch(
                    headless=False,
                    args=launch_args,
                )

                self._context = await self._browser.new_context(
                    viewport={"width": viewport_width, "height": viewport_height},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                self._page = await self._context.new_page()
                self._page.set_default_timeout(30000)

                print(
                    f"[Browser] Browser initialized with viewport: {viewport_width}x{viewport_height}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[Browser] New browser instance created",
                    file=sys.stderr,
                    flush=True,
                )

            self._initialized = True
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._event_loop = None

    async def _execute_js_from_file(self, filename: str, *args) -> Any:
        """Load and execute JavaScript from a file."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        script_path = BROWSER_TOOL_UTILS_DIR / filename
        if not script_path.exists():
            raise ToolError(f"Script file not found: {filename}")

        script = script_path.read_text()

        # Special handling for browser_dom_script.js
        if filename == "browser_dom_script.js":
            # The DOM script defines window.__generateAccessibilityTree function
            # We need to inject it and then call it
            filter_type = args[0] if args else ""
            combined_expression = f"""
                (function() {{
                    {script}
                    return window.__generateAccessibilityTree('{filter_type}');
                }})()
            """
            return await self._page.evaluate(combined_expression)
        else:
            # For other scripts, wrap as a function and call with arguments
            escaped_args = ", ".join(json.dumps(arg) for arg in args)
            js_expression = f"({script})({escaped_args})"
            return await self._page.evaluate(js_expression)

    async def _wait_for_page_ready(
        self,
        timeout_s: float = 3.0,
        poll_interval: float = 0.3,
        settle_delay_s: float = SCREENSHOT_SETTLE_DELAY_S,
    ) -> None:
        """Check whether the page has actually finished loading before a
        screenshot; if not, wait briefly and re-check rather than capturing a
        half-loaded page. Gives up after timeout_s - a screenshot of a
        still-loading (or frozen) page is still better than no screenshot.

        Wrapped in a hard asyncio.wait_for: if the page is genuinely frozen
        (JS engine stuck, a blocking native dialog, etc.) a single
        page.evaluate() call can hang indefinitely, which the elapsed-time
        bookkeeping alone wouldn't catch - the outer timeout guarantees this
        always returns.

        readyState alone isn't enough (see SCREENSHOT_SETTLE_DELAY_S) - after
        the poll gives up or succeeds, wait settle_delay_s more before
        returning, so the caller's screenshot has a better chance of showing
        actually-painted content.
        """
        if self._page is None:
            return

        async def _poll() -> None:
            while True:
                ready_state = await self._page.evaluate("document.readyState")
                if ready_state == "complete":
                    return
                await asyncio.sleep(poll_interval)

        try:
            await asyncio.wait_for(_poll(), timeout=timeout_s)
        except Exception:
            pass  # still loading, frozen, or navigated away mid-check - fall through

        if settle_delay_s:
            await asyncio.sleep(settle_delay_s)

    async def _take_screenshot(self, full_page: bool = False) -> ToolResult:
        """
        Take a visual screenshot of the current page.
        NOTE: This only returns an image, not text content.
        Use read_page or get_page_text to extract actual content.
        """
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._wait_for_page_ready()

            if full_page:
                page_height = await self._page.evaluate(
                    "document.documentElement.scrollHeight"
                )
                if page_height > MAX_FULL_PAGE_HEIGHT_PX:
                    raise ToolError(
                        f"Page is too tall for a full-page screenshot "
                        f"({page_height}px > {MAX_FULL_PAGE_HEIGHT_PX}px limit, "
                        f"likely an infinite-scroll page). Use scroll or "
                        f"get_page_text instead."
                    )

            screenshot_path = self.run_dir / f"screenshot_{uuid4().hex}.png"
            truncation_note = ""
            saved_note = f"Screenshot saved as {screenshot_path.name}\n"

            if full_page:
                await asyncio.wait_for(
                    self._page.screenshot(path=str(screenshot_path), full_page=True),
                    timeout=FULL_PAGE_SCREENSHOT_TIMEOUT_S,
                )
                if screenshot_path.stat().st_size > MAX_SCREENSHOT_RAW_BYTES:
                    clip_height = min(self.height * FULL_PAGE_FALLBACK_VIEWPORTS, page_height)
                    await self._page.screenshot(
                        path=str(screenshot_path),
                        clip={"x": 0, "y": 0, "width": self.width, "height": clip_height},
                    )
                    if screenshot_path.stat().st_size > MAX_SCREENSHOT_RAW_BYTES:
                        # Still too big (rare - very dense visual content).
                        # Fall back to a plain viewport screenshot, guaranteed small.
                        await self._page.screenshot(path=str(screenshot_path), full_page=False)
                        truncation_note = (
                            "Note: this page was too large to capture in full even "
                            "when clipped - showing only the current viewport. Use "
                            "scroll or get_page_text for the rest of the page.\n"
                        )
                    else:
                        truncation_note = (
                            f"Note: the full page was too large to send in one "
                            f"screenshot, so this shows only the top {clip_height}px "
                            f"(the part most likely to matter - headers, summaries, "
                            f"metadata). Use scroll or get_page_text for content "
                            f"further down the page.\n"
                        )
            else:
                await self._page.screenshot(path=str(screenshot_path), full_page=False)

            # Read the file and encode to base64
            screenshot_bytes = screenshot_path.read_bytes()
            image_base64 = base64.b64encode(screenshot_bytes).decode()

            return ToolResult(output=saved_note + truncation_note, error=None, base64_image=image_base64)
        except ToolError:
            raise
        except asyncio.TimeoutError as e:
            raise ToolError(
                f"Full-page screenshot timed out after "
                f"{FULL_PAGE_SCREENSHOT_TIMEOUT_S}s (page may be lazy-loading "
                f"content indefinitely). Use scroll or get_page_text instead."
            ) from e
        except Exception as e:
            raise ToolError(f"Failed to take screenshot: {str(e)}") from e

    async def _zoom_screenshot(
        self, x: int, y: int, width: int, height: int
    ) -> ToolResult:
        """Take a screenshot of a specific region."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._wait_for_page_ready()

            # Take screenshot with clipping
            screenshot_path = self.run_dir / f"zoom_screenshot_{uuid4().hex}.png"
            await self._page.screenshot(
                path=str(screenshot_path),
                clip={"x": x, "y": y, "width": width, "height": height},
            )

            # Read the file and encode to base64
            screenshot_bytes = screenshot_path.read_bytes()
            image_base64 = base64.b64encode(screenshot_bytes).decode()

            return ToolResult(output="", error=None, base64_image=image_base64)
        except Exception as e:
            raise ToolError(f"Failed to take zoom screenshot: {str(e)}") from e

    async def _navigate(self, url: str) -> ToolResult:
        """Navigate to a URL, or "back"/"forward" through browser history."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            if url == "back":
                await self._page.go_back(wait_until="domcontentloaded")
            elif url == "forward":
                await self._page.go_forward(wait_until="domcontentloaded")
            else:
                # Add protocol if missing
                if not url.startswith(("http://", "https://", "file://", "about:")):
                    url = f"https://{url}"
                await self._page.goto(url, wait_until="domcontentloaded")

            # Take screenshot after navigation - _take_screenshot itself
            # verifies the page actually finished loading first
            return await self._take_screenshot()

        except Exception as e:
            raise ToolError(f"Failed to navigate to {url}: {str(e)}") from e

    def _scale_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """
        Apply auto-scaling to coordinates using the CoordinateScaler.

        Claude's vision model interprets images at a different resolution than actual.
        We use empirically-derived base resolution for accurate coordinate mapping.

        Args:
            x: Original x coordinate
            y: Original y coordinate

        Returns:
            Tuple of (scaled_x, scaled_y)
        """
        # Get scale factors for this viewport
        scale_x, scale_y = CoordinateScaler.get_scale_factors(self.width, self.height)

        # Only log scale factors if they're being initialized
        if not hasattr(self, '_logged_scale_factors'):
            print(
                f"[Auto-Scale] Using scale factors: {scale_x:.3f}x, {scale_y:.3f}y",
                file=sys.stderr,
                flush=True,
            )
            self._logged_scale_factors = True

        # Apply scaling using CoordinateScaler
        scaled_x, scaled_y = CoordinateScaler.scale_coordinates(
            x, y, self.width, self.height
        )

        # Log if scaling was actually applied
        if scaled_x != x or scaled_y != y:
            print(
                f"[Auto-Scale] Scaled ({x}, {y}) -> ({scaled_x}, {scaled_y})",
                file=sys.stderr,
                flush=True,
            )

        return scaled_x, scaled_y

    async def _click(
        self,
        action: str,
        coordinate: Optional[tuple[int, int]] = None,
        ref: Optional[str] = None,
        text: Optional[str] = None,
    ) -> ToolResult:
        """Handle various click actions."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            button = "left"
            click_count = 1

            if action == "right_click":
                button = "right"
            elif action == "middle_click":
                button = "middle"
            elif action == "double_click":
                click_count = 2
            elif action == "triple_click":
                click_count = 3

            if coordinate:
                x, y = coordinate

                # Apply auto-scaling to coordinates
                x, y = self._scale_coordinates(x, y)

                # Validate coordinates are within viewport bounds
                viewport = self._page.viewport_size
                if viewport:
                    if x < 0 or x > viewport['width'] or y < 0 or y > viewport['height']:
                        print(
                            f"[Click] WARNING: Coordinates ({x}, {y}) are outside viewport "
                            f"({viewport['width']}x{viewport['height']})",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Still attempt the click but warn about potential issues
                        if x > viewport['width']:
                            print(
                                f"[Click] X coordinate {x} exceeds viewport width {viewport['width']}",
                                file=sys.stderr,
                                flush=True,
                            )
                        if y > viewport['height']:
                            print(
                                f"[Click] Y coordinate {y} exceeds viewport height {viewport['height']}",
                                file=sys.stderr,
                                flush=True,
                            )

                # Ensure the page has focus
                await self._page.bring_to_front()

                # Move mouse to position and click
                await self._page.mouse.move(x, y)
                await asyncio.sleep(0.01)  # Small delay to ensure mouse is positioned

                # Perform the click based on type
                await self._page.mouse.click(
                    x, y, button=button, click_count=click_count
                )
                return ToolResult(output=f"Clicked at ({x}, {y})", error=None)
            elif ref:
                # Use the browser_element_script.js to find and click element
                element_info = await self._execute_js_from_file(
                    "browser_element_script.js", ref
                )

                if not element_info.get("success", False):
                    raise ToolError(
                        element_info.get("message", "Failed to find element")
                    )

                # Get the coordinates from element_info
                click_x, click_y = element_info["coordinates"]

                # Move to element and click
                await self._page.mouse.move(click_x, click_y)
                await asyncio.sleep(0.1)
                await self._page.mouse.click(
                    click_x, click_y, button=button, click_count=click_count
                )
                return ToolResult(output=f"Clicked element with ref: {ref}", error=None)
            elif text:
                # Click on element containing text
                await self._page.click(
                    f"text={text}", button=button, click_count=click_count
                )
                return ToolResult(output=f"Clicked on text: {text}", error=None)
            else:
                raise ToolError(
                    "Either coordinate, ref, or text is required for click action"
                )

        except Exception as e:
            raise ToolError(f"Failed to perform {action}: {str(e)}") from e

    async def _type_text(self, text: str) -> ToolResult:
        """Type text into the focused element."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._page.keyboard.type(text)
            return ToolResult(output=f"Typed: {text}", error=None)
        except Exception as e:
            raise ToolError(f"Failed to type text: {str(e)}") from e

    async def _press_key(
        self, key: str, hold: bool = False, duration: float = 0.01
    ) -> ToolResult:
        """Press a keyboard key or key combination."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Load the key map
            from ..browser_tool_utils.browser_key_map import KEY_MAP

            def map_key(k: str) -> str:
                """Map a key name to Playwright's expected format."""
                key_info = KEY_MAP.get(k.lower())
                if key_info and "key" in key_info:
                    return key_info["key"]
                return k

            # Handle key combinations (e.g., "cmd+a", "ctrl+c")
            if "+" in key:
                parts = key.split("+")
                mapped_parts = [map_key(p) for p in parts]
                mapped_key = "+".join(mapped_parts)
                await self._page.keyboard.press(mapped_key)
                return ToolResult(output=f"Pressed key combination: {mapped_key}", error=None)

            # Map single key if needed
            key_info = KEY_MAP.get(key.lower())
            if key_info:
                key_to_press = key_info["code"] if "code" in key_info else key
            else:
                key_to_press = key

            if hold:
                await self._page.keyboard.down(key_to_press)
                await asyncio.sleep(duration)
                await self._page.keyboard.up(key_to_press)
                return ToolResult(
                    output=f"Held key '{key}' for {duration} seconds", error=None
                )
            else:
                await self._page.keyboard.press(key_to_press)
                return ToolResult(output=f"Pressed key: {key}", error=None)

        except Exception as e:
            raise ToolError(f"Failed to press key '{key}': {str(e)}") from e

    async def _scroll(
        self,
        coordinate: Optional[tuple[int, int]] = None,
        direction: Optional[str] = None,
        amount: Optional[int] = None,
    ) -> ToolResult:
        """Scroll the page or element."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            if not direction:
                direction = "down"
            if not amount:
                amount = 3  # Default scroll amount

            # Calculate scroll delta based on direction
            delta_x = 0
            delta_y = 0

            if direction == "up":
                delta_y = -amount * 100
            elif direction == "down":
                delta_y = amount * 100
            elif direction == "left":
                delta_x = -amount * 100
            elif direction == "right":
                delta_x = amount * 100

            if coordinate:
                x, y = coordinate
                await self._page.mouse.wheel(delta_x, delta_y)
            else:
                # Scroll the main page
                await self._page.evaluate(f"window.scrollBy({delta_x}, {delta_y})")

            # Wait for content to stabilize after scroll
            await asyncio.sleep(0.5)

            # Take screenshot to show new viewport content
            screenshot_result = await self._take_screenshot()
            return ToolResult(
                output=f"Scrolled {direction} by {amount} units",
                error=None,
                base64_image=screenshot_result.base64_image
            )

        except Exception as e:
            raise ToolError(f"Failed to scroll: {str(e)}") from e

    async def _scroll_to(self, ref: str) -> ToolResult:
        """Scroll to a specific element."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            element_info = await self._execute_js_from_file(
                "browser_element_script.js", ref
            )

            if not element_info["success"]:
                raise ToolError(element_info.get("message", "Failed to find element"))

            # Wait for content to stabilize after scroll
            await asyncio.sleep(0.5)

            # Take screenshot to show new viewport content
            screenshot_result = await self._take_screenshot()
            return ToolResult(
                output=f"Scrolled to element with ref: {ref}",
                error=None,
                base64_image=screenshot_result.base64_image
            )

        except Exception as e:
            raise ToolError(f"Failed to scroll to element: {str(e)}") from e

    async def _drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> ToolResult:
        """Perform a drag operation."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Apply auto-scaling to both start and end coordinates
            scaled_start_x, scaled_start_y = self._scale_coordinates(start_x, start_y)
            scaled_end_x, scaled_end_y = self._scale_coordinates(end_x, end_y)

            await self._page.mouse.move(scaled_start_x, scaled_start_y)
            await self._page.mouse.down()
            await self._page.mouse.move(scaled_end_x, scaled_end_y)
            await self._page.mouse.up()

            return ToolResult(
                output=f"Dragged from ({scaled_start_x}, {scaled_start_y}) to ({scaled_end_x}, {scaled_end_y})",
                error=None,
            )

        except Exception as e:
            raise ToolError(f"Failed to perform drag: {str(e)}") from e

    async def _mouse_down(self, x: int, y: int) -> ToolResult:
        """Press mouse button down."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Apply auto-scaling to coordinates
            scaled_x, scaled_y = self._scale_coordinates(x, y)

            await self._page.mouse.move(scaled_x, scaled_y)
            await self._page.mouse.down()
            return ToolResult(output=f"Mouse down at ({scaled_x}, {scaled_y})", error=None)

        except Exception as e:
            raise ToolError(f"Failed to perform mouse down: {str(e)}") from e

    async def _mouse_up(self, x: int, y: int) -> ToolResult:
        """Release mouse button."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Apply auto-scaling to coordinates
            scaled_x, scaled_y = self._scale_coordinates(x, y)

            await self._page.mouse.move(scaled_x, scaled_y)
            await self._page.mouse.up()
            return ToolResult(output=f"Mouse up at ({scaled_x}, {scaled_y})", error=None)

        except Exception as e:
            raise ToolError(f"Failed to perform mouse up: {str(e)}") from e

    async def _hover(
        self,
        coordinate: Optional[tuple[int, int]] = None,
        ref: Optional[str] = None,
    ) -> ToolResult:
        """
        Move the mouse cursor to a position without clicking.
        Useful for revealing tooltips, dropdown menus, or triggering hover states.
        """
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Prefer ref over coordinate (refs are more reliable)
            if ref:
                # Use the browser_element_script.js to find element coordinates
                element_info = await self._execute_js_from_file(
                    "browser_element_script.js", ref
                )

                if not element_info.get("success", False):
                    raise ToolError(
                        element_info.get("message", "Failed to find element")
                    )

                # Get the coordinates from element_info
                hover_x, hover_y = element_info["coordinates"]

                await self._page.bring_to_front()
                await self._page.mouse.move(hover_x, hover_y)
                # Wait for hover effects to render
                await asyncio.sleep(0.5)
                # Take screenshot to show hover result
                screenshot_result = await self._take_screenshot()
                return ToolResult(
                    output=f"Hovered over element with ref: {ref}",
                    error=None,
                    base64_image=screenshot_result.base64_image
                )
            elif coordinate:
                x, y = coordinate
                # Apply auto-scaling to coordinates
                scaled_x, scaled_y = self._scale_coordinates(x, y)

                await self._page.bring_to_front()
                await self._page.mouse.move(scaled_x, scaled_y)

                # Wait for hover effects to render
                await asyncio.sleep(0.3)
                # Take screenshot to show hover result
                screenshot_result = await self._take_screenshot()
                return ToolResult(
                    output=f"Hovered at ({scaled_x}, {scaled_y})",
                    error=None,
                    base64_image=screenshot_result.base64_image
                )
            else:
                raise ToolError(
                    "Either coordinate or ref is required for hover action"
                )

        except Exception as e:
            raise ToolError(f"Failed to perform hover: {str(e)}") from e

    async def _current_dom_text(self, filter_type: str = "all") -> str:
        """Full serialized accessibility tree text for the current page.
        Shared by _read_page, _find, and the automatic DOM-diff mechanism -
        the latter always uses the unfiltered "all" tree as its baseline
        regardless of what filter_type the model asks _read_page for, so the
        diff stays self-consistent even when the model requests a filtered
        view for its own display."""
        dom_tree = await self._execute_js_from_file("browser_dom_script.js", filter_type)
        if isinstance(dom_tree, dict) and "pageContent" in dom_tree:
            return dom_tree["pageContent"]
        if isinstance(dom_tree, dict):
            return json.dumps(dom_tree, indent=2)
        return str(dom_tree)

    def _diff_dom(self, old_tree: Optional[str], new_tree: str) -> str:
        """Line-level diff between two accessibility-tree snapshots. Refs are
        stable across calls for the same element (browser_dom_script.js
        reuses them via a WeakRef map keyed by element identity), so an
        unchanged element produces an unchanged line and this only surfaces
        genuinely added/removed/changed ones - not a full re-dump."""
        if old_tree is None or old_tree == new_tree:
            return ""
        diff_lines = difflib.unified_diff(
            old_tree.splitlines(), new_tree.splitlines(), lineterm="", n=0
        )
        changed = [
            line
            for line in diff_lines
            if line[:1] in ("+", "-") and not line.startswith(("+++", "---"))
        ]
        return "\n".join(changed)

    def _wrap_dom_note(self, label: str, content: str) -> str:
        """Wrap DOM content in the same __PAGE_EXTRACTED__/__FULL_CONTENT__
        marker pair _read_page uses, so message_renderer.py hides the raw
        tree/diff from the visible chat (showing just a one-line summary)
        while the model still gets the full text via the actual tool result."""
        truncated = len(content) > DOM_DIFF_MAX_CHARS
        if truncated:
            content = content[:DOM_DIFF_MAX_CHARS] + "\n... (truncated)"
        summary = f"{label} ({len(content):,} chars{', truncated' if truncated else ''})"
        return f"__PAGE_EXTRACTED__\n{summary}\n__FULL_CONTENT__\n{content}"

    async def _attach_dom_context(self, action: str, result: ToolResult) -> ToolResult:
        """Append DOM state to an action's result so the model doesn't have
        to remember to call read_page to see what changed - navigate gets a
        full fresh tree (a real page load), other DOM-mutating actions get a
        diff against the last-seen tree. Defensive by design: any failure
        here just skips the DOM note rather than breaking the action's own
        already-successful result."""
        if self._page is None or not result or result.error:
            return result

        try:
            if action == "navigate":
                full_tree = await self._current_dom_text("all")
                self._last_dom_snapshot = full_tree
                note = self._wrap_dom_note("DOM tree (page just loaded)", full_tree)
            elif action == "read_page":
                # _read_page already gave the model a full tree (possibly
                # filtered) - just keep the diff baseline in sync with it.
                self._last_dom_snapshot = await self._current_dom_text("all")
                return result
            elif action in DOM_MUTATING_ACTIONS:
                new_tree = await self._current_dom_text("all")
                diff_text = self._diff_dom(self._last_dom_snapshot, new_tree)
                self._last_dom_snapshot = new_tree
                if not diff_text:
                    return result
                note = self._wrap_dom_note("DOM changes since your last action", diff_text)
            else:
                return result
        except Exception:
            return result

        return result.replace(output=(result.output or "") + "\n\n" + note)

    async def _read_page(self, filter_type: str = "") -> ToolResult:
        """
        Extract the DOM tree with structured content and element references.
        USE THIS to analyze page structure and find specific elements.
        Returns a structured tree with text content, not just a screenshot.
        """
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._wait_for_page_ready(settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            full_content = await self._current_dom_text(filter_type)

            # Calculate content size for summary
            content_length = len(full_content)
            # Estimate token count
            # Note: For exact counts, use client.beta.messages.count_tokens API
            # This estimate uses ~3.5 chars/token which is typical for Claude with English text
            # Actual ratio varies by content type (code, languages, special characters)
            estimated_tokens = int(content_length / 3.5)

            # Create a summary for UI display
            summary = f"Extracted page DOM tree (~{estimated_tokens:,} tokens, {content_length:,} characters)"

            # Return the full content for the API but with a marker for the UI
            return ToolResult(
                output=f"__PAGE_EXTRACTED__\n{summary}\n__FULL_CONTENT__\n{full_content}",
                error=None
            )

        except Exception as e:
            raise ToolError(f"Failed to read page: {str(e)}") from e

    async def _get_page_text(self) -> ToolResult:
        """
        Extract ALL text content from the current page.
        USE THIS to read articles, posts, or any text content.
        Returns the actual text, not a screenshot.
        Perfect for reading Reddit posts, articles, etc.
        """
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._wait_for_page_ready(settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            # Use the browser_text_script.js from reference implementation
            result = await self._execute_js_from_file("browser_text_script.js")

            # Format the output like the reference implementation
            if isinstance(result, dict):
                full_content = f"""Title: {result.get("title", "N/A")}
URL: {result.get("url", "N/A")}
Source element: <{result.get("source", "unknown")}>
---
{result.get("text", "")}"""
            else:
                full_content = str(result)

            # Calculate content size for summary
            content_length = len(full_content)
            # Estimate token count
            # Note: For exact counts, use client.beta.messages.count_tokens API
            # This estimate uses ~3.5 chars/token which is typical for Claude with English text
            # Actual ratio varies by content type (code, languages, special characters)
            estimated_tokens = int(content_length / 3.5)

            # Create a summary for UI display
            title = result.get("title", "N/A") if isinstance(result, dict) else "N/A"
            url = result.get("url", "N/A") if isinstance(result, dict) else "N/A"
            summary = f"Extracted page text from: {title}\nURL: {url}\n(~{estimated_tokens:,} tokens, {content_length:,} characters)"

            # Return the full content for the API but with a marker for the UI
            return ToolResult(
                output=f"__TEXT_EXTRACTED__\n{summary}\n__FULL_CONTENT__\n{full_content}",
                error=None
            )

        except Exception as e:
            raise ToolError(f"Failed to get page text: {str(e)}") from e

    async def _find(self, search_query: str) -> ToolResult:
        """Find elements on the page matching the search query using AI."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # First get the DOM tree for analysis
            dom_tree_json = await self._current_dom_text("all")

            # Try to use Anthropic API if available
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    from anthropic import AsyncAnthropic

                    client = AsyncAnthropic(api_key=api_key)

                    prompt = f"""You are helping find elements on a web page. The user wants to find: "{search_query}"

Here is the accessibility tree of the page:
{dom_tree_json}

Find ALL elements that match the user's query. Return up to 20 most relevant matches, ordered by relevance.

Return your findings in this exact format (one line per matching element):

FOUND: <total_number_of_matching_elements>
SHOWING: <number_shown_up_to_20>
---
ref_X | role | name | type | reason why this matches
ref_Y | role | name | type | reason why this matches
...

If there are more than 20 matches, add this line at the end:
MORE: Use a more specific query to see additional results

If no matching elements are found, return only:
FOUND: 0
ERROR: explanation of why no elements were found"""

                    response = await client.messages.create(
                        model=FIND_MODEL,
                        max_tokens=800,
                        temperature=1.0,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Handle the response properly
                    first_content = response.content[0]
                    if hasattr(first_content, "text"):
                        response_text = first_content.text.strip()
                    else:
                        # Handle other content types if needed
                        response_text = str(first_content)
                    lines = [
                        line.strip()
                        for line in response_text.split("\n")
                        if line.strip()
                    ]

                    total_found = 0
                    elements = []
                    has_more = False
                    error_message = None

                    for line in lines:
                        if line.startswith("FOUND:"):
                            try:
                                total_found = int(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                total_found = 0
                        elif line.startswith("SHOWING:"):
                            pass
                        elif line.startswith("ERROR:"):
                            error_message = line[6:].strip()
                        elif line.startswith("MORE:"):
                            has_more = True
                        elif line.startswith("ref_") and "|" in line:
                            parts = [p.strip() for p in line.split("|")]
                            if len(parts) >= 4:
                                elements.append(
                                    {
                                        "ref": parts[0],
                                        "role": parts[1],
                                        "name": parts[2] if len(parts) > 2 else "",
                                        "type": parts[3] if len(parts) > 3 else "",
                                        "description": parts[4]
                                        if len(parts) > 4
                                        else "",
                                    }
                                )

                    if total_found == 0 or len(elements) == 0:
                        return ToolResult(
                            output=error_message or "No matching elements found",
                            error=None,
                        )

                    message = f"Found {total_found} matching element{'s' if total_found != 1 else ''}"
                    if has_more:
                        message += f" (showing first {len(elements)}, use a more specific query to narrow results)"

                    # Format elements for output
                    elements_output = []
                    for el in elements:
                        element_str = f"- {el['ref']}: {el['role']}"
                        if el.get("name"):
                            element_str += f" {el['name']}"
                        if el.get("type"):
                            element_str += f" {el['type']}"
                        if el.get("description"):
                            element_str += f" - {el['description']}"
                        elements_output.append(element_str)

                    elements_str = "\n".join(elements_output)
                    return ToolResult(output=f"{message}\n\n{elements_str}", error=None)

                except Exception:
                    pass  # Failed to use AI for find, falling back to simple search

            # Fallback to simple text search if AI is not available
            elements = await self._page.query_selector_all(
                f"*:has-text('{search_query}')"
            )

            if not elements:
                return ToolResult(
                    output=f"No matching elements found for: {search_query}", error=None
                )

            # For simple fallback, just report count (no ref_ids without AI analysis)
            return ToolResult(
                output=f"Found {len(elements)} matching element{'s' if len(elements) != 1 else ''} (Note: AI-based search with ref_ids requires ANTHROPIC_API_KEY)",
                error=None,
            )

        except Exception as e:
            raise ToolError(f"Failed to find elements: {str(e)}") from e

    async def _outline(self, ref: str) -> ToolResult:
        """Draw a bounding box around one element and screenshot it, then
        remove the box. A visual complement to find/execute_js: confirms
        which element a ref actually resolves to by making it directly
        visible, rather than trusting the ref's reported text/role alone.

        Named "outline", not "highlight" - a real run showed the model
        conflating our tool with a task's own use of "highlight" (e.g. a
        Wikipedia page's native highlighted-section behavior), reimplementing
        this exact box-drawing pattern by hand instead of searching the page
        for what the task actually meant. See browser_use_demo/loop.py."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            element_info = await self._execute_js_from_file(
                "browser_element_script.js", ref
            )
            if not element_info.get("success", False):
                raise ToolError(element_info.get("message", "Failed to find element"))

            rect = element_info["rect"]
            await self._page.evaluate(
                """(rect) => {
                    const existing = document.getElementById("__claude_outline_box__");
                    if (existing) existing.remove();
                    const box = document.createElement("div");
                    box.id = "__claude_outline_box__";
                    box.style.cssText = "position:fixed; left:" + (rect.left - 3) + "px; "
                        + "top:" + (rect.top - 3) + "px; width:" + (rect.width + 6) + "px; "
                        + "height:" + (rect.height + 6) + "px; border:3px solid #ff2d55; "
                        + "border-radius:3px; z-index:2147483647; pointer-events:none;";
                    document.body.appendChild(box);
                }""",
                rect,
            )

            try:
                screenshot_result = await self._take_screenshot()
            finally:
                await self._page.evaluate(
                    "document.getElementById('__claude_outline_box__')?.remove()"
                )

            element_desc = element_info.get("elementInfo", ref)
            return ToolResult(
                output=f"Outlined element with ref: {ref} ({element_desc})\n"
                + screenshot_result.output,
                error=None,
                base64_image=screenshot_result.base64_image,
            )

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to outline element: {str(e)}") from e

    async def _form_input(self, ref: str, value: Any) -> ToolResult:
        """Fill a form field with a value."""
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            # Use the browser_form_input_script.js from reference implementation
            result = await self._execute_js_from_file(
                "browser_form_input_script.js", ref, value
            )

            if isinstance(result, dict) and not result.get("success", False):
                raise ToolError(result.get("message", "Failed to fill form field"))

            return ToolResult(
                output=f"Filled form field {ref} with value: {value}", error=None
            )

        except Exception as e:
            raise ToolError(f"Failed to fill form field: {str(e)}") from e

    async def _wait(self, duration: float) -> ToolResult:
        """Wait for a specified duration."""
        try:
            await asyncio.sleep(duration)
            return ToolResult(
                output=f"Waited for {duration} second{'s' if duration != 1 else ''}",
                error=None,
            )
        except Exception as e:
            raise ToolError(f"Failed to wait: {str(e)}") from e

    async def _execute_js(self, code: str) -> ToolResult:
        """
        Execute JavaScript code in the page context.
        Returns the result of the last expression.
        """
        if self._page is None:
            raise ToolError("Browser not initialized")

        try:
            await self._wait_for_page_ready(settle_delay_s=TEXT_READ_SETTLE_DELAY_S)
            # Execute the code in page context
            # Playwright's evaluate handles async/await automatically
            result = await self._page.evaluate(code)

            # Format the result
            if result is None:
                result_str = "undefined"
            elif isinstance(result, (dict, list)):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)

            return ToolResult(output=result_str, error=None)

        except Exception as e:
            raise ToolError(f"JavaScript execution error: {str(e)}") from e

    async def __call__(self, *, action: Actions, **kwargs) -> ToolResult:
        """Public entry point. Runs the action via _dispatch, then attaches
        automatic DOM context (a full tree on navigate, a diff against the
        last-seen tree on other DOM-mutating actions) so the model doesn't
        have to remember to call read_page to see what an action changed -
        see _attach_dom_context."""
        result = await self._dispatch(action=action, **kwargs)
        return await self._attach_dom_context(action, result)

    async def _dispatch(
        self,
        *,
        action: Actions,
        text: Optional[str] = None,
        ref: Optional[str] = None,
        coordinate: Optional[tuple[int, int]] = None,
        start_coordinate: Optional[tuple[int, int]] = None,
        scroll_direction: Optional[Literal["up", "down", "left", "right"]] = None,
        scroll_amount: Optional[int] = None,
        duration: Optional[float] = None,
        value: Optional[Any] = None,
        region: Optional[tuple[int, int, int, int]] = None,
        full_page: bool = False,
        **kwargs,
    ) -> ToolResult:
        """
        Execute browser actions.

        Parameters:
        - action: The action to perform
        - text: Text input for type, key, navigate, find actions
        - ref: Element reference for element-based actions
        - coordinate: (x, y) coordinates for mouse actions
        - start_coordinate: Starting point for drag actions
        - scroll_direction: Direction for scroll action
        - scroll_amount: Amount to scroll
        - duration: Duration for wait or hold_key actions
        - value: Value for form_input action
        - region: (x, y, width, height) for zoom screenshot
        - full_page: For screenshot action, capture the entire page instead of viewport
        """

        # Ensure browser is running for all actions
        await self._ensure_browser()

        if action == "navigate":
            if not text:
                raise ToolError("URL is required for navigate action")
            return await self._navigate(text)

        elif action == "screenshot":
            return await self._take_screenshot(full_page=full_page)

        elif action == "zoom":
            if not region:
                raise ToolError(
                    "Region (x1, y1, x2, y2) is required for zoom action"
                )
            x1, y1, x2, y2 = region
            # Convert corner coordinates to x, y, width, height
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return await self._zoom_screenshot(x, y, width, height)

        elif action in [
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
        ]:
            return await self._click(action, coordinate, ref, text)

        elif action == "hover":
            return await self._hover(coordinate, ref)

        elif action == "type":
            if not text:
                raise ToolError("Text is required for type action")
            return await self._type_text(text)

        elif action == "key":
            if not text:
                raise ToolError("Key is required for key action")
            return await self._press_key(text)

        elif action == "hold_key":
            if not text:
                raise ToolError("Key is required for hold_key action")
            if not duration:
                duration = 1.0
            return await self._press_key(text, hold=True, duration=duration)

        elif action == "scroll":
            return await self._scroll(coordinate, scroll_direction, scroll_amount)

        elif action == "scroll_to":
            if not ref:
                raise ToolError("Element reference is required for scroll_to action")
            return await self._scroll_to(ref)

        elif action == "left_click_drag":
            if not start_coordinate or not coordinate:
                raise ToolError(
                    "Both start_coordinate and coordinate are required for drag action"
                )
            start_x, start_y = start_coordinate
            end_x, end_y = coordinate
            return await self._drag(start_x, start_y, end_x, end_y)

        elif action == "left_mouse_down":
            if not coordinate:
                raise ToolError("Coordinate is required for mouse_down action")
            x, y = coordinate
            return await self._mouse_down(x, y)

        elif action == "left_mouse_up":
            if not coordinate:
                raise ToolError("Coordinate is required for mouse_up action")
            x, y = coordinate
            return await self._mouse_up(x, y)

        elif action == "read_page":
            filter_type = text if text in ["interactive", ""] else ""
            return await self._read_page(filter_type)

        elif action == "get_page_text":
            return await self._get_page_text()

        elif action == "find":
            if not text:
                raise ToolError("Text is required for find action")
            return await self._find(text)

        elif action == "outline":
            if not ref:
                raise ToolError("Element reference is required for outline action")
            return await self._outline(ref)

        elif action == "form_input":
            if not ref:
                raise ToolError("Element reference is required for form_input action")
            if value is None:
                raise ToolError("Value is required for form_input action")
            return await self._form_input(ref, value)

        elif action == "wait":
            if not duration:
                duration = 1.0
            return await self._wait(duration)

        elif action == "execute_js":
            if not text:
                raise ToolError("JavaScript code is required for execute_js action")
            return await self._execute_js(text)

        else:
            raise ToolError(f"Unknown action: {action}")

    async def cleanup(self):
        """Cleanup method to ensure browser is closed properly."""
        # Clean up browser resources
        if self.cdp_url:
            # When connected to CDP server, just disconnect without closing tabs
            self._page = None
            self._context = None
            self._browser = None
        else:
            # For local browser, close everything
            if self._page:
                await self._page.close()
                self._page = None

            if self._context:
                await self._context.close()
                self._context = None

            if self._browser:
                await self._browser.close()
                self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._initialized = False
