"""
Agent orchestration for browser automation, built on the Claude Agent SDK.

The SDK handles the tool-calling loop, session/context management, and (once
wired up) subagents/hooks/permissions. We only own: the browser tool itself
(tools/browser.py), the system prompt, and translating the SDK's message
stream into the shapes browser_use_demo's Streamlit renderer already expects.
"""

from datetime import datetime
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

from .agent_sdk_bridge import MAX_BUFFER_SIZE, build_mcp_server
from .guardrails import GuardrailPolicy
from .model_config import SUBAGENT_MODEL, VERIFY_MODEL
from .model_config import resolve as resolve_model
from .run_logger import RunLogger
from .tools import BrowserTool, FileOutputTool
from .tools.batch_extract import BatchExtractTool, build_batch_extract_tool_fn
from .tools.script_runner import ScriptRunnerTool, build_run_script_tool_fn
from .tools.sub_browser_queue import (
    SubBrowserQueue,
    build_queue_clear_tool_fn,
    build_queue_pause_tool_fn,
    build_queue_resume_tool_fn,
    build_queue_screenshots_tool_fn,
    build_queue_status_tool_fn,
)
from .tools.subagent import DispatchSubagentsTool, build_dispatch_subagents_tool_fn
from .tools.verify import MAX_VERIFICATIONS_PER_SESSION, VerifyFindingTool, build_verify_finding_tool_fn

# Browser-specific system prompt. Passed as a plain string (not the
# `claude_code` preset) so the coordinator doesn't inherit Claude Code's
# coding-assistant framing, default tools, or system prompt.
BROWSER_SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You control a Chromium browser via Playwright automation.
* The current date is {datetime.today().strftime("%A, %B %-d, %Y")}.
</SYSTEM_CAPABILITY>

<TOOL_GUIDANCE>
You receive a screenshot at the start of each turn purely for orientation - to see whether you're already on the right page before deciding whether to navigate again. Do not use it to read text, locate elements, or decide where to click.

Navigation and interaction rely on the DOM, essentially always - not as a preference, as the default you don't deviate from without a specific reason. You get the DOM's current state automatically, without having to ask for it every time: a full tree the first time you land on a page (navigate), or whenever you explicitly call read_page, and a diff against what you last saw after every other action that can change the page (clicks, typing, scrolling, form_input, execute_js, etc.) - appended right to that action's own result. Use the element refs (ref_1, ref_2, ...) it gives you with your interaction tools (click, type, hover, form_input, scroll_to). Never click by raw (x, y) coordinate unless a specific element genuinely has no ref - and check with read_page before concluding that, don't assume it. Coordinate-clicking on a page you haven't actually inspected is exactly how tasks go wrong: clicking a red-herring link, landing on unrelated content, or building an answer from what a screenshot looked like instead of what the page actually contains.

Screenshots are for exactly two things: an explicitly requested visual deliverable, and a genuine last resort when the DOM isn't giving you useful signal (e.g. canvas-rendered content, or an accessibility tree that comes back empty/broken). They are not a parallel way to look around or extract information - never read text, counts, dates, or any other data off a screenshot when get_page_text, the DOM tree/diff, or execute_js could get it from the real page content instead. If you ever find yourself about to transcribe something from a screenshot into a final answer or file, stop and get it from the DOM first.

If content seems to be missing right after navigating or clicking - a DOM diff showing nothing new, get_page_text coming back sparse or with a transient loading/error placeholder, or a screenshot that looks blank - the page may genuinely still be loading, not actually empty. Before concluding the content isn't there, call wait for a few seconds and re-check, rather than giving up or reporting an absence you haven't actually confirmed.

Two different ways to locate something on a page, for two different needs. When you're looking for something by description ("the search box", "the citation that mentions Beevor") rather than an exact position, use find - it does semantic matching over the whole page for you. When you need an exact position or count ("item number N in a list"), that's a precise-counting problem, not a description-matching one - find isn't reliable for this (it's an LLM eyeballing a large dump of the page, which is exactly the kind of counting task LLMs get wrong), so write execute_js instead, and don't assume your selector's scope without checking it. If your execute_js code has more than one statement, or uses `return`, wrap it in an immediately-invoked function - `(() => {{ ...; return x; }})()` - not bare statements at the top level. A top-level `return` outside a function is a JavaScript syntax error and the call will fail immediately; this has been a repeated, avoidable failure. A single expression (e.g. `document.title`) doesn't need wrapping. A class-based selector (e.g. ".references li") matches descendants of EVERY element with that class combined into one list, even if there are multiple separate ones on the page (e.g. a short "Notes" list and a long "References" list can share the same class) - so indexing into it can silently land in the wrong place. Before trusting a positional index: confirm how many distinct containers your selector's root actually matches and that you've picked the right one specifically (e.g. query the containers themselves first - document.querySelectorAll('ol.references') - and pick the one whose size matches what you expect), then cross-check the specific element you land on against something you can directly observe (its actually-rendered/visible text, or an outline screenshot - see below) before treating it as the answer - don't stop at the first result that runs without erroring.

When content genuinely isn't reachable through the DOM at all - not just hard to find, but actually absent from it - use network_list_types, network_list, and network_inspect instead of digging further with execute_js/read_page. Two concrete cases where this matters: a same-origin iframe's content (execute_js/page.evaluate only ever touches the top frame, so that content is invisible to every DOM-reading action no matter how you query it, even though the iframe's own URL is visible); and a client-rendered page where the data you want only exists in a JSON response, never in the rendered HTML (the same class of page batch_extract can't read - see below). Start with network_list_types for a first-glance breakdown of what traffic exists, narrow down with network_list (filter by a URL/status/type substring), then network_inspect the specific response's body once you've found it. This reads raw HTTP responses, not what got rendered - a fundamentally different source of truth than every DOM-based action above, not a fallback flavor of the same thing.

When you have a batch of items that genuinely need per-item reasoning or interaction (forms, multi-step flows, judgment calls) - not just data extraction - use dispatch_subagents instead of processing them one at a time in a loop: it runs them concurrently, each with its own browser tab, all following the same shared instructions you give it. For pure data extraction at scale, see the scripting guidance below instead - it's far cheaper per item.

Don't scroll blindly in small increments to explore a page. If you don't already know exactly where your target is, call read_page (or re-check it) to find an element ref, or scroll_to a major structural landmark (a section heading, a "load more" control, etc.) and look at what's there - then decide your next move from what you actually see. Only use plain directional scroll when you're confident you're already close to the target and it'll take at most one or two calls to get there. If you're scrolling repeatedly without a specific ref in mind, stop and look at the DOM instead.

If a task involves several similar items and manually repeating the same steps on each would clearly take many tool calls, script it instead - this is a normal, common solution, not a last resort. Whether that's worth it depends on per-item effort as much as item count: even 5-10 items can be worth scripting if each requires several steps by hand. First inspect one example page/item manually to work out the recipe. Iterating on it - try it, look at what came back, fix it, try again - against that one example before applying it to the rest is expected; don't expect to get it right in one attempt.

First, a fork for the screenshot-only case (nothing else, no data pulled out of the page): which of run_script or queue_screenshots is better here is a question of scale, not just "screenshot only" by itself. For a small-to-medium batch - tens of items, not hundreds - run_script's simplicity is the better trade: it blocks until done and hands you one clean result, and for a batch that finishes in well under a minute that wait costs nothing real. Reserve queue_screenshots for genuinely large batches - roughly 100+ items - where run_script's blocking wait would mean sitting on one long call with no visibility into progress, and where being able to watch it happen live and keep working in the meantime actually earns its extra moving parts (a queue, a live panel, separate status checks). Keep reading here for run_script; skip to queue_screenshots below once a batch is large enough that this trade flips.

Default to run_script for get_page_text/execute_js data extraction: it replays a fixed sequence of steps (same action names as this tool - navigate, screenshot, get_page_text, execute_js, wait, scroll_to, network_list_types, network_list, network_inspect) once per item, over a pool of pages from your current session (inherits cookies/login), with no LLM cost per item. Whether it fetches or renders falls out of what you put in the script, not a separate choice: a script with no navigate step fetches each item as a URL via raw HTTP (fast, but only sees server-rendered HTML, not content that appears after client-side JS runs - same mechanism and same limitation as batch_extract, since that's literally what this is under the hood, and in fetch mode scroll_to/network_* aren't available - no live page to act on); a script that starts with navigate gets a real rendered page per item, so screenshot/get_page_text/scroll_to/network_*/JS that depends on client-rendering all work. scroll_to's text must be literal text that actually appears on the page (a substring match, not a description) - same semantics as queue_screenshots' scroll_to below. network_list_types/network_list/network_inspect work exactly like this tool's own network actions, but scoped to that one item's page and reusable across hundreds of items in one call - the natural way to pull the JSON-endpoint case (see above) out to scale instead of doing it one page at a time. batch_extract still exists as a narrower standalone tool for the fetch-only case if you prefer it, but run_script covers everything it does plus the render+screenshot+network-inspection case in one place. The browser tool's execute_js fetch() + Promise.all is also an option for same-origin bulk calls against the CURRENT page specifically (not run_script's per-item pages), but depends on that page's own context in ways that can silently break (fetch() has been observed failing outright - a hard "Failed to fetch" on every request - when run from a page a browser renders specially, like a raw JSON API response). None of these tools support real per-item interaction or judgment (forms, multi-step flows, deciding what to click) - that's still dispatch_subagents.

queue_screenshots is for the large-scale screenshot-only case above (roughly 100+ items) - drop in a list of items (each a short sequence of navigate/scroll_to/screenshot steps) and it returns immediately with queued item ids, it does not wait for them, unlike run_script which blocks until every item is done. Items run concurrently in the background (up to max_fanout, default 8) while you keep working on other things, AND the user can watch each one live in a panel next to the chat as it happens - something run_script has no way to offer, since it only reports back once everything has already finished; that live visibility, and not sitting on one huge blocking call, is specifically why this is worth its extra moving parts at this scale, and specifically why it's not worth reaching for below it. If a batch is going to hit many pages on the SAME site/domain, raise min_interval_s (seconds between starting successive items, default 0.5 - a real batch against several distinct sites got one of them to start rate-limiting the whole run once, at that same unthrottled pace) - max_fanout alone doesn't prevent this, it only caps how many run at once, not how fast new ones start as slots free up; a batch spread across unrelated domains doesn't need this raised. Note that scroll_to's target must be literal text that actually appears on the page (it's a substring match, not a description) - if you don't know the page's structure well enough to name a literal element by its real text, inspect one example item first (same as run_script's own advice) rather than guessing a vague description like "the top headline." Call queue_status when you want to check progress or read results yourself (e.g. before referencing a screenshot in your final message). queue_pause/queue_resume/queue_clear manage the queue itself. Still screenshots only, not data extraction (use run_script/batch_extract for that) and not interaction (use dispatch_subagents for that).

When you need to hand the user a deliverable file (a CSV, a report, extracted data), use the save_file tool. Don't try to trigger a browser download via execute_js - that saves inside the automated browser's own environment, not somewhere the user can find it.

The user only reads your final message - they do not scroll back up through the tool calls in between. Whatever deliverable your task actually produced (a screenshot, a saved file, anything else with a filename on disk), name its exact filename in your final message instead of referring to it vaguely (e.g. "the screenshot above" or "the file in the sidebar") - the app detects filenames you mention and surfaces them automatically, but only if you name them in your last message.

Before treating a specific element as evidence for a finding (e.g. "this is the citation/link/row that answers the question"), you can use outline(ref) to draw a bounding box around it and screenshot it - a fast visual gut-check that you landed on the right element and not a neighbor with similar text or structure. This is a visual aid, not a substitute for verify_finding below. Note: this tool draws a box - it does not search for or interact with anything a page itself calls "highlighted" (e.g. a Wikipedia page's own highlighted-section behavior after following a citation). If a task's own wording uses "highlight", treat that as the page's behavior to locate via the DOM, not a cue to reach for this tool.

Before presenting a finding as fact, check whether it's ALL THREE of: specific (a discrete, falsifiable value - a number, name, date, identifier - not a vague summary), checkable (there's a concrete way to confirm it independently of how you found it), and consequential (getting it wrong would make your final answer or deliverable wrong). Only when all three hold, call verify_finding to get an independent, fresh re-derivation before relying on it - a self-review in the same context tends to just confirm your own assumptions, since it's reasoning from the same trail that produced the answer; a fresh session with no visibility into your reasoning can actually catch it being wrong. This budget is small (only {MAX_VERIFICATIONS_PER_SESSION} uses per session) - spend it on the claim(s) that matter most to the final answer, not routine intermediate steps. If verification contradicts your finding, trust the independent result and redo the affected work. If you're out of budget for a claim that still needs checking, say so explicitly in your final answer rather than presenting it as verified.
</TOOL_GUIDANCE>

<TIPS>
* Prefer get_page_text over scrolling when looking for information - it's faster and more reliable
* Use screenshot with full_page: true to capture an entire page in one image instead of scrolling repeatedly
* Use execute_js to extract data from JavaScript variables, localStorage, or trigger behaviors not accessible through clicks
* Use full URLs with https://
* Use wait for slow-loading pages
* Use form_input with refs for form fields
* Use key for shortcuts (e.g., "ctrl+a")
* Close popups when they appear
* Verify actions succeeded before moving on
</TIPS>"""


def _combined_pre_tool_use_hook(run_logger: RunLogger, guardrail_policy: GuardrailPolicy):
    """One PreToolUse hook function that does both jobs in a fixed order:
    always log the call (run_logger), then let the guardrail decide whether
    it's allowed - see the "PreToolUse" hooks= comment in build_options for
    why this is one matcher instead of two."""

    async def _hook(input_data: dict, tool_use_id, context) -> dict:
        await run_logger.on_pre_tool_use(input_data, tool_use_id, context)
        return await guardrail_policy.on_pre_tool_use(input_data, tool_use_id, context)

    return _hook


def build_options(
    *,
    model: str,
    system_prompt_suffix: str,
    browser_tool: BrowserTool,
    file_output_tool: FileOutputTool,
    run_logger: RunLogger,
    api_key: str,
    max_turns: int = 200,
    max_budget_usd: Optional[float] = None,
    guardrail_policy: Optional[GuardrailPolicy] = None,
    sub_browser_queue: Optional[SubBrowserQueue] = None,
) -> ClaudeAgentOptions:
    """Build the SDK options for a browser-automation session.

    guardrail_policy: pass the caller's own GuardrailPolicy instance (see
    guardrails.py) if it wants to own restriction-mode state across
    reconnects (Streamlit does - it keeps one instance in session_state and
    flips `.mode` in place from the sidebar toggle, with no client reconnect
    needed for that to take effect). Defaults to a fresh "none" policy -
    i.e. no behavior change for any caller that doesn't pass one.

    sub_browser_queue: same reasoning as guardrail_policy - pass the
    caller's own SubBrowserQueue (see tools/sub_browser_queue.py) so pending/
    completed items and the fanout setting survive a reconnect and stay
    readable by the caller's own UI (Streamlit's sidebar panel reads the
    same instance's .snapshot()). Defaults to a fresh, empty queue.
    """
    system_prompt = BROWSER_SYSTEM_PROMPT
    if system_prompt_suffix:
        system_prompt = f"{system_prompt} {system_prompt_suffix}"
    guardrail_policy = guardrail_policy or GuardrailPolicy(
        mode="none", browser_tool=browser_tool, run_logger=run_logger
    )
    sub_browser_queue = sub_browser_queue or SubBrowserQueue(
        browser_tool=browser_tool, run_dir=browser_tool.run_dir
    )

    dispatch_subagents_tool = DispatchSubagentsTool(
        run_dir=browser_tool.run_dir,
        run_logger=run_logger,
        api_key=api_key,
        model=resolve_model(SUBAGENT_MODEL, model),
        base_system_prompt=BROWSER_SYSTEM_PROMPT,
    )
    batch_extract_tool = BatchExtractTool(
        browser_tool=browser_tool,
        file_output_tool=file_output_tool,
    )
    script_runner_tool = ScriptRunnerTool(
        browser_tool=browser_tool,
        file_output_tool=file_output_tool,
    )
    verify_finding_tool = VerifyFindingTool(
        run_dir=browser_tool.run_dir,
        run_logger=run_logger,
        api_key=api_key,
        model=resolve_model(VERIFY_MODEL, model),
        base_system_prompt=BROWSER_SYSTEM_PROMPT,
    )
    server = build_mcp_server(
        browser_tool,
        file_output_tool,
        extra_tools=[
            build_dispatch_subagents_tool_fn(dispatch_subagents_tool),
            build_batch_extract_tool_fn(batch_extract_tool),
            build_run_script_tool_fn(script_runner_tool),
            build_verify_finding_tool_fn(verify_finding_tool),
            build_queue_screenshots_tool_fn(sub_browser_queue),
            build_queue_status_tool_fn(sub_browser_queue),
            build_queue_clear_tool_fn(sub_browser_queue),
            build_queue_pause_tool_fn(sub_browser_queue),
            build_queue_resume_tool_fn(sub_browser_queue),
        ],
    )

    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers={"browser_use": server},
        # `tools=[]` disables the SDK's full built-in Claude Code toolset
        # (Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch, ...) -
        # `allowed_tools` alone does NOT do this, it only controls which
        # tools skip the permission prompt, not which tools exist (see the
        # SDK's own ClaudeAgentOptions.tools docstring). Without this, a real
        # run showed the model reaching for Bash/Read directly (curl-ing a
        # URL, cat-ing a saved tool-result file) when a browser action hit
        # friction - a much bigger capability surface than "browser
        # automation" was ever meant to have, and one that bypasses every
        # safeguard built around the MCP browser tools (rate limiting,
        # verify_finding, DOM-diff, run_logger's hooks).
        tools=[],
        allowed_tools=[
            "mcp__browser_use__browser",
            "mcp__browser_use__save_file",
            "mcp__browser_use__dispatch_subagents",
            "mcp__browser_use__batch_extract",
            "mcp__browser_use__run_script",
            "mcp__browser_use__verify_finding",
            "mcp__browser_use__queue_screenshots",
            "mcp__browser_use__queue_status",
            "mcp__browser_use__queue_clear",
            "mcp__browser_use__queue_pause",
            "mcp__browser_use__queue_resume",
        ],
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        max_buffer_size=MAX_BUFFER_SIZE,
        env={"ANTHROPIC_API_KEY": api_key},
        hooks={
            "UserPromptSubmit": [HookMatcher(hooks=[run_logger.on_user_prompt_submit])],
            # One matcher, not two, so the guardrail decision doesn't depend
            # on how the SDK/CLI combines multiple PreToolUse hooks on the
            # same event (undocumented in the Python SDK) - logging always
            # runs first, then the guardrail's allow/deny is the one that's
            # actually returned.
            "PreToolUse": [
                HookMatcher(
                    hooks=[
                        _combined_pre_tool_use_hook(run_logger, guardrail_policy)
                    ]
                )
            ],
            "PostToolUse": [HookMatcher(hooks=[run_logger.on_post_tool_use])],
            "PostToolUseFailure": [HookMatcher(hooks=[run_logger.on_post_tool_use_failure])],
            "Stop": [HookMatcher(hooks=[run_logger.on_stop])],
            "SubagentStart": [HookMatcher(hooks=[run_logger.on_subagent_start])],
            "SubagentStop": [HookMatcher(hooks=[run_logger.on_subagent_stop])],
        },
    )
