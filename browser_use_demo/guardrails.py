"""Deterministic guardrail checks for elevated-credential agent runs.

The problem this addresses: an operator gives the agent their own full-access
credentials so it can act as them, and needs some way to keep it from doing
something destructive/out-of-scope with that access. Two mechanisms were
deliberately ruled out - see TODO.md's write-up for the reasoning:

- A hand-built policy taxonomy (per-site/per-domain rules) - expensive to
  build up front, and still imperfect once built.
- A separate LLM classifier gating every action - nondeterministic, costs a
  real model call per gated action, and "the classifier said it was fine" is
  a weak answer for an audit-focused product compared to "this violated an
  explicit rule" or "a human signed off."

What's here instead: a small, fixed set of regex/keyword checks over the
*structured* tool call the model already produced (the action name and its
params - and, for click/form_input, the last-known DOM snapshot's text for
the targeted ref, so the check can see an element's accessible name/role
without re-deriving it). Cheap, deterministic, fully inspectable - a blocked
call always has a one-line "this exact pattern matched" answer. Not meant to
catch everything; meant to catch the tail, on the assumption (explicitly the
operator's own premise) that the agent is otherwise reasonably aligned.

Wired in as a PreToolUse hook via GuardrailPolicy, constructed once per
Streamlit session and passed into loop.build_options. Mutating `.mode` in
place (from the sidebar toggle) takes effect on the very next tool call - no
client reconnect needed, unlike model/max_turns changes.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

RestrictionMode = Literal["none", "all", "manual"]

_BROWSER_TOOL_NAME = "mcp__browser_use__browser"

# execute_js: flagged for a non-GET fetch/XHR call or a form .submit() - a
# JS-level write, not just reading the page.
_JS_DANGER_RE = re.compile(
    r"""fetch\s*\([^)]*method\s*:\s*['"](?:POST|PUT|PATCH|DELETE)['"]"""
    r"""|new\s+XMLHttpRequest|\.open\(\s*['"](?:POST|PUT|PATCH|DELETE)['"]"""
    r"""|\.submit\s*\(\s*\)""",
    re.IGNORECASE,
)

# navigate: flagged when the target URL's path itself looks destructive.
_NAVIGATE_DANGER_RE = re.compile(
    r"/(delete|remove|deactivate|unsubscribe|cancel)[-_/]", re.IGNORECASE
)

# left_click/right_click/middle_click/double_click/triple_click/form_input:
# flagged when the targeted element's accessible name/role looks
# destructive. Checked against the DOM snapshot line for that ref (falls
# back to the raw `text` param - e.g. click modifier keys - if no snapshot
# context is available, which just means this rule under-fires rather than
# raising on missing context).
_TARGETED_ACTIONS = frozenset(
    {"left_click", "right_click", "middle_click", "double_click", "triple_click", "form_input"}
)
_DANGEROUS_LABEL_RE = re.compile(
    r"delete|remove|deactivate|unsubscribe|permanently"
    r"|cancel (my |the )?(subscription|account|order|membership)"
    r"|confirm purchase|place order|checkout|pay now|submit payment"
    r"|transfer funds|wire transfer",
    re.IGNORECASE,
)


@dataclass
class GuardrailMatch:
    rule: str
    reason: str


def _ref_context(dom_snapshot: Optional[str], ref: Optional[str]) -> str:
    """The line(s) of the last DOM snapshot mentioning this ref, so a
    click/form_input target's accessible name/role can be checked even
    though the tool call itself only carries an opaque ref id. Best-effort:
    an empty result (no snapshot yet, or the ref isn't in it) just means the
    label check has nothing to match against, not an error."""
    if not dom_snapshot or not ref:
        return ""
    marker = f"[ref={ref}]"
    return "\n".join(line for line in dom_snapshot.splitlines() if marker in line)


def check_tool_call(
    tool_name: str, tool_input: dict[str, Any], dom_snapshot: Optional[str]
) -> Optional[GuardrailMatch]:
    """Deterministic pattern check - see module docstring. Returns None when
    nothing matched (the overwhelmingly common case)."""
    if tool_name != _BROWSER_TOOL_NAME:
        return None

    action = tool_input.get("action")
    text = tool_input.get("text") or ""

    if action == "execute_js" and _JS_DANGER_RE.search(text):
        return GuardrailMatch(
            rule="non_get_js_request",
            reason="execute_js contains a non-GET fetch/XHR call or a form .submit()",
        )

    if action == "navigate" and _NAVIGATE_DANGER_RE.search(text):
        return GuardrailMatch(
            rule="destructive_url_path",
            reason=f"navigate target's path looks destructive: {text!r}",
        )

    if action in _TARGETED_ACTIONS:
        context = _ref_context(dom_snapshot, tool_input.get("ref")) or text
        if context and _DANGEROUS_LABEL_RE.search(context):
            return GuardrailMatch(
                rule="destructive_target_label",
                reason=f"{action} targets an element whose label looks destructive: {context.strip()[:200]!r}",
            )

    return None


def fingerprint(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Stable id for 'the same call' - used by manual mode's approve-once
    bypass to recognize a retried action. Canonicalized JSON so key order in
    tool_input doesn't change the fingerprint."""
    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class GuardrailPolicy:
    """PreToolUse hook: runs check_tool_call, gated by `mode`.

    - "none": never intervenes - current/default behavior, zero overhead.
    - "all": any match is denied outright, every time.
    - "manual": any match is denied UNLESS its fingerprint is already in
      `approved`, in which case it's let through once (single-use - popped
      on the way through, so a later repeat of that same call needs
      re-approval, not a standing bypass). A denied call is also appended to
      `pending` for the UI to render and let a human approve.

    `browser_tool` is read (never mutated) purely for `_last_dom_snapshot`,
    to resolve a click/form_input ref's accessible name/role at check time.
    `pending`/`approved` are plain containers the caller (Streamlit) owns
    and renders/mutates directly from the sidebar - this class only appends
    to `pending` and pops from `approved`, so approval doesn't need any
    callback wired back into the hook itself.
    """

    mode: RestrictionMode
    browser_tool: Any  # BrowserTool - typed loosely to avoid an import cycle
    run_logger: Any = None  # RunLogger - optional, for the audit-log entry
    pending: list[dict[str, Any]] = field(default_factory=list)
    approved: set[str] = field(default_factory=set)

    async def on_pre_tool_use(self, input_data: dict, tool_use_id, context) -> dict:
        if self.mode == "none":
            return {}

        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input") or {}
        match = check_tool_call(tool_name, tool_input, getattr(self.browser_tool, "_last_dom_snapshot", None))
        if match is None:
            return {}

        fp = fingerprint(tool_name, tool_input)
        if self.mode == "manual" and fp in self.approved:
            self.approved.discard(fp)
            if self.run_logger is not None:
                await self.run_logger.log_guardrail_block(
                    mode=self.mode, rule=match.rule, reason=match.reason,
                    tool_name=tool_name, tool_input=tool_input, allowed=True,
                )
            return {}

        if self.mode == "manual":
            self.pending.append(
                {
                    "fingerprint": fp,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "rule": match.rule,
                    "reason": match.reason,
                }
            )
            reason = (
                f"Blocked by guardrail ({match.rule}): {match.reason}. This action "
                "needs manual approval - tell the user it's waiting for approval "
                "in the app, and don't attempt a workaround. If they approve it, "
                "retry this exact same action."
            )
        else:  # "all"
            reason = f"Blocked by guardrail ({match.rule}): {match.reason}"

        if self.run_logger is not None:
            await self.run_logger.log_guardrail_block(
                mode=self.mode, rule=match.rule, reason=match.reason,
                tool_name=tool_name, tool_input=tool_input, allowed=False,
            )

        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
