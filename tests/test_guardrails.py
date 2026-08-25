"""Tests for the guardrails module - deterministic pattern matching over
tool calls (check_tool_call/fingerprint) and the PreToolUse hook that gates
on it (GuardrailPolicy), across all three restriction modes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_use_demo.guardrails import GuardrailPolicy, check_tool_call, fingerprint

BROWSER_TOOL = "mcp__browser_use__browser"

DOM_SNAPSHOT = (
    '- button "Delete Account" [ref=ref_9] id="delete-btn"\n'
    '- button "Save Changes" [ref=ref_10] id="save-btn"\n'
    '- link "Learn more" [ref=ref_11] href="/about"\n'
)


class TestCheckToolCall:
    def test_ignores_other_tools(self):
        assert check_tool_call("mcp__browser_use__save_file", {"action": "delete"}, None) is None

    def test_plain_navigate_is_fine(self):
        assert check_tool_call(BROWSER_TOOL, {"action": "navigate", "text": "https://example.com/profile"}, None) is None

    def test_navigate_to_destructive_path_matches(self):
        match = check_tool_call(
            BROWSER_TOOL, {"action": "navigate", "text": "https://example.com/account/delete-confirm"}, None
        )
        assert match is not None
        assert match.rule == "destructive_url_path"

    def test_execute_js_get_fetch_is_fine(self):
        js = "fetch('/api/items').then(r => r.json())"
        assert check_tool_call(BROWSER_TOOL, {"action": "execute_js", "text": js}, None) is None

    def test_execute_js_post_fetch_matches(self):
        js = "fetch('/api/account', {method: 'POST', body: '{}'})"
        match = check_tool_call(BROWSER_TOOL, {"action": "execute_js", "text": js}, None)
        assert match is not None
        assert match.rule == "non_get_js_request"

    def test_execute_js_delete_fetch_matches(self):
        js = 'fetch("/api/account", {method: "DELETE"})'
        match = check_tool_call(BROWSER_TOOL, {"action": "execute_js", "text": js}, None)
        assert match is not None

    def test_execute_js_xhr_matches(self):
        js = "var x = new XMLHttpRequest(); x.open('POST', '/submit');"
        match = check_tool_call(BROWSER_TOOL, {"action": "execute_js", "text": js}, None)
        assert match is not None

    def test_execute_js_form_submit_matches(self):
        js = "document.getElementById('checkout-form').submit()"
        match = check_tool_call(BROWSER_TOOL, {"action": "execute_js", "text": js}, None)
        assert match is not None

    def test_click_on_dangerous_labeled_ref_matches(self):
        match = check_tool_call(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"}, DOM_SNAPSHOT)
        assert match is not None
        assert match.rule == "destructive_target_label"

    def test_click_on_benign_labeled_ref_is_fine(self):
        assert check_tool_call(BROWSER_TOOL, {"action": "left_click", "ref": "ref_10"}, DOM_SNAPSHOT) is None

    def test_click_with_no_dom_snapshot_is_fine(self):
        # No snapshot yet (e.g. first action of the session) - nothing to
        # check the label against, so this rule simply doesn't fire rather
        # than raising or false-positiving.
        assert check_tool_call(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"}, None) is None

    def test_click_on_ref_not_in_snapshot_is_fine(self):
        assert check_tool_call(BROWSER_TOOL, {"action": "left_click", "ref": "ref_999"}, DOM_SNAPSHOT) is None

    def test_form_input_on_dangerous_labeled_ref_matches(self):
        match = check_tool_call(
            BROWSER_TOOL, {"action": "form_input", "ref": "ref_9", "value": "yes"}, DOM_SNAPSHOT
        )
        assert match is not None

    def test_type_action_never_checked_against_labels(self):
        # `type` has no `ref` param (types into whatever's focused) - the
        # targeted-label rule only applies to ref-bearing actions.
        assert check_tool_call(BROWSER_TOOL, {"action": "type", "text": "delete everything"}, DOM_SNAPSHOT) is None


class TestFingerprint:
    def test_same_call_same_fingerprint(self):
        a = fingerprint(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"})
        b = fingerprint(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"})
        assert a == b

    def test_key_order_does_not_matter(self):
        a = fingerprint(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"})
        b = fingerprint(BROWSER_TOOL, {"ref": "ref_9", "action": "left_click"})
        assert a == b

    def test_different_input_different_fingerprint(self):
        a = fingerprint(BROWSER_TOOL, {"action": "left_click", "ref": "ref_9"})
        b = fingerprint(BROWSER_TOOL, {"action": "left_click", "ref": "ref_10"})
        assert a != b


def make_policy(mode: str, dom_snapshot=None, run_logger=None) -> GuardrailPolicy:
    browser_tool = MagicMock()
    browser_tool._last_dom_snapshot = dom_snapshot
    return GuardrailPolicy(mode=mode, browser_tool=browser_tool, run_logger=run_logger)


def pre_tool_input(action="left_click", ref="ref_9") -> dict:
    return {
        "tool_name": BROWSER_TOOL,
        "tool_use_id": "toolu_1",
        "tool_input": {"action": action, "ref": ref},
    }


class TestGuardrailPolicyNoneMode:
    @pytest.mark.asyncio
    async def test_never_intervenes_even_on_a_dangerous_call(self):
        policy = make_policy("none", dom_snapshot=DOM_SNAPSHOT)
        result = await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        assert result == {}
        assert policy.pending == []


class TestGuardrailPolicyAllMode:
    @pytest.mark.asyncio
    async def test_safe_call_passes_through(self):
        policy = make_policy("all", dom_snapshot=DOM_SNAPSHOT)
        result = await policy.on_pre_tool_use(pre_tool_input(ref="ref_10"), "toolu_1", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_dangerous_call_denied_every_time(self):
        policy = make_policy("all", dom_snapshot=DOM_SNAPSHOT)
        for _ in range(2):
            result = await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
            assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        # "all" mode never queues anything for approval - there's nothing to
        # approve, it's a hard block.
        assert policy.pending == []

    @pytest.mark.asyncio
    async def test_logs_the_block(self):
        run_logger = MagicMock()
        run_logger.log_guardrail_block = AsyncMock()
        policy = make_policy("all", dom_snapshot=DOM_SNAPSHOT, run_logger=run_logger)
        await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        run_logger.log_guardrail_block.assert_awaited_once()
        kwargs = run_logger.log_guardrail_block.await_args.kwargs
        assert kwargs["mode"] == "all"
        assert kwargs["allowed"] is False
        assert kwargs["rule"] == "destructive_target_label"


class TestGuardrailPolicyManualMode:
    @pytest.mark.asyncio
    async def test_dangerous_call_denied_and_queued(self):
        policy = make_policy("manual", dom_snapshot=DOM_SNAPSHOT)
        result = await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert len(policy.pending) == 1
        assert policy.pending[0]["rule"] == "destructive_target_label"
        assert policy.pending[0]["tool_input"] == {"action": "left_click", "ref": "ref_9"}

    @pytest.mark.asyncio
    async def test_approved_fingerprint_lets_the_retry_through(self):
        policy = make_policy("manual", dom_snapshot=DOM_SNAPSHOT)
        await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        fp = policy.pending[0]["fingerprint"]

        policy.approved.add(fp)
        result = await policy.on_pre_tool_use(pre_tool_input(), "toolu_2", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_approval_is_single_use(self):
        policy = make_policy("manual", dom_snapshot=DOM_SNAPSHOT)
        await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        fp = policy.pending[0]["fingerprint"]
        policy.approved.add(fp)

        # First retry after approval: allowed, and the approval is consumed.
        first = await policy.on_pre_tool_use(pre_tool_input(), "toolu_2", {})
        assert first == {}
        assert fp not in policy.approved

        # A second attempt of the exact same call needs a fresh approval.
        second = await policy.on_pre_tool_use(pre_tool_input(), "toolu_3", {})
        assert second["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_safe_call_never_queued(self):
        policy = make_policy("manual", dom_snapshot=DOM_SNAPSHOT)
        result = await policy.on_pre_tool_use(pre_tool_input(ref="ref_10"), "toolu_1", {})
        assert result == {}
        assert policy.pending == []

    @pytest.mark.asyncio
    async def test_logs_both_the_block_and_the_approved_retry(self):
        run_logger = MagicMock()
        run_logger.log_guardrail_block = AsyncMock()
        policy = make_policy("manual", dom_snapshot=DOM_SNAPSHOT, run_logger=run_logger)

        await policy.on_pre_tool_use(pre_tool_input(), "toolu_1", {})
        fp = policy.pending[0]["fingerprint"]
        policy.approved.add(fp)
        await policy.on_pre_tool_use(pre_tool_input(), "toolu_2", {})

        assert run_logger.log_guardrail_block.await_count == 2
        first_call, second_call = run_logger.log_guardrail_block.await_args_list
        assert first_call.kwargs["allowed"] is False
        assert second_call.kwargs["allowed"] is True
