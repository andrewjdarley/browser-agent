"""Tests for loop.build_options - the coordinator's ClaudeAgentOptions
construction."""

from unittest.mock import MagicMock, patch

import pytest

from browser_use_demo.agent_sdk_bridge import MAX_BUFFER_SIZE
from browser_use_demo.guardrails import GuardrailPolicy
from browser_use_demo.loop import BROWSER_SYSTEM_PROMPT, build_options
from browser_use_demo.run_logger import RunLogger
from browser_use_demo.tools import BrowserTool, FileOutputTool


def make_options(tmp_path, **overrides):
    kwargs = dict(
        model="claude-sonnet-4-5-20250929",
        system_prompt_suffix="",
        browser_tool=BrowserTool(run_dir=tmp_path),
        file_output_tool=FileOutputTool(run_dir=tmp_path),
        run_logger=RunLogger(tmp_path),
        api_key="test-key",
    )
    kwargs.update(overrides)
    return build_options(**kwargs)


class TestBuildOptions:
    def test_sets_generous_max_buffer_size(self, tmp_path):
        # Regression test: the SDK's stdio transport defaults to a 1 MiB
        # buffer, which a base64-encoded full_page screenshot of a
        # content-heavy page can exceed, killing the whole session (not
        # just one tool call). Confirmed against a real crash log - see
        # agent_sdk_bridge.MAX_BUFFER_SIZE's docstring/comment.
        options = make_options(tmp_path)
        assert options.max_buffer_size == MAX_BUFFER_SIZE
        assert options.max_buffer_size > 1024 * 1024

    def test_registers_all_tools(self, tmp_path):
        options = make_options(tmp_path)
        assert options.allowed_tools == [
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
        ]

    def test_disables_built_in_claude_code_toolset(self, tmp_path):
        # Regression test: allowed_tools alone doesn't restrict which tools
        # exist (a real run showed the model using Bash/Read directly - the
        # SDK's full built-in toolset was available the whole time since
        # `tools` was never set). `tools=[]` is the field that actually
        # disables it, per ClaudeAgentOptions.tools' own docstring.
        options = make_options(tmp_path)
        assert options.tools == []

    def test_api_key_passed_via_env_not_globally(self, tmp_path):
        options = make_options(tmp_path, api_key="sk-secret")
        assert options.env["ANTHROPIC_API_KEY"] == "sk-secret"

    def test_system_prompt_suffix_appended(self, tmp_path):
        options = make_options(tmp_path, system_prompt_suffix="Extra instructions.")
        assert options.system_prompt.startswith(BROWSER_SYSTEM_PROMPT)
        assert options.system_prompt.endswith("Extra instructions.")

    def test_empty_suffix_leaves_prompt_unchanged(self, tmp_path):
        options = make_options(tmp_path, system_prompt_suffix="")
        assert options.system_prompt == BROWSER_SYSTEM_PROMPT

    def test_prompt_uses_renamed_outline_action_not_highlight(self):
        # Regression guard for the highlight->outline rename (a real run
        # showed the model conflating our tool's "highlight" action with a
        # task's own use of that word) - the prompt should only reference
        # the new name as a tool invocation.
        assert "outline(ref)" in BROWSER_SYSTEM_PROMPT
        assert "highlight(ref)" not in BROWSER_SYSTEM_PROMPT

    def test_prompt_documents_automatic_dom_context(self):
        # The DOM-first rewrite should explain the automatic full-tree/diff
        # push, not just assert "use the DOM" as an unexplained mandate.
        assert "diff against what you last saw" in BROWSER_SYSTEM_PROMPT
        assert "Never click by raw (x, y) coordinate" in BROWSER_SYSTEM_PROMPT

    def test_prompt_warns_about_bare_return_in_execute_js(self):
        # Regression guard: three separate real runs hit the same JS syntax
        # error (a top-level `return` outside a function) in execute_js.
        assert "immediately-invoked function" in BROWSER_SYSTEM_PROMPT
        assert "top-level `return`" in BROWSER_SYSTEM_PROMPT

    def test_prompt_defaults_to_run_script_for_scripted_tasks(self):
        # Regression guard: a real run's hand-written execute_js fetch()
        # loop failed outright ("Failed to fetch" on every request) from a
        # page context that doesn't tolerate arbitrary fetches - a tool
        # isolated from page context (now run_script, which subsumes
        # batch_extract's own mechanism as its no-navigate fast path) should
        # be the stated default, not a same-origin-only manual fetch loop.
        assert "Default to run_script for get_page_text/execute_js data extraction" in BROWSER_SYSTEM_PROMPT
        # batch_extract still exists and is still mentioned as a narrower
        # standalone alternative, not silently dropped from the guidance.
        assert "batch_extract" in BROWSER_SYSTEM_PROMPT

    def test_prompt_forks_run_script_vs_queue_screenshots_on_scale(self):
        # Regression guard: the prompt must fork run_script vs
        # queue_screenshots on batch SCALE (roughly 100+ items), not just
        # "screenshot-only, always prefer queue_screenshots" - a small batch
        # doesn't need the extra moving parts (a queue, a live panel).
        assert "roughly 100+ items" in BROWSER_SYSTEM_PROMPT
        assert "run_script's simplicity is the better trade" in BROWSER_SYSTEM_PROMPT
        # The scale fork must appear before run_script's own pitch, not
        # after - otherwise the model anchors on run_script's default
        # framing first, the same failure mode as the original miss.
        assert BROWSER_SYSTEM_PROMPT.index("roughly 100+ items") < BROWSER_SYSTEM_PROMPT.index(
            "Default to run_script for get_page_text/execute_js data extraction"
        )
        assert "batch_extract" in BROWSER_SYSTEM_PROMPT

    def test_hooks_registered_for_run_logger(self, tmp_path):
        options = make_options(tmp_path)
        assert set(options.hooks.keys()) == {
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SubagentStart",
            "SubagentStop",
        }

    @pytest.mark.asyncio
    async def test_default_guardrail_policy_is_none_mode(self, tmp_path):
        # No guardrail_policy passed in -> build_options' own default ("none")
        # applies, so a call that would otherwise match a rule still goes
        # through - no behavior change for a caller that doesn't opt in.
        options = make_options(tmp_path)
        pre_tool_use_hook = options.hooks["PreToolUse"][0].hooks[0]
        result = await pre_tool_use_hook(
            {
                "tool_name": "mcp__browser_use__browser",
                "tool_input": {"action": "navigate", "text": "https://example.com/delete-account"},
            },
            "toolu_1",
            {},
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_passed_in_guardrail_policy_can_deny(self, tmp_path):
        browser_tool = BrowserTool(run_dir=tmp_path)
        policy = GuardrailPolicy(mode="all", browser_tool=browser_tool)
        options = make_options(tmp_path, browser_tool=browser_tool, guardrail_policy=policy)
        pre_tool_use_hook = options.hooks["PreToolUse"][0].hooks[0]
        result = await pre_tool_use_hook(
            {
                "tool_name": "mcp__browser_use__browser",
                "tool_input": {"action": "navigate", "text": "https://example.com/delete-account"},
            },
            "toolu_1",
            {},
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestModelResolution:
    """Subagent/verify workers default to the coordinator's own model
    (previously the only option); model_config's SUBAGENT_MODEL/VERIFY_MODEL
    env overrides let that be pinned independently - see test_model_config.py
    for the override-resolution logic itself."""

    def test_subagent_and_verify_default_to_coordinator_model(self, tmp_path):
        with patch("browser_use_demo.loop.DispatchSubagentsTool") as MockDispatch, patch(
            "browser_use_demo.loop.VerifyFindingTool"
        ) as MockVerify, patch("browser_use_demo.loop.build_dispatch_subagents_tool_fn"), patch(
            "browser_use_demo.loop.build_verify_finding_tool_fn"
        ), patch("browser_use_demo.loop.build_mcp_server"):
            make_options(tmp_path, model="claude-opus-5")

        assert MockDispatch.call_args.kwargs["model"] == "claude-opus-5"
        assert MockVerify.call_args.kwargs["model"] == "claude-opus-5"
