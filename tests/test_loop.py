"""Tests for loop.build_options - the coordinator's ClaudeAgentOptions
construction."""

from unittest.mock import MagicMock, patch

from browser_use_demo.agent_sdk_bridge import MAX_BUFFER_SIZE
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
            "mcp__browser_use__verify_finding",
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

    def test_prompt_defaults_to_batch_extract_for_concurrency(self):
        # Regression guard: a real run's hand-written execute_js fetch()
        # loop failed outright ("Failed to fetch" on every request) from a
        # page context that doesn't tolerate arbitrary fetches - batch_extract
        # (isolated from page context) should be the stated default, not a
        # same-origin-only alternative to a manual fetch loop.
        assert "Default to batch_extract for concurrent multi-URL fetching" in BROWSER_SYSTEM_PROMPT

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
