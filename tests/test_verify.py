"""Tests for VerifyFindingTool: an independent, fresh session re-derives one
claim - session-wide budget enforcement, isolation from dispatch/recursion,
and confirmed/contradicted result handling - without real API calls."""

import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from browser_use_demo.run_logger import RunLogger
from browser_use_demo.tools.verify import MAX_VERIFICATIONS_PER_SESSION, VerifyFindingTool


@contextlib.contextmanager
def patched_worker_deps():
    """Same reasoning as test_subagent.py's helper: no real Playwright
    browser should ever be launched, and cleanup() must be awaitable."""
    fake_browser_tool = MagicMock()
    fake_browser_tool.cleanup = AsyncMock()
    with patch(
        "browser_use_demo.tools.verify.BrowserTool", return_value=fake_browser_tool
    ), patch("browser_use_demo.tools.verify.FileOutputTool"), patch(
        "browser_use_demo.tools.verify.build_mcp_server", return_value=MagicMock()
    ):
        yield


def make_fake_client_cls(response_text):
    constructed_options = []

    class FakeClient:
        def __init__(self, options=None):
            self.options = options
            constructed_options.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, prompt):
            self._prompt = prompt

        async def receive_response(self):
            yield AssistantMessage(content=[TextBlock(text=response_text)], model="test-model")

    return FakeClient, constructed_options


@pytest.fixture
def run_logger(tmp_path):
    return RunLogger(tmp_path)


def make_verify_tool(tmp_path, run_logger):
    return VerifyFindingTool(
        run_dir=tmp_path,
        run_logger=run_logger,
        api_key="test-key",
        model="claude-haiku-4-5",
        base_system_prompt="<BASE PROMPT>",
    )


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_allows_up_to_the_cap(self, tmp_path, run_logger):
        text = '<result>{"confirmed": true, "actual_finding": "x", "explanation": "y"}</result>'
        FakeClient, _ = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            for _ in range(MAX_VERIFICATIONS_PER_SESSION):
                result = await tool(claim="x", context="y")
                assert result.error is None

    @pytest.mark.asyncio
    async def test_refuses_once_budget_exhausted(self, tmp_path, run_logger):
        text = '<result>{"confirmed": true, "actual_finding": "x", "explanation": "y"}</result>'
        FakeClient, _ = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            for _ in range(MAX_VERIFICATIONS_PER_SESSION):
                await tool(claim="x", context="y")
            result = await tool(claim="one too many", context="y")

        assert result.error is not None
        assert "budget" in result.error.lower()

    @pytest.mark.asyncio
    async def test_budget_is_not_refunded_on_a_failed_run(self, tmp_path, run_logger):
        """A verification attempt that itself errors (e.g. the sub-session
        crashes) still consumes budget - otherwise a broken verifier could
        be retried unboundedly, defeating the whole point of the cap."""

        class CrashingClient:
            def __init__(self, options=None):
                pass

            async def __aenter__(self):
                raise RuntimeError("boom")

            async def __aexit__(self, *exc):
                return False

        tool = make_verify_tool(tmp_path, run_logger)
        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", CrashingClient), patched_worker_deps():
            for _ in range(MAX_VERIFICATIONS_PER_SESSION):
                result = await tool(claim="x", context="y")
                assert result.error is not None  # each one fails...
            final = await tool(claim="x", context="y")

        assert "budget" in final.error.lower()  # ...but budget is still gone


class TestIndependentSessionConstruction:
    @pytest.mark.asyncio
    async def test_verifier_cannot_call_dispatch_or_batch_or_itself(self, tmp_path, run_logger):
        """The whole point is a bounded, isolated check - not a way to
        recursively spin up more agentic work."""
        text = '<result>{"confirmed": true, "actual_finding": "x", "explanation": "y"}</result>'
        FakeClient, constructed_options = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            await tool(claim="x", context="y")

        assert constructed_options[0].allowed_tools == ["mcp__browser_use__browser"]


class TestResultHandling:
    @pytest.mark.asyncio
    async def test_confirmed_result_is_reported(self, tmp_path, run_logger):
        text = '<result>{"confirmed": true, "actual_finding": "Beevor 2012", "explanation": "matched"}</result>'
        FakeClient, _ = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            result = await tool(claim="ref 275 is Beevor 2012", context="wikipedia page")

        parsed = json.loads(result.output)
        assert parsed["confirmed"] is True

    @pytest.mark.asyncio
    async def test_contradicted_result_is_reported(self, tmp_path, run_logger):
        text = '<result>{"confirmed": false, "actual_finding": "Chubarov 2001", "explanation": "id mismatch"}</result>'
        FakeClient, _ = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            result = await tool(claim="ref 275 is Beevor 2012", context="wikipedia page")

        parsed = json.loads(result.output)
        assert parsed["confirmed"] is False
        assert "Chubarov" in parsed["actual_finding"]

    @pytest.mark.asyncio
    async def test_missing_result_block_reported_as_error(self, tmp_path, run_logger):
        FakeClient, _ = make_fake_client_cls("I looked into it but wasn't sure.")
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            result = await tool(claim="x", context="y")

        parsed = json.loads(result.output)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_json_in_result_block_reported_as_error(self, tmp_path, run_logger):
        FakeClient, _ = make_fake_client_cls("<result>{not valid json</result>")
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            result = await tool(claim="x", context="y")

        parsed = json.loads(result.output)
        assert "error" in parsed


class TestRunLoggerIntegration:
    @pytest.mark.asyncio
    async def test_verification_is_logged(self, tmp_path, run_logger):
        text = '<result>{"confirmed": true, "actual_finding": "x", "explanation": "y"}</result>'
        FakeClient, _ = make_fake_client_cls(text)
        tool = make_verify_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.verify.ClaudeSDKClient", FakeClient), patched_worker_deps():
            await tool(claim="the claim", context="ctx")

        lines = (tmp_path / "run_log.jsonl").read_text().splitlines()
        events = [json.loads(l) for l in lines]
        verification_events = [e for e in events if e["event"] == "verification"]
        assert len(verification_events) == 1
        assert verification_events[0]["confirmed"] is True
        assert verification_events[0]["uses_remaining"] == MAX_VERIFICATIONS_PER_SESSION - 1
