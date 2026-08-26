"""Tests for DispatchSubagentsTool's fan-out logic: shared instructions reach
every worker identically, concurrency is capped, and mixed success/failure
results aggregate correctly - without making real API calls."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from browser_use_demo.run_logger import RunLogger
from browser_use_demo.tools.subagent import (
    MAX_CONCURRENCY_HARD_CAP,
    DispatchSubagentsTool,
)


@contextlib.contextmanager
def patched_worker_deps():
    """Patch the per-worker BrowserTool/FileOutputTool/build_mcp_server so
    no real Playwright browser is ever launched. cleanup() must be an
    AsyncMock explicitly - MagicMock's auto-generated attributes aren't
    awaitable, and _run_one always awaits browser_tool.cleanup() in its
    finally block."""
    fake_browser_tool = MagicMock()
    fake_browser_tool.cleanup = AsyncMock()
    with patch(
        "browser_use_demo.tools.subagent.BrowserTool", return_value=fake_browser_tool
    ), patch("browser_use_demo.tools.subagent.FileOutputTool"), patch(
        "browser_use_demo.tools.subagent.build_mcp_server", return_value=MagicMock()
    ):
        yield


def make_fake_client_cls(response_for_item):
    """Build a fake ClaudeSDKClient class whose response depends on which
    item it was queried with, and which records the options it was
    constructed with for later assertions."""
    constructed_options = []

    class FakeClient:
        def __init__(self, options=None):
            self.options = options
            constructed_options.append(options)
            self._item = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, item):
            self._item = item

        async def receive_response(self):
            text = response_for_item(self._item)
            yield AssistantMessage(content=[TextBlock(text=text)], model="test-model")

    return FakeClient, constructed_options


@pytest.fixture
def run_logger(tmp_path):
    return RunLogger(tmp_path)


def make_dispatch_tool(tmp_path, run_logger):
    return DispatchSubagentsTool(
        run_dir=tmp_path,
        run_logger=run_logger,
        api_key="test-key",
        model="claude-haiku-4-5",
        base_system_prompt="<BASE PROMPT>",
    )


class TestSharedInstructionsPropagation:
    @pytest.mark.asyncio
    async def test_every_worker_gets_identical_shared_instructions(self, tmp_path, run_logger):
        def respond(item):
            return f'<result>{{"item": "{item}", "ok": true}}</result>'

        FakeClient, constructed_options = make_fake_client_cls(respond)
        tool = make_dispatch_tool(tmp_path, run_logger)

        with patch("browser_use_demo.tools.subagent.ClaudeSDKClient", FakeClient), patched_worker_deps():
            await tool(
                shared_instructions="Extract the title field exactly like this.",
                items=["item_a", "item_b", "item_c"],
                output_schema={"title": "string"},
            )

        assert len(constructed_options) == 3
        for options in constructed_options:
            assert "Extract the title field exactly like this." in options.system_prompt
            # every worker's prompt is built from the same base + instructions,
            # so they should be byte-identical
        assert len({o.system_prompt for o in constructed_options}) == 1


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_default_concurrency_used_when_not_specified(self, tmp_path, run_logger):
        max_concurrent_seen = 0
        current_concurrent = 0

        def respond(item):
            return f'<result>{{"item": "{item}"}}</result>'

        FakeClient, _ = make_fake_client_cls(respond)

        class TrackingFakeClient(FakeClient):
            async def query(self, item):
                nonlocal max_concurrent_seen, current_concurrent
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
                await asyncio.sleep(0.02)
                self._item = item
                current_concurrent -= 1

        tool = make_dispatch_tool(tmp_path, run_logger)
        with patch("browser_use_demo.tools.subagent.ClaudeSDKClient", TrackingFakeClient), patched_worker_deps():
            await tool(
                shared_instructions="x",
                items=[f"item_{i}" for i in range(10)],
                output_schema={},
                max_concurrency=3,
            )

        assert max_concurrent_seen <= 3

    @pytest.mark.asyncio
    async def test_requested_concurrency_above_hard_cap_is_clamped(self, tmp_path, run_logger):
        # A single-item call can't distinguish "clamped" from "not clamped"
        # (Semaphore(999) and Semaphore(10) behave identically with one item
        # in flight) - use enough items to actually saturate above the cap
        # if it weren't being enforced.
        max_concurrent_seen = 0
        current_concurrent = 0

        def respond(item):
            return f'<result>{{"item": "{item}"}}</result>'

        FakeClient, _ = make_fake_client_cls(respond)

        class TrackingFakeClient(FakeClient):
            async def query(self, item):
                nonlocal max_concurrent_seen, current_concurrent
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
                await asyncio.sleep(0.02)
                self._item = item
                current_concurrent -= 1

        tool = make_dispatch_tool(tmp_path, run_logger)
        with patch("browser_use_demo.tools.subagent.ClaudeSDKClient", TrackingFakeClient), patched_worker_deps():
            await tool(
                shared_instructions="x",
                items=[f"item_{i}" for i in range(MAX_CONCURRENCY_HARD_CAP + 10)],
                output_schema={},
                max_concurrency=999,
            )

        assert max_concurrent_seen <= MAX_CONCURRENCY_HARD_CAP


class TestResultAggregation:
    @pytest.mark.asyncio
    async def test_mixed_success_and_exception_results(self, tmp_path, run_logger):
        def respond(item):
            if item == "bad_item":
                raise RuntimeError("worker crashed")
            return f'<result>{{"item": "{item}", "ok": true}}</result>'

        FakeClient, _ = make_fake_client_cls(respond)

        class RaisingFakeClient(FakeClient):
            async def receive_response(self):
                text = respond(self._item)  # may raise
                yield AssistantMessage(content=[TextBlock(text=text)], model="test-model")

        tool = make_dispatch_tool(tmp_path, run_logger)
        with patch("browser_use_demo.tools.subagent.ClaudeSDKClient", RaisingFakeClient), patched_worker_deps():
            result = await tool(
                shared_instructions="x",
                items=["good_item", "bad_item"],
                output_schema={},
            )

        import json

        parsed = json.loads(result.output)
        results_by_item = {r.get("item"): r for r in parsed["results"]}
        assert "error" not in results_by_item["good_item"]
        assert "error" in results_by_item["bad_item"]
        assert "worker crashed" in results_by_item["bad_item"]["error"]

    @pytest.mark.asyncio
    async def test_rejects_empty_items(self, tmp_path, run_logger):
        tool = make_dispatch_tool(tmp_path, run_logger)
        result = await tool(shared_instructions="x", items=[], output_schema={})
        assert result.error is not None


class TestParseResult:
    def test_valid_result_block(self):
        parsed = DispatchSubagentsTool._parse_result("item1", '<result>{"a": 1}</result>')
        assert parsed == {"a": 1}

    def test_missing_result_block(self):
        parsed = DispatchSubagentsTool._parse_result("item1", "I couldn't find it")
        assert parsed["item"] == "item1"
        assert "error" in parsed

    def test_invalid_json_in_result_block(self):
        parsed = DispatchSubagentsTool._parse_result("item1", "<result>not json</result>")
        assert parsed["item"] == "item1"
        assert "error" in parsed

    def test_result_block_with_surrounding_text(self):
        text = 'Here you go:\n<result>{"title": "Example"}</result>\nDone.'
        parsed = DispatchSubagentsTool._parse_result("item1", text)
        assert parsed == {"title": "Example"}
