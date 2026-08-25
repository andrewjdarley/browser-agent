"""Tests for RunLogger's JSONL output and hook callbacks."""

import json

import pytest

from browser_use_demo.run_logger import RunLogger


def read_events(run_dir) -> list[dict]:
    log_path = run_dir / "run_log.jsonl"
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


class TestRunLoggerBasics:
    def test_log_run_start_writes_valid_jsonl(self, tmp_path):
        logger = RunLogger(tmp_path)
        logger.log_run_start(model="claude-sonnet-4-5")
        events = read_events(tmp_path)
        assert len(events) == 1
        assert events[0]["event"] == "run_start"
        assert events[0]["model"] == "claude-sonnet-4-5"
        assert "timestamp" in events[0]

    def test_multiple_events_append_one_line_each(self, tmp_path):
        logger = RunLogger(tmp_path)
        logger.log_run_start(model="m")
        logger.log_error("boom", context="test")
        logger.log_run_end()
        events = read_events(tmp_path)
        assert [e["event"] for e in events] == ["run_start", "error", "run_end"]

    def test_never_logs_base64_image_data(self, tmp_path):
        logger = RunLogger(tmp_path)
        logger.log_run_start(model="m")
        raw_log = (tmp_path / "run_log.jsonl").read_text()
        assert "base64" not in raw_log.lower() or "base64_image" not in raw_log


class TestRunLoggerHooks:
    @pytest.mark.asyncio
    async def test_pre_tool_use_hook_logs_tool_call(self, tmp_path):
        logger = RunLogger(tmp_path)
        input_data = {
            "tool_use_id": "abc123",
            "tool_name": "mcp__browser_use__browser",
            "tool_input": {"action": "navigate", "text": "https://example.com"},
        }
        result = await logger.on_pre_tool_use(input_data, "abc123", None)
        assert result == {}  # observes only, never intervenes
        events = read_events(tmp_path)
        assert events[0]["event"] == "tool_call"
        assert events[0]["tool_name"] == "mcp__browser_use__browser"
        assert events[0]["tool_input"]["action"] == "navigate"

    @pytest.mark.asyncio
    async def test_post_tool_use_hook_detects_image_in_nested_shape(self, tmp_path):
        logger = RunLogger(tmp_path)
        input_data = {
            "tool_use_id": "abc123",
            "tool_name": "mcp__browser_use__browser",
            "tool_response": [
                {"type": "image", "source": {"type": "base64", "data": "xyz"}}
            ],
        }
        await logger.on_post_tool_use(input_data, "abc123", None)
        events = read_events(tmp_path)
        assert events[0]["has_image"] is True

    @pytest.mark.asyncio
    async def test_post_tool_use_hook_summarizes_text(self, tmp_path):
        logger = RunLogger(tmp_path)
        input_data = {
            "tool_use_id": "abc123",
            "tool_name": "mcp__browser_use__save_file",
            "tool_response": [{"type": "text", "text": "Saved file: out.csv"}],
        }
        await logger.on_post_tool_use(input_data, "abc123", None)
        events = read_events(tmp_path)
        assert events[0]["output_summary"] == "Saved file: out.csv"
        assert events[0]["has_image"] is False

    @pytest.mark.asyncio
    async def test_user_prompt_submit_hook(self, tmp_path):
        logger = RunLogger(tmp_path)
        await logger.on_user_prompt_submit({"prompt": "navigate somewhere"}, None, None)
        events = read_events(tmp_path)
        assert events[0]["event"] == "user_prompt"
        assert events[0]["prompt"] == "navigate somewhere"


class TestRunLoggerSubagentEvents:
    def test_dispatch_result_join_sequence(self, tmp_path):
        logger = RunLogger(tmp_path)
        logger.log_subagent_dispatch("d1", ["item1", "item2"], "do the thing")
        logger.log_subagent_result("d1", "item1", success=True)
        logger.log_subagent_result("d1", "item2", success=False)
        logger.log_subagent_join("d1", total=2, succeeded=1, failed=1)

        events = read_events(tmp_path)
        assert [e["event"] for e in events] == [
            "subagent_dispatch",
            "subagent_result",
            "subagent_result",
            "subagent_join",
        ]
        assert events[0]["items"] == ["item1", "item2"]
        assert events[3]["succeeded"] == 1
        assert events[3]["failed"] == 1
