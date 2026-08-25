"""Concurrent fan-out for batches of similar browser tasks (e.g. "extract the
same 4 fields from these 10 PR pages").

Uses independent ClaudeSDKClient sessions rather than the SDK's native
agents/AgentDefinition mechanism: a spike confirmed that native subagents
sharing one in-process MCP server all hit the SAME underlying tool-function
closure (same BrowserTool instance, same Playwright page) - concurrent
subagents would fight over one browser tab. Giving each fan-out worker its
own ClaudeSDKClient + its own BrowserTool/FileOutputTool sidesteps that
entirely, at the cost of orchestrating the fan-out ourselves instead of using
the SDK's built-in dispatcher.

The coordinator supplies shared_instructions and output_schema ONCE; every
worker gets the identical text, so workers don't each reinvent how to
extract fields from the same kind of page.
"""

import asyncio
import json
import re
from typing import Any, Optional
from uuid import uuid4

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock
from claude_agent_sdk import tool as sdk_tool

from ..agent_sdk_bridge import MAX_BUFFER_SIZE, build_mcp_server, tool_result_to_sdk_content
from ..run_logger import RunLogger
from .base import ToolError, ToolResult
from .browser import BrowserTool
from .file_output import FileOutputTool

MAX_CONCURRENCY_HARD_CAP = 10
DEFAULT_MAX_CONCURRENCY = 5
SUBAGENT_MAX_TURNS = 30

DISPATCH_SUBAGENTS_INPUT_SCHEMA: dict = {
    "properties": {
        "shared_instructions": {
            "description": (
                "The extraction/task recipe applied IDENTICALLY to every item "
                "- e.g. field names and how to find each one on this kind of "
                "page. Every worker gets this exact text, so be concrete "
                "enough that none of them have to guess or invent their own "
                "approach."
            ),
            "type": "string",
        },
        "items": {
            "description": "One entry per worker to dispatch, e.g. a list of URLs.",
            "type": "array",
            "items": {"type": "string"},
        },
        "output_schema": {
            "description": "JSON schema each worker's result must conform to, e.g. {\"pr_number\": \"string\", \"author\": \"string\"}.",
            "type": "object",
        },
        "max_concurrency": {
            "description": f"Max simultaneous workers (default {DEFAULT_MAX_CONCURRENCY}, hard-capped at {MAX_CONCURRENCY_HARD_CAP}).",
            "type": "integer",
        },
    },
    "required": ["shared_instructions", "items", "output_schema"],
    "type": "object",
}

DISPATCH_SUBAGENTS_DESCRIPTION = (
    "Process a batch of 3+ similar items (e.g. several PR pages, several "
    "product listings) concurrently instead of one at a time. Each item gets "
    "its own worker with its own browser tab, following the SAME shared "
    "instructions - so results are consistent across the batch. Returns a "
    "JSON object with one result (or error) per item, in the same order as "
    "the input. Prefer this over a sequential loop for 3+ similar items."
)

SUBAGENT_ROLE_ADDENDUM = """<SUBAGENT_ROLE>
You are one worker completing ONE item from a batch dispatched by a coordinator. Follow the shared instructions below exactly - use the same method every other worker in this batch is using, don't invent your own approach.

When you are done, respond with ONLY a JSON object matching the given output schema, wrapped in <result>...</result> tags, and nothing else after the closing tag.
</SUBAGENT_ROLE>"""


class DispatchSubagentsTool:
    """Fans out to N concurrent ClaudeSDKClient workers, each with an
    independent BrowserTool/FileOutputTool pair."""

    name = "dispatch_subagents"

    def __init__(
        self,
        *,
        run_dir,
        run_logger: RunLogger,
        api_key: str,
        model: str,
        base_system_prompt: str,
    ):
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.api_key = api_key
        self.model = model
        self.base_system_prompt = base_system_prompt

    async def __call__(
        self,
        *,
        shared_instructions: str,
        items: list[str],
        output_schema: dict[str, Any],
        max_concurrency: Optional[int] = None,
    ) -> ToolResult:
        if not items:
            return ToolResult(error="items must be a non-empty list")

        concurrency = max(1, min(max_concurrency or DEFAULT_MAX_CONCURRENCY, MAX_CONCURRENCY_HARD_CAP))
        dispatch_id = uuid4().hex
        self.run_logger.log_subagent_dispatch(dispatch_id, items, shared_instructions)

        semaphore = asyncio.Semaphore(concurrency)
        raw_results = await asyncio.gather(
            *(
                self._run_one(dispatch_id, item, shared_instructions, output_schema, semaphore)
                for item in items
            ),
            return_exceptions=True,
        )

        results = []
        succeeded = 0
        for item, raw in zip(items, raw_results):
            if isinstance(raw, BaseException):
                results.append({"item": item, "error": str(raw)})
            else:
                results.append(raw)
                if "error" not in raw:
                    succeeded += 1

        self.run_logger.log_subagent_join(
            dispatch_id, total=len(items), succeeded=succeeded, failed=len(items) - succeeded
        )

        return ToolResult(output=json.dumps({"results": results}, indent=2))

    async def _run_one(
        self,
        dispatch_id: str,
        item: str,
        shared_instructions: str,
        output_schema: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            browser_tool = BrowserTool(run_dir=self.run_dir)
            file_output_tool = FileOutputTool(run_dir=self.run_dir)
            try:
                server = build_mcp_server(browser_tool, file_output_tool)
                system_prompt = (
                    f"{self.base_system_prompt}\n\n{SUBAGENT_ROLE_ADDENDUM}\n\n"
                    f"<SHARED_INSTRUCTIONS>\n{shared_instructions}\n</SHARED_INSTRUCTIONS>\n\n"
                    f"<OUTPUT_SCHEMA>\n{json.dumps(output_schema)}\n</OUTPUT_SCHEMA>"
                )
                options = ClaudeAgentOptions(
                    model=self.model,
                    system_prompt=system_prompt,
                    mcp_servers={"browser_use": server},
                    allowed_tools=[
                        "mcp__browser_use__browser",
                        "mcp__browser_use__save_file",
                    ],
                    max_turns=SUBAGENT_MAX_TURNS,
                    max_buffer_size=MAX_BUFFER_SIZE,
                    env={"ANTHROPIC_API_KEY": self.api_key},
                )

                final_text_parts: list[str] = []
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(item)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    final_text_parts.append(block.text)

                result = self._parse_result(item, "\n".join(final_text_parts))
                self.run_logger.log_subagent_result(
                    dispatch_id, item, success="error" not in result
                )
                return result
            except Exception as e:
                self.run_logger.log_subagent_result(dispatch_id, item, success=False)
                return {"item": item, "error": str(e)}
            finally:
                await browser_tool.cleanup()

    @staticmethod
    def _parse_result(item: str, text: str) -> dict[str, Any]:
        match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
        if not match:
            return {"item": item, "error": "worker did not return a <result> block", "raw_output": text}
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {"item": item, "error": "worker's <result> block was not valid JSON", "raw_output": text}


def build_dispatch_subagents_tool_fn(dispatch_tool: DispatchSubagentsTool):
    """Wrap a DispatchSubagentsTool instance as an SDK-registrable @tool
    function, for the coordinator's own mcp_servers registration."""

    @sdk_tool("dispatch_subagents", DISPATCH_SUBAGENTS_DESCRIPTION, DISPATCH_SUBAGENTS_INPUT_SCHEMA)
    async def dispatch_subagents_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await dispatch_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return dispatch_subagents_fn
