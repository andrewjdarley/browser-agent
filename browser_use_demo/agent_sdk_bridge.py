"""Bridges our own tools (ToolResult-based) to the Claude Agent SDK's
in-process MCP tool format (plain content-block dicts).

Kept as a thin translation layer so the tools themselves (browser.py,
file_output.py) stay simple, framework-agnostic, and reusable outside the
SDK if needed.
"""

from typing import Any, Optional

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from .tools import ToolError, ToolResult
from .tools.browser import BROWSER_TOOL_DESCRIPTION, BROWSER_TOOL_INPUT_SCHEMA, BrowserTool
from .tools.file_output import SAVE_FILE_DESCRIPTION, SAVE_FILE_INPUT_SCHEMA, FileOutputTool

# The SDK's stdio transport to the Claude Code CLI subprocess defaults to a
# 1 MiB per-message buffer. The primary fix for oversized messages is
# BrowserTool._take_screenshot's own size check and retry-with-a-smaller-clip
# logic (tools/browser.py) - it keeps screenshots from ever getting this big
# in the first place, and a bigger image isn't more useful to the model
# anyway. This is just a backstop for other paths (e.g. a very large
# read_page DOM dump) that aren't screenshot-shaped and don't go through
# that logic - the old 1 MiB default has near-zero margin for those. Apply
# to every ClaudeAgentOptions we construct (coordinator and each subagent
# worker).
MAX_BUFFER_SIZE = 20 * 1024 * 1024  # 20 MiB


def strip_mcp_prefix(tool_name: str) -> str:
    """SDK tool calls arrive as "mcp__<server>__<tool>" (e.g.
    "mcp__browser_use__browser"). Strip that down to the plain tool name so
    the rest of the app (coordinate scaling checks, display) doesn't need to
    know about the MCP naming convention."""
    prefix = "mcp__browser_use__"
    return tool_name[len(prefix) :] if tool_name.startswith(prefix) else tool_name


def tool_result_to_sdk_content(result: ToolResult) -> dict[str, Any]:
    """Convert our ToolResult into the SDK's {"content": [...], "is_error": ...} shape.

    Strips the __PAGE_EXTRACTED__/__TEXT_EXTRACTED__/__FULL_CONTENT__ markers
    before sending to the model — those exist only so the Streamlit renderer
    can show a short summary instead of the full extracted page text.
    """
    content: list[dict[str, Any]] = []

    if result.output:
        output_text = result.output
        if "__PAGE_EXTRACTED__" in output_text or "__TEXT_EXTRACTED__" in output_text:
            if "__FULL_CONTENT__" in output_text:
                marker_pos = output_text.index("__FULL_CONTENT__")
                output_text = output_text[marker_pos + len("__FULL_CONTENT__") + 1 :]
        content.append({"type": "text", "text": output_text})

    if result.base64_image:
        content.append(
            {"type": "image", "data": result.base64_image, "mimeType": "image/png"}
        )

    if result.error:
        content.append({"type": "text", "text": f"Error: {result.error}"})

    return {"content": content, "is_error": bool(result.error)}


def sdk_content_to_tool_result(content: Any, is_error: bool | None) -> ToolResult:
    """Reverse of the above — reconstruct a ToolResult from a ToolResultBlock's
    content, so the existing Streamlit renderer (which expects ToolResult
    objects) doesn't need to change."""
    output_parts: list[str] = []
    base64_image: str | None = None
    error: str | None = None

    if isinstance(content, str):
        output_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                output_parts.append(block.get("text", ""))
            elif block_type == "image":
                # The CLI echoes tool-produced images back in the Anthropic
                # Messages API shape ({"source": {"data": ...}}), not the flat
                # {"data": ...} shape @tool functions send them in.
                source = block.get("source")
                if isinstance(source, dict):
                    base64_image = source.get("data")
                else:
                    base64_image = block.get("data")

    if is_error and output_parts:
        error = "\n".join(output_parts)
        output_parts = []

    return ToolResult(
        output="\n".join(output_parts) if output_parts else None,
        error=error,
        base64_image=base64_image,
    )


def build_browser_tool_fn(browser_tool: BrowserTool):
    """Wrap a BrowserTool instance as an SDK-registrable @tool function."""

    @tool("browser", BROWSER_TOOL_DESCRIPTION, BROWSER_TOOL_INPUT_SCHEMA)
    async def browser_tool_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await browser_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return browser_tool_fn


def build_save_file_tool_fn(file_output_tool: FileOutputTool):
    """Wrap a FileOutputTool instance as an SDK-registrable @tool function."""

    @tool("save_file", SAVE_FILE_DESCRIPTION, SAVE_FILE_INPUT_SCHEMA)
    async def save_file_tool_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await file_output_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return save_file_tool_fn


def build_mcp_server(
    browser_tool: BrowserTool,
    file_output_tool: FileOutputTool,
    extra_tools: Optional[list] = None,
) -> McpSdkServerConfig:
    """Assemble the in-process MCP server exposing our tools to the SDK.

    extra_tools lets callers (e.g. tools/subagent.py, for dispatch_subagents)
    add more @tool-wrapped functions without this module needing to import
    them - importing DispatchSubagentsTool here would be circular, since
    subagent.py itself needs build_mcp_server for each worker's own server.
    """
    return create_sdk_mcp_server(
        name="browser_use",
        tools=[
            build_browser_tool_fn(browser_tool),
            build_save_file_tool_fn(file_output_tool),
            *(extra_tools or []),
        ],
    )
