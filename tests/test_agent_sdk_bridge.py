"""Tests for the translation layer between our ToolResult and the Claude
Agent SDK's content-block format (agent_sdk_bridge.py)."""

from browser_use_demo.agent_sdk_bridge import (
    sdk_content_to_tool_result,
    strip_mcp_prefix,
    tool_result_to_sdk_content,
)
from browser_use_demo.tools import ToolResult


class TestStripMcpPrefix:
    def test_strips_known_prefix(self):
        assert strip_mcp_prefix("mcp__browser_use__browser") == "browser"
        assert strip_mcp_prefix("mcp__browser_use__save_file") == "save_file"

    def test_leaves_unprefixed_names_alone(self):
        assert strip_mcp_prefix("Bash") == "Bash"
        assert strip_mcp_prefix("browser") == "browser"


class TestToolResultToSdkContent:
    def test_text_only(self):
        result = ToolResult(output="hello")
        sdk_content = tool_result_to_sdk_content(result)
        assert sdk_content == {"content": [{"type": "text", "text": "hello"}], "is_error": False}

    def test_image_only(self):
        result = ToolResult(base64_image="abc123")
        sdk_content = tool_result_to_sdk_content(result)
        assert sdk_content["content"] == [
            {"type": "image", "data": "abc123", "mimeType": "image/png"}
        ]

    def test_error_sets_is_error_true(self):
        result = ToolResult(error="boom")
        sdk_content = tool_result_to_sdk_content(result)
        assert sdk_content["is_error"] is True
        assert {"type": "text", "text": "Error: boom"} in sdk_content["content"]

    def test_strips_page_extraction_markers_before_sending_to_model(self):
        raw = "__PAGE_EXTRACTED__\nExtracted page DOM tree (~10 tokens)\n__FULL_CONTENT__\nreal content here"
        result = ToolResult(output=raw)
        sdk_content = tool_result_to_sdk_content(result)
        text = sdk_content["content"][0]["text"]
        assert text == "real content here"
        assert "__PAGE_EXTRACTED__" not in text

    def test_empty_output_produces_no_text_block(self):
        # Matches _take_screenshot's ToolResult(output="", base64_image=...)
        result = ToolResult(output="", base64_image="abc")
        sdk_content = tool_result_to_sdk_content(result)
        assert sdk_content["content"] == [{"type": "image", "data": "abc", "mimeType": "image/png"}]


class TestSdkContentToToolResult:
    """The reverse direction - reconstructing a ToolResult from what the CLI
    echoes back through a ToolResultBlock/PostToolUse hook. Regression
    coverage for a real bug: the CLI normalizes images to the nested
    Anthropic Messages API shape ({"source": {"data": ...}}), not the flat
    {"data": ...} shape @tool functions send them out in - the first version
    of this function only handled the flat shape and silently dropped every
    screenshot."""

    def test_plain_string_content(self):
        result = sdk_content_to_tool_result("just text", is_error=None)
        assert result.output == "just text"
        assert result.base64_image is None

    def test_text_block_list(self):
        result = sdk_content_to_tool_result([{"type": "text", "text": "hi"}], is_error=None)
        assert result.output == "hi"

    def test_nested_source_image_shape(self):
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "REALDATA"},
            }
        ]
        result = sdk_content_to_tool_result(content, is_error=None)
        assert result.base64_image == "REALDATA"

    def test_flat_image_shape_fallback(self):
        content = [{"type": "image", "data": "FLATDATA", "mimeType": "image/png"}]
        result = sdk_content_to_tool_result(content, is_error=None)
        assert result.base64_image == "FLATDATA"

    def test_is_error_moves_text_to_error_field(self):
        result = sdk_content_to_tool_result(
            [{"type": "text", "text": "Error: something broke"}], is_error=True
        )
        assert result.error == "Error: something broke"
        assert result.output is None

    def test_none_content(self):
        result = sdk_content_to_tool_result(None, is_error=None)
        assert result.output is None
        assert result.base64_image is None

    def test_round_trip_image(self):
        original = ToolResult(base64_image="roundtrip-data")
        sdk_content = tool_result_to_sdk_content(original)
        reconstructed = sdk_content_to_tool_result(sdk_content["content"], is_error=False)
        # Not a full round trip (the CLI re-nests it in between), but our own
        # flat-shape fallback should still handle our own tool_result_to_sdk_content output
        assert reconstructed.base64_image == "roundtrip-data"
