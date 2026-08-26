"""Tests for MessageRenderer class with comprehensive edge case coverage."""

import pytest
from browser_use_demo.message_renderer import MessageRenderer, Sender
from browser_use_demo.tools import ToolResult


class TestRenderMethod:
    """Test the main render method with various inputs."""

    def test_render_string_message(self, mock_streamlit):
        """Test rendering a simple string message."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.USER, "Hello world")

        mock_streamlit["chat_message"].assert_called_with(Sender.USER)
        mock_streamlit["markdown"].assert_called_with("Hello world")

    def test_render_empty_string(self, mock_streamlit):
        """Test rendering an empty string (should skip)."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.USER, "")

        mock_streamlit["chat_message"].assert_not_called()

    def test_render_none_message(self, mock_streamlit):
        """Test rendering None message (should skip)."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, None)

        mock_streamlit["chat_message"].assert_not_called()

    def test_render_tool_result_with_output(self, mock_streamlit, sample_tool_result):
        """Test rendering ToolResult with output."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["success"])

        mock_streamlit["markdown"].assert_called_with("Success message")

    def test_render_tool_result_with_error(self, mock_streamlit, sample_tool_result):
        """Test rendering ToolResult with error."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["error"])

        mock_streamlit["error"].assert_called_with("Error message")

    def test_render_tool_result_with_image(self, mock_streamlit, sample_tool_result):
        """Test rendering ToolResult with image."""
        mock_streamlit["session_state"].hide_screenshots = False
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["with_image"])

        mock_streamlit["markdown"].assert_called_with("With screenshot")
        # Image should be decoded and displayed
        assert mock_streamlit["image"].called

    def test_render_tool_result_with_hidden_screenshots(
        self, mock_streamlit, sample_tool_result
    ):
        """Test that images are hidden when hide_screenshots is True."""
        mock_streamlit["session_state"].hide_screenshots = True
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["with_image"])

        # Should render text but not image
        mock_streamlit["markdown"].assert_called_with("With screenshot")
        mock_streamlit["image"].assert_not_called()

    def test_render_tool_result_image_only_skipped_when_hidden(
        self, mock_streamlit, sample_tool_result
    ):
        """Regression test: a ToolResult with only a screenshot (no output/
        error text) must be skipped entirely when hide_screenshots is True -
        _should_skip_message previously used hasattr(message, "error")/
        hasattr(message, "output"), which are always True on a ToolResult
        (they're always-present dataclass fields), so this never actually
        skipped anything."""
        mock_streamlit["session_state"].hide_screenshots = True
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["image_only"])

        mock_streamlit["chat_message"].assert_not_called()

    def test_render_tool_result_image_only_shown_when_not_hidden(
        self, mock_streamlit, sample_tool_result
    ):
        mock_streamlit["session_state"].hide_screenshots = False
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.TOOL, sample_tool_result["image_only"])

        assert mock_streamlit["image"].called

    def test_render_tool_result_with_appended_dom_context_shows_leading_text(
        self, mock_streamlit
    ):
        """BrowserTool._attach_dom_context appends a __PAGE_EXTRACTED__ block
        after an action's own result text (e.g. "Clicked element with ref:
        ref_5") rather than replacing it - regression coverage for the
        marker parser previously starting in_summary=False and silently
        dropping everything before the first marker line, which would have
        hidden the action's own result from the chat entirely."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        result = ToolResult(
            output=(
                "Clicked element with ref: ref_5\n\n"
                "__PAGE_EXTRACTED__\n"
                "DOM changes since your last action (42 chars)\n"
                "__FULL_CONTENT__\n"
                "+- dialog \"Confirm\" [ref=ref_9]"
            )
        )
        renderer.render(Sender.TOOL, result)

        rendered = [call.args[0] for call in mock_streamlit["markdown"].call_args_list]
        assert "Clicked element with ref: ref_5" in rendered
        assert "DOM changes since your last action (42 chars)" in rendered
        # the raw diff content itself must never reach the visible chat
        assert not any("dialog \"Confirm\"" in text for text in rendered)

    def test_render_dict_message_text_type(self, mock_streamlit):
        """Test rendering dictionary message with text type."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        message = {"type": "text", "text": "Hello from dict"}
        renderer.render(Sender.USER, message)

        mock_streamlit["write"].assert_called_with("Hello from dict")

    def test_render_dict_message_tool_use_type(self, mock_streamlit):
        """Test rendering dictionary message with tool_use type."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        message = {
            "type": "tool_use",
            "name": "browser_tool",
            "input": {"url": "example.com"},
        }
        renderer.render(Sender.BOT, message)

        expected_code = "Tool Use: browser_tool\nInput: {'url': 'example.com'}"
        mock_streamlit["code"].assert_called_with(expected_code)

    def test_render_dict_message_unknown_type(self, mock_streamlit):
        """Test rendering dictionary message with unknown type."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        message = {"type": "unknown", "data": "some data"}
        renderer.render(Sender.BOT, message)

        # Should fall back to generic write
        mock_streamlit["write"].assert_called_with(message)

class TestConversationHistory:
    """Test render_conversation_history method with various scenarios."""

    def test_render_single_message(self, mock_streamlit):
        """Test rendering single message in history."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [{"role": "user", "content": "Hello"}]
        renderer.render_conversation_history(messages)

        mock_streamlit["markdown"].assert_called_with("Hello")

    def test_render_multiple_messages(self, mock_streamlit, sample_messages):
        """Test rendering multiple messages with different roles."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render_conversation_history(sample_messages[:2])

        # Should render both messages
        assert mock_streamlit["markdown"].call_count >= 2

    def test_render_unknown_role(self, mock_streamlit):
        """Test handling messages with unknown roles."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [{"role": "unknown_role", "content": "Test"}]
        renderer.render_conversation_history(messages)

        # Should not crash, but won't render
        mock_streamlit["markdown"].assert_not_called()

    def test_render_none_content(self, mock_streamlit):
        """Test handling messages with None content."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [{"role": "user", "content": None}]
        renderer.render_conversation_history(messages)

        # Should handle gracefully without rendering
        mock_streamlit["markdown"].assert_not_called()

    def test_render_list_content(self, mock_streamlit):
        """Test rendering messages with list content."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"},
                ],
            }
        ]
        renderer.render_conversation_history(messages)

        # Should render both text blocks
        calls = mock_streamlit["markdown"].call_args_list
        assert any("First" in str(call) for call in calls)
        assert any("Second" in str(call) for call in calls)

    def test_skip_image_blocks_in_history(self, mock_streamlit):
        """Test that image blocks are skipped in conversation history."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Text message"},
                    {"type": "image", "source": "data:image/png;base64,abc"},
                ],
            }
        ]
        renderer.render_conversation_history(messages)

        # Should only render text, not image
        mock_streamlit["markdown"].assert_called_with("Text message")
        mock_streamlit["image"].assert_not_called()

    def test_tool_result_in_assistant_message(self, mock_streamlit, sample_tool_result):
        """Test rendering tool results from assistant messages."""
        mock_streamlit["session_state"].tools = {
            "tool_123": sample_tool_result["success"]
        }
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_result", "tool_use_id": "tool_123"}],
            }
        ]
        renderer.render_conversation_history(messages)

        # Should render the tool result from session state
        mock_streamlit["markdown"].assert_called_with("Success message")

    def test_missing_tool_in_session_state(self, mock_streamlit):
        """Test handling tool_use_id that doesn't exist in session state."""
        renderer = MessageRenderer(mock_streamlit["session_state"])
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_result", "tool_use_id": "nonexistent"}],
            }
        ]
        renderer.render_conversation_history(messages)

        # Should handle gracefully without crashing
        mock_streamlit["markdown"].assert_not_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_normalize_content_with_various_inputs(self, mock_streamlit):
        """Test _normalize_content with various input types."""
        renderer = MessageRenderer(mock_streamlit["session_state"])

        # String input
        assert renderer._normalize_content("test") == ["test"]

        # List input
        assert renderer._normalize_content([1, 2, 3]) == [1, 2, 3]

        # None input
        assert renderer._normalize_content(None) == [None]

        # Dict input
        assert renderer._normalize_content({"key": "value"}) == [{"key": "value"}]


class TestReferencedFileDetection:
    """A final assistant message that names a real deliverable file (by
    whatever filename the model chose to mention) should surface it inline,
    without us hardcoding which tool or naming scheme produced it."""

    def test_final_message_naming_a_real_file_offers_download(self, mock_streamlit):
        run_dir = mock_streamlit["run_dir"]
        (run_dir / "screenshot_abc123.png").write_bytes(b"\x89PNG\r\n")

        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, {"type": "text", "text": "See screenshot_abc123.png for the result."})

        mock_streamlit["download_button"].assert_called_once()
        assert mock_streamlit["download_button"].call_args.kwargs["file_name"] == "screenshot_abc123.png"

    def test_image_file_also_gets_a_thumbnail(self, mock_streamlit):
        run_dir = mock_streamlit["run_dir"]
        (run_dir / "chart.png").write_bytes(b"\x89PNG\r\n")

        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, {"type": "text", "text": "Saved as chart.png"})

        mock_streamlit["image"].assert_called_once()

    def test_non_image_file_gets_download_only_no_thumbnail(self, mock_streamlit):
        run_dir = mock_streamlit["run_dir"]
        (run_dir / "report.csv").write_bytes(b"a,b,c\n")

        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, {"type": "text", "text": "See report.csv"})

        mock_streamlit["download_button"].assert_called_once()
        mock_streamlit["image"].assert_not_called()

    def test_mentioning_a_nonexistent_file_renders_nothing_extra(self, mock_streamlit):
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, {"type": "text", "text": "e.g. see results.csv for details"})

        mock_streamlit["download_button"].assert_not_called()
        mock_streamlit["image"].assert_not_called()

    def test_plain_text_with_no_filenames_is_unaffected(self, mock_streamlit):
        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(Sender.BOT, {"type": "text", "text": "All done, no files here."})

        mock_streamlit["download_button"].assert_not_called()
        mock_streamlit["write"].assert_called_once_with("All done, no files here.")

    def test_same_filename_mentioned_twice_only_renders_once(self, mock_streamlit):
        run_dir = mock_streamlit["run_dir"]
        (run_dir / "data.json").write_bytes(b"{}")

        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(
            Sender.BOT,
            {"type": "text", "text": "Saved data.json. See data.json for the full result."},
        )

        mock_streamlit["download_button"].assert_called_once()

    def test_same_file_offered_twice_in_one_pass_does_not_collide(self, mock_streamlit):
        """Regression test for a real crash: save_file's own tool result and
        the model's final message both offering the same filename in one
        script run used to generate identical widget keys, which Streamlit
        rejects with DuplicateWidgetID."""
        run_dir = mock_streamlit["run_dir"]
        (run_dir / "elon_tweets_2026-08-24.csv").write_bytes(b"a,b\n1,2\n")

        renderer = MessageRenderer(mock_streamlit["session_state"])
        renderer.render(
            Sender.TOOL,
            ToolResult(output="Saved file: elon_tweets_2026-08-24.csv (8 bytes) at /tmp/x"),
        )
        renderer.render(
            Sender.BOT,
            {"type": "text", "text": "Done - see elon_tweets_2026-08-24.csv for the results."},
        )

        assert mock_streamlit["download_button"].call_count == 2
        keys = [c.kwargs["key"] for c in mock_streamlit["download_button"].call_args_list]
        assert keys[0] != keys[1]
