"""Message rendering for the Streamlit chat interface - separates
presentation from the main application logic in streamlit.py."""

import base64
import re
import uuid
from typing import cast

import streamlit as st
from anthropic.types.beta import BetaContentBlockParam

from browser_use_demo.tools import ToolResult
from browser_use_demo.tools.coordinate_scaling import CoordinateScaler

# Matches FileOutputTool's "Saved file: <name> (<size>) - ..." output text, so
# a save_file result can offer an inline download right in the chat instead
# of only being visible if the user finds the sidebar.
_SAVED_FILE_RE = re.compile(r"^Saved file: (.+?) \(")

# Candidate filename-shaped tokens in free text (name.ext) - deliberately not
# tied to any one tool's naming scheme (screenshot_*, save_file's own names,
# etc.). Whatever the model names in its final message, we check it against
# what's actually on disk in run_dir and only render what's really there.
_FILENAME_TOKEN_RE = re.compile(r"\b[\w\-]+\.[A-Za-z0-9]{1,8}\b")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


class Sender:
    """Message sender types."""

    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


class MessageRenderer:
    """Handles rendering of messages in the Streamlit chat interface."""

    def __init__(self, session_state):
        self.session_state = session_state

    def _scale_browser_coordinates(self, input_dict: dict) -> dict:
        """Scale a browser tool call's coordinates for display, so what's
        shown matches what the browser tool will actually use."""
        if not isinstance(input_dict, dict):
            return input_dict

        browser_tool = getattr(self.session_state, 'browser_tool', None)
        if not browser_tool:
            return input_dict

        import copy
        scaled_input = copy.deepcopy(input_dict)
        width, height = browser_tool.width, browser_tool.height

        if 'coordinate' in scaled_input:
            scaled_input['coordinate'] = CoordinateScaler.scale_coordinate_list(
                scaled_input['coordinate'], width, height
            )
        if 'start_coordinate' in scaled_input:
            scaled_input['start_coordinate'] = CoordinateScaler.scale_coordinate_list(
                scaled_input['start_coordinate'], width, height
            )

        return scaled_input

    def render(self, sender: str, message: str | BetaContentBlockParam | ToolResult):
        if self._should_skip_message(message):
            return
        with st.chat_message(sender):
            self._render_message_content(message)

    def _should_skip_message(self, message) -> bool:
        if not message:
            return True

        # Skip tool results that only have screenshots when screenshots are hidden
        is_tool_result = not isinstance(message, str | dict)
        if is_tool_result and self.session_state.hide_screenshots:
            return not message.error and not message.output

        return False

    def _render_message_content(self, message):
        renderers = {
            "tool_result": self._render_tool_result,
            "dict": self._render_dict_message,
            "string": lambda msg: st.markdown(msg),
        }

        if not isinstance(message, str | dict):
            renderers["tool_result"](cast(ToolResult, message))
        elif isinstance(message, dict):
            renderers["dict"](message)
        else:
            renderers["string"](message)

    def _render_tool_result(self, tool_result: ToolResult):
        if tool_result.output:
            output = tool_result.output
            # An extraction marker block may be the whole output (read_page/
            # get_page_text) or trail some real leading text (e.g.
            # BrowserTool._attach_dom_context appends one to an action's own
            # result, "Clicked element with ref: ref_5"). Show any leading
            # text normally, then only the marker block's one-line summary -
            # never the raw tree/diff dumped into the visible chat.
            marker_pos = next(
                (
                    output.index(marker)
                    for marker in ("__PAGE_EXTRACTED__", "__TEXT_EXTRACTED__")
                    if marker in output
                ),
                None,
            )
            if marker_pos is not None:
                leading = output[:marker_pos].rstrip()
                if leading:
                    st.markdown(leading)

                summary_lines = []
                in_summary = False
                for line in output[marker_pos:].split("\n"):
                    if "__PAGE_EXTRACTED__" in line or "__TEXT_EXTRACTED__" in line:
                        in_summary = True
                        continue
                    if "__FULL_CONTENT__" in line:
                        break
                    if in_summary:
                        summary_lines.append(line)

                if summary_lines:
                    st.markdown("\n".join(summary_lines))
            else:
                st.markdown(output)

            match = _SAVED_FILE_RE.match(tool_result.output)
            if match:
                self._render_file_reference(match.group(1))

        if tool_result.error:
            st.error(tool_result.error)
        if tool_result.base64_image and not self.session_state.hide_screenshots:
            st.image(base64.b64decode(tool_result.base64_image))

    def _render_file_reference(self, filename: str):
        """Offer a file that actually exists in run_dir as an immediate
        inline download (with a thumbnail if it's an image), so the user
        doesn't have to go find the sidebar for it."""
        path = self.session_state.run_dir / filename
        if not path.is_file():
            return
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            st.image(str(path))
        # The same filename can be offered more than once in one render pass
        # (e.g. once at the save_file tool result, again when the final
        # message names it) - a key derived only from filename/mtime isn't
        # unique across those call sites. A fresh UUID per call guarantees
        # uniqueness; download buttons have no cross-rerun state worth
        # preserving under a stable key.
        st.download_button(
            label=f"⬇️ Download {filename}",
            data=path.read_bytes(),
            file_name=filename,
            key=f"inline_download_{filename}_{uuid.uuid4().hex}",
        )

    def _render_referenced_files(self, text: str):
        """Scan free-form text (e.g. the model's own final message) for
        filenames and surface any that actually exist in run_dir. This lets
        the model point at whatever deliverable it produced - a screenshot,
        a saved file, anything else - without us hardcoding which tools or
        naming schemes produce a "real" deliverable."""
        seen: set[str] = set()
        for token in _FILENAME_TOKEN_RE.findall(text):
            if token in seen:
                continue
            seen.add(token)
            self._render_file_reference(token)

    def _render_text(self, text: str):
        """Render an assistant text message, then surface any real
        deliverable files it names (see _render_referenced_files) - the
        model's final message is what the user actually reads, so anything
        it points at should be reachable right there."""
        st.write(text)
        self._render_referenced_files(text)

    def _render_dict_message(self, message: dict):
        message_type = message.get("type", "")
        type_handlers = {
            "text": lambda: self._render_text(message["text"]),
            "tool_use": lambda: self._render_tool_use(message),
            "tool_result": lambda: self._render_stored_tool_result(message),
        }
        handler = type_handlers.get(message_type, lambda: st.write(message))
        handler()

    def _render_tool_use(self, message: dict):
        tool_name = message.get('name', 'unknown')
        tool_input = message.get('input', {})
        if tool_name == 'browser':
            tool_input = self._scale_browser_coordinates(tool_input)
        st.code(f"Tool Use: {tool_name}\nInput: {tool_input}")

    def _render_stored_tool_result(self, message: dict):
        tool_id = message.get("tool_use_id")
        if tool_id and tool_id in self.session_state.tools:
            self._render_tool_result(self.session_state.tools[tool_id])

    def render_conversation_history(self, messages: list):
        for message in messages:
            self._render_message_by_role(message)

    def _render_message_by_role(self, message: dict):
        role_handlers = {
            "user": lambda m: self._render_user_content(m["content"]),
            "assistant": lambda m: self._render_assistant_content(m["content"]),
        }
        handler = role_handlers.get(message["role"])
        if handler:
            handler(message)

    def _render_user_content(self, content):
        for item in self._normalize_content(content):
            if isinstance(item, dict) and item.get("type") == "image":
                continue  # images aren't re-shown in history

            if isinstance(item, dict):
                if item.get("type") == "text":
                    self.render(Sender.USER, item.get("text", ""))
                else:
                    self.render(Sender.USER, cast(BetaContentBlockParam, item))
            else:
                self.render(Sender.USER, item)

    def _render_assistant_content(self, content):
        for item in self._normalize_content(content):
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_id = item.get("tool_use_id")
                if tool_id and tool_id in self.session_state.tools:
                    self.render(Sender.TOOL, self.session_state.tools[tool_id])
            elif isinstance(item, dict):
                self.render(Sender.BOT, cast(BetaContentBlockParam, item))
            else:
                self.render(Sender.BOT, item)

    def _normalize_content(self, content):
        """A message's content can be a single item or a list - normalize
        to a list so callers don't need two code paths."""
        return content if isinstance(content, list) else [content]
