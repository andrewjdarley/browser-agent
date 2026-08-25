"""
Browser Use Demo - Streamlit interface for browser automation with Claude
"""

import asyncio
import base64
import io
import json
import os
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import PosixPath

import streamlit as st
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from browser_use_demo.agent_sdk_bridge import sdk_content_to_tool_result, strip_mcp_prefix
from browser_use_demo.guardrails import GuardrailPolicy
from browser_use_demo.loop import build_options
from browser_use_demo.message_renderer import MessageRenderer, Sender
from browser_use_demo.model_config import MAIN_MODEL
from browser_use_demo.run_context import get_run_dir, new_run_id
from browser_use_demo.run_logger import RunLogger
from browser_use_demo.text_utils import clean_text_extraction_markers
from browser_use_demo.tools import ToolResult
from browser_use_demo.tools.sub_browser_queue import SubBrowserQueue

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

# Process-wide (NOT st.session_state - a plain module global, shared across
# every session Streamlit's one server process handles), holding whichever
# session is currently the "main" one. Exists so the standalone
# ?view=queue page - a separate browser tab/iframe, and therefore a
# genuinely different session_state instance AND a different Streamlit
# session thread, with no access to the main session's own state - can
# still find and render the SAME SubBrowserQueue's live activity.
# Deliberately single-slot, not keyed by run_id: this app is a single-user,
# one-session-at-a-time demo (see image/static_content/index.html, which
# embeds this app's own /?view=queue as its third pane specifically to read
# this), so there's no real multi-session case to key against, and adding
# one would need the two iframes to coordinate on a run_id neither knows
# before the main session has actually started.
#
# "previews" holds the last-computed live-preview images (see
# render_queue_heartbeat), not just the queue object itself, because
# capturing them requires awaiting Playwright calls on Page objects bound
# to the MAIN session's own asyncio event loop - asyncio loops aren't
# thread-safe to drive from a different thread (which the ?view=queue
# session runs on), so that capture can only safely happen on the main
# session's own periodic tick. The queue-only view is a pure reader of
# whatever was last published here - never runs that capture itself.
_ACTIVE_SESSION = {"queue": None, "previews": {}}

STREAMLIT_STYLE = """
<style>
    /* Hide the streamlit deploy button */
    .stDeployButton {
        visibility: hidden;
    }
    section[data-testid="stSidebar"] {
        width: 360px !important;
    }
    /* Make the chat input stick to the bottom */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background: white;
        z-index: 999;
    }
</style>
"""

# Models for browser automation. First entry is the default (most capable)
# unless overridden via BROWSER_USE_MAIN_MODEL (model_config.MAIN_MODEL) -
# if that's set to something not already in this list, it's added so it's
# both the default and a selectable option, not just a mismatched default.
BROWSER_COMPATIBLE_MODELS = [
    "claude-opus-5",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
]
if MAIN_MODEL and MAIN_MODEL not in BROWSER_COMPATIBLE_MODELS:
    BROWSER_COMPATIBLE_MODELS = [MAIN_MODEL, *BROWSER_COMPATIBLE_MODELS]
DEFAULT_MODEL_INDEX = BROWSER_COMPATIBLE_MODELS.index(MAIN_MODEL) if MAIN_MODEL else 0


def setup_state():
    """Initialize session state variables."""
    # Import here to avoid circular imports when browser_tool lambda is evaluated
    from browser_use_demo.tools import BrowserTool, FileOutputTool

    # Computed up front (not as lambdas) since browser_tool/file_output_tool
    # below both need the same run_dir - avoids relying on dict iteration order.
    run_id = st.session_state.get("run_id") or new_run_id()
    run_dir = get_run_dir(run_id)

    # Define all defaults in one place - use lambdas for lazy evaluation of complex values
    defaults = {
        # UI State
        "messages": [],
        "system_prompt": "",
        "hide_screenshots": False,
        "rendered_message_count": 0,  # Track rendered messages to avoid re-rendering
        "last_error": None,  # Store last error message to display persistently
        "restriction_mode": "none",  # Guardrail toggle - see the sidebar widget below
        # API Configuration
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "max_turns": 200,
        "model": BROWSER_COMPATIBLE_MODELS[DEFAULT_MODEL_INDEX],
        # Runtime State
        "tools": {},
        "event_loop": None,  # Persistent event loop for async operations
        "chat_disabled": False,  # Simple flag to disable chat input
        "active_messages": [],  # Store messages for current interaction
        "active_response_container": None,  # Container reference for streaming responses
        "agent_client": None,  # Persistent ClaudeSDKClient, connected lazily on first message
        "agent_client_config": None,  # Config the client was built with, to detect sidebar changes
        # Output directory for this session's screenshots/saved files
        "run_id": run_id,
        "run_dir": run_dir,
        # Complex initialization - tools (inline lambdas)
        "browser_tool": lambda: BrowserTool(run_dir=run_dir),
        "file_output_tool": lambda: FileOutputTool(run_dir=run_dir),
        "run_logger": lambda: RunLogger(run_dir),
        # One instance for the whole session (not rebuilt on reconnect, unlike
        # build_options' other args) so the sidebar toggle can flip `.mode` in
        # place and pending approvals survive a model/max_turns-triggered
        # reconnect. Depends on browser_tool/run_logger already being set -
        # must stay after them (dict iteration order, see the loop below).
        "guardrail_policy": lambda: GuardrailPolicy(
            mode="none",
            browser_tool=st.session_state.browser_tool,
            run_logger=st.session_state.run_logger,
        ),
        # Same reasoning as guardrail_policy - one instance for the session,
        # its max_fanout flipped in place from the sidebar. Must also stay
        # after browser_tool in this dict.
        "sub_browser_queue": lambda: SubBrowserQueue(
            browser_tool=st.session_state.browser_tool,
            run_dir=run_dir,
            max_fanout=8,
        ),
    }

    # Apply all defaults - evaluate lambdas when needed
    for key, default_value in defaults.items():
        if key not in st.session_state:
            # If it's a callable (lambda), call it to get the actual value
            if callable(default_value):
                st.session_state[key] = default_value()
            else:
                st.session_state[key] = default_value


def create_transcript_zip(messages: list, include_images: bool = False) -> bytes:
    """Create a ZIP archive containing the transcript and optionally images.

    Args:
        messages: List of message dictionaries from session state
        include_images: Whether to include images as separate files

    Returns:
        Bytes of the ZIP archive
    """
    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if include_images:
            # Extract images and create transcript with file references
            transcript_json, image_files = extract_images_from_messages(messages)

            # Add images to ZIP
            for idx, img_data in enumerate(image_files):
                filename = f"images/screenshot_{idx+1:04d}.png"
                try:
                    img_bytes = base64.b64decode(img_data)
                    zip_file.writestr(filename, img_bytes)
                except Exception as e:
                    print(f"Error adding image to ZIP: {e}")

            # Add README
            readme_content = f"""Browser Use Demo - Conversation Transcript
Generated: {datetime.now().isoformat()}

This archive contains:
- transcript.json: The conversation transcript
- images/: {len(image_files)} screenshot images referenced in the transcript

The transcript is in JSON format with images stored as separate PNG files.
Image references in the transcript point to files in the images/ directory.
"""
            zip_file.writestr("README.txt", readme_content)
        else:
            # Just create transcript without images
            transcript_json = format_transcript_for_download(messages, False)

            readme_content = f"""Browser Use Demo - Conversation Transcript
Generated: {datetime.now().isoformat()}

This archive contains:
- transcript.json: The conversation transcript (text only)

The transcript is in JSON format and includes all text messages from the conversation.
"""
            zip_file.writestr("README.txt", readme_content)

        # Add the transcript JSON to the ZIP
        zip_file.writestr("transcript.json", transcript_json)

    # Get the ZIP file bytes
    zip_buffer.seek(0)
    return zip_buffer.read()


class ImageExtractor:
    """Helper class to extract images and track their file references."""

    def __init__(self):
        self.image_files = []
        self.image_counter = 0

    def extract_image(self, source: dict) -> dict:
        """Extract an image and return a file reference."""
        if source.get("type") == "base64":
            self.image_counter += 1
            self.image_files.append(source.get("data", ""))
            return {
                "type": "image",
                "file": f"images/screenshot_{self.image_counter:04d}.png"
            }
        else:
            return {"type": "image", "note": "No image data"}

    def process_image_content(self, item: dict) -> dict:
        """Process image content type."""
        source = item.get("source", {})
        return self.extract_image(source)

    def process_text_content(self, item: dict) -> dict:
        """Process text content type."""
        return {
            "type": "text",
            "text": clean_text_extraction_markers(item.get("text", ""))
        }

    def process_tool_use_content(self, item: dict) -> dict:
        """Process tool use content type."""
        return {
            "type": "tool_use",
            "name": item.get("name", ""),
            "input": item.get("input", {})
        }

    def process_tool_result_content(self, item: dict) -> dict:
        """Process tool result content type."""
        tool_content = []
        for content_item in item.get("content", []):
            if isinstance(content_item, dict):
                content_type = content_item.get("type")
                if content_type == "image":
                    source = content_item.get("source", {})
                    tool_content.append(self.extract_image(source))
                elif content_type == "text":
                    tool_content.append(self.process_text_content(content_item))
                else:
                    tool_content.append(content_item)

        return {
            "type": "tool_result",
            "tool_use_id": item.get("tool_use_id", ""),
            "content": tool_content
        }

    def process_default_content(self, item: dict) -> dict:
        """Default processor for unknown content types."""
        return _format_content_item(item, False)


def extract_images_from_messages(messages: list) -> tuple:
    """Extract images from messages and create transcript with file references.

    Returns:
        Tuple of (transcript_json, list_of_base64_image_data)
    """
    extractor = ImageExtractor()

    # Content type processors
    processors = {
        "image": extractor.process_image_content,
        "text": extractor.process_text_content,
        "tool_use": extractor.process_tool_use_content,
        "tool_result": extractor.process_tool_result_content,
    }

    def process_content(content):
        """Process content using appropriate processors."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            processed = []
            for item in content:
                if isinstance(item, dict):
                    content_type = item.get("type")
                    processor = processors.get(content_type, extractor.process_default_content)
                    processed.append(processor(item))
                else:
                    processed.append(str(item))
            return processed
        else:
            return str(content)

    # Build transcript
    transcript = {
        "timestamp": datetime.now().isoformat(),
        "format_version": "2.0",
        "image_storage": "separate_files",
        "conversation": []
    }

    # Process all messages
    for message in messages:
        cleaned_message = {
            "role": message.get("role"),
            "timestamp": datetime.now().isoformat(),
            "content": process_content(message.get("content", ""))
        }
        transcript["conversation"].append(cleaned_message)

    return json.dumps(transcript, indent=2, ensure_ascii=False), extractor.image_files


def format_transcript_for_download(messages: list, include_images: bool = False) -> str:
    """Format conversation messages into a readable transcript.

    Args:
        messages: List of message dictionaries from session state
        include_images: Whether to include base64 image data in the transcript

    Returns:
        Formatted JSON string of the conversation
    """
    transcript = {
        "timestamp": datetime.now().isoformat(),
        "format_version": "1.0",
        "includes_images": include_images,
        "conversation": []
    }

    for message in messages:
        cleaned_message = {
            "role": message.get("role"),
            "timestamp": datetime.now().isoformat(),
            "content": _format_message_content(message.get("content", ""), include_images)
        }
        transcript["conversation"].append(cleaned_message)

    return json.dumps(transcript, indent=2, ensure_ascii=False)


def _format_text_content(item: dict, include_images: bool = False) -> dict:
    """Format a text content block."""
    return {
        "type": "text",
        "text": clean_text_extraction_markers(item.get("text", ""))
    }


def _format_tool_use_content(item: dict, include_images: bool = False) -> dict:
    """Format a tool use content block."""
    return {
        "type": "tool_use",
        "name": item.get("name", ""),
        "input": item.get("input", {})
    }


def _format_tool_result_content(item: dict, include_images: bool = False) -> dict:
    """Format a tool result content block."""
    tool_content = []
    for content_item in item.get("content", []):
        if isinstance(content_item, dict):
            content_type = content_item.get("type")
            if content_type == "text":
                text = clean_text_extraction_markers(content_item.get("text", ""))
                tool_content.append({"type": "text", "text": text})
            elif content_type == "image":
                if include_images:
                    source = content_item.get("source", {})
                    if source.get("type") == "base64":
                        tool_content.append({
                            "type": "image",
                            "media_type": source.get("media_type", "image/png"),
                            "base64_data": source.get("data", "")
                        })
                else:
                    tool_content.append({"type": "image", "note": "Screenshot taken"})

    return {
        "type": "tool_result",
        "tool_use_id": item.get("tool_use_id", ""),
        "content": tool_content
    }


def _format_image_content(item: dict, include_images: bool = False) -> dict:
    """Format an image content block."""
    if include_images:
        source = item.get("source", {})
        if source.get("type") == "base64":
            return {
                "type": "image",
                "media_type": source.get("media_type", "image/png"),
                "base64_data": source.get("data", "")
            }
    return {"type": "image", "note": "Image/Screenshot included"}


def _format_default_content(item: dict, include_images: bool = False) -> dict:
    """Format unknown content types - fallback handler."""
    return item


# Strategy pattern: Map content types to their formatting functions
CONTENT_FORMATTERS = {
    "text": _format_text_content,
    "tool_use": _format_tool_use_content,
    "tool_result": _format_tool_result_content,
    "image": _format_image_content,
}


def _format_content_item(item, include_images: bool = False):
    """Format a single content item using the appropriate formatter.

    Uses the Strategy pattern to dispatch to the correct formatter based on content type.
    """
    if not isinstance(item, dict):
        return str(item)

    content_type = item.get("type")
    formatter = CONTENT_FORMATTERS.get(content_type, _format_default_content)
    return formatter(item, include_images)


def _format_message_content(content, include_images: bool = False):
    """Format message content based on its type.

    This is the main entry point that handles different content structures.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return [_format_content_item(item, include_images) for item in content]
    else:
        return str(content)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def render_artifacts_panel(run_dir):
    """Sidebar panel listing files the agent has written this session -
    screenshots and save_file outputs - each downloadable."""
    st.subheader("Session Files")

    files = sorted(
        (p for p in run_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        st.info("No files yet - screenshots and saved files will appear here", icon="🗂️")
        return

    for path in files:
        data = path.read_bytes()
        size_kb = len(data) / 1024
        if size_kb > 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        elif size_kb >= 1:
            size_str = f"{size_kb:.1f} KB"
        else:
            size_str = f"{len(data)} B"

        if path.suffix.lower() in _IMAGE_SUFFIXES:
            st.image(data, caption=f"{path.name} ({size_str})")

        st.download_button(
            label=f"⬇️ {path.name} ({size_str})",
            data=data,
            file_name=path.name,
            key=f"download_{path.name}",
            use_container_width=True,
        )


@st.fragment(run_every="2s")
def render_queue_heartbeat():
    """Lives in the MAIN chat session's own page (not the ?view=queue page -
    see render_sub_browser_panel/render_queue_only_view below) purely to keep
    driving the shared event loop and publishing fresh live previews into
    _ACTIVE_SESSION, independent of whether anyone has the dedicated queue
    view open. Two jobs, both real, neither skippable:

    1. pump_and_preview advances SubBrowserQueue's scheduled item tasks -
       asyncio.create_task only SCHEDULES an item's work, it doesn't run it,
       and nothing else keeps this session's loop spinning once the agent
       turn that queued the items has ended (see SubBrowserQueue.pump_async).
    2. Captures live previews (Playwright screenshot calls on Page objects
       bound to THIS session's own event loop) and publishes them to
       _ACTIVE_SESSION["previews"] - the ?view=queue page runs on a
       different Streamlit session thread, and asyncio loops aren't
       thread-safe to drive cross-thread, so it can only ever read what
       gets published here, never capture previews itself.

    Renders a one-line status + link, not the full grid - the dedicated
    /?view=queue page (see index.html's third pane) is where that lives.
    """
    queue = st.session_state.sub_browser_queue
    _ACTIVE_SESSION["queue"] = queue
    loop = get_or_create_event_loop()
    _ACTIVE_SESSION["previews"] = loop.run_until_complete(queue.pump_and_preview())

    snap = queue.snapshot()
    status_bits = [f"{snap['pending_count']} pending", f"{snap['in_progress_count']} running"]
    if snap["paused"]:
        status_bits.append("⏸ paused")
    st.caption(f"🧪 Sub-browser queue: {' · '.join(status_bits)} — live view: `/?view=queue`")


def render_sub_browser_panel():
    """The live grid itself - a pure reader of _ACTIVE_SESSION, safe to call
    from any session (in particular the standalone ?view=queue page, which
    runs on a different session thread than the one that actually captures
    these previews - see render_queue_heartbeat)."""
    queue = _ACTIVE_SESSION["queue"]
    if queue is None:
        st.info("Waiting for the main chat session to start...", icon="🧪")
        return

    previews = _ACTIVE_SESSION["previews"]
    snap = queue.snapshot()
    status_bits = [f"{snap['pending_count']} pending", f"{snap['in_progress_count']} running"]
    if snap["paused"]:
        status_bits.append("⏸ paused")
    st.caption(" · ".join(status_bits))

    if not previews:
        st.info("No sub-browsers running right now - ask the agent to queue some screenshots.", icon="🧪")
    else:
        cols = st.columns(2)
        for i, (item_id, b64_jpeg) in enumerate(previews.items()):
            with cols[i % len(cols)]:
                st.image(base64.b64decode(b64_jpeg), caption=item_id, use_container_width=True)

    if snap["completed"]:
        with st.expander(f"Finished ({len(snap['completed'])})", expanded=False):
            for item in snap["completed"][:20]:
                icon = "✅" if item["status"] == "done" else "⚠️"
                detail = f"{len(item['screenshots'])} screenshot(s)" if item["status"] == "done" else item["error"]
                st.caption(f"{icon} {item['id']} — {detail}")


@st.fragment(run_every="2s")
def _render_sub_browser_panel_fragment():
    render_sub_browser_panel()


def render_queue_only_view():
    """The standalone page served at /?view=queue - index.html's third pane
    points here (see image/static_content/index.html) so the live grid gets
    its own dedicated column instead of sharing space with the chat. No
    sidebar, no chat, no setup_state() (this session never needs its own
    BrowserTool/agent client - it's a read-only view onto the main
    session's queue, via _ACTIVE_SESSION)."""
    st.set_page_config(page_title="Sub-browser Queue", page_icon="🧪", layout="wide")
    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)
    st.title("🧪 Sub-browser Queue")
    _render_sub_browser_panel_fragment()


def authenticate():
    """Handle API key authentication."""
    if not st.session_state.api_key:
        st.error("Please provide your Anthropic API key in the sidebar")
        st.stop()
    return True


def get_or_create_event_loop():
    """Get existing event loop or create a new one if needed.

    This function ensures we have a valid event loop for async operations,
    reusing existing loops when possible to avoid Playwright issues with asyncio.run().

    Returns:
        The active asyncio event loop.
    """
    if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
        st.session_state.event_loop = asyncio.new_event_loop()

    asyncio.set_event_loop(st.session_state.event_loop)
    return st.session_state.event_loop


def _agent_config_key() -> tuple:
    """Config that requires reconnecting the SDK client if changed mid-session."""
    return (
        st.session_state.model,
        st.session_state.api_key,
        st.session_state.system_prompt,
        st.session_state.max_turns,
    )


async def get_or_create_agent_client() -> ClaudeSDKClient:
    """Get the persistent SDK client for this session, connecting on first use
    and reconnecting if model/API key/system prompt/max turns changed in the
    sidebar since the client was created (options are frozen at connect time)."""
    current_config = _agent_config_key()
    if (
        st.session_state.agent_client is not None
        and st.session_state.get("agent_client_config") != current_config
    ):
        await disconnect_agent_client()

    if st.session_state.agent_client is None:
        options = build_options(
            model=st.session_state.model,
            system_prompt_suffix=st.session_state.system_prompt,
            browser_tool=st.session_state.browser_tool,
            file_output_tool=st.session_state.file_output_tool,
            run_logger=st.session_state.run_logger,
            api_key=st.session_state.api_key,
            max_turns=st.session_state.max_turns,
            guardrail_policy=st.session_state.guardrail_policy,
            sub_browser_queue=st.session_state.sub_browser_queue,
        )
        client = ClaudeSDKClient(options=options)
        await client.connect()
        st.session_state.agent_client = client
        st.session_state.agent_client_config = current_config
        st.session_state.run_logger.log_run_start(model=st.session_state.model)
    return st.session_state.agent_client


async def disconnect_agent_client():
    """Disconnect and drop the persistent SDK client, if any."""
    client = st.session_state.get("agent_client")
    if client is not None:
        await client.disconnect()
        st.session_state.agent_client = None
        st.session_state.run_logger.log_run_end()


_API_ERROR_MESSAGES = {
    "authentication_failed": "Authentication failed - check your Anthropic API key in the sidebar.",
    "billing_error": "Billing error on your Anthropic account.",
    "rate_limit": "Rate limit exceeded. Please wait before sending another message.",
    "invalid_request": "Invalid request sent to the API.",
    "server_error": "Anthropic API server error.",
    "unknown": "Unknown API error.",
}


async def run_agent(user_input: str):
    """Run the browser automation agent with user input."""
    try:
        # Ensure chat is disabled while processing
        st.session_state.chat_disabled = True

        # Create message renderer
        renderer = MessageRenderer(st.session_state)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in active container
        with st.session_state.active_response_container:
            renderer.render(Sender.USER, user_input)

        # Clear active messages for new interaction
        st.session_state.active_messages = []

        client = await get_or_create_agent_client()
        await client.query(user_input)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                if message.error:
                    error_msg = _API_ERROR_MESSAGES.get(message.error, message.error)
                    st.session_state.last_error = {"message": error_msg, "traceback": None}
                    st.session_state.run_logger.log_error(error_msg, context="api_call")
                    with st.session_state.active_response_container:
                        st.error(error_msg)
                    continue

                content_blocks = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_block = {"type": "text", "text": block.text}
                        content_blocks.append(text_block)
                        with st.session_state.active_response_container:
                            renderer.render(Sender.BOT, text_block)
                    elif isinstance(block, ToolUseBlock):
                        tool_use_block = {
                            "type": "tool_use",
                            "id": block.id,
                            "name": strip_mcp_prefix(block.name),
                            "input": block.input,
                        }
                        content_blocks.append(tool_use_block)
                        with st.session_state.active_response_container:
                            renderer.render(Sender.BOT, tool_use_block)
                if content_blocks:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )

            elif isinstance(message, UserMessage):
                blocks = message.content if isinstance(message.content, list) else []
                content_blocks = []
                for block in blocks:
                    if isinstance(block, ToolResultBlock):
                        tool_result = sdk_content_to_tool_result(
                            block.content, block.is_error
                        )
                        st.session_state.tools[block.tool_use_id] = tool_result
                        content_blocks.append(
                            {"type": "tool_result", "tool_use_id": block.tool_use_id}
                        )
                        with st.session_state.active_response_container:
                            renderer.render(Sender.TOOL, tool_result)
                if content_blocks:
                    st.session_state.messages.append(
                        {"role": "user", "content": content_blocks}
                    )

            elif isinstance(message, ResultMessage) and message.is_error:
                error_msg = message.result or "The agent turn ended with an error."
                st.session_state.last_error = {"message": error_msg, "traceback": None}
                st.session_state.run_logger.log_error(error_msg, context="tool_execution")
                with st.session_state.active_response_container:
                    st.error(error_msg)

        # Re-enable chat input
        st.session_state.chat_disabled = False

        # Trigger a rerun to update the history display
        st.rerun()

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        error_traceback = traceback.format_exc()
        st.session_state.last_error = {"message": error_msg, "traceback": error_traceback}
        st.session_state.run_logger.log_error(error_msg, context="streamlit_run_agent")
        with st.session_state.active_response_container:
            st.error(error_msg)
            st.code(error_traceback)
        st.session_state.chat_disabled = False
        st.rerun()


def main():
    """Main application entry point."""
    # Standalone read-only queue view - a genuinely separate Streamlit
    # session (own thread, own session_state), never the main chat session
    # itself. Dispatched first, before set_page_config/setup_state, since
    # this path sets its own page config and doesn't need a BrowserTool/
    # agent client at all. See render_queue_only_view's docstring.
    if st.query_params.get("view") == "queue":
        render_queue_only_view()
        return

    st.set_page_config(
        page_title="Claude Browser Use Demo",
        page_icon="🌐",
        layout="wide"
    )

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    setup_state()


    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection (only browser-compatible models)
        st.selectbox(
            "Model", options=BROWSER_COMPATIBLE_MODELS, index=DEFAULT_MODEL_INDEX, key="model"
        )

        # API Key
        st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.api_key,
            key="api_key",
            help="Get your API key from https://console.anthropic.com",
        )

        # Max turns (Agent SDK bounds tool-use round trips, not raw tokens)
        st.number_input(
            "Max Turns",
            min_value=1,
            max_value=200,
            value=st.session_state.max_turns,
            step=1,
            key="max_turns",
            help="Maximum number of tool-use round trips per message",
        )

        # System prompt
        st.text_area(
            "Additional System Prompt",
            value=st.session_state.system_prompt,
            key="system_prompt",
            help="Add custom instructions for the browser agent",
        )

        # Hide screenshots
        st.checkbox(
            "Hide Screenshots",
            value=st.session_state.hide_screenshots,
            key="hide_screenshots",
            help="Hide screenshot outputs in the chat",
        )

        # Guardrails - restriction mode toggle. Mutating guardrail_policy.mode
        # in place takes effect on the agent's very next tool call; unlike
        # model/max_turns this doesn't need a client reconnect (see
        # GuardrailPolicy's docstring and _agent_config_key, which
        # deliberately doesn't include this).
        st.divider()
        st.subheader("🛡️ Guardrails")
        st.radio(
            "Action restriction",
            options=["none", "all", "manual"],
            format_func=lambda m: {
                "none": "None (default - unrestricted)",
                "all": "All (auto-block irreversible-looking actions)",
                "manual": "Manual (block, but you can approve)",
            }[m],
            key="restriction_mode",
            help=(
                "Deterministic pattern checks only (form submits, non-GET JS "
                "requests, destructive-looking click targets) - not a full "
                "policy engine and not an LLM judgment call. None: current "
                "behavior. All: matched actions are denied outright. Manual: "
                "matched actions are denied, but appear below the chat input "
                "for you to approve and let the agent retry."
            ),
        )
        st.session_state.guardrail_policy.mode = st.session_state.restriction_mode

        # No sidebar control for the sub-browser queue's fanout/pacing - it
        # never had anything worth showing here (the live grid lives at
        # /?view=queue, embedded as index.html's third pane - see
        # render_queue_only_view/render_sub_browser_panel), and max_fanout/
        # min_interval_s are now agent-settable directly via queue_screenshots'
        # own arguments (see tools/sub_browser_queue.py) instead of a human
        # dialing in a fixed value up front.

        # Conversation Management Section
        st.divider()
        st.subheader("💬 Conversation")

        # Download transcript options and button
        if st.session_state.messages:
            # Checkbox to include images
            include_images = st.checkbox(
                "Include images in transcript",
                value=False,
                help="Include screenshots as separate PNG files in a ZIP archive"
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if include_images:
                # Generate ZIP with images
                zip_data = create_transcript_zip(
                    st.session_state.messages,
                    include_images=True
                )

                # Show file size
                file_size_kb = len(zip_data) / 1024
                if file_size_kb > 1024:
                    size_str = f"{file_size_kb / 1024:.1f} MB"
                else:
                    size_str = f"{file_size_kb:.1f} KB"

                st.download_button(
                    label=f"📦 Download Transcript ZIP ({size_str})",
                    data=zip_data,
                    file_name=f"browser_demo_transcript_{timestamp}.zip",
                    mime="application/zip",
                    help=f"Download conversation with images as ZIP archive ({size_str})",
                    type="primary",
                    use_container_width=True,
                )
            else:
                # Generate JSON only
                transcript_json = format_transcript_for_download(
                    st.session_state.messages,
                    include_images=False
                )

                # Show file size
                file_size_kb = len(transcript_json.encode('utf-8')) / 1024
                if file_size_kb > 1024:
                    size_str = f"{file_size_kb / 1024:.1f} MB"
                else:
                    size_str = f"{file_size_kb:.1f} KB"

                st.download_button(
                    label=f"📄 Download Transcript JSON ({size_str})",
                    data=transcript_json,
                    file_name=f"browser_demo_transcript_{timestamp}.json",
                    mime="application/json",
                    help=f"Download conversation transcript as JSON ({size_str})",
                    type="primary",
                    use_container_width=True,
                )
        else:
            st.info("No messages to download yet", icon="💬")

        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
                st.session_state.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(st.session_state.event_loop)
            st.session_state.event_loop.run_until_complete(disconnect_agent_client())

            st.session_state.messages = []
            st.session_state.tools = {}
            st.session_state.rendered_message_count = 0
            st.session_state.active_messages = []
            st.session_state.chat_disabled = False
            st.rerun()

        # Reset browser to blank page
        if st.button("Reset Browser", type="secondary"):
            async def reset_browser():
                if st.session_state.browser_tool._page:
                    await st.session_state.browser_tool._page.goto("about:blank")

            if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
                st.session_state.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(st.session_state.event_loop)
            st.session_state.event_loop.run_until_complete(reset_browser())
            st.rerun()

        st.divider()
        render_artifacts_panel(st.session_state.run_dir)

    # Main chat interface
    st.title("🌐 Claude Browser Use Demo")
    st.markdown(
        "This demo showcases Claude's ability to interact with web browsers using "
        "Playwright automation. Ask Claude to navigate websites, fill forms, "
        "extract information, and more!"
    )

    # Authenticate
    if not authenticate():
        return

    render_queue_heartbeat()

    # Create container for conversation history
    history_container = st.container()

    # Display conversation history in the history container
    renderer = MessageRenderer(st.session_state)
    with history_container:
        renderer.render_conversation_history(st.session_state.messages)

    # Create container for active/streaming responses
    active_container = st.container()
    st.session_state.active_response_container = active_container

    # Show persistent error message if there is one
    if st.session_state.last_error:
        st.error(st.session_state.last_error["message"])
        if st.session_state.last_error["traceback"]:
            with st.expander("Show full traceback"):
                st.code(st.session_state.last_error["traceback"])
        if st.button("Clear Error"):
            st.session_state.last_error = None
            st.rerun()

    # Guardrail approvals pending (Manual restriction mode only - see the
    # sidebar toggle). Rendered here, in the main chat area, rather than the
    # sidebar - it's about a specific blocked action from the conversation,
    # not session configuration. Approving adds the call's fingerprint to
    # guardrail_policy.approved (single-use) - the agent still has to retry
    # the exact same action for it to actually go through, since the turn
    # that got denied has already ended by the time a human can click here.
    pending = st.session_state.guardrail_policy.pending
    if pending:
        st.warning(f"🛡️ {len(pending)} action(s) blocked by guardrails, awaiting approval")
        for entry in list(pending):
            action = entry["tool_input"].get("action", entry["tool_name"])
            with st.expander(f"{action} — {entry['rule']}", expanded=True):
                st.code(json.dumps(entry["tool_input"], indent=2), language="json")
                st.caption(entry["reason"])
                approve_col, dismiss_col = st.columns(2)
                if approve_col.button(
                    "✅ Approve", key=f"guardrail_approve_{entry['fingerprint']}", use_container_width=True
                ):
                    st.session_state.guardrail_policy.approved.add(entry["fingerprint"])
                    pending.remove(entry)
                    st.rerun()
                if dismiss_col.button(
                    "✖️ Dismiss", key=f"guardrail_dismiss_{entry['fingerprint']}", use_container_width=True
                ):
                    pending.remove(entry)
                    st.rerun()

    # Show status when chat is disabled
    if st.session_state.chat_disabled:
        st.info("🤖 Claude is currently processing your request. Please wait...")

    # Simple callback to disable chat input on submit
    def disable_chat_callback():
        st.session_state.chat_disabled = True

    # Simple chat input with disabled state
    prompt = st.chat_input(
        "Ask Claude to browse the web...",
        disabled=st.session_state.chat_disabled,
        on_submit=disable_chat_callback
    )

    if prompt:
        # Clear any previous error when starting a new request
        st.session_state.last_error = None
        # Process the prompt
        loop = get_or_create_event_loop()
        loop.run_until_complete(run_agent(prompt))


if __name__ == "__main__":
    main()
