"""Tests for small streamlit.py helpers not covered by test_message_renderer.py.

authenticate() and get_or_create_event_loop() are simple, standalone
functions worth covering directly; the bulk of streamlit.py's behavior
(message rendering, run_agent's SDK message-stream handling) is exercised
end-to-end against the real container rather than mocked here, since it's
mostly plumbing around the Claude Agent SDK's own typed message stream.
"""

import asyncio
from unittest.mock import patch

import pytest

from browser_use_demo import streamlit as app


class TestAuthenticate:
    def test_stops_when_api_key_missing(self, mock_streamlit):
        mock_streamlit["session_state"].api_key = ""
        with patch("streamlit.error") as mock_error, patch("streamlit.stop") as mock_stop:
            mock_stop.side_effect = SystemExit  # st.stop() halts execution
            with pytest.raises(SystemExit):
                app.authenticate()
            mock_error.assert_called_once()

    def test_passes_when_api_key_present(self, mock_streamlit):
        mock_streamlit["session_state"].api_key = "sk-ant-test"
        assert app.authenticate() is True


class TestGetOrCreateEventLoop:
    def test_creates_loop_when_none_exists(self, mock_streamlit):
        mock_streamlit["session_state"].event_loop = None
        loop = app.get_or_create_event_loop()
        try:
            assert isinstance(loop, asyncio.AbstractEventLoop)
            assert not loop.is_closed()
        finally:
            loop.close()

    def test_reuses_existing_open_loop(self, mock_streamlit):
        existing = asyncio.new_event_loop()
        mock_streamlit["session_state"].event_loop = existing
        try:
            loop = app.get_or_create_event_loop()
            assert loop is existing
        finally:
            existing.close()

    def test_replaces_closed_loop(self, mock_streamlit):
        closed = asyncio.new_event_loop()
        closed.close()
        mock_streamlit["session_state"].event_loop = closed
        loop = app.get_or_create_event_loop()
        try:
            assert loop is not closed
            assert not loop.is_closed()
        finally:
            loop.close()
