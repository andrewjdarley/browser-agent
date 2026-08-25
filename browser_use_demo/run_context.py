"""Per-run output directory.

Single source of truth for "where does this session's stuff go" - imported
by BrowserTool (screenshots), the save_file tool, and the Streamlit artifacts
panel, so all three always agree on the same directory.
"""

import secrets
from datetime import datetime
from pathlib import Path

from .tools.browser import OUTPUT_DIR


def new_run_id() -> str:
    """A run id that's chronologically sortable and collision-resistant
    enough for concurrent Streamlit sessions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{secrets.token_hex(4)}"


def get_run_dir(run_id: str) -> Path:
    """Return (creating if needed) the output directory for a run id."""
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
