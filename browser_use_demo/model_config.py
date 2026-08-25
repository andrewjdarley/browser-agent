"""Central place for which Claude model each part of the app uses.

The coordinator's own model is the one visible, user-facing choice (the
Streamlit sidebar dropdown - see streamlit.py's BROWSER_COMPATIBLE_MODELS).
Everything below is about the OTHER models running behind the scenes, which
used to either silently inherit the coordinator's model with no way to
change that, or (find's case) be hardcoded to a specific model string with
no way to change it at all short of editing code. All of these are
overridable via environment variables (picked up from .env - see
docker-compose.yml's env_file) without touching code.
"""

import os

# The coordinator's own default model - pre-selected in the Streamlit
# sidebar dropdown (still changeable there per session; this only sets what
# it starts on). Empty/unset means "use the first entry in
# streamlit.py's BROWSER_COMPATIBLE_MODELS list" - see streamlit.py, which
# also adds this to that list if it isn't already one of the presets.
MAIN_MODEL = os.environ.get("BROWSER_USE_MAIN_MODEL") or None

# Used by the `find` browser action (tools/browser.py's _find) for semantic
# element matching - a separate, direct Anthropic API call, not part of the
# coordinator's own SDK session, so it needs its own model rather than
# inheriting one. This is a fast, narrow "does this element match the
# description" task, not deep reasoning, so a fast/cheap model is a
# reasonable default - override for more accuracy on complex pages.
FIND_MODEL = os.environ.get("BROWSER_USE_FIND_MODEL", "claude-haiku-4-5-20251001")

# Model for each dispatch_subagents worker's independent session. Empty/unset
# (default) means "use whatever model the coordinator itself is running" -
# set this to pin workers to a specific model regardless of the
# coordinator's own choice (e.g. a cheaper model for a large fan-out).
SUBAGENT_MODEL = os.environ.get("BROWSER_USE_SUBAGENT_MODEL") or None

# Model for verify_finding's independent verification session. Empty/unset
# (default) means "use whatever model the coordinator itself is running."
# Since the whole point of verification is catching mistakes the
# coordinator's own model made, there's a reasonable case for pinning this
# to a different (or more capable) model than the coordinator - override to
# do that.
VERIFY_MODEL = os.environ.get("BROWSER_USE_VERIFY_MODEL") or None


def resolve(override: str | None, coordinator_model: str) -> str:
    """An override wins if set; otherwise fall back to the coordinator's
    own model. Shared by loop.py for both SUBAGENT_MODEL and VERIFY_MODEL."""
    return override or coordinator_model
