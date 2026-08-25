"""Lets the agent persist a generated deliverable (CSV, JSON, markdown report,
etc.) to a file the user can actually find and open - replacing the broken
pattern of triggering a browser download inside the containerized Chromium,
which saves into the container's filesystem rather than the host's."""

from pathlib import Path

from .base import ToolError, ToolResult

SAVE_FILE_INPUT_SCHEMA: dict = {
    "properties": {
        "filename": {
            "description": "Filename to save as, e.g. 'merged_prs.csv'. Basename only - no directories.",
            "type": "string",
        },
        "content": {
            "description": "The full text content to write (e.g. CSV, JSON, or markdown text).",
            "type": "string",
        },
    },
    "required": ["filename", "content"],
    "type": "object",
}

SAVE_FILE_DESCRIPTION = (
    "Save a generated text file (CSV, JSON, markdown report, etc.) so the "
    "user can download it. Use this for any deliverable file instead of "
    "trying to trigger a browser download - browser downloads save inside "
    "the automated browser's own environment, not somewhere the user can "
    "find them. The saved file appears in the Streamlit sidebar for download."
)


class FileOutputTool:
    """Writes text files into a run-scoped output directory."""

    name = "save_file"

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    async def __call__(self, *, filename: str, content: str) -> ToolResult:
        if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
            raise ToolError(
                f"Invalid filename {filename!r}: must be a plain filename "
                f"with no path separators."
            )

        path = self.run_dir / filename
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise ToolError(f"Failed to save {filename}: {e}") from e

        return ToolResult(
            output=f"Saved file: {filename} ({len(content)} bytes) at {path}"
        )
