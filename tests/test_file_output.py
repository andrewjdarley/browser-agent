"""Tests for FileOutputTool (the save_file tool)."""

import pytest

from browser_use_demo.tools.base import ToolError
from browser_use_demo.tools.file_output import FileOutputTool


class TestFileOutputTool:
    @pytest.mark.asyncio
    async def test_writes_file_into_run_dir(self, tmp_path):
        tool = FileOutputTool(run_dir=tmp_path)
        result = await tool(filename="out.csv", content="a,b\n1,2\n")
        written = tmp_path / "out.csv"
        assert written.read_text() == "a,b\n1,2\n"
        assert "out.csv" in result.output

    @pytest.mark.asyncio
    async def test_overwrites_existing_file(self, tmp_path):
        tool = FileOutputTool(run_dir=tmp_path)
        await tool(filename="out.txt", content="first")
        await tool(filename="out.txt", content="second")
        assert (tmp_path / "out.txt").read_text() == "second"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_filename",
        ["../escape.txt", "sub/dir.txt", "sub\\dir.txt", "", ".", ".."],
    )
    async def test_rejects_path_traversal_and_invalid_names(self, tmp_path, bad_filename):
        tool = FileOutputTool(run_dir=tmp_path)
        with pytest.raises(ToolError):
            await tool(filename=bad_filename, content="x")

    @pytest.mark.asyncio
    async def test_rejected_filename_never_escapes_run_dir(self, tmp_path):
        tool = FileOutputTool(run_dir=tmp_path)
        outside = tmp_path.parent / "escaped.txt"
        try:
            with pytest.raises(ToolError):
                await tool(filename="../escaped.txt", content="x")
            assert not outside.exists()
        finally:
            outside.unlink(missing_ok=True)
