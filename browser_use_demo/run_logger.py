"""Automatic, always-on, per-run action log - distinct from streamlit.py's
manual "Download Transcript" button (a user-triggered, full-fidelity export)
and the SDK's own internal session JSONL under ~/.claude/projects/ (not
meant for this app to read or rely on). Written incrementally to
run_log.jsonl in the run's output directory (survives a crash, colocated
with the run's screenshots/artifacts), one compact JSON object per line,
driven by SDK hooks. Never logs base64 image data - screenshots are already
on disk (file_path covers that) - so the log stays small and jq-able.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from .agent_sdk_bridge import sdk_content_to_tool_result

_SUMMARY_MAX_CHARS = 200


def _summarize_tool_output(tool_response: Any) -> tuple[Optional[str], bool]:
    """Extract (text_summary, has_image) from a PostToolUse hook's
    tool_response - the raw content list/string our @tool functions
    returned (same shape sdk_content_to_tool_result already parses for the
    Streamlit renderer, so reuse it instead of a second guess at the shape)."""
    result = sdk_content_to_tool_result(tool_response, is_error=None)
    summary = result.output
    if summary and len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[:_SUMMARY_MAX_CHARS] + "..."
    return summary, result.base64_image is not None


class RunLogger:
    """Writes hook-driven JSONL events for one run."""

    def __init__(self, run_dir: Path):
        self.log_path = run_dir / "run_log.jsonl"

    def _append(self, event: dict[str, Any]) -> None:
        event = {"timestamp": time.time(), **event}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_run_start(self, *, model: str) -> None:
        self._append({"event": "run_start", "model": model})

    def log_run_end(self) -> None:
        self._append({"event": "run_end"})

    def log_error(self, message: str, context: str) -> None:
        self._append({"event": "error", "message": message, "context": context})

    async def log_guardrail_block(
        self,
        *,
        mode: str,
        rule: str,
        reason: str,
        tool_name: Optional[str],
        tool_input: Any,
        allowed: bool,
    ) -> None:
        """A guardrail (see guardrails.py) matched a tool call. `allowed` is
        True only for a manual-mode call that had a prior approval consumed
        - a distinct, auditable event from a plain tool_error, since this is
        specifically the "did the elevated-credential agent try something
        the operator wanted gated" record an audit trail exists for."""
        self._append(
            {
                "event": "guardrail_block",
                "mode": mode,
                "rule": rule,
                "reason": reason,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "allowed": allowed,
            }
        )

    # --- Subagent fan-out events (called directly by DispatchSubagentsTool,
    # not via SDK hooks - these are independent ClaudeSDKClient sessions we
    # orchestrate ourselves, not the SDK's native subagent mechanism, so
    # there's no SubagentStart/SubagentStop hook firing for them) ---

    def log_subagent_dispatch(self, dispatch_id: str, items: list[str], shared_instructions: str) -> None:
        self._append(
            {
                "event": "subagent_dispatch",
                "dispatch_id": dispatch_id,
                "items": items,
                "shared_instructions_summary": shared_instructions[:200],
            }
        )

    def log_subagent_result(self, dispatch_id: str, item: str, success: bool) -> None:
        self._append(
            {
                "event": "subagent_result",
                "dispatch_id": dispatch_id,
                "item": item,
                "success": success,
            }
        )

    def log_subagent_join(self, dispatch_id: str, total: int, succeeded: int, failed: int) -> None:
        self._append(
            {
                "event": "subagent_join",
                "dispatch_id": dispatch_id,
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
            }
        )

    # --- Independent verification events (called directly by
    # VerifyFindingTool, same reasoning as the subagent events above: it's
    # our own orchestrated ClaudeSDKClient session, not something an SDK
    # hook fires for) ---

    def log_verification(self, claim: str, confirmed: Optional[bool], uses_remaining: int) -> None:
        self._append(
            {
                "event": "verification",
                "claim": claim[:200],
                "confirmed": confirmed,
                "uses_remaining": uses_remaining,
            }
        )

    # --- SDK hook callbacks: (input_data, tool_use_id, context) -> dict ---
    # All return {} (no intervention) - this logger only observes.

    async def on_user_prompt_submit(self, input_data: dict, tool_use_id, context) -> dict:
        self._append({"event": "user_prompt", "prompt": input_data.get("prompt", "")})
        return {}

    async def on_pre_tool_use(self, input_data: dict, tool_use_id, context) -> dict:
        self._append(
            {
                "event": "tool_call",
                "tool_use_id": input_data.get("tool_use_id") or tool_use_id,
                "tool_name": input_data.get("tool_name"),
                "tool_input": input_data.get("tool_input"),
            }
        )
        return {}

    async def on_post_tool_use(self, input_data: dict, tool_use_id, context) -> dict:
        summary, has_image = _summarize_tool_output(input_data.get("tool_response"))
        self._append(
            {
                "event": "tool_result",
                "tool_use_id": input_data.get("tool_use_id") or tool_use_id,
                "tool_name": input_data.get("tool_name"),
                "output_summary": summary,
                "has_image": has_image,
            }
        )
        return {}

    async def on_post_tool_use_failure(self, input_data: dict, tool_use_id, context) -> dict:
        self._append(
            {
                "event": "tool_error",
                "tool_use_id": input_data.get("tool_use_id") or tool_use_id,
                "tool_name": input_data.get("tool_name"),
            }
        )
        return {}

    async def on_stop(self, input_data: dict, tool_use_id, context) -> dict:
        self._append({"event": "turn_end"})
        return {}

    async def on_subagent_start(self, input_data: dict, tool_use_id, context) -> dict:
        self._append(
            {
                "event": "subagent_start",
                "agent_id": input_data.get("agent_id"),
                "agent_type": input_data.get("agent_type"),
            }
        )
        return {}

    async def on_subagent_stop(self, input_data: dict, tool_use_id, context) -> dict:
        self._append(
            {
                "event": "subagent_stop",
                "agent_id": input_data.get("agent_id"),
                "agent_type": input_data.get("agent_type"),
            }
        )
        return {}
