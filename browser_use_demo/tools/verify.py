"""Independent re-derivation of one specific claim, for when a self-review
in the same context wouldn't be trustworthy.

The coordinator re-checking its own conclusion in the same conversation is
weak: it's reasoning from the same tool-call trail and the same assumptions
that produced the (possibly wrong) answer, so it's disproportionately likely
to just confirm itself. This tool instead spins up ONE fresh ClaudeSDKClient
session - its own BrowserTool, no visibility into the coordinator's reasoning
- and asks it to independently determine whether a claim is true. It can
disagree, because it isn't anchored to how the original answer was reached.

Session-wide budget, not per-call: this is for the rare claim that's
specific, checkable, AND consequential - not a general habit, and not free
(it's a full extra agentic session). Capped at MAX_VERIFICATIONS_PER_SESSION
for the lifetime of one VerifyFindingTool instance (one instance per
coordinator session - see loop.py). Once exhausted, the tool returns an
error rather than silently no-op'ing, so the model knows to flag remaining
uncertainty explicitly instead of presenting an unchecked claim as verified.
"""

import json
import re
from typing import Any

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock
from claude_agent_sdk import tool as sdk_tool

from ..agent_sdk_bridge import MAX_BUFFER_SIZE, build_mcp_server, tool_result_to_sdk_content
from ..run_logger import RunLogger
from .base import ToolError, ToolResult
from .browser import BrowserTool
from .file_output import FileOutputTool

MAX_VERIFICATIONS_PER_SESSION = 2
VERIFY_MAX_TURNS = 20

VERIFY_FINDING_INPUT_SCHEMA: dict = {
    "properties": {
        "claim": {
            "description": (
                "The specific, checkable fact you found and are about to "
                "rely on, stated precisely (e.g. \"Reference 275 on the WW2 "
                "Wikipedia page is the Chubarov 2001 citation\")."
            ),
            "type": "string",
        },
        "context": {
            "description": (
                "Enough context for an independent reviewer to check this "
                "from scratch - where to look (a URL), and what 'correct' "
                "means here. Do NOT explain the method you used to reach "
                "your answer - the point is a genuinely independent check, "
                "not a review of your reasoning."
            ),
            "type": "string",
        },
    },
    "required": ["claim", "context"],
    "type": "object",
}

VERIFY_FINDING_DESCRIPTION = (
    "Get an independent, fresh re-derivation of ONE specific claim before "
    "relying on it. Use this only when a finding is ALL THREE of: specific "
    "(a discrete, falsifiable value - a number, name, date, identifier - not "
    "a vague summary), checkable (there's a concrete way to confirm it "
    "independently of how you found it), and consequential (getting it "
    "wrong would make your final answer or deliverable wrong). Not for "
    "routine intermediate steps or subjective judgments - most findings "
    f"don't need this. Budget: {MAX_VERIFICATIONS_PER_SESSION} uses per "
    "session, so spend it on the claim(s) that matter most."
)

VERIFIER_SYSTEM_PROMPT_ADDENDUM = """<VERIFIER_ROLE>
You are an independent reviewer checking ONE specific claim someone else made. You have not seen their reasoning or method - your job is to determine the truth for yourself, using your own judgment about how to check it, not to review their logic. Don't assume the claim is correct just because it's stated confidently - actively look for evidence that would confirm or contradict it.

When you're done, respond with ONLY a JSON object wrapped in <result>...</result> tags, matching: {"confirmed": true|false, "actual_finding": "<what you independently found>", "explanation": "<brief reasoning>"}. Nothing else after the closing tag.
</VERIFIER_ROLE>"""


class VerifyFindingTool:
    """Runs one independent verification session per call, capped at
    MAX_VERIFICATIONS_PER_SESSION for this instance's lifetime."""

    name = "verify_finding"

    def __init__(
        self,
        *,
        run_dir,
        run_logger: RunLogger,
        api_key: str,
        model: str,
        base_system_prompt: str,
    ):
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.api_key = api_key
        self.model = model
        self.base_system_prompt = base_system_prompt
        self._uses = 0

    async def __call__(self, *, claim: str, context: str) -> ToolResult:
        if self._uses >= MAX_VERIFICATIONS_PER_SESSION:
            return ToolResult(
                error=(
                    f"Verification budget exhausted ({MAX_VERIFICATIONS_PER_SESSION} "
                    "per session already used). If this claim still needs checking, "
                    "say so explicitly in your final answer instead of presenting it "
                    "as verified."
                )
            )
        self._uses += 1
        uses_remaining = MAX_VERIFICATIONS_PER_SESSION - self._uses

        browser_tool = BrowserTool(run_dir=self.run_dir)
        file_output_tool = FileOutputTool(run_dir=self.run_dir)
        try:
            server = build_mcp_server(browser_tool, file_output_tool)
            system_prompt = f"{self.base_system_prompt}\n\n{VERIFIER_SYSTEM_PROMPT_ADDENDUM}"
            options = ClaudeAgentOptions(
                model=self.model,
                system_prompt=system_prompt,
                mcp_servers={"browser_use": server},
                # Deliberately browser-only: no save_file (nothing to
                # produce), no dispatch_subagents/batch_extract/
                # verify_finding (keep this a quick, bounded check, and
                # avoid recursive verification).
                allowed_tools=["mcp__browser_use__browser"],
                max_turns=VERIFY_MAX_TURNS,
                max_buffer_size=MAX_BUFFER_SIZE,
                env={"ANTHROPIC_API_KEY": self.api_key},
            )
            prompt = (
                f"Claim to verify: {claim}\n\nContext: {context}\n\n"
                "Independently determine whether this claim is true."
            )

            final_text_parts: list[str] = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text_parts.append(block.text)

            result = self._parse_result("\n".join(final_text_parts))
            self.run_logger.log_verification(
                claim=claim,
                confirmed=result.get("confirmed"),
                uses_remaining=uses_remaining,
            )
            return ToolResult(output=json.dumps(result, indent=2))
        except Exception as e:
            self.run_logger.log_verification(claim=claim, confirmed=None, uses_remaining=uses_remaining)
            return ToolResult(error=f"Verification failed to run: {e}")
        finally:
            await browser_tool.cleanup()

    @staticmethod
    def _parse_result(text: str) -> dict[str, Any]:
        match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
        if not match:
            return {"error": "verifier did not return a <result> block", "raw_output": text}
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {"error": "verifier's <result> block was not valid JSON", "raw_output": text}


def build_verify_finding_tool_fn(verify_tool: VerifyFindingTool):
    """Wrap a VerifyFindingTool instance as an SDK-registrable @tool
    function, for the coordinator's own mcp_servers registration."""

    @sdk_tool("verify_finding", VERIFY_FINDING_DESCRIPTION, VERIFY_FINDING_INPUT_SCHEMA)
    async def verify_finding_fn(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await verify_tool(**args)
        except ToolError as e:
            return tool_result_to_sdk_content(ToolResult(error=e.message))
        return tool_result_to_sdk_content(result)

    return verify_finding_fn
