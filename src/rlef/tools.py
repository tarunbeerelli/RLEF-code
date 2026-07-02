"""
tools.py — Strict gRPC Routing Layer for RLEF-Code

This layer is now locked down for the H200. It acts purely as a client router.
It intercepts tool calls from env.py and forwards execution strictly to the
isolated gRPC sandbox server.
"""

from dataclasses import dataclass, field
from enum import Enum

# Import your existing gRPC client function (adjust the import if your bridge uses a different function name)
try:
    from rlef.reward_bridge import reward_func
except ImportError:
    # Fallback definition just in case the import fails, ensuring we NEVER use local subprocess
    def reward_func(code: str, timeout: int = 10):
        raise RuntimeError(
            "gRPC Bridge not found. Local execution is disabled for security."
        )


# ── 1. Result Types ──────────────────────────────────────────────────────────


class ToolName(str, Enum):
    EXECUTE = "execute"
    LINT = "lint"
    TESTS = "generate_tests"


@dataclass
class ToolResult:
    tool: ToolName
    success: bool
    output: str
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    lint_errors: list[dict] = field(default_factory=list)
    generated_tests: str = ""


# ── 2. The Strict Router ──────────────────────────────────────────────────────


def call_tool(
    tool_name: str,
    code: str = "",
    problem: str = "",
    timeout: int = 10,
) -> ToolResult:
    """
    Central dispatch. This NEVER runs code locally.
    It routes everything to the gRPC sandbox.
    """
    try:
        name = ToolName(tool_name)
    except ValueError:
        return ToolResult(
            tool=ToolName.EXECUTE,
            success=False,
            output=f"System Error: Unknown tool '{tool_name}'. Available: execute, lint, generate_tests",
        )

    # 1. Execute: Route to gRPC Server
    if name == ToolName.EXECUTE:
        try:
            # We assume your gRPC client returns a dict with stdout, stderr, etc.
            # Map it cleanly to our ToolResult dataclass.
            grpc_response = reward_func(code=code, timeout=timeout)

            return ToolResult(
                tool=ToolName.EXECUTE,
                success=grpc_response.get("success", False),
                output=grpc_response.get("feedback", ""),
                stdout=grpc_response.get("stdout", ""),
                stderr=grpc_response.get("stderr", ""),
                error=grpc_response.get("error_type", ""),
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.EXECUTE,
                success=False,
                output=f"Sandbox Connection Error: {str(e)}",
                error="gRPC_Connection_Failed",
            )

    # 2. Lint (Deprecated as a standalone tool, handled by env.py / execute natively)
    elif name == ToolName.LINT:
        return ToolResult(
            tool=ToolName.LINT,
            success=False,
            output="Warning: The 'lint' tool is deprecated. Use 'execute' to check your syntax.",
        )

    # 3. Generate Tests (Now a mock step, since env.py handles the Two-Sided Oracle logic)
    elif name == ToolName.TESTS:
        return ToolResult(
            tool=ToolName.TESTS,
            success=True,
            output="Internal Engine: Testing framework initialized. Awaiting test code.",
        )

    return ToolResult(
        tool=ToolName.EXECUTE, success=False, output="Critical router failure."
    )
