"""Structured error types for the edge-agent framework.

Inspired by claude-code's layered error hierarchy: each error carries
context that is useful for debugging and safe for telemetry / logging.
"""

from __future__ import annotations


class EdgeAgentError(Exception):
    """Base class for all edge-agent errors."""


class ConfigError(EdgeAgentError):
    """Invalid or missing configuration."""

    def __init__(self, key: str, message: str) -> None:
        self.key = key
        super().__init__(f"Config[{key}]: {message}")


class ProviderError(EdgeAgentError):
    """A provider (System 1 / System 2) failed."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ToolExecutionError(EdgeAgentError):
    """A tool call failed."""

    def __init__(self, tool_name: str, cause: Exception | None = None) -> None:
        self.tool_name = tool_name
        self.cause = cause
        msg = f"Tool '{tool_name}' failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class InvalidTransition(EdgeAgentError):
    """State machine received an illegal transition."""

    def __init__(self, from_state: str, to_state: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"{from_state} -> {to_state}")


class ChannelError(EdgeAgentError):
    """A communication channel encountered an error."""

    def __init__(self, channel: str, message: str) -> None:
        self.channel = channel
        super().__init__(f"Channel[{channel}]: {message}")
