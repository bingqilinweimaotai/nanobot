"""Configuration for bounded multi-agent team tools."""

from __future__ import annotations

from pydantic import Field

from nanobot.config_base import Base


class MultiAgentToolConfig(Base):
    """Limits and capability policy for foreground and background team runs."""

    enable: bool = False
    max_tasks: int = Field(default=8, ge=1, le=32)
    max_concurrency: int = Field(default=3, ge=1, le=16)
    max_iterations_per_task: int = Field(default=40, ge=1, le=200)
    max_tool_result_chars: int = Field(default=16_000, ge=1_000, le=128_000)
    max_total_tokens: int = Field(default=100_000, ge=1_000)
    max_stored_runs: int = Field(default=200, ge=1, le=10_000)
    task_timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)
    max_dependency_chars: int = Field(default=12_000, ge=1_000, le=128_000)
    max_delegation_depth: int = Field(default=2, ge=0, le=5)
    max_message_chars: int = Field(default=4_000, ge=100, le=32_000)
    serialize_writes: bool = True
    fail_on_tool_error: bool = True

    research_tools: list[str] = Field(default_factory=lambda: [
        "find_files",
        "list_dir",
        "read_file",
        "search_files",
        "web_fetch",
        "web_search",
    ])
    review_tools: list[str] = Field(default_factory=lambda: [
        "find_files",
        "list_dir",
        "read_file",
        "search_files",
    ])

    @property
    def capability_profiles(self) -> set[str]:
        return {"general", "implement", "research", "review"}

    @property
    def write_profiles(self) -> set[str]:
        return {"general", "implement"}

    def tools_for_profile(self, profile: str, available: set[str]) -> set[str]:
        """Return the admin-approved tool names for a capability profile."""
        if profile in self.write_profiles:
            return set(available)
        configured = self.research_tools if profile == "research" else self.review_tools
        return set(configured) & available
