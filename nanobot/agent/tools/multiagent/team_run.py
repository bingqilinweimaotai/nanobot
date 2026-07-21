"""Core-scope tool that runs a bounded foreground team."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import TeamPlanError
from nanobot.agent.tools.multiagent.models import TeamTaskSpec
from nanobot.agent.tools.multiagent.plan_schema import team_tasks_schema
from nanobot.agent.tools.multiagent.service import TeamRunService, shared_team_run_service
from nanobot.agent.tools.schema import (
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import current_workspace_scope


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema(
            "Overall goal shared by every worker.",
            min_length=1,
        ),
        tasks=team_tasks_schema(),
        max_concurrency=IntegerSchema(
            description=(
                "Optional per-run concurrency limit. It may lower, but not exceed, "
                "the configured maximum."
            ),
            minimum=1,
            maximum=16,
            nullable=True,
        ),
        required=["goal", "tasks"],
    )
)
class TeamRunTool(Tool):
    """Execute an explicit multi-agent task graph in the foreground."""

    config_key = "multiagent"

    @classmethod
    def config_cls(cls):
        return MultiAgentToolConfig

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        config = getattr(ctx.config, "multiagent", None)
        return bool(config and config.enable)

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(shared_team_run_service(ctx))

    def __init__(
        self,
        service: TeamRunService,
    ) -> None:
        self.service = service

    @property
    def config(self) -> MultiAgentToolConfig:
        return self.service.config

    @property
    def worker_runner(self):
        """Compatibility handle for tests and extensions customizing worker execution."""
        return self.service.worker_runner

    @property
    def name(self) -> str:
        return "team_run"

    @property
    def description(self) -> str:
        return (
            "Run a bounded team of specialized agents against an explicit dependency graph. "
            "Independent tasks execute concurrently; dependent tasks receive successful "
            "upstream results. Use research/review profiles for read-only work and implement "
            "only when workspace changes are required. The call waits for the team to finish."
        )

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(
        self,
        goal: str,
        tasks: list[dict[str, Any]],
        max_concurrency: int | None = None,
        **_: Any,
    ) -> str:
        request = current_request_context()
        if request is None or request.runtime is None:
            return ToolResult.error("Error: team_run requires an active model runtime")
        if max_concurrency is not None:
            if max_concurrency < 1:
                return ToolResult.error("Error: requested team concurrency must be at least 1")
            if max_concurrency > self.config.max_concurrency:
                return ToolResult.error(
                    "Error: requested team concurrency exceeds configured maximum "
                    f"({max_concurrency}/{self.config.max_concurrency})"
                )

        try:
            if not isinstance(tasks, list):
                raise ValueError("tasks must be a list")
            task_specs = [TeamTaskSpec.from_payload(task) for task in tasks]
            result = await self.service.run_foreground(
                goal=goal,
                tasks=task_specs,
                request=request,
                workspace_scope=current_workspace_scope(),
                max_concurrency=max_concurrency,
            )
        except TeamPlanError as exc:
            return ToolResult.error(f"Error: invalid team plan: {exc}")
        except ValueError as exc:
            return ToolResult.error(f"Error: invalid team task: {exc}")
        except RuntimeError as exc:
            return ToolResult.error(f"Error: team run failed: {exc}")
        return result.to_json()
