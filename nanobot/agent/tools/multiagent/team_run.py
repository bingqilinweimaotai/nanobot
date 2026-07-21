"""Core-scope tool that runs a bounded foreground team."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import TeamCoordinator, TeamPlanError
from nanobot.agent.tools.multiagent.models import TeamTaskSpec
from nanobot.agent.tools.multiagent.plan_schema import team_tasks_schema
from nanobot.agent.tools.multiagent.worker import TeamTokenBudget, TeamWorkerRunner
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
        return cls(
            workspace=Path(ctx.workspace),
            tools_config=ctx.config,
            config=ctx.config.multiagent,
        )

    def __init__(
        self,
        *,
        workspace: Path,
        tools_config: Any,
        config: MultiAgentToolConfig,
    ) -> None:
        self.workspace = workspace
        self.tools_config = tools_config
        self.config = config
        self.worker_runner = TeamWorkerRunner(
            workspace=workspace,
            tools_config=tools_config,
            config=config,
        )

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
            coordinator = TeamCoordinator(
                max_tasks=self.config.max_tasks,
                max_concurrency=max_concurrency or self.config.max_concurrency,
                task_timeout_s=self.config.task_timeout_seconds,
                serialize_writes=self.config.serialize_writes,
                capability_profiles=self.config.capability_profiles,
                write_profiles=self.config.write_profiles,
                max_delegation_depth=self.config.max_delegation_depth,
                max_message_chars=self.config.max_message_chars,
            )
            budget = TeamTokenBudget(self.config.max_total_tokens)

            async def run_worker(state, team_goal, task, dependencies):
                return await self.worker_runner.run(
                    state=state,
                    goal=team_goal,
                    task=task,
                    dependencies=dependencies,
                    runtime=request.runtime,
                    parent_request=request,
                    workspace_scope=current_workspace_scope(),
                    budget=budget,
                )

            result = await coordinator.run(goal, task_specs, run_worker)
        except TeamPlanError as exc:
            return ToolResult.error(f"Error: invalid team plan: {exc}")
        except ValueError as exc:
            return ToolResult.error(f"Error: invalid team task: {exc}")
        return result.to_json()
