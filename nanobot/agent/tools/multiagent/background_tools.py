"""Core tools for starting and controlling durable background team runs."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import TeamPlanError
from nanobot.agent.tools.multiagent.models import TeamTaskSpec
from nanobot.agent.tools.multiagent.plan_schema import team_tasks_schema
from nanobot.agent.tools.multiagent.service import (
    TeamRunAccessError,
    TeamRunService,
    request_owner_key,
    shared_team_run_service,
)
from nanobot.agent.tools.schema import (
    IntegerSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import current_workspace_scope


class _TeamServiceTool(Tool):
    config_key = "multiagent"

    def __init__(self, service: TeamRunService) -> None:
        self.service = service

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


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema("Overall goal shared by every worker.", min_length=1),
        tasks=team_tasks_schema(),
        max_concurrency=IntegerSchema(
            description="Optional concurrency limit no higher than the configured maximum.",
            minimum=1,
            maximum=16,
            nullable=True,
        ),
        required=["goal", "tasks"],
    )
)
class TeamStartTool(_TeamServiceTool):
    @property
    def name(self) -> str:
        return "team_start"

    @property
    def description(self) -> str:
        return (
            "Start a durable multi-agent task graph in the background and return its run id. "
            "Use team_status to poll, team_wait to wait or resume after interruption, and "
            "team_cancel to stop it."
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
        if request is None:
            return ToolResult.error("Error: team_start requires an active request context")
        try:
            if not isinstance(tasks, list):
                raise ValueError("tasks must be a list")
            specs = [TeamTaskSpec.from_payload(task) for task in tasks]
            stored = await self.service.start(
                goal=goal,
                tasks=specs,
                request=request,
                workspace_scope=current_workspace_scope(),
                max_concurrency=max_concurrency,
            )
        except (TeamPlanError, ValueError) as exc:
            return ToolResult.error(f"Error: invalid team run: {exc}")
        return json.dumps(stored.to_payload(), ensure_ascii=False, indent=2)


@tool_parameters(
    tool_parameters_schema(
        run_id=StringSchema("Team run id returned by team_start.", min_length=1),
        required=["run_id"],
    )
)
class TeamStatusTool(_TeamServiceTool):
    @property
    def name(self) -> str:
        return "team_status"

    @property
    def description(self) -> str:
        return "Read the current state and available result of a background team run."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, run_id: str, **_: Any) -> str:
        request = current_request_context()
        if request is None:
            return ToolResult.error("Error: team_status requires an active request context")
        try:
            stored = self.service.status(run_id.strip(), request_owner_key(request))
        except TeamRunAccessError as exc:
            return ToolResult.error(f"Error: {exc}")
        return json.dumps(stored.to_payload(), ensure_ascii=False, indent=2)


@tool_parameters(
    tool_parameters_schema(
        run_id=StringSchema("Team run id returned by team_start.", min_length=1),
        timeout_seconds=NumberSchema(
            description="How long to wait in this call; the run continues after timeout.",
            minimum=0,
            maximum=120,
        ),
        required=["run_id"],
    )
)
class TeamWaitTool(_TeamServiceTool):
    @property
    def name(self) -> str:
        return "team_wait"

    @property
    def description(self) -> str:
        return (
            "Wait briefly for a background team run. If a prior process interruption left it "
            "paused, resume pending nodes with the current model runtime; completed nodes are "
            "not repeated."
        )

    async def execute(
        self,
        run_id: str,
        timeout_seconds: float = 30,
        **_: Any,
    ) -> str:
        request = current_request_context()
        if request is None:
            return ToolResult.error("Error: team_wait requires an active request context")
        try:
            stored = await self.service.wait(
                run_id=run_id.strip(),
                request=request,
                workspace_scope=current_workspace_scope(),
                timeout_seconds=timeout_seconds,
            )
        except (TeamRunAccessError, ValueError) as exc:
            return ToolResult.error(f"Error: {exc}")
        return json.dumps(stored.to_payload(), ensure_ascii=False, indent=2)


@tool_parameters(
    tool_parameters_schema(
        run_id=StringSchema("Team run id returned by team_start.", min_length=1),
        required=["run_id"],
    )
)
class TeamCancelTool(_TeamServiceTool):
    @property
    def name(self) -> str:
        return "team_cancel"

    @property
    def description(self) -> str:
        return "Cancel a queued, running, or paused background team run owned by this session."

    @property
    def exclusive(self) -> bool:
        return True

    async def execute(self, run_id: str, **_: Any) -> str:
        request = current_request_context()
        if request is None:
            return ToolResult.error("Error: team_cancel requires an active request context")
        try:
            stored = await self.service.cancel(run_id.strip(), request_owner_key(request))
        except TeamRunAccessError as exc:
            return ToolResult.error(f"Error: {exc}")
        return json.dumps(stored.to_payload(), ensure_ascii=False, indent=2)
