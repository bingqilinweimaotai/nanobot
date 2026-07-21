"""Worker-only tools for team messaging and dynamic delegation."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.multiagent.state import TeamCollaborationError, TeamRunState
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema


def _json_response(*, ok: bool, **payload: Any) -> str:
    return json.dumps({"ok": ok, **payload}, ensure_ascii=False)


class _TeamWorkerTool(Tool):
    _plugin_discoverable = False

    def __init__(self, state: TeamRunState, task_id: str) -> None:
        self.state = state
        self.task_id = task_id


@tool_parameters(
    tool_parameters_schema(
        recipient_id=StringSchema("Task id of the worker that should receive the message."),
        content=StringSchema("Focused evidence, question, or coordination note."),
        required=["recipient_id", "content"],
    )
)
class TeamSendTool(_TeamWorkerTool):
    name = "team_send"
    description = (
        "Send a short message to another active worker in this team run. "
        "Messages are delivered at that worker's next runner checkpoint."
    )

    async def execute(self, recipient_id: str, content: str, **_: Any) -> str:
        try:
            message = self.state.send_message(
                sender_id=self.task_id,
                recipient_id=recipient_id,
                content=content,
            )
        except TeamCollaborationError as exc:
            return _json_response(ok=False, error=str(exc))
        return _json_response(ok=True, message=message.to_payload())


@tool_parameters(
    tool_parameters_schema(
        limit=IntegerSchema(
            5,
            description="Maximum queued messages to return.",
            minimum=1,
            maximum=10,
        ),
    )
)
class TeamReceiveTool(_TeamWorkerTool):
    name = "team_receive"
    description = "Read and remove currently queued messages from your team mailbox. Never blocks."

    async def execute(self, limit: int = 5, **_: Any) -> str:
        messages = self.state.drain_messages(self.task_id, limit=limit)
        return _json_response(
            ok=True,
            messages=[message.to_payload() for message in messages],
        )


@tool_parameters(
    tool_parameters_schema(
        role=StringSchema("Role for the delegated worker."),
        instruction=StringSchema("Self-contained instruction for the delegated worker."),
        capability_profile=StringSchema(
            "Capability profile for the delegated worker.",
            enum=["general", "implement", "research", "review"],
        ),
        required=["role", "instruction", "capability_profile"],
    )
)
class TeamDelegateTool(_TeamWorkerTool):
    name = "team_delegate"
    description = (
        "Add one downstream worker task when the current assignment reveals necessary new work. "
        "The delegated task starts after your task finishes; do not wait for it or delegate duplicates."
    )

    async def execute(
        self,
        role: str,
        instruction: str,
        capability_profile: str,
        **_: Any,
    ) -> str:
        try:
            task = self.state.delegate(
                parent_id=self.task_id,
                role=role,
                instruction=instruction,
                capability_profile=capability_profile,
            )
        except TeamCollaborationError as exc:
            return _json_response(ok=False, error=str(exc))
        return _json_response(
            ok=True,
            task={
                "id": task.task_id,
                "role": task.role,
                "dependsOn": list(task.depends_on),
                "capabilityProfile": task.capability_profile,
            },
        )


@tool_parameters(tool_parameters_schema())
class TeamTaskStatusTool(_TeamWorkerTool):
    name = "team_task_status"
    description = "Inspect task ids, roles, dependency edges, depths, and current team status."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **_: Any) -> str:
        return json.dumps(self.state.status_payload(), ensure_ascii=False)


def register_collaboration_tools(
    registry: Any,
    *,
    state: TeamRunState,
    task_id: str,
) -> None:
    """Register the worker's run-local collaboration tools."""
    for tool_cls in (TeamSendTool, TeamReceiveTool, TeamDelegateTool, TeamTaskStatusTool):
        registry.register(tool_cls(state, task_id))
