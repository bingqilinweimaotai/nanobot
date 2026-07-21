"""Data contracts for multi-agent team runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TeamTaskStatus(StrEnum):
    """Terminal state of one task in a team run."""

    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TeamTaskSpec:
    """One node in the team task graph."""

    task_id: str
    role: str
    instruction: str
    depends_on: tuple[str, ...] = ()
    capability_profile: str = "research"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TeamTaskSpec:
        """Parse the tool-call representation of a task."""
        if not isinstance(payload, dict):
            raise ValueError("each team task must be an object")
        task_id = str(payload.get("id") or "").strip()
        role = str(payload.get("role") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        raw_dependencies = payload.get("dependsOn", payload.get("depends_on", []))
        if raw_dependencies is None:
            raw_dependencies = []
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) for item in raw_dependencies
        ):
            raise ValueError(f"task {task_id or '<unknown>'}: dependsOn must be a string list")
        capability_profile = str(
            payload.get("capabilityProfile", payload.get("capability_profile", "research"))
            or "research"
        ).strip()
        return cls(
            task_id=task_id,
            role=role,
            instruction=instruction,
            depends_on=tuple(item.strip() for item in raw_dependencies if item.strip()),
            capability_profile=capability_profile,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "role": self.role,
            "instruction": self.instruction,
            "dependsOn": list(self.depends_on),
            "capabilityProfile": self.capability_profile,
        }


@dataclass(slots=True)
class TeamTaskResult:
    """Structured result returned by one worker."""

    task_id: str
    role: str
    status: TeamTaskStatus
    content: str = ""
    stop_reason: str | None = None
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    late_messages: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TeamTaskResult:
        return cls(
            task_id=str(payload["id"]),
            role=str(payload["role"]),
            status=TeamTaskStatus(str(payload["status"])),
            content=str(payload.get("content") or ""),
            stop_reason=(
                str(payload["stopReason"])
                if payload.get("stopReason") is not None
                else None
            ),
            tools_used=[str(item) for item in payload.get("toolsUsed", [])],
            usage={str(key): int(value) for key, value in payload.get("usage", {}).items()},
            error=str(payload["error"]) if payload.get("error") is not None else None,
            late_messages=[
                {str(key): str(value) for key, value in message.items()}
                for message in payload.get("lateMessages", [])
            ],
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.task_id,
            "role": self.role,
            "status": self.status.value,
            "content": self.content,
            "toolsUsed": list(self.tools_used),
            "usage": dict(self.usage),
        }
        if self.stop_reason:
            payload["stopReason"] = self.stop_reason
        if self.error:
            payload["error"] = self.error
        if self.late_messages:
            payload["lateMessages"] = list(self.late_messages)
        return payload


@dataclass(slots=True)
class TeamRunResult:
    """Aggregate result returned to the main agent by ``team_run``."""

    run_id: str
    status: str
    tasks: list[TeamTaskResult]
    total_usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TeamRunResult:
        return cls(
            run_id=str(payload["runId"]),
            status=str(payload["status"]),
            tasks=[TeamTaskResult.from_payload(item) for item in payload.get("tasks", [])],
            total_usage={
                str(key): int(value) for key, value in payload.get("totalUsage", {}).items()
            },
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "status": self.status,
            "tasks": [task.to_payload() for task in self.tasks],
            "totalUsage": dict(self.total_usage),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)


def accumulate_usage(results: list[TeamTaskResult]) -> dict[str, int]:
    """Sum numeric usage fields across task results."""
    total: dict[str, int] = {}
    for result in results:
        for key, value in result.usage.items():
            total[key] = total.get(key, 0) + int(value)
    return total
