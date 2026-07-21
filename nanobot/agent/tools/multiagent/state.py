"""Mutable run-local collaboration state for a multi-agent team."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nanobot.agent.tools.multiagent.models import TeamTaskResult, TeamTaskSpec


class TeamCollaborationError(ValueError):
    """Raised when an agent requests an invalid collaboration operation."""


@dataclass(frozen=True, slots=True)
class TeamMessage:
    sender_id: str
    recipient_id: str
    content: str

    def to_payload(self) -> dict[str, str]:
        return {
            "senderId": self.sender_id,
            "recipientId": self.recipient_id,
            "content": self.content,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TeamMessage:
        return cls(
            sender_id=str(payload["senderId"]),
            recipient_id=str(payload["recipientId"]),
            content=str(payload["content"]),
        )


class TeamRunState:
    """Task registry, mailboxes, and delegation policy for one team run.

    All methods are synchronous by design. The state is owned by one asyncio
    event loop and every mutation completes without an ``await``, making each
    operation atomic relative to other workers in the same run.
    """

    _TERMINAL_STATUSES = {"blocked", "completed", "failed"}

    def __init__(
        self,
        *,
        run_id: str,
        tasks: list[TeamTaskSpec],
        max_tasks: int,
        max_delegation_depth: int,
        capability_profiles: set[str],
        max_message_chars: int,
        on_change: Callable[[TeamRunState], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.max_tasks = max_tasks
        self.max_delegation_depth = max_delegation_depth
        self.capability_profiles = set(capability_profiles)
        self.max_message_chars = max_message_chars
        self._tasks = {task.task_id: task for task in tasks}
        self._depths = {task.task_id: 0 for task in tasks}
        self._statuses = {task.task_id: "pending" for task in tasks}
        self._results: dict[str, TeamTaskResult] = {}
        self._mailboxes = {task.task_id: [] for task in tasks}
        self._delegation_fingerprints: set[tuple[str, str, str, str]] = set()
        self._on_change = on_change

    def set_on_change(self, callback: Callable[[TeamRunState], None] | None) -> None:
        self._on_change = callback

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change(self)

    @property
    def tasks(self) -> list[TeamTaskSpec]:
        return list(self._tasks.values())

    def has_task(self, task_id: str) -> bool:
        return task_id in self._tasks

    def mark_running(self, task_id: str) -> None:
        if task_id in self._statuses:
            self._statuses[task_id] = "running"
            self._notify_change()

    def record_result(self, result: TeamTaskResult) -> None:
        self._results[result.task_id] = result
        self._statuses[result.task_id] = result.status.value
        self._notify_change()

    def result_for(self, task_id: str) -> TeamTaskResult | None:
        return self._results.get(task_id)

    def delegate(
        self,
        *,
        parent_id: str,
        role: str,
        instruction: str,
        capability_profile: str,
    ) -> TeamTaskSpec:
        """Append a unique child task that runs after its delegating parent."""
        if parent_id not in self._tasks:
            raise TeamCollaborationError(f"unknown delegating task: {parent_id}")
        role = role.strip()
        instruction = instruction.strip()
        capability_profile = capability_profile.strip()
        if not role:
            raise TeamCollaborationError("delegated task requires a role")
        if not instruction:
            raise TeamCollaborationError("delegated task requires an instruction")
        if capability_profile not in self.capability_profiles:
            allowed = ", ".join(sorted(self.capability_profiles))
            raise TeamCollaborationError(
                f"unknown capability profile {capability_profile!r}; allowed: {allowed}"
            )
        depth = self._depths[parent_id] + 1
        if depth > self.max_delegation_depth:
            raise TeamCollaborationError(
                f"delegation depth limit reached ({depth}/{self.max_delegation_depth})"
            )
        if len(self._tasks) >= self.max_tasks:
            raise TeamCollaborationError(
                f"team task limit reached ({len(self._tasks)}/{self.max_tasks})"
            )
        fingerprint = (parent_id, role, instruction, capability_profile)
        if fingerprint in self._delegation_fingerprints:
            raise TeamCollaborationError("duplicate delegation from this task")

        task_id = f"delegate-{uuid.uuid4().hex[:8]}"
        task = TeamTaskSpec(
            task_id=task_id,
            role=role,
            instruction=instruction,
            depends_on=(parent_id,),
            capability_profile=capability_profile,
        )
        self._delegation_fingerprints.add(fingerprint)
        self._tasks[task_id] = task
        self._depths[task_id] = depth
        self._statuses[task_id] = "pending"
        self._mailboxes[task_id] = []
        self._notify_change()
        return task

    def send_message(self, *, sender_id: str, recipient_id: str, content: str) -> TeamMessage:
        if sender_id not in self._tasks:
            raise TeamCollaborationError(f"unknown sender task: {sender_id}")
        if recipient_id not in self._tasks:
            raise TeamCollaborationError(f"unknown recipient task: {recipient_id}")
        if sender_id == recipient_id:
            raise TeamCollaborationError("a task cannot send a team message to itself")
        if self._statuses[recipient_id] in self._TERMINAL_STATUSES:
            raise TeamCollaborationError(f"recipient task {recipient_id} has already finished")
        content = content.strip()
        if not content:
            raise TeamCollaborationError("team message content cannot be empty")
        if len(content) > self.max_message_chars:
            raise TeamCollaborationError(
                f"team message exceeds configured limit of {self.max_message_chars} characters"
            )
        message = TeamMessage(sender_id, recipient_id, content)
        self._mailboxes[recipient_id].append(message)
        self._notify_change()
        return message

    def drain_messages(self, task_id: str, *, limit: int) -> list[TeamMessage]:
        if task_id not in self._mailboxes:
            raise TeamCollaborationError(f"unknown mailbox task: {task_id}")
        if limit < 1:
            return []
        mailbox = self._mailboxes[task_id]
        drained = mailbox[:limit]
        del mailbox[:limit]
        if drained:
            self._notify_change()
        return drained

    def to_snapshot(self) -> dict[str, Any]:
        """Return the complete durable state needed to resume this run."""
        return {
            "schemaVersion": 1,
            "runId": self.run_id,
            "limits": {
                "maxTasks": self.max_tasks,
                "maxDelegationDepth": self.max_delegation_depth,
                "maxMessageChars": self.max_message_chars,
            },
            "capabilityProfiles": sorted(self.capability_profiles),
            "tasks": [task.to_payload() for task in self._tasks.values()],
            "depths": dict(self._depths),
            "statuses": dict(self._statuses),
            "results": {
                task_id: result.to_payload() for task_id, result in self._results.items()
            },
            "mailboxes": {
                task_id: [message.to_payload() for message in messages]
                for task_id, messages in self._mailboxes.items()
            },
            "delegationFingerprints": [
                list(fingerprint) for fingerprint in sorted(self._delegation_fingerprints)
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        *,
        on_change: Callable[[TeamRunState], None] | None = None,
    ) -> TeamRunState:
        """Restore a checkpoint, returning interrupted workers to pending."""
        limits = payload.get("limits", {})
        tasks = [TeamTaskSpec.from_payload(item) for item in payload.get("tasks", [])]
        state = cls(
            run_id=str(payload["runId"]),
            tasks=tasks,
            max_tasks=int(limits["maxTasks"]),
            max_delegation_depth=int(limits["maxDelegationDepth"]),
            capability_profiles={str(item) for item in payload.get("capabilityProfiles", [])},
            max_message_chars=int(limits["maxMessageChars"]),
            on_change=on_change,
        )
        raw_depths = payload.get("depths", {})
        raw_statuses = payload.get("statuses", {})
        state._depths = {
            task.task_id: int(raw_depths.get(task.task_id, 0)) for task in tasks
        }
        state._statuses = {
            task.task_id: (
                "pending"
                if raw_statuses.get(task.task_id, "pending") == "running"
                else str(raw_statuses.get(task.task_id, "pending"))
            )
            for task in tasks
        }
        state._results = {
            str(task_id): TeamTaskResult.from_payload(result)
            for task_id, result in payload.get("results", {}).items()
        }
        raw_mailboxes = payload.get("mailboxes", {})
        state._mailboxes = {
            task.task_id: [
                TeamMessage.from_payload(message)
                for message in raw_mailboxes.get(task.task_id, [])
            ]
            for task in tasks
        }
        state._delegation_fingerprints = {
            tuple(str(item) for item in fingerprint)
            for fingerprint in payload.get("delegationFingerprints", [])
            if len(fingerprint) == 4
        }
        return state

    def status_payload(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "tasks": [
                {
                    "id": task.task_id,
                    "role": task.role,
                    "status": self._statuses[task.task_id],
                    "depth": self._depths[task.task_id],
                    "dependsOn": list(task.depends_on),
                }
                for task in self._tasks.values()
            ],
        }
