"""Shared execution engine for foreground and background team runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import TeamCoordinator
from nanobot.agent.tools.multiagent.models import TeamRunResult, TeamTaskResult, TeamTaskStatus
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.worker import TeamTokenBudget, TeamWorkerRunner
from nanobot.security.workspace_access import WorkspaceScope

WorkerSlot = Callable[[], AbstractAsyncContextManager[None]]


class TeamExecutionEngine:
    """Apply workspace-level resource policy while executing one run state."""

    def __init__(
        self,
        *,
        config: MultiAgentToolConfig,
        worker_runner: TeamWorkerRunner,
        agent_worker_slot: WorkerSlot | None = None,
    ) -> None:
        self.config = config
        self.worker_runner = worker_runner
        self.agent_worker_slot = agent_worker_slot
        self._worker_semaphore = asyncio.Semaphore(config.max_concurrency)
        self._write_lock = asyncio.Lock()

    def _coordinator(self, state: TeamRunState, max_concurrency: int) -> TeamCoordinator:
        # Existing nodes remain resumable after an administrator lowers maxTasks;
        # the restored state still uses the current limit for any new delegation.
        validation_limit = max(self.config.max_tasks, len(state.tasks))
        return TeamCoordinator(
            max_tasks=validation_limit,
            max_concurrency=max_concurrency,
            task_timeout_s=self.config.task_timeout_seconds,
            serialize_writes=self.config.serialize_writes,
            capability_profiles=self.config.capability_profiles,
            write_profiles=self.config.write_profiles,
            max_delegation_depth=self.config.max_delegation_depth,
            max_message_chars=self.config.max_message_chars,
            worker_semaphore=self._worker_semaphore,
            write_lock=self._write_lock,
        )

    async def run(
        self,
        *,
        goal: str,
        state: TeamRunState,
        request: RequestContext,
        workspace_scope: WorkspaceScope,
        max_concurrency: int,
        consumed_tokens: int = 0,
    ) -> TeamRunResult:
        """Execute one state through the shared runner and resource gates."""
        if request.runtime is None:
            raise ValueError("team execution requires an active model runtime")
        budget = TeamTokenBudget(self.config.max_total_tokens)
        budget.used = consumed_tokens
        coordinator = self._coordinator(state, max_concurrency)

        async def run_worker(team_state, team_goal, task, dependencies):
            if await budget.is_exhausted():
                return TeamTaskResult(
                    task_id=task.task_id,
                    role=task.role,
                    status=TeamTaskStatus.FAILED,
                    stop_reason="budget_exhausted",
                    error=(
                        "team token budget already exhausted "
                        f"({budget.used}/{budget.limit})"
                    ),
                )

            async def invoke():
                return await self.worker_runner.run(
                    state=team_state,
                    goal=team_goal,
                    task=task,
                    dependencies=dependencies,
                    runtime=request.runtime,
                    parent_request=request,
                    workspace_scope=workspace_scope,
                    budget=budget,
                )

            if self.agent_worker_slot is None:
                return await invoke()
            async with self.agent_worker_slot():
                return await invoke()

        return await coordinator.run(
            goal,
            state.tasks,
            run_worker,
            state=state,
        )
