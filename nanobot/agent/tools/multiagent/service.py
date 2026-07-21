"""Shared lifecycle service for durable background team runs."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import TeamCoordinator, validate_task_graph
from nanobot.agent.tools.multiagent.models import TeamTaskSpec
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.store import StoredTeamRun, TeamRunStore
from nanobot.agent.tools.multiagent.worker import TeamTokenBudget, TeamWorkerRunner
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import (
    WorkspaceScope,
    default_workspace_scope,
)

_TERMINAL_RUN_STATUSES = {"cancelled", "completed", "failed", "partial"}
_SERVICES: dict[str, TeamRunService] = {}


class TeamRunAccessError(ValueError):
    """Raised when a caller cannot inspect or resume a durable run."""


def request_owner_key(request: RequestContext) -> str:
    return request.session_key or f"{request.channel}:{request.chat_id}"


def shared_team_run_service(ctx: Any) -> TeamRunService:
    """Return one service shared by all team lifecycle tools in a registry."""
    workspace = Path(ctx.workspace).expanduser().resolve(strict=False)
    key = str(workspace)
    service = _SERVICES.get(key)
    if service is None:
        service = TeamRunService(
            workspace=workspace,
            tools_config=ctx.config,
            config=ctx.config.multiagent,
            exec_session_manager=getattr(ctx, "exec_session_manager", None),
        )
        _SERVICES[key] = service
    return service


class TeamRunService:
    """Own background tasks and checkpoint them for explicit later resumption."""

    def __init__(
        self,
        *,
        workspace: Path,
        tools_config: ToolsConfig,
        config: MultiAgentToolConfig,
        exec_session_manager: Any | None = None,
        store: TeamRunStore | None = None,
        worker_runner: TeamWorkerRunner | None = None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve(strict=False)
        self.tools_config = tools_config
        self.config = config
        self.store = store or TeamRunStore(
            self.workspace / "multiagent" / "runs.sqlite3",
            max_stored_runs=config.max_stored_runs,
        )
        self.worker_runner = worker_runner or TeamWorkerRunner(
            workspace=self.workspace,
            tools_config=tools_config,
            config=config,
            exec_session_manager=exec_session_manager,
        )
        self._active: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()

    def _scope(self, scope: WorkspaceScope | None) -> WorkspaceScope:
        return scope or default_workspace_scope(
            self.workspace,
            self.tools_config.restrict_to_workspace,
        )

    def _coordinator(self, max_concurrency: int) -> TeamCoordinator:
        return TeamCoordinator(
            max_tasks=self.config.max_tasks,
            max_concurrency=max_concurrency,
            task_timeout_s=self.config.task_timeout_seconds,
            serialize_writes=self.config.serialize_writes,
            capability_profiles=self.config.capability_profiles,
            write_profiles=self.config.write_profiles,
            max_delegation_depth=self.config.max_delegation_depth,
            max_message_chars=self.config.max_message_chars,
        )

    def _validate_concurrency(self, max_concurrency: int | None) -> int:
        concurrency = max_concurrency or self.config.max_concurrency
        if concurrency < 1:
            raise ValueError("requested team concurrency must be at least 1")
        if concurrency > self.config.max_concurrency:
            raise ValueError(
                "requested team concurrency exceeds configured maximum "
                f"({concurrency}/{self.config.max_concurrency})"
            )
        return concurrency

    def _owned(self, run_id: str, owner_session_key: str) -> StoredTeamRun:
        stored = self.store.get(run_id)
        if stored is None:
            raise TeamRunAccessError(f"unknown team run: {run_id}")
        if stored.owner_session_key != owner_session_key:
            raise TeamRunAccessError("team run belongs to a different session")
        return stored

    @staticmethod
    def _validate_resume_scope(stored: StoredTeamRun, scope: WorkspaceScope) -> None:
        current_path = scope.project_path.resolve(strict=False)
        stored_path = Path(stored.workspace_path).resolve(strict=False)
        if current_path != stored_path or scope.access_mode != stored.access_mode:
            raise TeamRunAccessError(
                "team run must be resumed from its original workspace and access mode"
            )

    async def start(
        self,
        *,
        goal: str,
        tasks: list[TeamTaskSpec],
        request: RequestContext,
        workspace_scope: WorkspaceScope | None,
        max_concurrency: int | None = None,
    ) -> StoredTeamRun:
        if request.runtime is None:
            raise ValueError("team_start requires an active model runtime")
        concurrency = self._validate_concurrency(max_concurrency)
        validate_task_graph(
            tasks,
            max_tasks=self.config.max_tasks,
            capability_profiles=self.config.capability_profiles,
        )
        if not goal.strip():
            raise ValueError("team run requires a non-empty goal")

        scope = self._scope(workspace_scope)
        run_id = uuid.uuid4().hex[:12]
        state = TeamRunState(
            run_id=run_id,
            tasks=tasks,
            max_tasks=self.config.max_tasks,
            max_delegation_depth=self.config.max_delegation_depth,
            capability_profiles=self.config.capability_profiles,
            max_message_chars=self.config.max_message_chars,
        )
        self.store.create(
            run_id=run_id,
            owner_session_key=request_owner_key(request),
            goal=goal.strip(),
            workspace_path=str(scope.project_path),
            access_mode=scope.access_mode,
            max_concurrency=concurrency,
            snapshot=state.to_snapshot(),
        )
        state.set_on_change(
            lambda changed: self.store.checkpoint(
                run_id,
                snapshot=changed.to_snapshot(),
                status="running",
            )
        )
        self._launch(
            stored=self._owned(run_id, request_owner_key(request)),
            state=state,
            request=request,
            scope=scope,
        )
        return self._owned(run_id, request_owner_key(request))

    def _launch(
        self,
        *,
        stored: StoredTeamRun,
        state: TeamRunState,
        request: RequestContext,
        scope: WorkspaceScope,
    ) -> asyncio.Task[None]:
        existing = self._active.get(stored.run_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(
            self._run(stored=stored, state=state, request=request, scope=scope),
            name=f"nanobot-team-run-{stored.run_id}",
        )
        self._active[stored.run_id] = task

        def cleanup(completed: asyncio.Task[None]) -> None:
            if self._active.get(stored.run_id) is completed:
                self._active.pop(stored.run_id, None)
            try:
                error = completed.exception()
            except asyncio.CancelledError:
                return
            if error is not None:
                logger.error("Background team run {} failed: {}", stored.run_id, error)

        task.add_done_callback(cleanup)
        return task

    @staticmethod
    def _consumed_tokens(state: TeamRunState) -> int:
        total = 0
        for task in state.tasks:
            result = state.result_for(task.task_id)
            if result is None:
                continue
            usage = result.usage
            total += int(
                usage.get("total_tokens", 0)
                or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
        return total

    async def _run(
        self,
        *,
        stored: StoredTeamRun,
        state: TeamRunState,
        request: RequestContext,
        scope: WorkspaceScope,
    ) -> None:
        run_id = stored.run_id
        self.store.checkpoint(run_id, snapshot=state.to_snapshot(), status="running")
        budget = TeamTokenBudget(self.config.max_total_tokens)
        budget.used = self._consumed_tokens(state)
        coordinator = self._coordinator(stored.max_concurrency)

        async def run_worker(team_state, team_goal, task, dependencies):
            return await self.worker_runner.run(
                state=team_state,
                goal=team_goal,
                task=task,
                dependencies=dependencies,
                runtime=request.runtime,
                parent_request=request,
                workspace_scope=scope,
                budget=budget,
            )

        try:
            result = await coordinator.run(
                stored.goal,
                state.tasks,
                run_worker,
                state=state,
            )
        except asyncio.CancelledError:
            status = "cancelled" if run_id in self._cancel_requested else "paused"
            self.store.finish(run_id, status=status, snapshot=state.to_snapshot())
            raise
        except Exception as exc:
            self.store.finish(
                run_id,
                status="failed",
                snapshot=state.to_snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            self.store.finish(
                run_id,
                status=result.status,
                snapshot=state.to_snapshot(),
                result=result,
            )
        finally:
            self._cancel_requested.discard(run_id)

    def status(self, run_id: str, owner_session_key: str) -> StoredTeamRun:
        return self._owned(run_id, owner_session_key)

    async def wait(
        self,
        *,
        run_id: str,
        request: RequestContext,
        workspace_scope: WorkspaceScope | None,
        timeout_seconds: float | None,
    ) -> StoredTeamRun:
        owner = request_owner_key(request)
        stored = self._owned(run_id, owner)
        if stored.status in _TERMINAL_RUN_STATUSES:
            return stored
        if request.runtime is None:
            raise ValueError("team_wait requires an active model runtime to resume a run")
        scope = self._scope(workspace_scope)
        self._validate_resume_scope(stored, scope)

        task = self._active.get(run_id)
        if task is None or task.done():
            state = TeamRunState.from_snapshot(stored.snapshot)
            state.set_on_change(
                lambda changed: self.store.checkpoint(
                    run_id,
                    snapshot=changed.to_snapshot(),
                    status="running",
                )
            )
            task = self._launch(stored=stored, state=state, request=request, scope=scope)
        try:
            if timeout_seconds is None:
                await asyncio.shield(task)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
        except TimeoutError:
            pass
        return self._owned(run_id, owner)

    async def cancel(self, run_id: str, owner_session_key: str) -> StoredTeamRun:
        stored = self._owned(run_id, owner_session_key)
        if stored.status in _TERMINAL_RUN_STATUSES:
            return stored
        task = self._active.get(run_id)
        if task is None or task.done():
            self.store.finish(
                run_id,
                status="cancelled",
                snapshot=stored.snapshot,
            )
        else:
            self._cancel_requested.add(run_id)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self._owned(run_id, owner_session_key)
