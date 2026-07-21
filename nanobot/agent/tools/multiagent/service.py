"""Shared lifecycle service for foreground and durable background team runs."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.coordinator import validate_task_graph
from nanobot.agent.tools.multiagent.engine import TeamExecutionEngine, WorkerSlot
from nanobot.agent.tools.multiagent.models import TeamRunResult, TeamTaskSpec
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.store import (
    StoredTeamRun,
    TeamRunLeaseError,
    TeamRunStore,
)
from nanobot.agent.tools.multiagent.worker import TeamWorkerRunner
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import (
    WorkspaceScope,
    default_workspace_scope,
)

_TERMINAL_RUN_STATUSES = {"cancelled", "completed", "failed", "partial"}


class TeamRunAccessError(ValueError):
    """Raised when a caller cannot inspect or resume a durable run."""


def request_owner_key(request: RequestContext) -> str:
    return request.session_key or f"{request.channel}:{request.chat_id}"


def shared_team_run_service(ctx: Any) -> TeamRunService:
    """Return one service shared by team tools built from the same ToolContext."""
    service = getattr(ctx, "team_run_service", None)
    if service is not None:
        return service

    workspace = Path(ctx.workspace).expanduser().resolve(strict=False)
    manager = getattr(ctx, "subagent_manager", None)
    worker_slot: WorkerSlot | None = None
    if manager is not None and callable(getattr(manager, "worker_slot", None)):
        ensure_capacity = getattr(manager, "ensure_worker_capacity", None)
        if callable(ensure_capacity):
            ensure_capacity(ctx.config.multiagent.max_concurrency)
        worker_slot = manager.worker_slot
    service = TeamRunService(
        workspace=workspace,
        tools_config=ctx.config,
        config=ctx.config.multiagent,
        agent_worker_slot=worker_slot,
    )
    ctx.team_run_service = service
    return service


class TeamRunService:
    """Own execution, persistence, cancellation, and resumption for team runs."""

    def __init__(
        self,
        *,
        workspace: Path,
        tools_config: ToolsConfig,
        config: MultiAgentToolConfig,
        store: TeamRunStore | None = None,
        worker_runner: TeamWorkerRunner | None = None,
        engine: TeamExecutionEngine | None = None,
        agent_worker_slot: WorkerSlot | None = None,
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
        )
        self.engine = engine or TeamExecutionEngine(
            config=config,
            worker_runner=self.worker_runner,
            agent_worker_slot=agent_worker_slot,
        )
        self.runner_id = uuid.uuid4().hex
        self._active: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()

    def _scope(self, scope: WorkspaceScope | None) -> WorkspaceScope:
        return scope or default_workspace_scope(
            self.workspace,
            self.tools_config.restrict_to_workspace,
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

    def _validate_active_capacity(self) -> None:
        active = sum(1 for task in self._active.values() if not task.done())
        if active >= self.config.max_active_runs:
            raise ValueError(
                "active team run limit reached "
                f"({active}/{self.config.max_active_runs})"
            )

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

    def _restore_state(self, stored: StoredTeamRun) -> TeamRunState:
        return TeamRunState.from_snapshot(
            stored.snapshot,
            max_tasks=self.config.max_tasks,
            max_delegation_depth=self.config.max_delegation_depth,
            capability_profiles=self.config.capability_profiles,
            max_message_chars=self.config.max_message_chars,
        )

    def _checkpoint_callback(self, run_id: str):
        def checkpoint(changed: TeamRunState) -> None:
            self.store.checkpoint(
                run_id,
                snapshot=changed.to_snapshot(),
                status="running",
                runner_id=self.runner_id,
            )

        return checkpoint

    async def _start_run(
        self,
        *,
        goal: str,
        tasks: list[TeamTaskSpec],
        request: RequestContext,
        workspace_scope: WorkspaceScope | None,
        max_concurrency: int | None,
    ) -> tuple[StoredTeamRun, asyncio.Task[None]]:
        if request.runtime is None:
            raise ValueError("team execution requires an active model runtime")
        concurrency = self._validate_concurrency(max_concurrency)
        validate_task_graph(
            tasks,
            max_tasks=self.config.max_tasks,
            capability_profiles=self.config.capability_profiles,
        )
        if not goal.strip():
            raise ValueError("team run requires a non-empty goal")
        self._validate_active_capacity()

        scope = self._scope(workspace_scope)
        owner = request_owner_key(request)
        state: TeamRunState | None = None
        run_id = ""
        for _ in range(3):
            run_id = uuid.uuid4().hex[:12]
            state = TeamRunState(
                run_id=run_id,
                tasks=tasks,
                max_tasks=self.config.max_tasks,
                max_delegation_depth=self.config.max_delegation_depth,
                capability_profiles=self.config.capability_profiles,
                max_message_chars=self.config.max_message_chars,
            )
            try:
                self.store.create(
                    run_id=run_id,
                    owner_session_key=owner,
                    goal=goal.strip(),
                    workspace_path=str(scope.project_path),
                    access_mode=scope.access_mode,
                    max_concurrency=concurrency,
                    snapshot=state.to_snapshot(),
                    runner_id=self.runner_id,
                )
                break
            except sqlite3.IntegrityError:
                state = None
        if state is None:
            raise RuntimeError("could not allocate a unique team run id")

        state.set_on_change(self._checkpoint_callback(run_id))
        stored = self._owned(run_id, owner)
        task = self._launch(stored=stored, state=state, request=request, scope=scope)
        return self._owned(run_id, owner), task

    async def start(
        self,
        *,
        goal: str,
        tasks: list[TeamTaskSpec],
        request: RequestContext,
        workspace_scope: WorkspaceScope | None,
        max_concurrency: int | None = None,
    ) -> StoredTeamRun:
        """Start a durable run and return without waiting for its workers."""
        stored, _ = await self._start_run(
            goal=goal,
            tasks=tasks,
            request=request,
            workspace_scope=workspace_scope,
            max_concurrency=max_concurrency,
        )
        return stored

    async def run_foreground(
        self,
        *,
        goal: str,
        tasks: list[TeamTaskSpec],
        request: RequestContext,
        workspace_scope: WorkspaceScope | None,
        max_concurrency: int | None = None,
    ) -> TeamRunResult:
        """Run through the same durable lifecycle while the caller waits."""
        stored, task = await self._start_run(
            goal=goal,
            tasks=tasks,
            request=request,
            workspace_scope=workspace_scope,
            max_concurrency=max_concurrency,
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                await self.cancel(stored.run_id, request_owner_key(request))
                raise
            finished = self._owned(stored.run_id, request_owner_key(request))
            raise RuntimeError(
                finished.error or f"team run stopped unexpectedly ({finished.status})"
            ) from None
        finished = self._owned(stored.run_id, request_owner_key(request))
        if finished.result is None:
            raise RuntimeError(finished.error or "team run ended without a result")
        return finished.result

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
        self._validate_active_capacity()
        try:
            self.store.claim(stored.run_id, self.runner_id)
        except TeamRunLeaseError as exc:
            raise TeamRunAccessError(str(exc)) from exc
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
                logger.error("Team run {} failed: {}", stored.run_id, error)

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

    async def _heartbeat(
        self,
        run_id: str,
        owner_task: asyncio.Task[Any],
        lease_lost: asyncio.Event,
    ) -> None:
        interval = max(0.001, self.store.lease_seconds / 3)
        try:
            while True:
                await asyncio.sleep(interval)
                self.store.heartbeat(run_id, self.runner_id)
        except TeamRunLeaseError:
            lease_lost.set()
            owner_task.cancel()
        except Exception:
            logger.exception("Team run {} heartbeat failed; stopping its workers", run_id)
            lease_lost.set()
            owner_task.cancel()

    async def _run(
        self,
        *,
        stored: StoredTeamRun,
        state: TeamRunState,
        request: RequestContext,
        scope: WorkspaceScope,
    ) -> None:
        run_id = stored.run_id
        lease_lost = asyncio.Event()
        owner_task = asyncio.current_task()
        assert owner_task is not None
        heartbeat = asyncio.create_task(
            self._heartbeat(run_id, owner_task, lease_lost),
            name=f"nanobot-team-heartbeat-{run_id}",
        )
        try:
            self.store.checkpoint(
                run_id,
                snapshot=state.to_snapshot(),
                status="running",
                runner_id=self.runner_id,
            )
            result = await self.engine.run(
                goal=stored.goal,
                state=state,
                request=request,
                workspace_scope=scope,
                max_concurrency=min(stored.max_concurrency, self.config.max_concurrency),
                consumed_tokens=self._consumed_tokens(state),
            )
        except asyncio.CancelledError:
            if not lease_lost.is_set():
                status = "cancelled" if run_id in self._cancel_requested else "paused"
                self.store.finish(
                    run_id,
                    status=status,
                    snapshot=state.to_snapshot(),
                    runner_id=self.runner_id,
                )
            raise
        except TeamRunLeaseError:
            logger.warning("Stopped team run {} after losing its process lease", run_id)
        except Exception as exc:
            try:
                self.store.finish(
                    run_id,
                    status="failed",
                    snapshot=state.to_snapshot(),
                    error=f"{type(exc).__name__}: {exc}",
                    runner_id=self.runner_id,
                )
            except TeamRunLeaseError:
                logger.warning("Could not persist failure for team run {} after lease loss", run_id)
        else:
            self.store.finish(
                run_id,
                status=result.status,
                snapshot=state.to_snapshot(),
                result=result,
                runner_id=self.runner_id,
            )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
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
            state = self._restore_state(stored)
            state.set_on_change(self._checkpoint_callback(run_id))
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
        if task is not None and not task.done():
            self._cancel_requested.add(run_id)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            current = self._owned(run_id, owner_session_key)
            if current.status not in _TERMINAL_RUN_STATUSES:
                self.store.finish(
                    run_id,
                    status="cancelled",
                    snapshot=current.snapshot,
                    runner_id=self.runner_id,
                )
            self._cancel_requested.discard(run_id)
        else:
            try:
                self.store.claim(run_id, self.runner_id)
                self.store.finish(
                    run_id,
                    status="cancelled",
                    snapshot=stored.snapshot,
                    runner_id=self.runner_id,
                )
            except TeamRunLeaseError as exc:
                raise TeamRunAccessError(str(exc)) from exc
        return self._owned(run_id, owner_session_key)
