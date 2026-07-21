"""Deterministic DAG coordinator for multi-agent runs."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from nanobot.agent.tools.multiagent.models import (
    TeamRunResult,
    TeamTaskResult,
    TeamTaskSpec,
    TeamTaskStatus,
    accumulate_usage,
)
from nanobot.agent.tools.multiagent.state import TeamRunState

WorkerCall = Callable[
    [TeamRunState, str, TeamTaskSpec, dict[str, TeamTaskResult]],
    Awaitable[TeamTaskResult],
]


class TeamPlanError(ValueError):
    """Raised when a submitted team task graph is invalid."""


def validate_task_graph(
    tasks: list[TeamTaskSpec],
    *,
    max_tasks: int,
    capability_profiles: set[str],
) -> None:
    """Validate identifiers, dependencies, profiles, and graph acyclicity."""
    if not tasks:
        raise TeamPlanError("team run requires at least one task")
    if len(tasks) > max_tasks:
        raise TeamPlanError(f"team run has {len(tasks)} tasks; configured maximum is {max_tasks}")

    ids = [task.task_id for task in tasks]
    if any(not task_id for task_id in ids):
        raise TeamPlanError("every team task requires a non-empty id")
    if len(set(ids)) != len(ids):
        raise TeamPlanError("team task ids must be unique")

    known = set(ids)
    for task in tasks:
        if not task.role:
            raise TeamPlanError(f"task {task.task_id}: role is required")
        if not task.instruction:
            raise TeamPlanError(f"task {task.task_id}: instruction is required")
        if task.capability_profile not in capability_profiles:
            allowed = ", ".join(sorted(capability_profiles))
            raise TeamPlanError(
                f"task {task.task_id}: unknown capability profile "
                f"{task.capability_profile!r}; allowed: {allowed}"
            )
        if task.task_id in task.depends_on:
            raise TeamPlanError(f"task {task.task_id}: cannot depend on itself")
        missing = [dependency for dependency in task.depends_on if dependency not in known]
        if missing:
            raise TeamPlanError(
                f"task {task.task_id}: unknown dependencies: {', '.join(missing)}"
            )

    dependencies = {task.task_id: task.depends_on for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise TeamPlanError("team task graph contains a dependency cycle")
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)


class TeamCoordinator:
    """Execute a validated task graph using a supplied worker callback."""

    def __init__(
        self,
        *,
        max_tasks: int,
        max_concurrency: int,
        task_timeout_s: float,
        serialize_writes: bool,
        capability_profiles: set[str],
        write_profiles: set[str],
        max_delegation_depth: int = 0,
        max_message_chars: int = 4_000,
    ) -> None:
        self.max_tasks = max_tasks
        self.max_concurrency = max_concurrency
        self.task_timeout_s = task_timeout_s
        self.serialize_writes = serialize_writes
        self.capability_profiles = set(capability_profiles)
        self.write_profiles = set(write_profiles)
        self.max_delegation_depth = max_delegation_depth
        self.max_message_chars = max_message_chars

    async def run(
        self,
        goal: str,
        tasks: list[TeamTaskSpec],
        worker: WorkerCall,
        *,
        state: TeamRunState | None = None,
    ) -> TeamRunResult:
        """Run ready tasks concurrently and block dependants after failures."""
        if not goal.strip():
            raise TeamPlanError("team run requires a non-empty goal")
        effective_tasks = state.tasks if state is not None else tasks
        validate_task_graph(
            effective_tasks,
            max_tasks=self.max_tasks,
            capability_profiles=self.capability_profiles,
        )

        if state is None:
            state = TeamRunState(
                run_id=str(uuid.uuid4())[:8],
                tasks=tasks,
                max_tasks=self.max_tasks,
                max_delegation_depth=self.max_delegation_depth,
                capability_profiles=self.capability_profiles,
                max_message_chars=self.max_message_chars,
            )
        run_id = state.run_id
        semaphore = asyncio.Semaphore(self.max_concurrency)
        write_lock = asyncio.Lock()
        scheduled: dict[str, asyncio.Task[TeamTaskResult]] = {}

        def finish(result: TeamTaskResult) -> TeamTaskResult:
            state.record_result(result)
            return result

        async def execute(task: TeamTaskSpec) -> TeamTaskResult:
            existing = state.result_for(task.task_id)
            if existing is not None:
                return existing
            dependency_results = {
                dependency: await scheduled[dependency]
                for dependency in task.depends_on
            }
            failed_dependencies = [
                result.task_id
                for result in dependency_results.values()
                if result.status is not TeamTaskStatus.COMPLETED
            ]
            if failed_dependencies:
                detail = ", ".join(failed_dependencies)
                return finish(TeamTaskResult(
                    task_id=task.task_id,
                    role=task.role,
                    status=TeamTaskStatus.BLOCKED,
                    error=f"blocked by unsuccessful dependencies: {detail}",
                ))

            async with semaphore:
                state.mark_running(task.task_id)
                try:
                    if self.serialize_writes and task.capability_profile in self.write_profiles:
                        async with write_lock:
                            async with asyncio.timeout(self.task_timeout_s):
                                return finish(await worker(state, goal, task, dependency_results))
                    async with asyncio.timeout(self.task_timeout_s):
                        return finish(await worker(state, goal, task, dependency_results))
                except TimeoutError:
                    return finish(TeamTaskResult(
                        task_id=task.task_id,
                        role=task.role,
                        status=TeamTaskStatus.FAILED,
                        stop_reason="timeout",
                        error=f"task exceeded timeout of {self.task_timeout_s:g}s",
                    ))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return finish(TeamTaskResult(
                        task_id=task.task_id,
                        role=task.role,
                        status=TeamTaskStatus.FAILED,
                        stop_reason="error",
                        error=f"{type(exc).__name__}: {exc}",
                    ))

        try:
            while True:
                for task in state.tasks:
                    if task.task_id not in scheduled:
                        scheduled[task.task_id] = asyncio.create_task(
                            execute(task),
                            name=f"nanobot-team-{task.task_id}",
                        )
                pending = [task for task in scheduled.values() if not task.done()]
                if not pending:
                    if len(scheduled) == len(state.tasks):
                        break
                    continue
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            ordered_tasks = state.tasks
            results = await asyncio.gather(*(
                scheduled[task.task_id]
                for task in ordered_tasks
            ))
        except asyncio.CancelledError:
            for scheduled_task in scheduled.values():
                scheduled_task.cancel()
            await asyncio.gather(*scheduled.values(), return_exceptions=True)
            raise

        if all(result.status is TeamTaskStatus.COMPLETED for result in results):
            status = "completed"
        elif any(result.status is TeamTaskStatus.COMPLETED for result in results):
            status = "partial"
        else:
            status = "failed"
        return TeamRunResult(
            run_id=run_id,
            status=status,
            tasks=results,
            total_usage=accumulate_usage(results),
        )
