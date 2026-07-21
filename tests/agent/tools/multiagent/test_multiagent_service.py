from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.models import (
    TeamTaskResult,
    TeamTaskSpec,
    TeamTaskStatus,
)
from nanobot.agent.tools.multiagent.service import TeamRunAccessError, TeamRunService
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.store import TeamRunStore
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import build_workspace_scope


def _request(session_key: str = "cli:one") -> RequestContext:
    return RequestContext(
        channel="cli",
        chat_id="one",
        session_key=session_key,
        runtime=MagicMock(),
    )


def _task(
    task_id: str,
    depends_on: tuple[str, ...] = (),
    profile: str = "research",
) -> TeamTaskSpec:
    return TeamTaskSpec(task_id, "researcher", f"run {task_id}", depends_on, profile)


class _Worker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.budget_used: list[int] = []
        self.entered = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def run(self, **kwargs):
        task = kwargs["task"]
        self.calls.append(task.task_id)
        self.budget_used.append(kwargs["budget"].used)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return TeamTaskResult(
            task.task_id,
            task.role,
            TeamTaskStatus.COMPLETED,
            content=f"done-{task.task_id}",
        )


def _service(
    tmp_path,
    worker: _Worker,
    store: TeamRunStore | None = None,
    **config_overrides,
) -> TeamRunService:
    config = MultiAgentToolConfig(enable=True, **config_overrides)
    tools_config = ToolsConfig(multiagent=config, restrict_to_workspace=True)
    return TeamRunService(
        workspace=tmp_path,
        tools_config=tools_config,
        config=config,
        worker_runner=worker,
        store=store,
    )


@pytest.mark.asyncio
async def test_start_wait_completes_and_persists_result(tmp_path) -> None:
    worker = _Worker()
    service = _service(tmp_path, worker)

    started = await service.start(
        goal="goal",
        tasks=[_task("a"), _task("b", ("a",))],
        request=_request(),
        workspace_scope=None,
    )
    finished = await service.wait(
        run_id=started.run_id,
        request=_request(),
        workspace_scope=None,
        timeout_seconds=1,
    )

    assert finished.status == "completed"
    assert [task.task_id for task in finished.result.tasks] == ["a", "b"]
    assert worker.calls == ["a", "b"]


@pytest.mark.asyncio
async def test_cancel_stops_active_worker(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker)
    started = await service.start(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    )
    await asyncio.wait_for(worker.entered.wait(), timeout=1)

    cancelled = await service.cancel(started.run_id, "cli:one")

    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_immediately_after_start_marks_queued_run_cancelled(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker)
    started = await service.start(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    )

    cancelled = await service.cancel(started.run_id, "cli:one")

    assert cancelled.status == "cancelled"
    assert started.run_id not in service._cancel_requested


@pytest.mark.asyncio
async def test_foreground_cancellation_cancels_durable_run(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker)
    foreground = asyncio.create_task(service.run_foreground(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    ))
    await asyncio.wait_for(worker.entered.wait(), timeout=1)
    run_id = next(iter(service._active))

    foreground.cancel()
    with pytest.raises(asyncio.CancelledError):
        await foreground

    assert service.status(run_id, "cli:one").status == "cancelled"


@pytest.mark.asyncio
async def test_internal_foreground_task_cancellation_is_reported_as_error(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker)
    foreground = asyncio.create_task(service.run_foreground(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    ))
    await asyncio.wait_for(worker.entered.wait(), timeout=1)
    run_id, internal_task = next(iter(service._active.items()))

    internal_task.cancel()

    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        await foreground
    assert service.status(run_id, "cli:one").status == "paused"


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_background_run(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker)
    started = await service.start(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    )
    await asyncio.wait_for(worker.entered.wait(), timeout=1)

    waiting = await service.wait(
        run_id=started.run_id,
        request=_request(),
        workspace_scope=None,
        timeout_seconds=0.001,
    )
    assert waiting.status == "running"

    worker.release.set()
    finished = await service.wait(
        run_id=started.run_id,
        request=_request(),
        workspace_scope=None,
        timeout_seconds=1,
    )
    assert finished.status == "completed"


@pytest.mark.asyncio
async def test_resume_skips_completed_nodes(tmp_path) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3")
    state = TeamRunState(
        run_id="resumable",
        tasks=[_task("a"), _task("b", ("a",))],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=100,
    )
    state.record_result(
        TeamTaskResult(
            "a",
            "researcher",
            TeamTaskStatus.COMPLETED,
            content="done-a",
            usage={"total_tokens": 7},
        )
    )
    store.create(
        run_id="resumable",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=2,
        snapshot=state.to_snapshot(),
    )
    store.checkpoint("resumable", snapshot=state.to_snapshot(), status="paused")
    worker = _Worker()
    service = _service(tmp_path, worker, store)

    finished = await service.wait(
        run_id="resumable",
        request=_request(),
        workspace_scope=None,
        timeout_seconds=1,
    )

    assert finished.status == "completed"
    assert worker.calls == ["b"]
    assert worker.budget_used == [7]


@pytest.mark.asyncio
async def test_distinct_runs_share_global_concurrency_limit(tmp_path) -> None:
    class ConcurrentWorker(_Worker):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0

        async def run(self, **kwargs):
            task = kwargs["task"]
            self.calls.append(task.task_id)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    worker = ConcurrentWorker()
    service = _service(tmp_path, worker, max_concurrency=2)
    first = await service.start(
        goal="first",
        tasks=[_task("a"), _task("b")],
        request=_request(),
        workspace_scope=None,
    )
    second = await service.start(
        goal="second",
        tasks=[_task("c"), _task("d")],
        request=_request(),
        workspace_scope=None,
    )

    await asyncio.gather(
        service.wait(
            run_id=first.run_id,
            request=_request(),
            workspace_scope=None,
            timeout_seconds=1,
        ),
        service.wait(
            run_id=second.run_id,
            request=_request(),
            workspace_scope=None,
            timeout_seconds=1,
        ),
    )

    assert worker.peak == 2


@pytest.mark.asyncio
async def test_write_workers_are_serialized_across_runs(tmp_path) -> None:
    class WriteWorker(_Worker):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0

        async def run(self, **kwargs):
            task = kwargs["task"]
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    worker = WriteWorker()
    service = _service(tmp_path, worker, max_concurrency=2)
    first = await service.start(
        goal="first",
        tasks=[_task("a", profile="implement")],
        request=_request(),
        workspace_scope=None,
    )
    second = await service.start(
        goal="second",
        tasks=[_task("b", profile="implement")],
        request=_request(),
        workspace_scope=None,
    )

    await asyncio.gather(
        service.wait(
            run_id=first.run_id,
            request=_request(),
            workspace_scope=None,
            timeout_seconds=1,
        ),
        service.wait(
            run_id=second.run_id,
            request=_request(),
            workspace_scope=None,
            timeout_seconds=1,
        ),
    )

    assert worker.peak == 1


@pytest.mark.asyncio
async def test_resume_clamps_old_run_to_current_concurrency_and_task_policy(tmp_path) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3")
    tasks = [_task("a"), _task("b")]
    state = TeamRunState(
        run_id="old-policy",
        tasks=tasks,
        max_tasks=8,
        max_delegation_depth=3,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=4_000,
    )
    store.create(
        run_id="old-policy",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=3,
        snapshot=state.to_snapshot(),
    )
    store.checkpoint("old-policy", snapshot=state.to_snapshot(), status="paused")

    class ConcurrentWorker(_Worker):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0

        async def run(self, **kwargs):
            task = kwargs["task"]
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    worker = ConcurrentWorker()
    service = _service(
        tmp_path,
        worker,
        store,
        max_concurrency=1,
        max_tasks=1,
        max_delegation_depth=0,
        max_message_chars=100,
    )

    finished = await service.wait(
        run_id="old-policy",
        request=_request(),
        workspace_scope=None,
        timeout_seconds=1,
    )

    assert finished.status == "completed"
    assert worker.peak == 1


@pytest.mark.asyncio
async def test_resume_does_not_spend_more_tokens_after_current_budget_is_exhausted(
    tmp_path,
) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3")
    state = TeamRunState(
        run_id="old-budget",
        tasks=[_task("a"), _task("b", ("a",))],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=4_000,
    )
    state.record_result(TeamTaskResult(
        "a",
        "researcher",
        TeamTaskStatus.COMPLETED,
        usage={"total_tokens": 2_000},
    ))
    store.create(
        run_id="old-budget",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=1,
        snapshot=state.to_snapshot(),
    )
    store.checkpoint("old-budget", snapshot=state.to_snapshot(), status="paused")
    worker = _Worker()
    service = _service(tmp_path, worker, store, max_total_tokens=1_000)

    finished = await service.wait(
        run_id="old-budget",
        request=_request(),
        workspace_scope=None,
        timeout_seconds=1,
    )

    assert finished.status == "partial"
    assert worker.calls == []
    assert finished.result.tasks[1].stop_reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_active_run_limit_rejects_unbounded_background_runs(tmp_path) -> None:
    worker = _Worker()
    worker.release = asyncio.Event()
    service = _service(tmp_path, worker, max_active_runs=1)
    first = await service.start(
        goal="first",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    )

    with pytest.raises(ValueError, match="active team run limit"):
        await service.start(
            goal="second",
            tasks=[_task("b")],
            request=_request(),
            workspace_scope=None,
        )

    await service.cancel(first.run_id, "cli:one")


@pytest.mark.asyncio
async def test_live_run_cannot_be_resumed_by_second_service(tmp_path) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3", lease_seconds=1)
    first_worker = _Worker()
    first_worker.release = asyncio.Event()
    first = _service(tmp_path, first_worker, store)
    started = await first.start(
        goal="goal",
        tasks=[_task("a")],
        request=_request(),
        workspace_scope=None,
    )
    await asyncio.wait_for(first_worker.entered.wait(), timeout=1)
    second = _service(tmp_path, _Worker(), store)

    with pytest.raises(TeamRunAccessError, match="another process"):
        await second.wait(
            run_id=started.run_id,
            request=_request(),
            workspace_scope=None,
            timeout_seconds=0,
        )

    await first.cancel(started.run_id, "cli:one")


@pytest.mark.asyncio
async def test_owner_and_workspace_are_enforced_on_resume(tmp_path) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3")
    state = TeamRunState(
        run_id="owned",
        tasks=[_task("a")],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"research"},
        max_message_chars=100,
    )
    store.create(
        run_id="owned",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=1,
        snapshot=state.to_snapshot(),
    )
    store.checkpoint("owned", snapshot=state.to_snapshot(), status="paused")
    service = _service(tmp_path, _Worker(), store)

    with pytest.raises(TeamRunAccessError, match="different session"):
        service.status("owned", "cli:two")

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(TeamRunAccessError, match="original workspace"):
        await service.wait(
            run_id="owned",
            request=_request(),
            workspace_scope=build_workspace_scope(other, "restricted"),
            timeout_seconds=0,
        )
