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


def _task(task_id: str, depends_on: tuple[str, ...] = ()) -> TeamTaskSpec:
    return TeamTaskSpec(task_id, "researcher", f"run {task_id}", depends_on)


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


def _service(tmp_path, worker: _Worker, store: TeamRunStore | None = None) -> TeamRunService:
    config = MultiAgentToolConfig(enable=True)
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
