from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.tools.multiagent.coordinator import (
    TeamCoordinator,
    TeamPlanError,
    validate_task_graph,
)
from nanobot.agent.tools.multiagent.models import (
    TeamTaskResult,
    TeamTaskSpec,
    TeamTaskStatus,
)
from nanobot.agent.tools.multiagent.state import TeamRunState


def _task(
    task_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    profile: str = "research",
) -> TeamTaskSpec:
    return TeamTaskSpec(
        task_id=task_id,
        role="tester",
        instruction=f"run {task_id}",
        depends_on=depends_on,
        capability_profile=profile,
    )


def _coordinator(**overrides) -> TeamCoordinator:
    options = {
        "max_tasks": 8,
        "max_concurrency": 3,
        "task_timeout_s": 1.0,
        "serialize_writes": True,
        "capability_profiles": {"general", "implement", "research", "review"},
        "write_profiles": {"implement"},
    }
    options.update(overrides)
    return TeamCoordinator(**options)


def test_validate_task_graph_rejects_cycle() -> None:
    tasks = [_task("a", depends_on=("b",)), _task("b", depends_on=("a",))]

    with pytest.raises(TeamPlanError, match="cycle"):
        validate_task_graph(tasks, max_tasks=8, capability_profiles={"research"})


def test_validate_task_graph_rejects_unknown_profile() -> None:
    tasks = [_task("a", profile="admin")]

    with pytest.raises(TeamPlanError, match="unknown capability profile"):
        validate_task_graph(tasks, max_tasks=8, capability_profiles={"research"})


@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently() -> None:
    entered: set[str] = set()
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def worker(_run_id, goal, task, dependencies):
        assert goal == "goal"
        assert dependencies == {}
        entered.add(task.task_id)
        if len(entered) == 2:
            both_entered.set()
        await release.wait()
        return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    run = asyncio.create_task(_coordinator().run("goal", [_task("a"), _task("b")], worker))
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    result = await run

    assert result.status == "completed"
    assert [task.task_id for task in result.tasks] == ["a", "b"]


@pytest.mark.asyncio
async def test_dependency_receives_completed_result() -> None:
    calls: list[str] = []

    async def worker(_run_id, _goal, task, dependencies):
        calls.append(task.task_id)
        if task.task_id == "b":
            assert dependencies["a"].content == "result-a"
        return TeamTaskResult(
            task.task_id,
            task.role,
            TeamTaskStatus.COMPLETED,
            content=f"result-{task.task_id}",
        )

    result = await _coordinator().run(
        "goal",
        [_task("a"), _task("b", depends_on=("a",))],
        worker,
    )

    assert calls == ["a", "b"]
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_failed_dependency_blocks_dependant_but_not_independent_branch() -> None:
    calls: list[str] = []

    async def worker(_run_id, _goal, task, _dependencies):
        calls.append(task.task_id)
        status = TeamTaskStatus.FAILED if task.task_id == "a" else TeamTaskStatus.COMPLETED
        return TeamTaskResult(task.task_id, task.role, status)

    result = await _coordinator().run(
        "goal",
        [_task("a"), _task("b", depends_on=("a",)), _task("c")],
        worker,
    )

    by_id = {task.task_id: task for task in result.tasks}
    assert calls == ["a", "c"]
    assert by_id["b"].status is TeamTaskStatus.BLOCKED
    assert result.status == "partial"


@pytest.mark.asyncio
async def test_write_profiles_are_serialized() -> None:
    active = 0
    peak = 0

    async def worker(_run_id, _goal, task, _dependencies):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    await _coordinator().run(
        "goal",
        [_task("a", profile="implement"), _task("b", profile="implement")],
        worker,
    )

    assert peak == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_to_running_workers() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker(_run_id, _goal, task, _dependencies):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    run = asyncio.create_task(_coordinator().run("goal", [_task("a")], worker))
    await asyncio.wait_for(entered.wait(), timeout=1)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_unhandled_task_failure_cancels_sibling_workers() -> None:
    tasks = [_task("a"), _task("b")]
    sibling_entered = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    state = TeamRunState(
        run_id="run-1",
        tasks=tasks,
        max_tasks=8,
        max_delegation_depth=0,
        capability_profiles={"research"},
        max_message_chars=100,
    )

    def fail_checkpoint(changed: TeamRunState) -> None:
        if changed.result_for("a") is not None:
            raise OSError("checkpoint unavailable")

    state.set_on_change(fail_checkpoint)

    async def worker(_state, _goal, task, _dependencies):
        if task.task_id == "a":
            await sibling_entered.wait()
            return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)
        sibling_entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    with pytest.raises(OSError, match="checkpoint unavailable"):
        await asyncio.wait_for(
            _coordinator().run("goal", tasks, worker, state=state),
            timeout=1,
        )
    assert sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_task_timeout_becomes_failed_result() -> None:
    async def worker(_run_id, _goal, task, _dependencies):
        await asyncio.sleep(1)
        return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    result = await _coordinator(task_timeout_s=0.01).run(
        "goal",
        [_task("a")],
        worker,
    )

    assert result.status == "failed"
    assert result.tasks[0].stop_reason == "timeout"


@pytest.mark.asyncio
async def test_dynamic_delegation_adds_downstream_task_without_deadlock() -> None:
    calls: list[str] = []

    async def worker(state, _goal, task, dependencies):
        calls.append(task.task_id)
        if task.task_id == "a":
            state.delegate(
                parent_id="a",
                role="reviewer",
                instruction="review a",
                capability_profile="review",
            )
        else:
            assert dependencies["a"].content == "result-a"
        return TeamTaskResult(
            task.task_id,
            task.role,
            TeamTaskStatus.COMPLETED,
            content=f"result-{task.task_id}",
        )

    result = await _coordinator(
        max_concurrency=1,
        max_delegation_depth=2,
    ).run("goal", [_task("a")], worker)

    assert result.status == "completed"
    assert calls[0] == "a"
    assert len(calls) == 2
    assert result.tasks[1].task_id.startswith("delegate-")
