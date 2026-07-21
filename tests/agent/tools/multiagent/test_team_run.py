from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.context import RequestContext, ToolContext, request_context
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.models import TeamTaskResult, TeamTaskStatus
from nanobot.agent.tools.multiagent.plan_schema import team_tasks_schema
from nanobot.agent.tools.multiagent.service import TeamRunService, shared_team_run_service
from nanobot.agent.tools.multiagent.team_run import TeamRunTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime() -> LLMRuntime:
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test-model", context_window_tokens=32_000)


def _tool(tmp_path, **config_overrides) -> TeamRunTool:
    config = MultiAgentToolConfig(enable=True, **config_overrides)
    tools_config = ToolsConfig(multiagent=config)
    return TeamRunTool(TeamRunService(
        workspace=tmp_path,
        tools_config=tools_config,
        config=config,
    ))


def test_team_run_is_disabled_by_default(tmp_path) -> None:
    registry = ToolRegistry()

    ToolLoader().load(
        ToolContext(config=ToolsConfig(), workspace=str(tmp_path)),
        registry,
    )

    assert not {
        "team_run",
        "team_start",
        "team_status",
        "team_wait",
        "team_cancel",
    } & set(registry.tool_names)


def test_team_run_registers_when_enabled(tmp_path) -> None:
    registry = ToolRegistry()
    config = ToolsConfig(multiagent=MultiAgentToolConfig(enable=True))

    ToolLoader().load(ToolContext(config=config, workspace=str(tmp_path)), registry)

    assert {
        "team_run",
        "team_start",
        "team_status",
        "team_wait",
        "team_cancel",
    } <= set(registry.tool_names)


def test_multiagent_config_accepts_camel_case() -> None:
    config = ToolsConfig.model_validate({
        "multiagent": {
            "enable": True,
            "maxTasks": 4,
            "maxConcurrency": 2,
            "maxActiveRuns": 3,
            "maxDelegationDepth": 1,
            "maxStoredRuns": 50,
        },
    })

    assert config.multiagent.enable is True
    assert config.multiagent.max_tasks == 4
    assert config.multiagent.max_concurrency == 2
    assert config.multiagent.max_active_runs == 3
    assert config.multiagent.max_delegation_depth == 1
    assert config.multiagent.max_stored_runs == 50


def test_task_schema_describes_capability_profiles_as_built_in() -> None:
    schema = team_tasks_schema().to_json_schema()

    profile = schema["items"]["properties"]["capabilityProfile"]
    assert profile["description"].startswith("Built-in capability profile")
    assert profile["enum"] == ["general", "implement", "research", "review"]


@pytest.mark.asyncio
async def test_background_lifecycle_tools_share_service(tmp_path) -> None:
    registry = ToolRegistry()
    config = ToolsConfig(
        multiagent=MultiAgentToolConfig(enable=True),
        restrict_to_workspace=True,
    )
    ToolLoader().load(ToolContext(config=config, workspace=str(tmp_path)), registry)
    start_tool = registry.get("team_start")
    wait_tool = registry.get("team_wait")
    status_tool = registry.get("team_status")
    run_tool = registry.get("team_run")
    assert start_tool is not None
    assert wait_tool is not None
    assert status_tool is not None
    assert run_tool is not None
    assert start_tool.service is wait_tool.service is status_tool.service is run_tool.service

    async def worker_result(**kwargs):
        task = kwargs["task"]
        return TeamTaskResult(task.task_id, task.role, TeamTaskStatus.COMPLETED)

    start_tool.service.worker_runner.run = AsyncMock(side_effect=worker_result)
    with request_context(RequestContext(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        runtime=_runtime(),
    )):
        started = json.loads(await start_tool.execute(
            goal="background goal",
            tasks=[{"id": "a", "role": "researcher", "instruction": "research"}],
        ))
        finished = json.loads(await wait_tool.execute(
            run_id=started["runId"],
            timeout_seconds=1,
        ))
        status = json.loads(await status_tool.execute(run_id=started["runId"]))

    assert finished["status"] == "completed"
    assert status["result"]["tasks"][0]["id"] == "a"


def test_new_tool_context_gets_fresh_service_and_current_config(tmp_path) -> None:
    first_registry = ToolRegistry()
    first_config = ToolsConfig(
        multiagent=MultiAgentToolConfig(enable=True, max_concurrency=1),
    )
    ToolLoader().load(ToolContext(config=first_config, workspace=str(tmp_path)), first_registry)

    second_registry = ToolRegistry()
    second_config = ToolsConfig(
        multiagent=MultiAgentToolConfig(enable=True, max_concurrency=2),
    )
    ToolLoader().load(ToolContext(config=second_config, workspace=str(tmp_path)), second_registry)

    first_service = first_registry.get("team_run").service
    second_service = second_registry.get("team_run").service
    assert first_service is not second_service
    assert second_service.config.max_concurrency == 2


def test_team_service_joins_subagent_worker_capacity_pool(tmp_path) -> None:
    class Manager:
        def __init__(self) -> None:
            self.capacity = 1

        def ensure_worker_capacity(self, capacity: int) -> None:
            self.capacity = max(self.capacity, capacity)

        @asynccontextmanager
        async def worker_slot(self):
            yield

    manager = Manager()
    config = ToolsConfig(multiagent=MultiAgentToolConfig(enable=True, max_concurrency=3))
    context = ToolContext(
        config=config,
        workspace=str(tmp_path),
        subagent_manager=manager,
    )

    service = shared_team_run_service(context)

    assert manager.capacity == 3
    assert service.engine.agent_worker_slot == manager.worker_slot


def test_agent_loop_wires_one_service_and_shared_worker_capacity(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    config = ToolsConfig(multiagent=MultiAgentToolConfig(enable=True, max_concurrency=3))

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=config,
        max_concurrent_subagents=1,
    )

    assert loop.tools.get("team_run").service is loop.tools.get("team_start").service
    assert loop.subagents._worker_capacity == 3


@pytest.mark.asyncio
async def test_team_run_executes_dependency_graph_and_returns_json(tmp_path) -> None:
    tool = _tool(tmp_path)

    async def worker_result(**kwargs):
        task = kwargs["task"]
        return TeamTaskResult(
            task_id=task.task_id,
            role=task.role,
            status=TeamTaskStatus.COMPLETED,
            content=f"done-{task.task_id}",
            usage={"total_tokens": 5},
        )

    tool.worker_runner.run = AsyncMock(side_effect=worker_result)
    tasks = [
        {"id": "research", "role": "researcher", "instruction": "find evidence"},
        {
            "id": "review",
            "role": "reviewer",
            "instruction": "review evidence",
            "dependsOn": ["research"],
            "capabilityProfile": "review",
        },
    ]

    with request_context(RequestContext(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        runtime=_runtime(),
    )):
        raw = await tool.execute(goal="produce answer", tasks=tasks)

    payload = json.loads(raw)
    assert payload["status"] == "completed"
    assert [task["id"] for task in payload["tasks"]] == ["research", "review"]
    assert payload["totalUsage"]["total_tokens"] == 10
    second_dependencies = tool.worker_runner.run.await_args_list[1].kwargs["dependencies"]
    assert second_dependencies["research"].content == "done-research"


@pytest.mark.asyncio
async def test_team_run_rejects_cycle_before_starting_workers(tmp_path) -> None:
    tool = _tool(tmp_path)
    tool.worker_runner.run = AsyncMock()

    with request_context(RequestContext(
        channel="cli",
        chat_id="direct",
        runtime=_runtime(),
    )):
        result = await tool.execute(
            goal="goal",
            tasks=[
                {"id": "a", "role": "one", "instruction": "a", "dependsOn": ["b"]},
                {"id": "b", "role": "two", "instruction": "b", "dependsOn": ["a"]},
            ],
        )

    assert result.is_error
    assert "dependency cycle" in result
    tool.worker_runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_run_cannot_raise_configured_concurrency(tmp_path) -> None:
    tool = _tool(tmp_path, max_concurrency=2)

    with request_context(RequestContext(
        channel="cli",
        chat_id="direct",
        runtime=_runtime(),
    )):
        result = await tool.execute(
            goal="goal",
            tasks=[{"id": "a", "role": "one", "instruction": "a"}],
            max_concurrency=3,
        )

    assert result.is_error
    assert "exceeds configured maximum" in result


@pytest.mark.asyncio
async def test_team_run_requires_active_runtime(tmp_path) -> None:
    tool = _tool(tmp_path)

    result = await tool.execute(
        goal="goal",
        tasks=[{"id": "a", "role": "one", "instruction": "a"}],
    )

    assert result.is_error
    assert "active model runtime" in result
