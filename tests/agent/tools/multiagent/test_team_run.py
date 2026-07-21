from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.context import RequestContext, ToolContext, request_context
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.models import TeamTaskResult, TeamTaskStatus
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
    return TeamRunTool(workspace=tmp_path, tools_config=tools_config, config=config)


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
            "maxDelegationDepth": 1,
            "maxStoredRuns": 50,
        },
    })

    assert config.multiagent.enable is True
    assert config.multiagent.max_tasks == 4
    assert config.multiagent.max_concurrency == 2
    assert config.multiagent.max_delegation_depth == 1
    assert config.multiagent.max_stored_runs == 50


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
    assert start_tool is not None
    assert wait_tool is not None
    assert status_tool is not None
    assert start_tool.service is wait_tool.service is status_tool.service

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
