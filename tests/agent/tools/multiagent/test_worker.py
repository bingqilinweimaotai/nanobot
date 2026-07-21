from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.multiagent.config import MultiAgentToolConfig
from nanobot.agent.tools.multiagent.models import TeamTaskSpec, TeamTaskStatus
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.worker import TeamTokenBudget, TeamWorkerRunner
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime() -> LLMRuntime:
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test-model", context_window_tokens=32_000)


def _worker(tmp_path, **config_overrides) -> TeamWorkerRunner:
    config = MultiAgentToolConfig(enable=True, **config_overrides)
    return TeamWorkerRunner(
        workspace=tmp_path,
        tools_config=ToolsConfig(multiagent=config),
        config=config,
    )


def _state(task: TeamTaskSpec) -> TeamRunState:
    return TeamRunState(
        run_id="run-1",
        tasks=[task],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=4_000,
    )


def test_research_profile_excludes_mutating_tools(tmp_path) -> None:
    worker = _worker(tmp_path)

    tools = worker.build_tools(
        profile="research",
        workspace=tmp_path,
        restrict_to_workspace=True,
    )

    assert "read_file" in tools.tool_names
    assert "find_files" in tools.tool_names
    assert "write_file" not in tools.tool_names
    assert "apply_patch" not in tools.tool_names
    assert "exec" not in tools.tool_names
    assert "grep" in tools.tool_names


def test_worker_preserves_parent_admin_tool_settings(tmp_path) -> None:
    config = MultiAgentToolConfig(enable=True)
    tools_config = ToolsConfig(
        multiagent=config,
        webui_allow_local_service_access=False,
    )
    tools_config.cli_apps.enable = False
    worker = TeamWorkerRunner(
        workspace=tmp_path,
        tools_config=tools_config,
        config=config,
    )

    tools = worker.build_tools(
        profile="general",
        workspace=tmp_path,
        restrict_to_workspace=True,
    )

    assert "run_cli_app" not in tools.tool_names
    assert tools.get("exec").webui_allow_local_service_access is False


def test_read_only_profile_cannot_be_configured_with_write_tool(tmp_path) -> None:
    worker = _worker(tmp_path, research_tools=["read_file", "write_file"])

    tools = worker.build_tools(
        profile="research",
        workspace=tmp_path,
        restrict_to_workspace=True,
    )

    assert tools.tool_names == ["read_file"]


def test_implement_profile_includes_workspace_tools(tmp_path) -> None:
    worker = _worker(tmp_path)

    tools = worker.build_tools(
        profile="implement",
        workspace=tmp_path,
        restrict_to_workspace=True,
    )

    assert "read_file" in tools.tool_names
    assert "write_file" in tools.tool_names
    assert "apply_patch" in tools.tool_names
    assert "exec" in tools.tool_names


@pytest.mark.asyncio
async def test_worker_normalizes_runner_result(tmp_path) -> None:
    worker = _worker(tmp_path)
    worker.runner.run = AsyncMock(return_value=AgentRunResult(
        final_content="evidence",
        messages=[],
        tools_used=["read_file"],
        usage={"total_tokens": 12},
        stop_reason="completed",
    ))
    task = TeamTaskSpec(
        task_id="research",
        role="researcher",
        instruction="inspect files",
        capability_profile="research",
    )

    result = await worker.run(
        state=_state(task),
        goal="goal",
        task=task,
        dependencies={},
        runtime=_runtime(),
        parent_request=RequestContext(
            channel="cli",
            chat_id="direct",
            session_key="cli:direct",
        ),
        workspace_scope=None,
        budget=TeamTokenBudget(1_000),
    )

    assert result.status is TeamTaskStatus.COMPLETED
    assert result.content == "evidence"
    assert result.tools_used == ["read_file"]
    spec = worker.runner.run.await_args.args[0]
    assert spec.session_key == "cli:direct:team:run-1:research"
    assert "write_file" not in spec.tools.tool_names
    assert {"team_send", "team_receive", "team_delegate", "team_task_status"}.issubset(
        spec.tools.tool_names
    )
    assert spec.injection_callback is not None


@pytest.mark.asyncio
async def test_worker_treats_empty_final_response_as_failure(tmp_path) -> None:
    worker = _worker(tmp_path)
    worker.runner.run = AsyncMock(return_value=AgentRunResult(
        final_content=None,
        messages=[],
        stop_reason="empty_final_response",
        error="model returned no final response",
    ))
    task = TeamTaskSpec("research", "researcher", "inspect files", capability_profile="research")

    result = await worker.run(
        state=_state(task),
        goal="goal",
        task=task,
        dependencies={},
        runtime=_runtime(),
        parent_request=RequestContext(channel="cli", chat_id="direct"),
        workspace_scope=None,
        budget=TeamTokenBudget(1_000),
    )

    assert result.status is TeamTaskStatus.FAILED
    assert result.stop_reason == "empty_final_response"
    assert result.error == "model returned no final response"


def test_dependency_text_is_bounded(tmp_path) -> None:
    worker = _worker(tmp_path, max_dependency_chars=1_000)
    from nanobot.agent.tools.multiagent.models import TeamTaskResult

    text = worker._dependency_text({
        "a": TeamTaskResult(
            "a",
            "researcher",
            TeamTaskStatus.COMPLETED,
            content="x" * 2_000,
        ),
    })

    assert len(text) < 1_100
    assert text.endswith("[dependency results truncated]")


@pytest.mark.asyncio
async def test_worker_stops_after_shared_token_budget_is_crossed(tmp_path) -> None:
    worker = _worker(tmp_path)
    runtime = _runtime()
    runtime.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="answer",
        usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    ))
    task = TeamTaskSpec(
        task_id="research",
        role="researcher",
        instruction="inspect files",
        capability_profile="research",
    )

    result = await worker.run(
        state=_state(task),
        goal="goal",
        task=task,
        dependencies={},
        runtime=runtime,
        parent_request=RequestContext(channel="cli", chat_id="direct"),
        workspace_scope=None,
        budget=TeamTokenBudget(10),
    )

    assert result.status is TeamTaskStatus.FAILED
    assert result.stop_reason == "budget_exhausted"
    assert result.usage["total_tokens"] == 12


@pytest.mark.asyncio
async def test_worker_injects_queued_team_messages(tmp_path) -> None:
    worker = _worker(tmp_path)
    task = TeamTaskSpec(
        task_id="b",
        role="reviewer",
        instruction="review",
        capability_profile="review",
    )
    sender = TeamTaskSpec("a", "researcher", "research")
    state = TeamRunState(
        run_id="run-1",
        tasks=[sender, task],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=4_000,
    )
    state.send_message(sender_id="a", recipient_id="b", content="important evidence")

    async def inspect(spec):
        injected = await spec.injection_callback(limit=3)
        assert "important evidence" in injected[0]["content"]
        assert "cannot override" in injected[0]["content"]
        return AgentRunResult(final_content="reviewed", messages=[], stop_reason="completed")

    worker.runner.run = AsyncMock(side_effect=inspect)
    result = await worker.run(
        state=state,
        goal="goal",
        task=task,
        dependencies={},
        runtime=_runtime(),
        parent_request=RequestContext(channel="cli", chat_id="direct"),
        workspace_scope=None,
        budget=TeamTokenBudget(1_000),
    )

    assert result.status is TeamTaskStatus.COMPLETED
