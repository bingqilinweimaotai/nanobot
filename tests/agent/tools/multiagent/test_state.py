from __future__ import annotations

import json

import pytest

from nanobot.agent.tools.multiagent.collaboration_tools import (
    TeamDelegateTool,
    TeamReceiveTool,
    TeamSendTool,
    TeamTaskStatusTool,
)
from nanobot.agent.tools.multiagent.models import (
    TeamTaskResult,
    TeamTaskSpec,
    TeamTaskStatus,
)
from nanobot.agent.tools.multiagent.state import TeamCollaborationError, TeamRunState


def _task(task_id: str) -> TeamTaskSpec:
    return TeamTaskSpec(task_id, "researcher", f"run {task_id}")


def test_finishing_task_rejects_late_messages() -> None:
    state = TeamRunState(
        run_id="run-1",
        tasks=[_task("a"), _task("b")],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"research"},
        max_message_chars=100,
    )
    state.mark_running("b")
    state.mark_finishing("b")

    with pytest.raises(TeamCollaborationError, match="already finished"):
        state.send_message(sender_id="a", recipient_id="b", content="too late")


def test_message_racing_with_terminal_record_is_preserved_in_result() -> None:
    state = TeamRunState(
        run_id="run-1",
        tasks=[_task("a"), _task("b")],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"research"},
        max_message_chars=100,
    )
    state.mark_running("b")
    state.send_message(sender_id="a", recipient_id="b", content="last evidence")
    result = TeamTaskResult("b", "researcher", TeamTaskStatus.COMPLETED, content="answer")

    state.record_result(result)

    assert result.late_messages == [{
        "senderId": "a",
        "recipientId": "b",
        "content": "last evidence",
    }]
    assert state.to_snapshot()["mailboxes"]["b"] == []


def test_late_messages_survive_result_serialization() -> None:
    result = TeamTaskResult(
        "b",
        "researcher",
        TeamTaskStatus.COMPLETED,
        late_messages=[{
            "senderId": "a",
            "recipientId": "b",
            "content": "last evidence",
        }],
    )

    restored = TeamTaskResult.from_payload(result.to_payload())

    assert restored.late_messages == result.late_messages


def test_resume_uses_current_policy_limits() -> None:
    original = TeamRunState(
        run_id="run-1",
        tasks=[_task("a")],
        max_tasks=8,
        max_delegation_depth=3,
        capability_profiles={"research", "review"},
        max_message_chars=4_000,
    )

    restored = TeamRunState.from_snapshot(
        original.to_snapshot(),
        max_tasks=2,
        max_delegation_depth=1,
        capability_profiles={"research"},
        max_message_chars=200,
    )

    assert restored.max_tasks == 2
    assert restored.max_delegation_depth == 1
    assert restored.capability_profiles == {"research"}
    assert restored.max_message_chars == 200


def _state(*tasks: TeamTaskSpec, max_tasks: int = 8, max_depth: int = 2) -> TeamRunState:
    return TeamRunState(
        run_id="run-1",
        tasks=list(tasks),
        max_tasks=max_tasks,
        max_delegation_depth=max_depth,
        capability_profiles={"general", "implement", "research", "review"},
        max_message_chars=100,
    )


def test_mailbox_delivers_in_order_and_drains() -> None:
    state = _state(_task("a"), _task("b"))

    state.send_message(sender_id="a", recipient_id="b", content="one")
    state.send_message(sender_id="a", recipient_id="b", content="two")

    first = state.drain_messages("b", limit=1)
    second = state.drain_messages("b", limit=5)
    assert [message.content for message in first] == ["one"]
    assert [message.content for message in second] == ["two"]


def test_mailbox_rejects_unknown_or_finished_recipient() -> None:
    state = _state(_task("a"), _task("b"))

    with pytest.raises(TeamCollaborationError, match="unknown recipient"):
        state.send_message(sender_id="a", recipient_id="missing", content="hello")

    state.record_result(TeamTaskResult("b", "researcher", TeamTaskStatus.COMPLETED))
    with pytest.raises(TeamCollaborationError, match="already finished"):
        state.send_message(sender_id="a", recipient_id="b", content="late")


def test_delegation_is_bounded_and_deduplicated() -> None:
    state = _state(_task("a"), max_tasks=3, max_depth=1)

    child = state.delegate(
        parent_id="a",
        role="reviewer",
        instruction="review result",
        capability_profile="review",
    )

    assert child.depends_on == ("a",)
    with pytest.raises(TeamCollaborationError, match="duplicate"):
        state.delegate(
            parent_id="a",
            role="reviewer",
            instruction="review result",
            capability_profile="review",
        )
    with pytest.raises(TeamCollaborationError, match="depth limit"):
        state.delegate(
            parent_id=child.task_id,
            role="researcher",
            instruction="go deeper",
            capability_profile="research",
        )


def test_snapshot_roundtrip_preserves_results_mailboxes_and_delegation() -> None:
    state = _state(_task("a"), _task("b"))
    child = state.delegate(
        parent_id="a",
        role="reviewer",
        instruction="review result",
        capability_profile="review",
    )
    state.send_message(sender_id="a", recipient_id="b", content="evidence")
    state.record_result(TeamTaskResult("a", "researcher", TeamTaskStatus.COMPLETED))
    state.mark_running(child.task_id)

    restored = TeamRunState.from_snapshot(state.to_snapshot())

    assert restored.result_for("a").status is TeamTaskStatus.COMPLETED
    assert restored.drain_messages("b", limit=1)[0].content == "evidence"
    statuses = {task["id"]: task["status"] for task in restored.status_payload()["tasks"]}
    assert statuses[child.task_id] == "pending"
    with pytest.raises(TeamCollaborationError, match="duplicate"):
        restored.delegate(
            parent_id="a",
            role="reviewer",
            instruction="review result",
            capability_profile="review",
        )


@pytest.mark.asyncio
async def test_collaboration_tools_share_run_state() -> None:
    state = _state(_task("a"), _task("b"))

    sent = json.loads(await TeamSendTool(state, "a").execute("b", "evidence"))
    received = json.loads(await TeamReceiveTool(state, "b").execute())
    delegated = json.loads(await TeamDelegateTool(state, "a").execute(
        role="reviewer",
        instruction="review evidence",
        capability_profile="review",
    ))
    status = json.loads(await TeamTaskStatusTool(state, "a").execute())

    assert sent["ok"] is True
    assert received["messages"][0]["content"] == "evidence"
    assert delegated["ok"] is True
    assert len(status["tasks"]) == 3
