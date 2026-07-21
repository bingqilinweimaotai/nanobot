from __future__ import annotations

from nanobot.agent.tools.multiagent.models import TeamTaskSpec
from nanobot.agent.tools.multiagent.state import TeamRunState
from nanobot.agent.tools.multiagent.store import TeamRunStore


def _snapshot() -> dict:
    return TeamRunState(
        run_id="run-1",
        tasks=[TeamTaskSpec("a", "researcher", "research")],
        max_tasks=8,
        max_delegation_depth=2,
        capability_profiles={"research"},
        max_message_chars=100,
    ).to_snapshot()


def test_store_roundtrip_and_stale_run_pause(tmp_path) -> None:
    path = tmp_path / "runs.sqlite3"
    store = TeamRunStore(path)
    store.create(
        run_id="run-1",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=2,
        snapshot=_snapshot(),
    )

    assert store.get("run-1").status == "queued"

    reopened = TeamRunStore(path)
    stored = reopened.get("run-1")
    assert stored.status == "paused"
    assert stored.owner_session_key == "cli:one"
    assert stored.snapshot["runId"] == "run-1"


def test_store_prunes_old_terminal_runs_but_keeps_active_runs(tmp_path) -> None:
    store = TeamRunStore(tmp_path / "runs.sqlite3", max_stored_runs=2)
    for run_id in ("old", "new"):
        snapshot = _snapshot()
        snapshot["runId"] = run_id
        store.create(
            run_id=run_id,
            owner_session_key="cli:one",
            goal="goal",
            workspace_path=str(tmp_path),
            access_mode="restricted",
            max_concurrency=1,
            snapshot=snapshot,
        )
        if run_id == "old":
            store.finish(run_id, status="completed", snapshot=snapshot)

    third = _snapshot()
    third["runId"] = "third"
    store.create(
        run_id="third",
        owner_session_key="cli:one",
        goal="goal",
        workspace_path=str(tmp_path),
        access_mode="restricted",
        max_concurrency=1,
        snapshot=third,
    )

    assert store.get("old") is None
    assert store.get("new") is not None
    assert store.get("third") is not None
