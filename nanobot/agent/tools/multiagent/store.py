"""SQLite persistence for resumable multi-agent team runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.multiagent.models import TeamRunResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StoredTeamRun:
    run_id: str
    owner_session_key: str
    status: str
    goal: str
    workspace_path: str
    access_mode: str
    max_concurrency: int
    snapshot: dict[str, Any]
    result: TeamRunResult | None
    error: str | None
    created_at: str
    updated_at: str

    def to_payload(self) -> dict[str, Any]:
        depths = self.snapshot.get("depths", {})
        statuses = self.snapshot.get("statuses", {})

        def visible_status(task_id: str | None) -> str:
            status = statuses.get(task_id, "pending")
            if self.status == "paused" and status == "running":
                return "pending"
            if self.status == "cancelled" and status in {"pending", "running"}:
                return "cancelled"
            return status

        payload: dict[str, Any] = {
            "runId": self.run_id,
            "status": self.status,
            "goal": self.goal,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "tasks": [
                {
                    "id": task.get("id"),
                    "role": task.get("role"),
                    "capabilityProfile": task.get("capabilityProfile"),
                    "status": visible_status(task.get("id")),
                    "depth": depths.get(task.get("id"), 0),
                    "dependsOn": task.get("dependsOn", []),
                }
                for task in self.snapshot.get("tasks", [])
            ],
        }
        if self.result is not None:
            payload["result"] = self.result.to_payload()
        if self.error:
            payload["error"] = self.error
        return payload


class TeamRunStore:
    """Small durable store; every mutation commits before returning."""

    def __init__(self, path: Path, *, max_stored_runs: int = 200) -> None:
        self.path = path
        self.max_stored_runs = max_stored_runs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_runs (
                    run_id TEXT PRIMARY KEY,
                    owner_session_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    access_mode TEXT NOT NULL,
                    max_concurrency INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = _now()
            connection.execute(
                """
                UPDATE team_runs
                SET status = 'paused', updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )

    def create(
        self,
        *,
        run_id: str,
        owner_session_key: str,
        goal: str,
        workspace_path: str,
        access_mode: str,
        max_concurrency: int,
        snapshot: dict[str, Any],
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO team_runs (
                    run_id, owner_session_key, status, goal, workspace_path,
                    access_mode, max_concurrency, snapshot_json, result_json,
                    error, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    run_id,
                    owner_session_key,
                    goal,
                    workspace_path,
                    access_mode,
                    max_concurrency,
                    json.dumps(snapshot, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            total = int(connection.execute("SELECT COUNT(*) FROM team_runs").fetchone()[0])
            excess = total - self.max_stored_runs
            if excess > 0:
                connection.execute(
                    """
                    DELETE FROM team_runs
                    WHERE run_id IN (
                        SELECT run_id FROM team_runs
                        WHERE status IN ('cancelled', 'completed', 'failed', 'partial')
                        ORDER BY updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def checkpoint(
        self,
        run_id: str,
        *,
        snapshot: dict[str, Any],
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        assignments = ["snapshot_json = ?", "updated_at = ?"]
        values: list[Any] = [json.dumps(snapshot, ensure_ascii=False), _now()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        values.append(run_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE team_runs SET {', '.join(assignments)} WHERE run_id = ?",  # noqa: S608
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        snapshot: dict[str, Any],
        result: TeamRunResult | None = None,
        error: str | None = None,
    ) -> None:
        result_json = (
            json.dumps(result.to_payload(), ensure_ascii=False) if result is not None else None
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE team_runs
                SET status = ?, snapshot_json = ?, result_json = ?, error = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    json.dumps(snapshot, ensure_ascii=False),
                    result_json,
                    error,
                    _now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    def get(self, run_id: str) -> StoredTeamRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM team_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        raw_result = json.loads(row["result_json"]) if row["result_json"] else None
        return StoredTeamRun(
            run_id=row["run_id"],
            owner_session_key=row["owner_session_key"],
            status=row["status"],
            goal=row["goal"],
            workspace_path=row["workspace_path"],
            access_mode=row["access_mode"],
            max_concurrency=int(row["max_concurrency"]),
            snapshot=json.loads(row["snapshot_json"]),
            result=TeamRunResult.from_payload(raw_result) if raw_result is not None else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
