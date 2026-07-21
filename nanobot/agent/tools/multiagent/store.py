"""SQLite persistence for resumable multi-agent team runs."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.multiagent.models import TeamRunResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TeamRunLeaseError(RuntimeError):
    """Raised when a process tries to mutate a run owned by another live process."""


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
    runner_id: str | None = None
    lease_expires_at: float | None = None

    def to_payload(self) -> dict[str, Any]:
        depths = self.snapshot.get("depths", {})
        statuses = self.snapshot.get("statuses", {})

        def visible_status(task_id: str | None) -> str:
            status = statuses.get(task_id, "pending")
            if self.status == "paused" and status in {"running", "finishing"}:
                return "pending"
            if self.status == "cancelled" and status in {"pending", "running", "finishing"}:
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
    """Small durable store with explicit connection and process-lease lifecycles."""

    def __init__(
        self,
        path: Path,
        *,
        max_stored_runs: int = 200,
        lease_seconds: float = 30.0,
    ) -> None:
        self.path = path
        self.max_stored_runs = max_stored_runs
        self.lease_seconds = lease_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
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
                    updated_at TEXT NOT NULL,
                    runner_id TEXT,
                    lease_expires_at REAL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(team_runs)").fetchall()
            }
            for name, column_type in (
                ("runner_id", "TEXT"),
                ("lease_expires_at", "REAL"),
            ):
                if name not in columns:
                    try:
                        connection.execute(f"ALTER TABLE team_runs ADD COLUMN {name} {column_type}")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column name" not in str(exc).lower():
                            raise
            connection.execute(
                """
                UPDATE team_runs
                SET status = 'paused', runner_id = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE status IN ('queued', 'running')
                  AND (runner_id IS NULL OR lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (_now(), time.time()),
            )

    def _prune(self, connection: sqlite3.Connection) -> None:
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
        runner_id: str | None = None,
    ) -> None:
        now = _now()
        lease_expires_at = time.time() + self.lease_seconds if runner_id is not None else None
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO team_runs (
                    run_id, owner_session_key, status, goal, workspace_path,
                    access_mode, max_concurrency, snapshot_json, result_json,
                    error, created_at, updated_at, runner_id, lease_expires_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
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
                    runner_id,
                    lease_expires_at,
                ),
            )
            self._prune(connection)

    def claim(self, run_id: str, runner_id: str) -> None:
        now_epoch = time.time()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE team_runs
                SET status = 'running', runner_id = ?, lease_expires_at = ?, updated_at = ?
                WHERE run_id = ?
                  AND status IN ('queued', 'paused', 'running')
                  AND (
                      runner_id IS NULL OR runner_id = ?
                      OR lease_expires_at IS NULL OR lease_expires_at <= ?
                  )
                """,
                (
                    runner_id,
                    now_epoch + self.lease_seconds,
                    _now(),
                    run_id,
                    runner_id,
                    now_epoch,
                ),
            )
            if cursor.rowcount != 1:
                raise TeamRunLeaseError(f"team run {run_id} is active in another process")

    def heartbeat(self, run_id: str, runner_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE team_runs
                SET lease_expires_at = ?, updated_at = ?
                WHERE run_id = ? AND runner_id = ? AND status = 'running'
                """,
                (time.time() + self.lease_seconds, _now(), run_id, runner_id),
            )
            if cursor.rowcount != 1:
                raise TeamRunLeaseError(f"lost lease for team run {run_id}")

    def checkpoint(
        self,
        run_id: str,
        *,
        snapshot: dict[str, Any],
        status: str | None = None,
        error: str | None = None,
        runner_id: str | None = None,
    ) -> None:
        assignments = ["snapshot_json = ?", "updated_at = ?"]
        values: list[Any] = [json.dumps(snapshot, ensure_ascii=False), _now()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        where = "run_id = ?"
        values.append(run_id)
        if runner_id is not None:
            assignments.append("lease_expires_at = ?")
            values.insert(-1, time.time() + self.lease_seconds)
            where += " AND runner_id = ?"
            values.append(runner_id)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE team_runs SET {', '.join(assignments)} WHERE {where}",  # noqa: S608
                values,
            )
            if cursor.rowcount != 1:
                if runner_id is not None:
                    raise TeamRunLeaseError(f"lost lease for team run {run_id}")
                raise KeyError(run_id)

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        snapshot: dict[str, Any],
        result: TeamRunResult | None = None,
        error: str | None = None,
        runner_id: str | None = None,
    ) -> None:
        result_json = (
            json.dumps(result.to_payload(), ensure_ascii=False) if result is not None else None
        )
        where = "run_id = ?"
        values: list[Any] = [
            status,
            json.dumps(snapshot, ensure_ascii=False),
            result_json,
            error,
            _now(),
            run_id,
        ]
        if runner_id is not None:
            where += " AND runner_id = ?"
            values.append(runner_id)
        with self._connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE team_runs
                SET status = ?, snapshot_json = ?, result_json = ?, error = ?, updated_at = ?,
                    runner_id = NULL, lease_expires_at = NULL
                WHERE {where}
                """,  # noqa: S608
                values,
            )
            if cursor.rowcount != 1:
                if runner_id is not None:
                    raise TeamRunLeaseError(f"lost lease for team run {run_id}")
                raise KeyError(run_id)

    def get(self, run_id: str) -> StoredTeamRun | None:
        with self._connection() as connection:
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
            runner_id=row["runner_id"],
            lease_expires_at=(
                float(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
        )
