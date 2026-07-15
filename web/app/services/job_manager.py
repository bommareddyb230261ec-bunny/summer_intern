import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


VALID_STATUSES = {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}


class JobManager:
    """SQLite-backed job state shared by routes, background tasks, and workers."""

    def __init__(self, db_path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "web" / "app" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or data_dir / "jobs.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                stage TEXT NOT NULL,
                message TEXT NOT NULL,
                result TEXT,
                error TEXT,
                query_path TEXT,
                video_paths TEXT,
                video_uploaded INTEGER NOT NULL DEFAULT 0,
                results_ready INTEGER NOT NULL DEFAULT 0,
                results_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                failed_at TEXT
            )
            """
        )
        self._connection.commit()

    def create_job(
        self,
        *,
        job_id: str | None = None,
        stage: str = "QUEUED",
        message: str = "Job queued.",
    ) -> dict[str, Any]:
        job_id = job_id or str(uuid.uuid4())
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, progress, stage, message, result, error,
                    query_path, video_paths, video_uploaded, results_ready,
                    results_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "QUEUED",
                    0,
                    stage,
                    message,
                    self._to_json(None),
                    None,
                    None,
                    self._to_json([]),
                    0,
                    0,
                    None,
                    now,
                    now,
                ),
            )
            self._connection.commit()
        logger.info("[JOB CREATED] job_id=%s stage=%s", job_id, stage)
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        if not updates:
            return self.get_job(job_id)

        if "status" in updates:
            updates["status"] = str(updates["status"]).upper()
            if updates["status"] not in VALID_STATUSES:
                raise ValueError(f"Invalid job status: {updates['status']}")

        timestamp = self._now()
        updates["updated_at"] = timestamp

        status_value = updates.get("status")
        if status_value == "RUNNING" and "started_at" not in updates:
            updates["started_at"] = timestamp
        if status_value == "COMPLETED":
            updates.setdefault("progress", 100)
            updates["completed_at"] = timestamp
        if status_value == "FAILED":
            updates["failed_at"] = timestamp

        serializable = dict(updates)
        for key in ("result", "video_paths"):
            if key in serializable:
                serializable[key] = self._to_json(serializable[key])

        columns = ", ".join(f"{key} = ?" for key in serializable)
        values = list(serializable.values())
        values.append(job_id)

        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE jobs SET {columns} WHERE job_id = ?",
                values,
            )
            self._connection.commit()

        if cursor.rowcount == 0:
            return None

        logger.info(
            "[JOB UPDATED] job_id=%s status=%s progress=%s stage=%s",
            job_id,
            updates.get("status"),
            updates.get("progress"),
            updates.get("stage"),
        )
        return self.get_job(job_id)

    def status_payload(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "message": job["message"],
            "result": job["result"],
        }

    def mark_failed(self, job_id: str, message: str, error: str | None = None) -> None:
        self.update_job(
            job_id,
            status="FAILED",
            stage="FAILED",
            message=message,
            error=error or message,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _from_json(value: str | None, default: Any) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        job["result"] = self._from_json(job.get("result"), None)
        job["video_paths"] = self._from_json(job.get("video_paths"), [])
        job["video_uploaded"] = bool(job.get("video_uploaded"))
        job["results_ready"] = bool(job.get("results_ready"))
        return job


job_manager = JobManager()
