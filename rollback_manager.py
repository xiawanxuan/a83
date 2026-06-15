import json
import logging
import logging.handlers
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from sync_executor import SyncPlan


class SyncPhase(str, Enum):
    INIT = "INIT"
    METADATA_COLLECT = "METADATA_COLLECT"
    DIFF_COMPARE = "DIFF_COMPARE"
    SCRIPT_GENERATE = "SCRIPT_GENERATE"
    SYNC_EXECUTE = "SYNC_EXECUTE"
    ROLLBACK = "ROLLBACK"
    COMPLETE = "COMPLETE"


class SyncStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class PhaseLogEntry:
    phase: SyncPhase
    status: SyncStatus
    started_at: str
    completed_at: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class SyncRunLog:
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    overall_status: SyncStatus = SyncStatus.PENDING
    run_mode: str = "preview"
    phases: List[PhaseLogEntry] = field(default_factory=list)
    metadata_snapshot: Dict[str, Any] = field(default_factory=dict)
    diffs: List[Dict[str, Any]] = field(default_factory=list)
    sync_plan: Optional[Dict[str, Any]] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "overall_status": self.overall_status.value,
            "run_mode": self.run_mode,
            "phases": [p.to_dict() for p in self.phases],
            "metadata_snapshot": self.metadata_snapshot,
            "diffs": self.diffs,
            "sync_plan": self.sync_plan,
            "execution_history": self.execution_history,
            "rollback_history": self.rollback_history,
            "error_message": self.error_message,
        }


class ChangeLogger:
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs", config: Optional[Dict[str, Any]] = None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._setup_logger(log_level, config)
        self.current_run: Optional[SyncRunLog] = None
        self.current_phase: Optional[PhaseLogEntry] = None

    def _setup_logger(self, log_level_str: str, config: Optional[Dict[str, Any]]) -> logging.Logger:
        logger = logging.getLogger("smart_manufacturing_sync")
        logger.setLevel(getattr(logging, log_level_str.upper(), logging.INFO))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_filename_pattern = (
            (config or {}).get("log_filename_pattern", "sync_%Y%m%d_%H%M%S.log")
            if config else "sync_%Y%m%d_%H%M%S.log"
        )
        max_bytes = ((config or {}).get("max_log_size_mb", 100) or 100) * 1024 * 1024
        backup_count = (config or {}).get("backup_count", 30) or 30

        current_log_name = datetime.now().strftime(log_filename_pattern)
        log_path = os.path.join(self.log_dir, current_log_name)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        ms = datetime.now().microsecond
        return f"run_{ts}_{ms:06d}"

    def start_run(self, run_mode: str = "preview") -> str:
        run_id = self._generate_run_id()
        self.current_run = SyncRunLog(
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            overall_status=SyncStatus.RUNNING,
            run_mode=run_mode,
        )
        self.info(f"{'=' * 60}")
        self.info(f"Starting sync run ID: {run_id} | Mode: {run_mode}")
        self.info(f"{'=' * 60}")
        return run_id

    def start_phase(self, phase: SyncPhase, details: Optional[Dict[str, Any]] = None) -> None:
        self.current_phase = PhaseLogEntry(
            phase=phase,
            status=SyncStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            details=details or {},
        )
        if self.current_run:
            self.current_run.phases.append(self.current_phase)
        self.info(f"--- Phase START: {phase.value} ---")
        if details:
            for k, v in details.items():
                self.debug(f"  {k}: {v}")

    def complete_phase(self, details: Optional[Dict[str, Any]] = None) -> None:
        if self.current_phase:
            self.current_phase.status = SyncStatus.SUCCESS
            self.current_phase.completed_at = datetime.now().isoformat()
            if details:
                self.current_phase.details.update(details)
            self.info(f"--- Phase COMPLETE: {self.current_phase.phase.value} ---")
            if details:
                for k, v in details.items():
                    self.info(f"  {k}: {v}")

    def fail_phase(self, error: Exception, details: Optional[Dict[str, Any]] = None) -> None:
        if self.current_phase:
            self.current_phase.status = SyncStatus.FAILED
            self.current_phase.completed_at = datetime.now().isoformat()
            self.current_phase.error = str(error)
            if details:
                self.current_phase.details.update(details)
            self.error(f"--- Phase FAILED: {self.current_phase.phase.value} ---")
            self.error(f"Error: {error}")
            self.error(f"Stack trace:\n{traceback.format_exc()}")

    def record_metadata_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if self.current_run:
            self.current_run.metadata_snapshot = snapshot
        pg_count = len(snapshot.get("postgresql_tables", []))
        mongo_count = len(snapshot.get("mongodb_collections", []))
        self.info(f"Metadata snapshot: {pg_count} PG tables, {mongo_count} MongoDB collections")

    def record_diffs(self, diffs: List[Any]) -> None:
        if self.current_run:
            self.current_run.diffs = [d.to_dict() for d in diffs]
        total_changes = sum(
            len(d.field_diffs) + len(d.index_diffs) + len(d.sharding_diffs)
            for d in diffs
        )
        self.info(f"Diffs detected: {len(diffs)} mapping pairs, {total_changes} total changes")

    def record_sync_plan(self, plan: SyncPlan) -> None:
        if self.current_run:
            self.current_run.sync_plan = plan.to_dict()
        self.info(f"Sync plan generated: {len(plan.operations)} operations")

    def record_execution_history(self, history: List[Dict[str, Any]]) -> None:
        if self.current_run:
            self.current_run.execution_history.extend(history)
        success = sum(1 for h in history if h.get("status") == "SUCCESS")
        failed = sum(1 for h in history if h.get("status") == "FAILED")
        self.info(f"Execution history: {success} success, {failed} failed")

    def record_rollback_history(self, history: List[Dict[str, Any]]) -> None:
        if self.current_run:
            self.current_run.rollback_history.extend(history)
        success = sum(1 for h in history if h.get("status") == "SUCCESS")
        failed = sum(1 for h in history if h.get("status") == "FAILED")
        self.info(f"Rollback history: {success} success, {failed} failed")

    def complete_run(self, status: SyncStatus, error: Optional[str] = None) -> None:
        if self.current_run:
            self.current_run.overall_status = status
            self.current_run.completed_at = datetime.now().isoformat()
            self.current_run.error_message = error
            self._persist_run_log()

        self.info(f"{'=' * 60}")
        if status == SyncStatus.SUCCESS:
            self.info(f"Sync run COMPLETED SUCCESSFULLY")
        elif status == SyncStatus.ROLLED_BACK:
            self.warning(f"Sync run FAILED but ROLLED BACK successfully")
        else:
            self.error(f"Sync run FAILED: {error or 'Unknown error'}")
        if self.current_run:
            self.info(f"Run ID: {self.current_run.run_id}")
            self.info(f"Started:   {self.current_run.started_at}")
            self.info(f"Completed: {self.current_run.completed_at}")
        self.info(f"{'=' * 60}")

    def _persist_run_log(self) -> None:
        if not self.current_run:
            return
        run_log_path = os.path.join(self.log_dir, f"run_{self.current_run.run_id}.json")
        try:
            with open(run_log_path, "w", encoding="utf-8") as f:
                json.dump(self.current_run.to_dict(), f, ensure_ascii=False, indent=2)
            self.debug(f"Run log persisted to: {run_log_path}")
        except Exception as e:
            self.error(f"Failed to persist run log: {e}")

    def get_log_callback(self) -> Callable[[str], None]:
        return self.info

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)


class RollbackManager:
    def __init__(self, logger: ChangeLogger):
        self.logger = logger
        self.log_dir = logger.log_dir

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        runs = []
        if not os.path.exists(self.log_dir):
            return runs
        for f in sorted(os.listdir(self.log_dir), reverse=True):
            if f.startswith("run_") and f.endswith(".json"):
                try:
                    path = os.path.join(self.log_dir, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    runs.append({
                        "run_id": data.get("run_id"),
                        "started_at": data.get("started_at"),
                        "completed_at": data.get("completed_at"),
                        "status": data.get("overall_status"),
                        "run_mode": data.get("run_mode"),
                        "num_ops": len((data.get("sync_plan") or {}).get("operations", [])),
                    })
                    if len(runs) >= limit:
                        break
                except Exception:
                    continue
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.log_dir, f"run_{run_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def generate_rollback_script(self, run_id: str) -> Optional[Dict[str, List[str]]]:
        run = self.get_run(run_id)
        if not run:
            return None

        plan = run.get("sync_plan") or {}
        operations = plan.get("operations", [])

        sql_rollback: List[str] = []
        mongo_rollback: List[str] = []

        for op in reversed(operations):
            if op.get("rollback_sql"):
                sql_rollback.append(f"-- Op: {op.get('id')} | ROLLBACK {op.get('operation_type')}")
                sql_rollback.append(op["rollback_sql"])
                sql_rollback.append("")
            if op.get("rollback_mongo"):
                mongo_rollback.append(f"// Op: {op.get('id')} | ROLLBACK {op.get('operation_type')}")
                mongo_rollback.append(op["rollback_mongo"])
                mongo_rollback.append("")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sql_file = os.path.join(self.log_dir, f"rollback_{run_id}_sql_{ts}.sql")
        mongo_file = os.path.join(self.log_dir, f"rollback_{run_id}_mongo_{ts}.js")

        sql_header = [
            f"-- Rollback SQL for run: {run_id}",
            f"-- Generated at: {ts}",
            f"-- Original run status: {run.get('overall_status')}",
            "",
        ]
        mongo_header = [
            f"// Rollback MongoShell script for run: {run_id}",
            f"// Generated at: {ts}",
            f"// Original run status: {run.get('overall_status')}",
            "",
        ]

        with open(sql_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sql_header + sql_rollback))
        with open(mongo_file, "w", encoding="utf-8") as f:
            f.write("\n".join(mongo_header + mongo_rollback))

        self.logger.info(f"Rollback scripts generated:")
        self.logger.info(f"  SQL:    {sql_file}")
        self.logger.info(f"  Mongo:  {mongo_file}")

        return {"sql_rollback": [sql_file], "mongo_rollback": [mongo_file]}
