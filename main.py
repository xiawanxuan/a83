import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

from config_manager import ConfigManager
from db_adapters import MongoDBAdapter, PostgreSQLAdapter
from diff_engine import DiffEngine, TableCollectionDiff
from metadata_collector import MetadataCollector, MongoCollectionMetadata, TableMetadata
from rollback_manager import ChangeLogger, RollbackManager, SyncPhase, SyncStatus
from sync_executor import SyncExecutor, SyncPlan, SyncScriptGenerator


class SmartManufacturingSync:
    def __init__(
        self,
        config_path: str = "config/sync_rules.json",
        db_ini_path: str = "config/database.ini",
    ):
        self.config_path = config_path
        self.db_ini_path = db_ini_path

        self._validate_config_files()

        self.config_manager = ConfigManager(config_path)
        self._validate_sync_config()

        self.pg_adapter = PostgreSQLAdapter(db_ini_path)
        self.mongo_adapter = MongoDBAdapter(db_ini_path)

        log_config = self.config_manager.sync_config.logging_config
        log_level = log_config.get("level", "INFO")
        log_dir = log_config.get("log_directory", "logs")

        self.logger = ChangeLogger(log_level=log_level, log_dir=log_dir, config=log_config)
        self.metadata_collector = MetadataCollector(
            self.pg_adapter, self.mongo_adapter, self.config_manager
        )
        self.diff_engine = DiffEngine(self.config_manager)
        self.script_generator = SyncScriptGenerator(self.config_manager)
        self.sync_executor = SyncExecutor(
            self.pg_adapter, self.mongo_adapter, self.logger.get_log_callback()
        )
        self.rollback_manager = RollbackManager(self.logger)

    def _validate_config_files(self) -> None:
        import os
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Sync config file not found: {self.config_path}")
        if not os.path.exists(self.db_ini_path):
            raise FileNotFoundError(f"Database config file not found: {self.db_ini_path}")

    def _validate_sync_config(self) -> None:
        errors = self.config_manager.validate_config()
        if errors:
            error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

    def test_connections(self) -> bool:
        self.logger.info("Testing database connections...")
        pg_ok = self.pg_adapter.test_connection()
        mongo_ok = self.mongo_adapter.test_connection()
        self.logger.info(f"  PostgreSQL: {'OK' if pg_ok else 'FAILED'}")
        self.logger.info(f"  MongoDB:    {'OK' if mongo_ok else 'FAILED'}")
        return pg_ok and mongo_ok

    def run_sync(
        self,
        run_mode: Optional[str] = None,
        export_scripts: bool = True,
    ) -> int:
        mode = (run_mode or self.config_manager.sync_config.run_mode).lower()
        run_id = self.logger.start_run(run_mode=mode)

        try:
            if not self.test_connections():
                raise ConnectionError("Database connection test failed")

            self.logger.start_phase(SyncPhase.METADATA_COLLECT)
            pg_tables: List[TableMetadata] = self.metadata_collector.collect_pg_tables()
            mongo_colls: List[MongoCollectionMetadata] = self.metadata_collector.collect_mongo_collections()
            snapshot = self.metadata_collector.collect_all()
            self.logger.record_metadata_snapshot(snapshot)
            self.logger.complete_phase(details={
                "pg_tables_collected": len(pg_tables),
                "mongo_collections_collected": len(mongo_colls),
            })

            self.logger.start_phase(SyncPhase.DIFF_COMPARE)
            diffs: List[TableCollectionDiff] = self.diff_engine.compare_all(pg_tables, mongo_colls)
            self.logger.record_diffs(diffs)

            if not diffs:
                self.logger.info("No differences detected. Nothing to sync.")
                self.logger.complete_phase(details={"diffs_found": 0})
                self.logger.complete_run(SyncStatus.SUCCESS)
                return 0

            self.logger.complete_phase(details={"diffs_found": len(diffs)})

            self.logger.start_phase(SyncPhase.SCRIPT_GENERATE)
            plan: SyncPlan = self.script_generator.generate_from_diffs(diffs)
            self.logger.record_sync_plan(plan)
            self.logger.complete_phase(details={"operations_generated": len(plan.operations)})

            if export_scripts:
                sql_file, mongo_file = self.sync_executor.export_sql_scripts(plan)
                self.logger.info(f"Generated SQL script: {sql_file}")
                self.logger.info(f"Generated MongoShell script: {mongo_file}")

            self.logger.start_phase(SyncPhase.SYNC_EXECUTE)
            if mode == "preview":
                self.logger.info("RUN MODE: PREVIEW - No changes will be applied")
                self.logger.info("")
                self.logger.info("=== SYNC OPERATIONS PREVIEW ===")
                for i, op in enumerate(plan.operations, 1):
                    self.logger.info(f"")
                    self.logger.info(f"[{i}] Op ID:    {op.id}")
                    self.logger.info(f"    Target DB: {op.target_db}")
                    self.logger.info(f"    Operation: {op.operation_type}")
                    self.logger.info(f"    Object:    {op.object_name}")
                    if op.sql_script:
                        self.logger.info(f"    SQL:")
                        for line in op.sql_script.splitlines():
                            self.logger.info(f"      {line}")
                    if op.mongo_script:
                        self.logger.info(f"    MongoShell:")
                        for line in op.mongo_script.splitlines():
                            self.logger.info(f"      {line}")
                self.logger.info("")
                self.logger.info("=== END OF PREVIEW ===")
                self.logger.info("Use --mode execute to apply changes.")
                self.logger.complete_phase(details={"mode": "preview", "ops_previewed": len(plan.operations)})
                self.logger.complete_run(SyncStatus.SUCCESS)
                return 0

            if mode == "execute":
                self.logger.info("RUN MODE: EXECUTE - Changes will be applied to databases")
                success, history, error = self.sync_executor.execute_plan(plan)
                self.logger.record_execution_history(history)

                if success:
                    self.logger.complete_phase(details={
                        "ops_executed": len(history),
                        "status": "all_success",
                    })
                    self.logger.complete_run(SyncStatus.SUCCESS)
                    return 0
                else:
                    self.logger.fail_phase(Exception(error or "Unknown execution error"))
                    if self.config_manager.sync_config.auto_rollback_on_failure:
                        self.logger.start_phase(SyncPhase.ROLLBACK)
                        self.logger.warning("Auto-rollback triggered...")
                        rb_success, rb_history, rb_error = self.sync_executor.rollback_failed()
                        self.logger.record_rollback_history(rb_history)
                        if rb_success:
                            self.logger.complete_phase(details={"ops_rolled_back": len(rb_history)})
                            self.logger.complete_run(SyncStatus.ROLLED_BACK, error=error)
                        else:
                            self.logger.fail_phase(Exception(rb_error or "Rollback error"))
                            self.logger.complete_run(SyncStatus.FAILED, error=f"Execution failed: {error}; Rollback failed: {rb_error}")
                    else:
                        self.logger.warning("Auto-rollback disabled. Manual rollback required.")
                        self.logger.complete_run(SyncStatus.FAILED, error=error)
                    return 1

            raise ValueError(f"Invalid run_mode: {mode}")

        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            if self.logger.current_run:
                for phase in self.logger.current_run.phases:
                    if phase.status.value == SyncStatus.RUNNING.value:
                        phase.status = SyncStatus.FAILED
                        phase.error = str(e)
                        phase.completed_at = datetime.now().isoformat()
            self.logger.complete_run(SyncStatus.FAILED, error=str(e))
            return 1

        finally:
            self.mongo_adapter.close()

    def rollback_run(self, run_id: str) -> int:
        self.logger.info(f"Generating rollback scripts for run: {run_id}")
        result = self.rollback_manager.generate_rollback_script(run_id)
        if result:
            self.logger.info("Rollback scripts generated successfully.")
            self.logger.info("Review and execute scripts manually if needed.")
            return 0
        else:
            self.logger.error(f"Run ID not found: {run_id}")
            return 1

    def list_runs(self, limit: int = 20) -> int:
        runs = self.rollback_manager.list_runs(limit=limit)
        if not runs:
            self.logger.info("No run history found.")
            return 0

        self.logger.info(f"{'Run ID':<30} {'Started':<20} {'Status':<14} {'Mode':<10} {'Ops':<6}")
        self.logger.info("-" * 82)
        for r in runs:
            self.logger.info(
                f"{r.get('run_id', ''):<30} "
                f"{(r.get('started_at') or '')[:19]:<20} "
                f"{r.get('status', ''):<14} "
                f"{r.get('run_mode', ''):<10} "
                f"{r.get('num_ops', 0):<6}"
            )
        return 0

    def show_diff(self, limit: Optional[int] = None) -> int:
        if not self.test_connections():
            self.logger.error("Database connection failed")
            return 1

        pg_tables = self.metadata_collector.collect_pg_tables()
        mongo_colls = self.metadata_collector.collect_mongo_collections()
        diffs = self.diff_engine.compare_all(pg_tables, mongo_colls)

        if not diffs:
            self.logger.info("No differences found between PostgreSQL and MongoDB.")
            return 0

        total = len(diffs)
        shown = 0
        for d in diffs:
            if limit and shown >= limit:
                break
            self.logger.info("")
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"Mapping: PG table '{d.pg_table}' <-> Mongo collection '{d.mongo_collection}'")
            self.logger.info(f"Direction: {d.direction}")
            self.logger.info(f"{'=' * 60}")

            if d.field_diffs:
                self.logger.info(f"  Field changes: {len(d.field_diffs)}")
                for fd in d.field_diffs:
                    action_symbol = {
                        "add_field": "[+]",
                        "drop_field": "[-]",
                        "modify_field": "[~]",
                    }.get(fd.action, "[?]")
                    target_db = "PG" if fd.target_db_type == "postgresql" else "MG"
                    src = fd.source_type or "N/A"
                    tgt = fd.target_type or "N/A"
                    self.logger.info(f"    {action_symbol} ({target_db}) {fd.field_name}: {src} -> {tgt}")

            if d.index_diffs:
                self.logger.info(f"  Index changes: {len(d.index_diffs)}")
                for idx in d.index_diffs:
                    action_symbol = {
                        "add_index": "[+]",
                        "drop_index": "[-]",
                    }.get(idx.action, "[?]")
                    target_db = "PG" if idx.target_db_type == "postgresql" else "MG"
                    cols = ", ".join(idx.columns)
                    uniq = " UNIQUE" if idx.is_unique else ""
                    self.logger.info(f"    {action_symbol} ({target_db}) {idx.index_name}: ({cols}){uniq}")

            if d.sharding_diffs:
                self.logger.info(f"  Sharding/partition changes: {len(d.sharding_diffs)}")
                for sd in d.sharding_diffs:
                    target_db = "PG" if sd.target_db_type == "postgresql" else "MG"
                    self.logger.info(f"    [!] ({target_db}) {sd.action}: {sd.extra}")

            shown += 1

        if limit and limit < total:
            self.logger.info(f"... ({total - shown} more mapping pairs not shown)")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart_manufacturing_sync",
        description="智能制造混合数据库数据中台 - PostgreSQL 与 MongoDB 双向同步工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sync_parser = subparsers.add_parser("sync", help="Run synchronization process")
    sync_parser.add_argument(
        "--mode",
        choices=["preview", "execute"],
        default=None,
        help="Run mode: preview (default) or execute",
    )
    sync_parser.add_argument(
        "--config",
        default="config/sync_rules.json",
        help="Path to sync rules JSON config",
    )
    sync_parser.add_argument(
        "--db-ini",
        default="config/database.ini",
        help="Path to database INI config",
    )
    sync_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export SQL/MongoShell scripts to files",
    )

    diff_parser = subparsers.add_parser("diff", help="Show differences without generating scripts")
    diff_parser.add_argument(
        "--config", default="config/sync_rules.json", help="Path to sync rules JSON config"
    )
    diff_parser.add_argument(
        "--db-ini", default="config/database.ini", help="Path to database INI config"
    )
    diff_parser.add_argument("--limit", type=int, default=None, help="Limit number of mappings shown")

    list_parser = subparsers.add_parser("list-runs", help="List recent sync runs")
    list_parser.add_argument(
        "--config", default="config/sync_rules.json", help="Path to sync rules JSON config"
    )
    list_parser.add_argument(
        "--db-ini", default="config/database.ini", help="Path to database INI config"
    )
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show")

    rb_parser = subparsers.add_parser("rollback", help="Generate rollback scripts for a run")
    rb_parser.add_argument("run_id", help="Run ID to generate rollback scripts for")
    rb_parser.add_argument(
        "--config", default="config/sync_rules.json", help="Path to sync rules JSON config"
    )
    rb_parser.add_argument(
        "--db-ini", default="config/database.ini", help="Path to database INI config"
    )

    test_parser = subparsers.add_parser("test-connection", help="Test database connections")
    test_parser.add_argument(
        "--config", default="config/sync_rules.json", help="Path to sync rules JSON config"
    )
    test_parser.add_argument(
        "--db-ini", default="config/database.ini", help="Path to database INI config"
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    config_path = getattr(args, "config", "config/sync_rules.json")
    db_ini_path = getattr(args, "db_ini", "config/database.ini")

    try:
        sync_app = SmartManufacturingSync(config_path=config_path, db_ini_path=db_ini_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Failed to initialize sync application: {e}")
        return 1

    try:
        if args.command == "sync":
            return sync_app.run_sync(
                run_mode=args.mode,
                export_scripts=not args.no_export,
            )
        elif args.command == "diff":
            return sync_app.show_diff(limit=args.limit)
        elif args.command == "list-runs":
            return sync_app.list_runs(limit=args.limit)
        elif args.command == "rollback":
            return sync_app.rollback_run(args.run_id)
        elif args.command == "test-connection":
            return 0 if sync_app.test_connections() else 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
