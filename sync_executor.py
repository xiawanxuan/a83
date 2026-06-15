import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from config_manager import ConfigManager
from db_adapters import MongoDBAdapter, PostgreSQLAdapter
from diff_engine import (
    FieldDiff,
    IndexDiff,
    ShardingDiff,
    TableCollectionDiff,
)


@dataclass
class SyncOperation:
    id: str
    order: int
    target_db: str
    operation_type: str
    object_name: str
    sql_script: Optional[str] = None
    mongo_script: Optional[str] = None
    rollback_sql: Optional[str] = None
    rollback_mongo: Optional[str] = None
    diff_ref: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "target_db": self.target_db,
            "operation_type": self.operation_type,
            "object_name": self.object_name,
            "sql_script": self.sql_script,
            "mongo_script": self.mongo_script,
            "rollback_sql": self.rollback_sql,
            "rollback_mongo": self.rollback_mongo,
        }


@dataclass
class SyncPlan:
    operations: List[SyncOperation] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_operation(self, op: SyncOperation) -> None:
        op.order = len(self.operations) + 1
        self.operations.append(op)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "operations": [op.to_dict() for op in self.operations],
        }

    def save_to_file(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class SyncScriptGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self._op_counter = 0

    def _next_id(self) -> str:
        self._op_counter += 1
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"op_{ts}_{self._op_counter:04d}"

    def generate_from_diffs(self, diffs: List[TableCollectionDiff]) -> SyncPlan:
        plan = SyncPlan()

        for diff in diffs:
            self._generate_timeseries_scripts(diff, plan)
            self._generate_field_scripts(diff, plan)
            self._generate_index_scripts(diff, plan)
            self._generate_sharding_scripts(diff, plan)

        return plan

    def _generate_timeseries_scripts(self, diff: TableCollectionDiff, plan: SyncPlan) -> None:
        for sd in diff.sharding_diffs:
            if sd.action == "create_timeseries" and sd.target_db_type == "mongodb":
                coll = diff.mongo_collection
                time_field = sd.extra.get("time_field", "timestamp")
                meta_field = sd.extra.get("meta_field", "metadata")
                granularity = sd.extra.get("granularity", "seconds")
                bucket_span = sd.extra.get("bucket_max_span_seconds", 3600)

                mongo_script = (
                    f"db.createCollection('{coll}', {{\n"
                    f"  timeseries: {{\n"
                    f"    timeField: '{time_field}',\n"
                    f"    metaField: '{meta_field}',\n"
                    f"    granularity: '{granularity}',\n"
                    f"    bucketMaxSpanSeconds: {bucket_span}\n"
                    f"  }}\n"
                    f"}});"
                )
                rollback = f"db.{coll}.drop();"
                plan.add_operation(SyncOperation(
                    id=self._next_id(),
                    order=0,
                    target_db="mongodb",
                    operation_type="CREATE_TIMESERIES_COLLECTION",
                    object_name=coll,
                    mongo_script=mongo_script,
                    rollback_mongo=rollback,
                    diff_ref=sd,
                ))

            if sd.action == "create_partitioned_table" and sd.target_db_type == "postgresql":
                table = diff.pg_table
                time_field = sd.extra.get("time_field", "measurement_time")
                sql = (
                    f"CREATE TABLE IF NOT EXISTS {table} (\n"
                    f"    id SERIAL,\n"
                    f"    {time_field} TIMESTAMP WITH TIME ZONE NOT NULL\n"
                    f") PARTITION BY RANGE ({time_field});"
                )
                rollback = f"DROP TABLE IF EXISTS {table};"
                plan.add_operation(SyncOperation(
                    id=self._next_id(),
                    order=0,
                    target_db="postgresql",
                    operation_type="CREATE_PARTITIONED_TABLE",
                    object_name=table,
                    sql_script=sql,
                    rollback_sql=rollback,
                    diff_ref=sd,
                ))

    def _generate_field_scripts(self, diff: TableCollectionDiff, plan: SyncPlan) -> None:
        for fd in diff.field_diffs:
            if fd.target_db_type == "postgresql":
                self._pg_field_op(diff, fd, plan)
            elif fd.target_db_type == "mongodb":
                self._mongo_field_op(diff, fd, plan)

    def _pg_field_op(self, diff: TableCollectionDiff, fd: FieldDiff, plan: SyncPlan) -> None:
        table = diff.pg_table
        if fd.action == "add_field":
            col_type = fd.target_type or "VARCHAR(255)"
            nullable = "NULL" if fd.extra.get("is_nullable", True) else "NOT NULL"
            sql = f"ALTER TABLE {table} ADD COLUMN {fd.field_name} {col_type} {nullable};"
            rollback = f"ALTER TABLE {table} DROP COLUMN IF EXISTS {fd.field_name};"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="postgresql",
                operation_type="ADD_COLUMN",
                object_name=f"{table}.{fd.field_name}",
                sql_script=sql,
                rollback_sql=rollback,
                diff_ref=fd,
            ))
        elif fd.action == "modify_field":
            col_type = fd.target_type or "VARCHAR(255)"
            using_clause = self._build_pg_using_clause(fd.field_name, fd.extra.get("current_pg_type", ""), col_type)
            sql = f"ALTER TABLE {table} ALTER COLUMN {fd.field_name} TYPE {col_type}{using_clause};"
            orig_type = fd.extra.get("current_pg_type", "VARCHAR(255)")
            rollback = f"ALTER TABLE {table} ALTER COLUMN {fd.field_name} TYPE {orig_type};"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="postgresql",
                operation_type="ALTER_COLUMN_TYPE",
                object_name=f"{table}.{fd.field_name}",
                sql_script=sql,
                rollback_sql=rollback,
                diff_ref=fd,
            ))
        elif fd.action == "drop_field":
            if fd.extra.get("is_primary_key", False):
                return
            sql = f"ALTER TABLE {table} DROP COLUMN IF EXISTS {fd.field_name};"
            col_type = fd.source_type or "VARCHAR(255)"
            rollback = f"ALTER TABLE {table} ADD COLUMN {fd.field_name} {col_type};"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="postgresql",
                operation_type="DROP_COLUMN",
                object_name=f"{table}.{fd.field_name}",
                sql_script=sql,
                rollback_sql=rollback,
                diff_ref=fd,
            ))

    def _build_pg_using_clause(self, col: str, src_type: str, tgt_type: str) -> str:
        src_lower = src_type.lower()
        tgt_lower = tgt_type.lower()
        if "varchar" in tgt_lower or "character varying" in tgt_lower:
            return f" USING {col}::VARCHAR"
        if "integer" in tgt_lower or "bigint" in tgt_lower:
            return f" USING {col}::BIGINT"
        if "double precision" in tgt_lower or "numeric" in tgt_lower:
            return f" USING {col}::NUMERIC"
        if "timestamp" in tgt_lower:
            if "date" in src_lower:
                return f" USING {col}::TIMESTAMP WITH TIME ZONE"
            return f" USING to_timestamp({col}::text, 'YYYY-MM-DD HH24:MI:SS')"
        if "boolean" in tgt_lower:
            return f" USING {col}::BOOLEAN"
        if "jsonb" in tgt_lower:
            return f" USING {col}::JSONB"
        return ""

    def _mongo_field_op(self, diff: TableCollectionDiff, fd: FieldDiff, plan: SyncPlan) -> None:
        coll = diff.mongo_collection
        if fd.action == "add_field":
            bson_type = fd.target_type or "string"
            default_val = self._mongo_default_for_type(bson_type)
            mongo_script = (
                f"db.{coll}.updateMany(\n"
                f"  {{ {fd.field_name}: {{ $exists: false }} }},\n"
                f"  {{ $set: {{ {fd.field_name}: {default_val} }} }},\n"
                f"  {{ upsert: false, multi: true }}\n"
                f");"
            )
            rollback = (
                f"db.{coll}.updateMany(\n"
                f"  {{ }},\n"
                f"  {{ $unset: {{ {fd.field_name}: '' }} }}\n"
                f");"
            )
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="mongodb",
                operation_type="ADD_FIELD",
                object_name=f"{coll}.{fd.field_name}",
                mongo_script=mongo_script,
                rollback_mongo=rollback,
                diff_ref=fd,
            ))
        elif fd.action == "modify_field":
            src_type = fd.source_type or "string"
            tgt_type = fd.target_type or "string"
            conversion = self._mongo_conversion(fd.field_name, src_type, tgt_type)
            mongo_script = (
                f"db.{coll}.updateMany(\n"
                f"  {{ {fd.field_name}: {{ $exists: true }} }},\n"
                f"  [\n"
                f"    {{ $set: {{ {fd.field_name}: {conversion} }} }}\n"
                f"  ]\n"
                f");"
            )
            reverse = self._mongo_conversion(fd.field_name, tgt_type, src_type)
            rollback = (
                f"db.{coll}.updateMany(\n"
                f"  {{ {fd.field_name}: {{ $exists: true }} }},\n"
                f"  [\n"
                f"    {{ $set: {{ {fd.field_name}: {reverse} }} }}\n"
                f"  ]\n"
                f");"
            )
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="mongodb",
                operation_type="CONVERT_FIELD_TYPE",
                object_name=f"{coll}.{fd.field_name}",
                mongo_script=mongo_script,
                rollback_mongo=rollback,
                diff_ref=fd,
            ))
        elif fd.action == "drop_field":
            mongo_script = (
                f"db.{coll}.updateMany(\n"
                f"  {{ }},\n"
                f"  {{ $unset: {{ {fd.field_name}: '' }} }}\n"
                f");"
            )
            rollback = (
                f"db.{coll}.updateMany(\n"
                f"  {{ }},\n"
                f"  {{ $set: {{ {fd.field_name}: null }} }}\n"
                f");"
            )
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="mongodb",
                operation_type="DROP_FIELD",
                object_name=f"{coll}.{fd.field_name}",
                mongo_script=mongo_script,
                rollback_mongo=rollback,
                diff_ref=fd,
            ))

    def _mongo_default_for_type(self, bson_type: str) -> str:
        defaults = {
            "string": "''",
            "int": "NumberInt(0)",
            "long": "NumberLong(0)",
            "double": "0.0",
            "decimal": "NumberDecimal('0')",
            "bool": "false",
            "date": "new Date()",
            "object": "{}",
            "array": "[]",
            "objectid": "ObjectId()",
            "bindata": "BinData(0, '')",
        }
        return defaults.get(bson_type.lower(), "null")

    def _mongo_conversion(self, field: str, src: str, tgt: str) -> str:
        field_ref = f"${field}"
        src_l = src.lower()
        tgt_l = tgt.lower()
        if tgt_l in ("string",):
            return f"{{ $toString: {field_ref} }}"
        if tgt_l in ("int", "long"):
            return f"{{ $toLong: {field_ref} }}"
        if tgt_l in ("double", "decimal"):
            return f"{{ $toDouble: {field_ref} }}"
        if tgt_l in ("date",):
            if src_l in ("string",):
                return f"{{ $toDate: {field_ref} }}"
            return f"{{ $convert: {{ input: {field_ref}, to: 'date', onError: null }} }}"
        if tgt_l in ("bool", "boolean"):
            return f"{{ $toBool: {field_ref} }}"
        return field_ref

    def _generate_index_scripts(self, diff: TableCollectionDiff, plan: SyncPlan) -> None:
        for idx_diff in diff.index_diffs:
            if idx_diff.target_db_type == "postgresql":
                self._pg_index_op(diff, idx_diff, plan)
            elif idx_diff.target_db_type == "mongodb":
                self._mongo_index_op(diff, idx_diff, plan)

    def _pg_index_op(self, diff: TableCollectionDiff, idx: IndexDiff, plan: SyncPlan) -> None:
        table = diff.pg_table
        if idx.action == "add_index":
            unique = "UNIQUE" if idx.is_unique else ""
            cols_sql = ", ".join(idx.columns)
            sql = f"CREATE {unique} INDEX IF NOT EXISTS {idx.index_name} ON {table} ({cols_sql});"
            rollback = f"DROP INDEX IF EXISTS {idx.index_name};"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="postgresql",
                operation_type="CREATE_INDEX",
                object_name=f"{table}.{idx.index_name}",
                sql_script=sql,
                rollback_sql=rollback,
                diff_ref=idx,
            ))
        elif idx.action == "drop_index":
            sql = f"DROP INDEX IF EXISTS {idx.index_name};"
            unique = "UNIQUE" if idx.is_unique else ""
            cols_sql = ", ".join(idx.columns)
            rollback = f"CREATE {unique} INDEX {idx.index_name} ON {table} ({cols_sql});"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="postgresql",
                operation_type="DROP_INDEX",
                object_name=f"{table}.{idx.index_name}",
                sql_script=sql,
                rollback_sql=rollback,
                diff_ref=idx,
            ))

    def _mongo_index_op(self, diff: TableCollectionDiff, idx: IndexDiff, plan: SyncPlan) -> None:
        coll = diff.mongo_collection
        if idx.action == "add_index":
            keys_obj = "{" + ", ".join([f"'{c}': 1" for c in idx.columns]) + "}"
            options_parts = []
            if idx.is_unique:
                options_parts.append("unique: true")
            options_parts.append(f"name: '{idx.index_name}'")
            ttl = idx.extra.get("expire_after_seconds")
            if ttl:
                options_parts.append(f"expireAfterSeconds: {ttl}")
            options_str = ", ".join(options_parts)
            mongo_script = f"db.{coll}.createIndex({keys_obj}, {{ {options_str} }});"
            rollback = f"db.{coll}.dropIndex('{idx.index_name}');"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="mongodb",
                operation_type="CREATE_INDEX",
                object_name=f"{coll}.{idx.index_name}",
                mongo_script=mongo_script,
                rollback_mongo=rollback,
                diff_ref=idx,
            ))
        elif idx.action == "drop_index":
            mongo_script = f"db.{coll}.dropIndex('{idx.index_name}');"
            keys_obj = "{" + ", ".join([f"'{c}': 1" for c in idx.columns]) + "}"
            options_str = f"name: '{idx.index_name}'"
            if idx.is_unique:
                options_str = "unique: true, " + options_str
            rollback = f"db.{coll}.createIndex({keys_obj}, {{ {options_str} }});"
            plan.add_operation(SyncOperation(
                id=self._next_id(),
                order=0,
                target_db="mongodb",
                operation_type="DROP_INDEX",
                object_name=f"{coll}.{idx.index_name}",
                mongo_script=mongo_script,
                rollback_mongo=rollback,
                diff_ref=idx,
            ))

    def _generate_sharding_scripts(self, diff: TableCollectionDiff, plan: SyncPlan) -> None:
        for sd in diff.sharding_diffs:
            if sd.action == "add_bucket_range" and sd.target_db_type == "mongodb":
                coll = diff.mongo_collection
                from_date = sd.extra.get("from", "")
                to_date = sd.extra.get("to", "")
                bucket_size = sd.extra.get("bucket_size", "1h")
                table_name = f"{coll}_{from_date.replace('-', '')}"
                mongo_script = (
                    f"db.runCommand({{\n"
                    f"  create: '{table_name}',\n"
                    f"  viewOn: '{coll}',\n"
                    f"  pipeline: [\n"
                    f"    {{ $match: {{\n"
                    f"      timestamp: {{\n"
                    f"        $gte: ISODate('{from_date}T00:00:00Z'),\n"
                    f"        $lt: ISODate('{to_date}T00:00:00Z')\n"
                    f"      }}\n"
                    f"    }}]\n"
                    f"}});"
                )
                rollback = f"db.{table_name}.drop();"
                plan.add_operation(SyncOperation(
                    id=self._next_id(),
                    order=0,
                    target_db="mongodb",
                    operation_type="CREATE_BUCKET_VIEW",
                    object_name=table_name,
                    mongo_script=mongo_script,
                    rollback_mongo=rollback,
                    diff_ref=sd,
                ))

            if sd.action == "add_partition_key" and sd.target_db_type == "postgresql":
                table = diff.pg_table
                time_field = sd.extra.get("time_field", "measurement_time")
                sql = f"ALTER TABLE {table} ADD PRIMARY KEY (id, {time_field});"
                rollback = f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_pkey;"
                plan.add_operation(SyncOperation(
                    id=self._next_id(),
                    order=0,
                    target_db="postgresql",
                    operation_type="ADD_PARTITION_KEY",
                    object_name=table,
                    sql_script=sql,
                    rollback_sql=rollback,
                    diff_ref=sd,
                ))


class SyncExecutor:
    def __init__(
        self,
        pg_adapter: PostgreSQLAdapter,
        mongo_adapter: MongoDBAdapter,
        logger: Optional[Callable] = None,
    ):
        self.pg = pg_adapter
        self.mongo = mongo_adapter
        self.logger = logger or (lambda msg: None)
        self.execution_history: List[Dict[str, Any]] = []
        self.failed_op: Optional[SyncOperation] = None
        self.ops_executed: List[SyncOperation] = []

    def execute_plan(self, plan: SyncPlan) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        self.execution_history = []
        self.failed_op = None
        self.ops_executed = []

        for op in plan.operations:
            try:
                self.logger(f"Executing [{op.id}] {op.operation_type} on {op.object_name}")
                result = self._execute_operation(op)
                self.execution_history.append({
                    "op_id": op.id,
                    "operation_type": op.operation_type,
                    "object_name": op.object_name,
                    "target_db": op.target_db,
                    "status": "SUCCESS",
                    "result": str(result) if result else None,
                    "timestamp": datetime.now().isoformat(),
                })
                self.ops_executed.append(op)
                self.logger(f"  -> Success")
            except Exception as e:
                self.failed_op = op
                self.execution_history.append({
                    "op_id": op.id,
                    "operation_type": op.operation_type,
                    "object_name": op.object_name,
                    "target_db": op.target_db,
                    "status": "FAILED",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                self.logger(f"  -> FAILED: {e}")
                return False, self.execution_history, str(e)

        return True, self.execution_history, None

    def _execute_operation(self, op: SyncOperation) -> Any:
        if op.target_db == "postgresql":
            if op.sql_script:
                self.pg.execute_script(op.sql_script)
                return f"SQL executed successfully"
        elif op.target_db == "mongodb":
            if op.mongo_script:
                return self._execute_mongo_script(op.mongo_script)
        return None

    def _execute_mongo_script(self, script: str) -> Any:
        return self.mongo.execute_mongoshell_script(script)

    def rollback_failed(self) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        rollback_results = []
        if not self.failed_op:
            return True, rollback_results, None

        for op in reversed(self.ops_executed):
            try:
                self.logger(f"Rolling back [{op.id}] {op.operation_type}")
                rb_result = self._execute_rollback(op)
                rollback_results.append({
                    "op_id": op.id,
                    "operation_type": f"ROLLBACK_{op.operation_type}",
                    "object_name": op.object_name,
                    "target_db": op.target_db,
                    "status": "SUCCESS",
                    "result": str(rb_result) if rb_result else None,
                    "timestamp": datetime.now().isoformat(),
                })
                self.logger(f"  -> Rollback success")
            except Exception as e:
                rollback_results.append({
                    "op_id": op.id,
                    "operation_type": f"ROLLBACK_{op.operation_type}",
                    "object_name": op.object_name,
                    "target_db": op.target_db,
                    "status": "FAILED",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                self.logger(f"  -> Rollback FAILED: {e}")
                return False, rollback_results, str(e)

        return True, rollback_results, None

    def _execute_rollback(self, op: SyncOperation) -> Any:
        if op.target_db == "postgresql":
            if op.rollback_sql:
                self.pg.execute_script(op.rollback_sql)
                return "Rollback SQL executed"
        elif op.target_db == "mongodb":
            if op.rollback_mongo:
                return self._execute_mongo_script(op.rollback_mongo)
        return None

    def export_sql_scripts(self, plan: SyncPlan, output_dir: str = "scripts") -> Tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        sql_file = os.path.join(output_dir, f"sync_sql_{ts}.sql")
        mongo_file = os.path.join(output_dir, f"sync_mongo_{ts}.js")

        sql_lines = ["-- PostgreSQL Synchronization Script", f"-- Generated at: {ts}", ""]
        mongo_lines = ["// MongoDB Synchronization Script", f"// Generated at: {ts}", ""]

        for op in plan.operations:
            if op.sql_script:
                sql_lines.append(f"-- Op: {op.id} | {op.operation_type} | {op.object_name}")
                sql_lines.append(op.sql_script)
                sql_lines.append("")
            if op.mongo_script:
                mongo_lines.append(f"// Op: {op.id} | {op.operation_type} | {op.object_name}")
                mongo_lines.append(op.mongo_script)
                mongo_lines.append("")

        with open(sql_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sql_lines))
        with open(mongo_file, "w", encoding="utf-8") as f:
            f.write("\n".join(mongo_lines))

        return sql_file, mongo_file
