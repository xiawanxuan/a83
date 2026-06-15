"""
MongoDB 存储容量前置校验子模块

功能:
    - 复用 metadata_collector 采集的容量元数据
    - 基于 diff 结果预估本次同步新增的数据量
    - 检查每个分片磁盘利用率、shard maxSize
    - 支持 warn / block 两种拦截级别
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config_manager import ConfigManager
from diff_engine import (
    FieldDiff,
    IndexDiff,
    ShardingDiff,
    TableCollectionDiff,
)
from metadata_collector import MongoCollectionMetadata, TableMetadata


@dataclass
class CollectionStorageForecast:
    collection_name: str
    target_db: str
    current_size_bytes: int = 0
    projected_growth_bytes: int = 0
    projected_total_bytes: int = 0
    added_fields_bytes: int = 0
    added_indexes_bytes: int = 0
    migration_rebuild_bytes: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "target_db": self.target_db,
            "current_size_bytes": self.current_size_bytes,
            "projected_growth_bytes": self.projected_growth_bytes,
            "projected_total_bytes": self.projected_total_bytes,
            "added_fields_bytes": self.added_fields_bytes,
            "added_indexes_bytes": self.added_indexes_bytes,
            "migration_rebuild_bytes": self.migration_rebuild_bytes,
            "warnings": self.warnings,
        }


@dataclass
class ShardStorageStatus:
    shard_name: str
    current_size_bytes: int = 0
    limit_bytes: Optional[int] = None
    projected_size_bytes: int = 0
    usage_ratio: float = 0.0
    projected_usage_ratio: float = 0.0
    level: str = "ok"
    message: str = ""
    projected_delta_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_name": self.shard_name,
            "current_size_bytes": self.current_size_bytes,
            "limit_bytes": self.limit_bytes,
            "projected_size_bytes": self.projected_size_bytes,
            "usage_ratio": round(self.usage_ratio, 4),
            "projected_usage_ratio": round(self.projected_usage_ratio, 4),
            "level": self.level,
            "message": self.message,
            "projected_delta_bytes": self.projected_delta_bytes,
        }


@dataclass
class StorageValidationResult:
    passed: bool = True
    enforcement: str = "block"
    overall_level: str = "ok"
    per_collection: List[CollectionStorageForecast] = field(default_factory=list)
    per_shard: List[ShardStorageStatus] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    warning_reasons: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    total_projected_growth_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "enforcement": self.enforcement,
            "overall_level": self.overall_level,
            "per_collection": [c.to_dict() for c in self.per_collection],
            "per_shard": [s.to_dict() for s in self.per_shard],
            "blocking_reasons": self.blocking_reasons,
            "warning_reasons": self.warning_reasons,
            "info": self.info,
            "total_projected_growth_bytes": self.total_projected_growth_bytes,
        }

    def format_human_readable(self) -> str:
        lines = []
        lines.append("=== Mongo 存储容量前置校验报告 ===")
        lines.append(f"执行模式: {self.enforcement.upper()}   总体状态: {self.overall_level.upper()}")
        lines.append(
            "预计总容量增幅: "
            f"{self._fmt_bytes(self.total_projected_growth_bytes)}"
        )
        lines.append("")
        if self.per_collection:
            lines.append("--- 集合级预测 ---")
            for c in self.per_collection:
                lines.append(
                    f"  [{c.target_db.upper()}] {c.collection_name}: "
                    f"当前 {self._fmt_bytes(c.current_size_bytes)}  "
                    f"+{self._fmt_bytes(c.projected_growth_bytes)}  "
                    f"→ {self._fmt_bytes(c.projected_total_bytes)}"
                )
                for w in c.warnings:
                    lines.append(f"      ! {w}")
        if self.per_shard:
            lines.append("")
            lines.append("--- 分片级状态 ---")
            for s in self.per_shard:
                limit_s = self._fmt_bytes(s.limit_bytes) if s.limit_bytes else "无上限"
                lines.append(
                    f"  [{s.level.upper():8s}] {s.shard_name}: "
                    f"{self._fmt_bytes(s.current_size_bytes)} / {limit_s} "
                    f"({s.usage_ratio*100:.1f}% → {s.projected_usage_ratio*100:.1f}%)"
                )
                if s.message:
                    lines.append(f"      . {s.message}")
        if self.warning_reasons:
            lines.append("")
            lines.append("--- 警告 ---")
            for w in self.warning_reasons:
                lines.append(f"  ! {w}")
        if self.blocking_reasons:
            lines.append("")
            lines.append("--- 拦截原因 ---")
            for b in self.blocking_reasons:
                lines.append(f"  X {b}")
        if self.info:
            lines.append("")
            lines.append("--- 提示 ---")
            for i in self.info:
                lines.append(f"  - {i}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n is None:
            return "0B"
        n = int(n)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        size = float(n)
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.2f}{units[i]}"


class StorageValidator:
    def __init__(
        self,
        config: ConfigManager,
    ):
        self.config = config
        self.warn_threshold = config.get_disk_usage_warning_threshold()
        self.block_threshold = config.get_disk_usage_block_threshold()
        self.index_overhead = config.get_index_overhead_multiplier()
        self.migration_overhead = config.get_migration_overhead_multiplier()
        self.per_coll_max = config.get_per_collection_max_size_bytes()
        self.enforcement = config.get_enforcement_mode()
        self.capacity_overrides = config.get_storage_capacity_overrides()

    def _fmt_bytes(self, n: int) -> str:
        return StorageValidationResult._fmt_bytes(int(n))

    def run_validation(
        self,
        diffs: List[TableCollectionDiff],
        pg_tables: List[TableMetadata],
        mongo_colls: List[MongoCollectionMetadata],
        storage_overview: Optional[Dict[str, Any]] = None,
    ) -> StorageValidationResult:
        result = StorageValidationResult(enforcement=self.enforcement)

        if not self.config.is_storage_validation_enabled():
            result.info.append("storage_validation.enabled=false，已跳过容量校验")
            result.passed = True
            result.overall_level = "skipped"
            return result

        mongo_coll_map = {c.collection_name: c for c in mongo_colls}
        pg_table_map = {t.table_name: t for t in pg_tables}

        target_collections: Dict[str, Tuple[str, Optional[MongoCollectionMetadata], Optional[TableMetadata], List[TableCollectionDiff]]] = {}

        for diff in diffs:
            if diff.mongo_collection not in target_collections:
                target_collections[diff.mongo_collection] = (
                    diff.mongo_collection,
                    mongo_coll_map.get(diff.mongo_collection),
                    pg_table_map.get(diff.pg_table),
                    [],
                )
            target_collections[diff.mongo_collection][3].append(diff)

        result.info.append(
            f"参与同步集合数: {len(target_collections)}；"
            f"warn_threshold={self.warn_threshold*100:.0f}%, "
            f"block_threshold={self.block_threshold*100:.0f}%"
        )

        shard_current: Dict[str, int] = {}
        shard_limit: Dict[str, int] = {}
        shard_project_delta: Dict[str, int] = {}

        if storage_overview:
            db_shards = storage_overview.get("shards", []) or []
            for shard in db_shards:
                sid = shard.get("shard_id") or shard.get("id") or ""
                if not sid:
                    continue
                if sid in self.capacity_overrides:
                    shard_limit[str(sid)] = self.capacity_overrides[str(sid)]
                elif shard.get("max_size_bytes"):
                    shard_limit[str(sid)] = shard["max_size_bytes"]
                fs_total = storage_overview.get("fs_total_size_bytes")
                if sid not in shard_limit and fs_total:
                    shard_limit[str(sid)] = int(fs_total * self.block_threshold)

        for coll_name, (_, mongo_md, pg_md, coll_diffs) in target_collections.items():
            forecast = self._forecast_collection(
                coll_name, mongo_md, pg_md, coll_diffs
            )
            result.per_collection.append(forecast)
            result.total_projected_growth_bytes += forecast.projected_growth_bytes

            if forecast.projected_total_bytes > 0 and self.per_coll_max:
                if forecast.projected_total_bytes > self.per_coll_max:
                    msg = (
                        f"集合 {coll_name} 预测容量 "
                        f"{self._fmt_bytes(forecast.projected_total_bytes)} "
                        f"超过单集合上限 {self._fmt_bytes(self.per_coll_max)}"
                    )
                    result.blocking_reasons.append(msg)
                    forecast.warnings.append(msg)

            if mongo_md and mongo_md.shard_chunks:
                shard_count = max(1, len(mongo_md.shard_chunks))
                per_shard_delta = forecast.projected_growth_bytes // shard_count
                for sc in mongo_md.shard_chunks:
                    shard_name = sc.shard_name
                    shard_current[shard_name] = (
                        shard_current.get(shard_name, 0) + sc.estimated_size_bytes
                    )
                    shard_project_delta[shard_name] = (
                        shard_project_delta.get(shard_name, 0) + per_shard_delta
                    )
            elif mongo_md and storage_overview:
                default_shard = "__unsharded__"
                shard_current[default_shard] = (
                    shard_current.get(default_shard, 0)
                    + mongo_md.total_size_bytes
                )
                shard_project_delta[default_shard] = (
                    shard_project_delta.get(default_shard, 0)
                    + forecast.projected_growth_bytes
                )
                fs_total = storage_overview.get("fs_total_size_bytes")
                if fs_total and default_shard not in shard_limit:
                    shard_limit[default_shard] = int(fs_total * self.block_threshold)

        if not shard_current and storage_overview:
            default_shard = "__global__"
            shard_current[default_shard] = storage_overview.get(
                "total_size_bytes", storage_overview.get("storage_size_bytes", 0)
            )
            fs_total = storage_overview.get("fs_total_size_bytes")
            if fs_total:
                shard_limit[default_shard] = int(fs_total * self.block_threshold)
            shard_project_delta[default_shard] = result.total_projected_growth_bytes

        overall_level = "ok"
        for shard_name in sorted(set(shard_current.keys()) | set(shard_limit.keys())):
            current = shard_current.get(shard_name, 0)
            limit = shard_limit.get(shard_name)
            delta = shard_project_delta.get(shard_name, 0)
            projected = current + delta

            usage = 0.0
            proj_usage = 0.0
            if limit and limit > 0:
                usage = current / limit
                proj_usage = projected / limit

            level = "ok"
            message = ""
            if limit is None:
                if storage_overview and storage_overview.get("fs_total_size_bytes"):
                    fs_used = storage_overview.get("fs_used_size_bytes", 0)
                    fs_total = storage_overview.get("fs_total_size_bytes", 0)
                    if fs_total:
                        usage = fs_used / fs_total
                        projected_fs = fs_used + delta
                        proj_usage = projected_fs / fs_total
                        limit = int(fs_total * self.block_threshold)
                        if proj_usage > self.block_threshold:
                            level = "block"
                            message = (
                                f"文件系统预测占用 {proj_usage*100:.1f}% "
                                f"超过拦截线 {self.block_threshold*100:.0f}%"
                            )
                        elif proj_usage > self.warn_threshold:
                            level = "warn"
                            message = (
                                f"文件系统预测占用 {proj_usage*100:.1f}% "
                                f"超过警告线 {self.warn_threshold*100:.0f}%"
                            )
                else:
                    level = "unknown"
                    message = "未获取到分片容量上限，跳过严格校验"
            else:
                if proj_usage > self.block_threshold:
                    level = "block"
                    message = (
                        f"预测容量 {self._fmt_bytes(projected)} 超过分片上限 "
                        f"{self._fmt_bytes(limit)} 的 "
                        f"{self.block_threshold*100:.0f}%"
                    )
                elif proj_usage > self.warn_threshold:
                    level = "warn"
                    message = (
                        f"预测容量 {self._fmt_bytes(projected)} 超过分片上限 "
                        f"{self._fmt_bytes(limit)} 的 "
                        f"{self.warn_threshold*100:.0f}%"
                    )
                elif usage > self.block_threshold:
                    level = "warn"
                    message = (
                        f"当前已达 {usage*100:.1f}%，接近拦截线，建议先清退历史数据"
                    )

            status = ShardStorageStatus(
                shard_name=shard_name,
                current_size_bytes=current,
                limit_bytes=limit,
                projected_size_bytes=projected,
                usage_ratio=usage,
                projected_usage_ratio=proj_usage,
                level=level,
                message=message,
                projected_delta_bytes=delta,
            )
            result.per_shard.append(status)

            if level == "block":
                overall_level = "block"
                result.blocking_reasons.append(f"[{shard_name}] {message}")
            elif level == "warn" and overall_level != "block":
                overall_level = "warn"
                result.warning_reasons.append(f"[{shard_name}] {message}")
            elif level == "unknown" and overall_level == "ok":
                overall_level = "unknown"

        if self.enforcement == "warn_only":
            if overall_level == "block":
                result.warning_reasons.extend(result.blocking_reasons)
                result.blocking_reasons = []
                overall_level = "warn" if overall_level != "ok" else overall_level
                result.passed = True
            else:
                result.passed = True
        else:
            result.passed = (
                overall_level != "block" and len(result.blocking_reasons) == 0
            )

        if overall_level == "ok" and not result.warning_reasons:
            result.info.append("所有目标集合与分片均通过容量检查")
        result.overall_level = overall_level
        return result

    def _forecast_collection(
        self,
        coll_name: str,
        mongo_md: Optional[MongoCollectionMetadata],
        pg_md: Optional[TableMetadata],
        diffs: List[TableCollectionDiff],
    ) -> CollectionStorageForecast:
        forecast = CollectionStorageForecast(
            collection_name=coll_name,
            target_db="mongodb",
        )
        if mongo_md:
            forecast.current_size_bytes = int(mongo_md.total_size_bytes or 0)
            doc_count = int(mongo_md.document_count or 0)
            avg_doc = int(mongo_md.avg_document_size_bytes or 0)
        else:
            doc_count = 0
            avg_doc = 0

        added_fields_bytes = 0
        added_indexes_bytes = 0
        migration_bytes = 0

        for diff in diffs:
            for fd in diff.field_diffs:
                if fd.target_db_type != "mongodb":
                    continue
                if fd.action == "add_field":
                    field_width_est = self._estimate_field_width_bytes(fd)
                    added_fields_bytes += field_width_est * max(doc_count, 1)
                elif fd.action == "modify_field":
                    src_w = self._estimate_type_width_bytes(fd.source_type or "")
                    tgt_w = self._estimate_type_width_bytes(fd.target_type or "")
                    delta_per_doc = max(0, tgt_w - src_w)
                    added_fields_bytes += delta_per_doc * doc_count

            for idx in diff.index_diffs:
                if idx.target_db_type != "mongodb":
                    continue
                if idx.action == "add_index":
                    est = self._estimate_index_bytes(idx, mongo_md)
                    added_indexes_bytes += int(est * self.index_overhead)
                elif idx.action == "modify_index":
                    est = self._estimate_index_bytes(idx, mongo_md)
                    migration_bytes += int(est * self.migration_overhead)
                elif idx.action == "drop_index":
                    pass

            for sd in diff.sharding_diffs:
                if sd.target_db_type != "mongodb":
                    continue
                if sd.action in (
                    "modify_time_field",
                    "modify_meta_field",
                    "modify_granularity",
                    "modify_bucket_span",
                    "shard_key_mismatch",
                    "collection_type_mismatch",
                    "modify_bucket_range",
                ):
                    current_total = forecast.current_size_bytes
                    migration_bytes += int(current_total * self.migration_overhead)
                    forecast.warnings.append(
                        f"动作 {sd.action} 需要全集合重写，预计产生 "
                        f"{self._fmt_bytes(current_total)} × "
                        f"{self.migration_overhead:.1f} 临时空间"
                    )
                if sd.action in ("create_timeseries", "create_bucket_range"):
                    if mongo_md is None:
                        if pg_md:
                            approx = self._pg_table_est_size(pg_md)
                            added_fields_bytes += approx
                            migration_bytes += int(approx * 0.5)

        forecast.added_fields_bytes = added_fields_bytes
        forecast.added_indexes_bytes = added_indexes_bytes
        forecast.migration_rebuild_bytes = migration_bytes
        forecast.projected_growth_bytes = (
            added_fields_bytes + added_indexes_bytes + migration_bytes
        )
        forecast.projected_total_bytes = (
            forecast.current_size_bytes + forecast.projected_growth_bytes
        )
        return forecast

    def _estimate_field_width_bytes(self, fd: FieldDiff) -> int:
        width = 64
        if fd.target_type:
            width = self._estimate_type_width_bytes(fd.target_type)
        elif fd.source_type:
            width = self._estimate_type_width_bytes(fd.source_type)
        return max(16, width)

    def _estimate_type_width_bytes(self, type_str: str) -> int:
        t = (type_str or "").lower()
        if not t:
            return 64
        if t in ("bool", "boolean"):
            return 1
        if t in ("tinyint", "smallint", "int2"):
            return 2
        if t in ("int", "integer", "int4", "numberint", "int32"):
            return 4
        if t in ("bigint", "long", "numberlong", "int8", "timestamp"):
            return 8
        if t in ("float", "float4", "double", "float8", "decimal", "numberdouble"):
            return 8
        if t in ("date", "datetime", "time", "timetz"):
            return 8
        if t in ("oid", "objectid"):
            return 12
        if t in ("uuid"):
            return 16
        if t.startswith("varchar") or t.startswith("character varying") or t in ("string", "text"):
            m = self._extract_len(t)
            return m if m else 128
        if t.startswith("char"):
            m = self._extract_len(t)
            return m if m else 32
        if t in ("json", "jsonb", "object", "objectid", "document", "dict"):
            return 512
        if t in ("array", "list"):
            return 256
        if t in ("binary", "bytea", "blob"):
            return 1024
        return 64

    @staticmethod
    def _extract_len(type_str: str) -> int:
        import re

        m = re.search(r"\((\d+)\)", type_str)
        if m:
            return int(m.group(1))
        return 0

    def _estimate_index_bytes(
        self, idx: IndexDiff, mongo_md: Optional[MongoCollectionMetadata]
    ) -> int:
        if mongo_md is None:
            doc_count = 100_000
            est_key_width = 16
        else:
            doc_count = max(1, int(mongo_md.document_count or 0))
            est_key_width = max(
                8, int(mongo_md.avg_document_size_bytes or 128) // 4
            )
        key_width = 0
        for col in idx.columns:
            key_width += self._estimate_type_width_bytes(
                self._infer_column_type(col, mongo_md)
            )
        if key_width == 0:
            key_width = est_key_width
        per_row = max(16, key_width + 32)
        return per_row * doc_count

    @staticmethod
    def _infer_column_type(col: str, mongo_md: Optional[MongoCollectionMetadata]) -> str:
        if mongo_md is None:
            return "string"
        fi = mongo_md.get_field(col)
        if fi:
            return fi.bson_type or "string"
        if col.lower() in ("id", "_id"):
            return "objectid"
        if "time" in col.lower() or "date" in col.lower():
            return "date"
        if "count" in col.lower() or col.lower().endswith("_id"):
            return "long"
        return "string"

    def _pg_table_est_size(self, pg_md: TableMetadata) -> int:
        total = 0
        for col in pg_md.columns:
            total += self._estimate_type_width_bytes(col.data_type)
        return max(total, 256) * 100_000
