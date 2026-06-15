from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config_manager import ConfigManager
from metadata_collector import (
    ColumnInfo,
    IndexInfo,
    MongoCollectionMetadata,
    MongoFieldInfo,
    MongoIndexInfo,
    TableMetadata,
)


@dataclass
class FieldDiff:
    action: str
    field_name: str
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    target_db_type: str = "postgresql"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "field_name": self.field_name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "target_db_type": self.target_db_type,
            "extra": self.extra,
        }


@dataclass
class IndexDiff:
    action: str
    index_name: str
    target_db_type: str = "postgresql"
    columns: List[str] = field(default_factory=list)
    is_unique: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "index_name": self.index_name,
            "target_db_type": self.target_db_type,
            "columns": self.columns,
            "is_unique": self.is_unique,
            "extra": self.extra,
        }


@dataclass
class ShardingDiff:
    action: str
    target_db_type: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_db_type": self.target_db_type,
            "extra": self.extra,
        }


@dataclass
class TableCollectionDiff:
    pg_table: str
    mongo_collection: str
    direction: str
    field_diffs: List[FieldDiff] = field(default_factory=list)
    index_diffs: List[IndexDiff] = field(default_factory=list)
    sharding_diffs: List[ShardingDiff] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.field_diffs or self.index_diffs or self.sharding_diffs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pg_table": self.pg_table,
            "mongo_collection": self.mongo_collection,
            "direction": self.direction,
            "field_diffs": [f.to_dict() for f in self.field_diffs],
            "index_diffs": [i.to_dict() for i in self.index_diffs],
            "sharding_diffs": [s.to_dict() for s in self.sharding_diffs],
        }


class DiffEngine:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def compare_pair(
        self,
        pg_table: Optional[TableMetadata],
        mongo_coll: Optional[MongoCollectionMetadata],
    ) -> TableCollectionDiff:
        pg_table_name = pg_table.table_name if pg_table else ""
        mongo_coll_name = mongo_coll.collection_name if mongo_coll else ""

        mapping = self.config.get_mapping_for_pg_table(pg_table_name)
        direction = self.config.sync_config.sync_direction

        diff = TableCollectionDiff(
            pg_table=pg_table_name,
            mongo_collection=mongo_coll_name,
            direction=direction,
        )

        if not pg_table and mongo_coll:
            if self.config.is_mongo_to_pg_only() or self.config.is_bidirectional():
                diff.field_diffs.extend(self._diff_fields_mongo_to_pg(mongo_coll, None))
                diff.index_diffs.extend(self._diff_indexes_mongo_to_pg(mongo_coll, None))
                diff.sharding_diffs.extend(self._diff_sharding_mongo_to_pg(mongo_coll, None))
            return diff

        if pg_table and not mongo_coll:
            if self.config.is_pg_to_mongo_only() or self.config.is_bidirectional():
                diff.field_diffs.extend(self._diff_fields_pg_to_mongo(pg_table, None))
                diff.index_diffs.extend(self._diff_indexes_pg_to_mongo(pg_table, None))
                diff.sharding_diffs.extend(self._diff_sharding_pg_to_mongo(pg_table, None))
            return diff

        sync_fields = mapping.sync_fields if mapping else True
        sync_indexes = mapping.sync_indexes if mapping else True
        sync_sharding = mapping.sync_sharding if mapping else True

        if self.config.is_pg_to_mongo_only() or self.config.is_bidirectional():
            if sync_fields:
                diff.field_diffs.extend(self._diff_fields_pg_to_mongo(pg_table, mongo_coll))
            if sync_indexes:
                diff.index_diffs.extend(self._diff_indexes_pg_to_mongo(pg_table, mongo_coll))
            if sync_sharding:
                diff.sharding_diffs.extend(self._diff_sharding_pg_to_mongo(pg_table, mongo_coll))

        if self.config.is_mongo_to_pg_only() or self.config.is_bidirectional():
            if sync_fields:
                diff.field_diffs.extend(self._diff_fields_mongo_to_pg(mongo_coll, pg_table))
            if sync_indexes:
                diff.index_diffs.extend(self._diff_indexes_mongo_to_pg(mongo_coll, pg_table))
            if sync_sharding:
                diff.sharding_diffs.extend(self._diff_sharding_mongo_to_pg(mongo_coll, pg_table))

        return diff

    def compare_all(
        self,
        pg_tables: List[TableMetadata],
        mongo_colls: List[MongoCollectionMetadata],
    ) -> List[TableCollectionDiff]:
        diffs = []
        pg_map = {t.table_name: t for t in pg_tables}
        mongo_map = {c.collection_name: c for c in mongo_colls}

        processed = set()

        for mapping in self.config.sync_config.table_collection_mappings:
            pg_table = pg_map.get(mapping.pg_table)
            mongo_coll = mongo_map.get(mapping.mongo_collection)
            if pg_table or mongo_coll:
                diffs.append(self.compare_pair(pg_table, mongo_coll))
                processed.add((mapping.pg_table, mapping.mongo_collection))

        for table in pg_tables:
            mapping = self.config.get_mapping_for_pg_table(table.table_name)
            if mapping:
                continue
            coll_name = table.table_name
            mongo_coll = mongo_map.get(coll_name)
            if (table.table_name, coll_name) in processed:
                continue
            diffs.append(self.compare_pair(table, mongo_coll))
            processed.add((table.table_name, coll_name))

        for coll in mongo_colls:
            mapping = self.config.get_mapping_for_mongo_collection(coll.collection_name)
            if mapping:
                continue
            tbl_name = coll.collection_name
            if tbl_name in pg_map:
                continue
            if (tbl_name, coll.collection_name) in processed:
                continue
            diffs.append(self.compare_pair(None, coll))
            processed.add((tbl_name, coll.collection_name))

        return [d for d in diffs if d.has_changes()]

    def _diff_fields_pg_to_mongo(
        self, pg: TableMetadata, mongo: Optional[MongoCollectionMetadata]
    ) -> List[FieldDiff]:
        diffs = []
        mongo_fields = {}
        if mongo:
            mongo_fields = {f.name: f for f in mongo.fields}

        for pg_col in pg.columns:
            expected_mongo_type = self.config.get_pg_to_mongo_type(pg_col.data_type)
            if pg_col.name not in mongo_fields:
                diffs.append(FieldDiff(
                    action="add_field",
                    field_name=pg_col.name,
                    target_db_type="mongodb",
                    target_type=expected_mongo_type,
                    source_type=pg_col.data_type,
                    extra={"is_nullable": pg_col.is_nullable, "is_primary_key": pg_col.is_primary_key},
                ))
            else:
                mongo_field = mongo_fields[pg_col.name]
                if not self._types_compatible_mongo(expected_mongo_type, mongo_field.bson_type):
                    diffs.append(FieldDiff(
                        action="modify_field",
                        field_name=pg_col.name,
                        target_db_type="mongodb",
                        target_type=expected_mongo_type,
                        source_type=pg_col.data_type,
                        extra={"current_mongo_type": mongo_field.bson_type},
                    ))

        if mongo:
            for mongo_field in mongo.fields:
                if mongo_field.is_time_field or mongo_field.is_meta_field:
                    continue
                if not self.config.should_sync_field(mongo_field.name):
                    continue
                pg_col = pg.get_column(mongo_field.name)
                if not pg_col:
                    diffs.append(FieldDiff(
                        action="drop_field",
                        field_name=mongo_field.name,
                        target_db_type="mongodb",
                        source_type=mongo_field.bson_type,
                        extra={},
                    ))

        return diffs

    def _diff_fields_mongo_to_pg(
        self, mongo: MongoCollectionMetadata, pg: Optional[TableMetadata]
    ) -> List[FieldDiff]:
        diffs = []
        pg_cols = {}
        if pg:
            pg_cols = {c.name: c for c in pg.columns}

        for mongo_field in mongo.fields:
            if not self.config.should_sync_field(mongo_field.name):
                continue
            expected_pg_type = self.config.get_mongo_to_pg_type(mongo_field.bson_type)
            if mongo_field.name not in pg_cols:
                diffs.append(FieldDiff(
                    action="add_field",
                    field_name=mongo_field.name,
                    target_db_type="postgresql",
                    target_type=expected_pg_type,
                    source_type=mongo_field.bson_type,
                    extra={"is_time_field": mongo_field.is_time_field, "is_meta_field": mongo_field.is_meta_field},
                ))
            else:
                pg_col = pg_cols[mongo_field.name]
                if not self._types_compatible_pg(expected_pg_type, pg_col.data_type):
                    diffs.append(FieldDiff(
                        action="modify_field",
                        field_name=mongo_field.name,
                        target_db_type="postgresql",
                        target_type=expected_pg_type,
                        source_type=mongo_field.bson_type,
                        extra={"current_pg_type": pg_col.data_type},
                    ))

        if pg:
            for pg_col in pg.columns:
                if not self.config.should_sync_field(pg_col.name):
                    continue
                mongo_field = mongo.get_field(pg_col.name)
                if not mongo_field:
                    diffs.append(FieldDiff(
                        action="drop_field",
                        field_name=pg_col.name,
                        target_db_type="postgresql",
                        source_type=pg_col.data_type,
                        extra={"is_primary_key": pg_col.is_primary_key},
                    ))

        return diffs

    def _types_compatible_mongo(self, expected: str, actual: str) -> bool:
        if expected.lower() == actual.lower():
            return True
        compatible = {
            "string": ["string", "objectId", "uuid"],
            "date": ["date", "timestamp"],
            "int": ["int", "long", "double"],
            "long": ["long", "int", "double"],
            "double": ["double", "int", "long", "decimal"],
            "object": ["object", "array"],
        }
        return actual.lower() in compatible.get(expected.lower(), [expected.lower()])

    def _types_compatible_pg(self, expected: str, actual: str) -> bool:
        if expected.lower() == actual.lower():
            return True
        if "character varying" in actual.lower() and "varchar" in expected.lower():
            return True
        if "timestamp" in actual.lower() and "timestamp" in expected.lower():
            return True
        if "numeric" in actual.lower() and "numeric" in expected.lower():
            return True
        compatible = {
            "integer": ["integer", "bigint", "smallint", "numeric"],
            "bigint": ["bigint", "integer", "numeric"],
            "double precision": ["double precision", "numeric", "real"],
            "jsonb": ["jsonb", "json"],
            "varchar(255)": ["character varying", "varchar", "text"],
        }
        return actual.lower() in compatible.get(expected.lower(), [expected.lower()])

    def _diff_indexes_pg_to_mongo(
        self, pg: TableMetadata, mongo: Optional[MongoCollectionMetadata]
    ) -> List[IndexDiff]:
        diffs = []
        mongo_indexes = {}
        if mongo:
            mongo_indexes = {idx.name: idx for idx in mongo.indexes}

        for pg_idx in pg.indexes:
            if pg_idx.is_primary:
                continue
            idx_name = self._normalize_index_name(pg_idx.name, pg.table_name, "mongo")
            if idx_name not in mongo_indexes and pg_idx.name not in mongo_indexes:
                diffs.append(IndexDiff(
                    action="add_index",
                    index_name=idx_name,
                    target_db_type="mongodb",
                    columns=list(pg_idx.columns),
                    is_unique=pg_idx.is_unique,
                    extra={"original_pg_name": pg_idx.name, "pg_index_type": pg_idx.index_type},
                ))

        recommended = self.config.get_recommended_indexes()
        for rec in recommended:
            rec_fields = rec.get("fields", [])
            exists = False
            if mongo:
                for mi in mongo.indexes:
                    mi_fields = [list(k.keys())[0] for k in mi.keys]
                    if mi_fields == rec_fields:
                        exists = True
                        break
            if not exists:
                idx_name = f"idx_{pg.table_name}_{'_'.join(rec_fields)}"
                diffs.append(IndexDiff(
                    action="add_index",
                    index_name=idx_name,
                    target_db_type="mongodb",
                    columns=rec_fields,
                    is_unique=rec.get("unique", False),
                    extra={"recommended": True},
                ))

        if mongo:
            for mi in mongo.indexes:
                if mi.name.startswith("_id_"):
                    continue
                mi_fields = [list(k.keys())[0] for k in mi.keys]
                found = False
                for pg_idx in pg.indexes:
                    if list(pg_idx.columns) == mi_fields:
                        found = True
                        break
                if not found:
                    diffs.append(IndexDiff(
                        action="drop_index",
                        index_name=mi.name,
                        target_db_type="mongodb",
                        columns=mi_fields,
                        is_unique=mi.is_unique,
                        extra={},
                    ))

        return diffs

    def _diff_indexes_mongo_to_pg(
        self, mongo: MongoCollectionMetadata, pg: Optional[TableMetadata]
    ) -> List[IndexDiff]:
        diffs = []
        pg_indexes = {}
        if pg:
            pg_indexes = {idx.name: idx for idx in pg.indexes}

        for mi in mongo.indexes:
            if mi.name.startswith("_id_"):
                continue
            mi_fields = [list(k.keys())[0] for k in mi.keys]
            idx_name = self._normalize_index_name(mi.name, mongo.collection_name, "pg")
            found = False
            if pg:
                for pidx in pg.indexes:
                    if list(pidx.columns) == mi_fields:
                        found = True
                        break
            if not found:
                diffs.append(IndexDiff(
                    action="add_index",
                    index_name=idx_name,
                    target_db_type="postgresql",
                    columns=mi_fields,
                    is_unique=mi.is_unique,
                    extra={"original_mongo_name": mi.name, "ttl": mi.is_ttl, "expire_after_seconds": mi.expire_after_seconds},
                ))

        if pg:
            for pidx in pg.indexes:
                if pidx.is_primary:
                    continue
                found = False
                for mi in mongo.indexes:
                    mi_fields = [list(k.keys())[0] for k in mi.keys]
                    if list(pidx.columns) == mi_fields:
                        found = True
                        break
                if not found:
                    diffs.append(IndexDiff(
                        action="drop_index",
                        index_name=pidx.name,
                        target_db_type="postgresql",
                        columns=list(pidx.columns),
                        is_unique=pidx.is_unique,
                        extra={"pg_index_type": pidx.index_type},
                    ))

        return diffs

    def _normalize_index_name(self, original: str, table: str, target: str) -> str:
        prefix = self.config.sync_config.index_rules.get("default_index_prefix", "idx_")
        name = original
        for suffix in ["_pkey", "_idx", "_key", "_unq"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        prefixes_to_strip = ["idx_", "pk_", "uk_", "ix_", "i_"]
        for p in prefixes_to_strip:
            if name.startswith(p):
                name = name[len(p):]
                break
        return f"{prefix}{table}_{name}"

    def _diff_sharding_pg_to_mongo(
        self, pg: TableMetadata, mongo: Optional[MongoCollectionMetadata]
    ) -> List[ShardingDiff]:
        diffs = []
        if not mongo or not mongo.is_time_series:
            time_field = self.config.get_time_field()
            granularity = self.config.get_granularity()
            bucket_max_span = self.config.sync_config.time_series_config.get("bucket_max_span_seconds", 3600)
            meta_field = self.config.get_meta_field()
            diffs.append(ShardingDiff(
                action="create_timeseries",
                target_db_type="mongodb",
                extra={
                    "time_field": time_field,
                    "meta_field": meta_field,
                    "granularity": granularity,
                    "bucket_max_span_seconds": bucket_max_span,
                },
            ))
        else:
            expected_granularity = self.config.get_granularity()
            if mongo.granularity and mongo.granularity != expected_granularity:
                diffs.append(ShardingDiff(
                    action="modify_granularity",
                    target_db_type="mongodb",
                    extra={
                        "current": mongo.granularity,
                        "expected": expected_granularity,
                    },
                ))
            expected_bucket = self.config.sync_config.time_series_config.get("bucket_max_span_seconds", 3600)
            if mongo.bucket_max_span_seconds and mongo.bucket_max_span_seconds != expected_bucket:
                diffs.append(ShardingDiff(
                    action="modify_bucket_span",
                    target_db_type="mongodb",
                    extra={
                        "current": mongo.bucket_max_span_seconds,
                        "expected": expected_bucket,
                    },
                ))

        buckets_config = self.config.sync_config.time_series_config.get("bucket_ranges", [])
        if buckets_config and mongo:
            configured_froms = {b.from_date for b in mongo.buckets}
            for bc in buckets_config:
                if bc.get("from", "") not in configured_froms:
                    diffs.append(ShardingDiff(
                        action="add_bucket_range",
                        target_db_type="mongodb",
                        extra={
                            "from": bc.get("from", ""),
                            "to": bc.get("to", ""),
                            "bucket_size": bc.get("bucket_size", ""),
                        },
                    ))
        return diffs

    def _diff_sharding_mongo_to_pg(
        self, mongo: MongoCollectionMetadata, pg: Optional[TableMetadata]
    ) -> List[ShardingDiff]:
        diffs = []
        if mongo.is_time_series:
            time_field = mongo.time_field or self.config.get_time_field()
            if pg and pg.primary_key_columns:
                if time_field not in pg.primary_key_columns:
                    diffs.append(ShardingDiff(
                        action="add_partition_key",
                        target_db_type="postgresql",
                        extra={"time_field": time_field},
                    ))
            if pg is None:
                diffs.append(ShardingDiff(
                    action="create_partitioned_table",
                    target_db_type="postgresql",
                    extra={"time_field": time_field},
                ))
        return diffs
