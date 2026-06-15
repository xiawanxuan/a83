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
    column_directions: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "index_name": self.index_name,
            "target_db_type": self.target_db_type,
            "columns": self.columns,
            "is_unique": self.is_unique,
            "column_directions": self.column_directions,
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

    def _make_index_signature(
        self, columns: List[str], directions: Optional[List[str]] = None
    ) -> tuple:
        sig = []
        for i, col in enumerate(columns):
            direction = "ASC"
            if directions and i < len(directions):
                direction = directions[i]
            sig.append((col, direction))
        return tuple(sig)

    def _mongo_keys_to_signature(self, keys: List[Dict[str, int]]) -> tuple:
        sig = []
        for key_dict in keys:
            for field_name, direction in key_dict.items():
                dir_str = "DESC" if direction == -1 else "ASC"
                sig.append((field_name, dir_str))
        return tuple(sig)

    def _diff_indexes_pg_to_mongo(
        self, pg: TableMetadata, mongo: Optional[MongoCollectionMetadata]
    ) -> List[IndexDiff]:
        diffs = []

        mongo_sig_map: Dict[tuple, Any] = {}
        if mongo:
            for mi in mongo.indexes:
                sig = self._mongo_keys_to_signature(mi.keys)
                mongo_sig_map[sig] = mi

        pg_sig_set: set = set()
        for pg_idx in pg.indexes:
            if pg_idx.is_primary:
                continue
            pg_sig = self._make_index_signature(pg_idx.columns, pg_idx.column_directions)
            pg_sig_set.add(pg_sig)

            matching_mongo = mongo_sig_map.get(pg_sig)

            if matching_mongo is None:
                matched_by_cols = None
                if mongo:
                    pg_col_set = tuple(c for c, _ in pg_sig)
                    for mi in mongo.indexes:
                        mi_col_set = tuple(list(k.keys())[0] for k in mi.keys)
                        if mi_col_set == pg_col_set:
                            matched_by_cols = mi
                            break

                if matched_by_cols is not None:
                    mi_sig = self._mongo_keys_to_signature(matched_by_cols.keys)
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matched_by_cols.name,
                        target_db_type="mongodb",
                        columns=list(pg_idx.columns),
                        is_unique=pg_idx.is_unique,
                        column_directions=list(pg_idx.column_directions),
                        extra={
                            "change": "direction_mismatch",
                            "current_sig": list(mi_sig),
                            "expected_sig": list(pg_sig),
                            "original_pg_name": pg_idx.name,
                        },
                    ))
                else:
                    idx_name = self._normalize_index_name(pg_idx.name, pg.table_name, "mongo")
                    diffs.append(IndexDiff(
                        action="add_index",
                        index_name=idx_name,
                        target_db_type="mongodb",
                        columns=list(pg_idx.columns),
                        is_unique=pg_idx.is_unique,
                        column_directions=list(pg_idx.column_directions),
                        extra={"original_pg_name": pg_idx.name, "pg_index_type": pg_idx.index_type},
                    ))
            else:
                if pg_idx.is_unique and not matching_mongo.is_unique:
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matching_mongo.name,
                        target_db_type="mongodb",
                        columns=list(pg_idx.columns),
                        is_unique=True,
                        column_directions=list(pg_idx.column_directions),
                        extra={"change": "add_uniqueness", "original_pg_name": pg_idx.name},
                    ))
                elif not pg_idx.is_unique and matching_mongo.is_unique:
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matching_mongo.name,
                        target_db_type="mongodb",
                        columns=list(pg_idx.columns),
                        is_unique=False,
                        column_directions=list(pg_idx.column_directions),
                        extra={"change": "remove_uniqueness", "original_pg_name": pg_idx.name},
                    ))

        recommended = self.config.get_recommended_indexes()
        recommended_sigs: set = set()
        for rec in recommended:
            rec_fields = rec.get("fields", [])
            rec_sig = self._make_index_signature(rec_fields)
            recommended_sigs.add(rec_sig)
            if rec_sig in mongo_sig_map:
                continue
            prefix_covered = False
            for existing_sig in mongo_sig_map:
                if existing_sig[:len(rec_sig)] == rec_sig:
                    prefix_covered = True
                    break
            if prefix_covered:
                continue
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
                mi_sig = self._mongo_keys_to_signature(mi.keys)
                if mi_sig in pg_sig_set:
                    continue
                mi_col_tuple = tuple(list(k.keys())[0] for k in mi.keys)
                found_by_cols = False
                for pg_idx in pg.indexes:
                    if pg_idx.is_primary:
                        continue
                    if tuple(pg_idx.columns) == mi_col_tuple:
                        found_by_cols = True
                        break
                if found_by_cols:
                    continue
                if mi_sig in recommended_sigs:
                    continue
                mi_col_tuple_check = tuple(list(k.keys())[0] for k in mi.keys)
                rec_covered = False
                for rsig in recommended_sigs:
                    rcols = tuple(c for c, _ in rsig)
                    if rcols == mi_col_tuple_check:
                        rec_covered = True
                        break
                if rec_covered:
                    continue
                mi_fields = [list(k.keys())[0] for k in mi.keys]
                mi_directions = ["DESC" if list(k.values())[0] == -1 else "ASC" for k in mi.keys]
                diffs.append(IndexDiff(
                    action="drop_index",
                    index_name=mi.name,
                    target_db_type="mongodb",
                    columns=mi_fields,
                    is_unique=mi.is_unique,
                    column_directions=mi_directions,
                    extra={},
                ))

        return diffs

    def _diff_indexes_mongo_to_pg(
        self, mongo: MongoCollectionMetadata, pg: Optional[TableMetadata]
    ) -> List[IndexDiff]:
        diffs = []

        pg_sig_map: Dict[tuple, Any] = {}
        if pg:
            for pidx in pg.indexes:
                sig = self._make_index_signature(pidx.columns, pidx.column_directions)
                pg_sig_map[sig] = pidx

        mongo_sig_set: set = set()
        for mi in mongo.indexes:
            if mi.name.startswith("_id_"):
                continue
            mi_sig = self._mongo_keys_to_signature(mi.keys)
            mongo_sig_set.add(mi_sig)
            mi_fields = [list(k.keys())[0] for k in mi.keys]
            mi_directions = ["DESC" if list(k.values())[0] == -1 else "ASC" for k in mi.keys]

            matching_pg = pg_sig_map.get(mi_sig)

            if matching_pg is None:
                matched_by_cols = None
                if pg:
                    mi_col_set = tuple(list(k.keys())[0] for k in mi.keys)
                    for pidx in pg.indexes:
                        if pidx.is_primary:
                            continue
                        if tuple(pidx.columns) == mi_col_set:
                            matched_by_cols = pidx
                            break

                if matched_by_cols is not None:
                    pg_sig = self._make_index_signature(matched_by_cols.columns, matched_by_cols.column_directions)
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matched_by_cols.name,
                        target_db_type="postgresql",
                        columns=mi_fields,
                        is_unique=mi.is_unique,
                        column_directions=mi_directions,
                        extra={
                            "change": "direction_mismatch",
                            "current_sig": list(pg_sig),
                            "expected_sig": list(mi_sig),
                            "original_mongo_name": mi.name,
                        },
                    ))
                else:
                    idx_name = self._normalize_index_name(mi.name, mongo.collection_name, "pg")
                    diffs.append(IndexDiff(
                        action="add_index",
                        index_name=idx_name,
                        target_db_type="postgresql",
                        columns=mi_fields,
                        is_unique=mi.is_unique,
                        column_directions=mi_directions,
                        extra={"original_mongo_name": mi.name, "ttl": mi.is_ttl, "expire_after_seconds": mi.expire_after_seconds},
                    ))
            else:
                if mi.is_unique and not matching_pg.is_unique:
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matching_pg.name,
                        target_db_type="postgresql",
                        columns=mi_fields,
                        is_unique=True,
                        column_directions=mi_directions,
                        extra={"change": "add_uniqueness", "original_mongo_name": mi.name},
                    ))
                elif not mi.is_unique and matching_pg.is_unique:
                    diffs.append(IndexDiff(
                        action="modify_index",
                        index_name=matching_pg.name,
                        target_db_type="postgresql",
                        columns=mi_fields,
                        is_unique=False,
                        column_directions=mi_directions,
                        extra={"change": "remove_uniqueness", "original_mongo_name": mi.name},
                    ))

        if pg:
            for pidx in pg.indexes:
                if pidx.is_primary:
                    continue
                pg_sig = self._make_index_signature(pidx.columns, pidx.column_directions)
                if pg_sig in mongo_sig_set:
                    continue
                pg_col_tuple = tuple(pidx.columns)
                found_by_cols = False
                for mi in mongo.indexes:
                    if mi.name.startswith("_id_"):
                        continue
                    mi_col_tuple = tuple(list(k.keys())[0] for k in mi.keys)
                    if mi_col_tuple == pg_col_tuple:
                        found_by_cols = True
                        break
                if found_by_cols:
                    continue
                diffs.append(IndexDiff(
                    action="drop_index",
                    index_name=pidx.name,
                    target_db_type="postgresql",
                    columns=list(pidx.columns),
                    is_unique=pidx.is_unique,
                    column_directions=list(pidx.column_directions),
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
        time_field = self.config.get_time_field()
        meta_field = self.config.get_meta_field()
        granularity = self.config.get_granularity()
        bucket_max_span = self.config.sync_config.time_series_config.get("bucket_max_span_seconds", 3600)

        if not mongo:
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
            return diffs

        if not mongo.is_time_series:
            diffs.append(ShardingDiff(
                action="collection_type_mismatch",
                target_db_type="mongodb",
                extra={
                    "current_type": "regular",
                    "expected_type": "timeseries",
                    "collection": mongo.collection_name,
                    "expected_time_field": time_field,
                    "expected_meta_field": meta_field,
                    "expected_granularity": granularity,
                },
            ))
            return diffs

        if mongo.time_field and mongo.time_field != time_field:
            diffs.append(ShardingDiff(
                action="modify_time_field",
                target_db_type="mongodb",
                extra={"current": mongo.time_field, "expected": time_field},
            ))

        if mongo.meta_field and mongo.meta_field != meta_field:
            diffs.append(ShardingDiff(
                action="modify_meta_field",
                target_db_type="mongodb",
                extra={"current": mongo.meta_field, "expected": meta_field},
            ))

        if mongo.granularity and mongo.granularity != granularity:
            diffs.append(ShardingDiff(
                action="modify_granularity",
                target_db_type="mongodb",
                extra={"current": mongo.granularity, "expected": granularity},
            ))

        if mongo.bucket_max_span_seconds and mongo.bucket_max_span_seconds != bucket_max_span:
            diffs.append(ShardingDiff(
                action="modify_bucket_span",
                target_db_type="mongodb",
                extra={"current": mongo.bucket_max_span_seconds, "expected": bucket_max_span},
            ))

        buckets_config = self.config.sync_config.time_series_config.get("bucket_ranges", [])
        if buckets_config:
            configured_map: Dict[tuple, str] = {}
            for bc in buckets_config:
                key = (bc.get("from", ""), bc.get("to", ""))
                configured_map[key] = bc.get("bucket_size", "")

            existing_map: Dict[tuple, str] = {}
            for b in mongo.buckets:
                key = (b.from_date, b.to_date)
                existing_map[key] = b.bucket_size

            for key, expected_size in configured_map.items():
                if key not in existing_map:
                    diffs.append(ShardingDiff(
                        action="add_bucket_range",
                        target_db_type="mongodb",
                        extra={"from": key[0], "to": key[1], "bucket_size": expected_size},
                    ))
                elif existing_map[key] != expected_size:
                    diffs.append(ShardingDiff(
                        action="modify_bucket_range",
                        target_db_type="mongodb",
                        extra={
                            "from": key[0],
                            "to": key[1],
                            "current_size": existing_map[key],
                            "expected_size": expected_size,
                        },
                    ))

            for key, current_size in existing_map.items():
                if key not in configured_map:
                    diffs.append(ShardingDiff(
                        action="remove_bucket_range",
                        target_db_type="mongodb",
                        extra={"from": key[0], "to": key[1], "bucket_size": current_size},
                    ))

        if mongo.is_sharded:
            expected_shard_key = self.config.sync_config.time_series_config.get("shard_key")
            if expected_shard_key and mongo.shard_key != expected_shard_key:
                diffs.append(ShardingDiff(
                    action="shard_key_mismatch",
                    target_db_type="mongodb",
                    extra={"current": mongo.shard_key, "expected": expected_shard_key},
                ))

        return diffs

    def _diff_sharding_mongo_to_pg(
        self, mongo: MongoCollectionMetadata, pg: Optional[TableMetadata]
    ) -> List[ShardingDiff]:
        diffs = []

        if not mongo.is_time_series:
            return diffs

        time_field = mongo.time_field or self.config.get_time_field()

        if pg is None:
            diffs.append(ShardingDiff(
                action="create_partitioned_table",
                target_db_type="postgresql",
                extra={
                    "time_field": time_field,
                    "meta_field": mongo.meta_field,
                    "granularity": mongo.granularity,
                },
            ))
            if mongo.buckets:
                partition_ranges = []
                for b in mongo.buckets:
                    partition_ranges.append({
                        "from": b.from_date,
                        "to": b.to_date,
                        "bucket_size": b.bucket_size,
                    })
                diffs.append(ShardingDiff(
                    action="create_time_partitions",
                    target_db_type="postgresql",
                    extra={"time_field": time_field, "partition_ranges": partition_ranges},
                ))
            return diffs

        has_partition_key = time_field in pg.primary_key_columns
        if not has_partition_key:
            diffs.append(ShardingDiff(
                action="add_partition_key",
                target_db_type="postgresql",
                extra={"time_field": time_field},
            ))

        is_partitioned = False
        for pidx in pg.indexes:
            if pidx.is_primary and time_field in pidx.columns:
                is_partitioned = True
                break
        if not is_partitioned and not has_partition_key:
            diffs.append(ShardingDiff(
                action="add_partition_key",
                target_db_type="postgresql",
                extra={"time_field": time_field},
            ))

        if mongo.buckets:
            partition_ranges = []
            for b in mongo.buckets:
                partition_ranges.append({
                    "from": b.from_date,
                    "to": b.to_date,
                    "bucket_size": b.bucket_size,
                })
            diffs.append(ShardingDiff(
                action="create_time_partitions",
                target_db_type="postgresql",
                extra={"time_field": time_field, "partition_ranges": partition_ranges},
            ))

        if mongo.is_sharded and mongo.shard_key:
            shard_fields = list(mongo.shard_key.keys())
            pg_idx_covers_shard = False
            for pidx in pg.indexes:
                if pidx.is_primary:
                    continue
                if all(f in pidx.columns for f in shard_fields):
                    pg_idx_covers_shard = True
                    break
            if not pg_idx_covers_shard:
                diffs.append(ShardingDiff(
                    action="add_shard_key_index",
                    target_db_type="postgresql",
                    extra={"shard_key": mongo.shard_key, "shard_fields": shard_fields},
                ))

        return diffs
