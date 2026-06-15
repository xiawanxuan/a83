import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from db_adapters import MongoDBAdapter, PostgreSQLAdapter
from config_manager import ConfigManager


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool = True
    default_value: Optional[str] = None
    character_maximum_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    is_primary_key: bool = False
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_nullable": self.is_nullable,
            "default_value": self.default_value,
            "character_maximum_length": self.character_maximum_length,
            "numeric_precision": self.numeric_precision,
            "numeric_scale": self.numeric_scale,
            "is_primary_key": self.is_primary_key,
            "comment": self.comment,
        }


@dataclass
class IndexInfo:
    name: str
    columns: List[str]
    is_unique: bool = False
    is_primary: bool = False
    index_type: str = "btree"
    where_clause: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "is_unique": self.is_unique,
            "is_primary": self.is_primary,
            "index_type": self.index_type,
            "where_clause": self.where_clause,
        }


@dataclass
class TableMetadata:
    table_name: str
    schema: str
    columns: List[ColumnInfo] = field(default_factory=list)
    indexes: List[IndexInfo] = field(default_factory=list)
    primary_key_columns: List[str] = field(default_factory=list)
    comment: Optional[str] = None

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def get_index(self, name: str) -> Optional[IndexInfo]:
        for idx in self.indexes:
            if idx.name == name:
                return idx
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "schema": self.schema,
            "columns": [c.to_dict() for c in self.columns],
            "indexes": [idx.to_dict() for idx in self.indexes],
            "primary_key_columns": self.primary_key_columns,
            "comment": self.comment,
        }


@dataclass
class MongoFieldInfo:
    name: str
    bson_type: str
    description: Optional[str] = None
    is_time_field: bool = False
    is_meta_field: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bson_type": self.bson_type,
            "description": self.description,
            "is_time_field": self.is_time_field,
            "is_meta_field": self.is_meta_field,
        }


@dataclass
class MongoIndexInfo:
    name: str
    keys: List[Dict[str, int]]
    is_unique: bool = False
    is_ttl: bool = False
    expire_after_seconds: Optional[int] = None
    partial_filter_expression: Optional[Dict[str, Any]] = None
    sparse: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "keys": self.keys,
            "is_unique": self.is_unique,
            "is_ttl": self.is_ttl,
            "expire_after_seconds": self.expire_after_seconds,
            "partial_filter_expression": self.partial_filter_expression,
            "sparse": self.sparse,
        }


@dataclass
class TimeSeriesBucket:
    from_date: str
    to_date: str
    bucket_size: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_date": self.from_date,
            "to_date": self.to_date,
            "bucket_size": self.bucket_size,
        }


@dataclass
class MongoCollectionMetadata:
    collection_name: str
    is_time_series: bool = False
    time_field: Optional[str] = None
    meta_field: Optional[str] = None
    granularity: Optional[str] = None
    bucket_max_span_seconds: Optional[int] = None
    fields: List[MongoFieldInfo] = field(default_factory=list)
    indexes: List[MongoIndexInfo] = field(default_factory=list)
    buckets: List[TimeSeriesBucket] = field(default_factory=list)
    is_sharded: bool = False
    shard_key: Optional[Dict[str, int]] = None

    def get_field(self, name: str) -> Optional[MongoFieldInfo]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def get_index(self, name: str) -> Optional[MongoIndexInfo]:
        for idx in self.indexes:
            if idx.name == name:
                return idx
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "is_time_series": self.is_time_series,
            "time_field": self.time_field,
            "meta_field": self.meta_field,
            "granularity": self.granularity,
            "bucket_max_span_seconds": self.bucket_max_span_seconds,
            "fields": [f.to_dict() for f in self.fields],
            "indexes": [idx.to_dict() for idx in self.indexes],
            "buckets": [b.to_dict() for b in self.buckets],
            "is_sharded": self.is_sharded,
            "shard_key": self.shard_key,
        }


class MetadataCollector:
    def __init__(
        self,
        pg_adapter: PostgreSQLAdapter,
        mongo_adapter: MongoDBAdapter,
        config_manager: ConfigManager,
    ):
        self.pg = pg_adapter
        self.mongo = mongo_adapter
        self.config = config_manager

    def collect_pg_tables(self) -> List[TableMetadata]:
        schema = self.pg.get_schema()
        tables_sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        tables_data = self.pg.query_all(tables_sql, (schema,))
        results = []
        for row in tables_data:
            table_name = row["table_name"]
            if not self.config.should_sync_table(table_name):
                continue
            metadata = self._collect_single_pg_table(schema, table_name)
            if metadata:
                results.append(metadata)
        return results

    def _collect_single_pg_table(self, schema: str, table_name: str) -> Optional[TableMetadata]:
        try:
            metadata = TableMetadata(table_name=table_name, schema=schema)

            columns_sql = """
                SELECT
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    c.character_maximum_length,
                    c.numeric_precision,
                    c.numeric_scale,
                    pgd.description AS comment
                FROM information_schema.columns c
                LEFT JOIN pg_catalog.pg_statio_all_tables st
                    ON st.schemaname = c.table_schema
                    AND st.relname = c.table_name
                LEFT JOIN pg_catalog.pg_description pgd
                    ON pgd.objoid = st.relid
                    AND pgd.objsubid = c.ordinal_position
                WHERE c.table_schema = %s AND c.table_name = %s
                ORDER BY c.ordinal_position
            """
            columns = self.pg.query_all(columns_sql, (schema, table_name))
            for col in columns:
                if not self.config.should_sync_field(col["column_name"]):
                    continue
                column_info = ColumnInfo(
                    name=col["column_name"],
                    data_type=col["data_type"],
                    is_nullable=col["is_nullable"] == "YES",
                    default_value=col["column_default"],
                    character_maximum_length=col["character_maximum_length"],
                    numeric_precision=col["numeric_precision"],
                    numeric_scale=col["numeric_scale"],
                    comment=col["comment"],
                )
                metadata.columns.append(column_info)

            pk_sql = """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = %s
                  AND tc.table_name = %s
                  AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position
            """
            pk_rows = self.pg.query_all(pk_sql, (schema, table_name))
            metadata.primary_key_columns = [r["column_name"] for r in pk_rows]
            for col in metadata.columns:
                if col.name in metadata.primary_key_columns:
                    col.is_primary_key = True

            indexes_sql = """
                SELECT
                    i.relname AS index_name,
                    a.attname AS column_name,
                    ix.indisunique AS is_unique,
                    ix.indisprimary AS is_primary,
                    am.amname AS index_type,
                    array_position(ix.indkey, a.attnum) AS col_position
                FROM pg_index ix
                JOIN pg_class t ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_am am ON am.oid = i.relam
                JOIN pg_namespace ns ON ns.oid = t.relnamespace
                JOIN pg_attribute a
                    ON a.attrelid = t.oid
                    AND a.attnum = ANY(ix.indkey)
                WHERE ns.nspname = %s
                  AND t.relname = %s
                ORDER BY i.relname, array_position(ix.indkey, a.attnum)
            """
            idx_rows = self.pg.query_all(indexes_sql, (schema, table_name))
            idx_map: Dict[str, Dict[str, Any]] = {}
            for r in idx_rows:
                idx_name = r["index_name"]
                if idx_name not in idx_map:
                    idx_map[idx_name] = {
                        "name": idx_name,
                        "columns": [],
                        "is_unique": r["is_unique"],
                        "is_primary": r["is_primary"],
                        "index_type": r["index_type"],
                    }
                idx_map[idx_name]["columns"].append(r["column_name"])
            for idx_data in idx_map.values():
                idx_info = IndexInfo(**idx_data)
                metadata.indexes.append(idx_info)

            return metadata
        except Exception as e:
            print(f"Error collecting metadata for table {table_name}: {e}")
            return None

    def collect_mongo_collections(self) -> List[MongoCollectionMetadata]:
        db = self.mongo.get_database()
        collection_names = db.list_collection_names()
        results = []
        for name in collection_names:
            if not self.config.should_sync_collection(name):
                continue
            metadata = self._collect_single_mongo_collection(db, name)
            if metadata:
                results.append(metadata)
        return results

    def _collect_single_mongo_collection(
        self, db: Any, collection_name: str
    ) -> Optional[MongoCollectionMetadata]:
        try:
            collection = db[collection_name]
            options = db.command("listCollections", filter={"name": collection_name})
            cursor = options.get("cursor", {})
            first_batch = cursor.get("firstBatch", [])
            collection_options = {}
            if first_batch:
                collection_options = first_batch[0].get("options", {})

            is_ts = collection_options.get("timeseries", None) is not None
            metadata = MongoCollectionMetadata(
                collection_name=collection_name,
                is_time_series=is_ts,
            )

            if is_ts:
                ts_opts = collection_options["timeseries"]
                metadata.time_field = ts_opts.get("timeField")
                metadata.meta_field = ts_opts.get("metaField")
                metadata.granularity = ts_opts.get("granularity")
                metadata.bucket_max_span_seconds = ts_opts.get("bucketMaxSpanSeconds")

            fields_map: Dict[str, str] = {}
            try:
                pipeline = [
                    {"$project": {"arrayofkeyvalue": {"$objectToArray": "$$ROOT"}}},
                    {"$unwind": "$arrayofkeyvalue"},
                    {"$group": {
                        "_id": "$arrayofkeyvalue.k",
                        "types": {"$addToSet": {"$type": "$arrayofkeyvalue.v"}}
                    }},
                ]
                sample_size = 1000
                try:
                    pipeline.insert(0, {"$limit": sample_size})
                except Exception:
                    pass

                field_infos = list(collection.aggregate(pipeline, allowDiskUse=True))
                for fi in field_infos:
                    fname = fi["_id"]
                    if not self.config.should_sync_field(fname):
                        continue
                    ftypes = fi["types"]
                    bson_type = ftypes[0] if ftypes else "object"
                    field_info = MongoFieldInfo(
                        name=fname,
                        bson_type=bson_type,
                    )
                    if is_ts and fname == metadata.time_field:
                        field_info.is_time_field = True
                    if is_ts and fname == metadata.meta_field:
                        field_info.is_meta_field = True
                    metadata.fields.append(field_info)
                    fields_map[fname] = bson_type
            except Exception:
                pass

            if not metadata.fields and is_ts:
                time_f = metadata.time_field or "timestamp"
                metadata.fields.append(MongoFieldInfo(
                    name=time_f,
                    bson_type="date",
                    is_time_field=True,
                ))
                if metadata.meta_field:
                    metadata.fields.append(MongoFieldInfo(
                        name=metadata.meta_field,
                        bson_type="object",
                        is_meta_field=True,
                    ))

            try:
                indexes_info = collection.index_information()
                for idx_name, idx_data in indexes_info.items():
                    keys = idx_data.get("key", [])
                    key_list = [{k: v} for k, v in keys]
                    idx_info = MongoIndexInfo(
                        name=idx_name,
                        keys=key_list,
                        is_unique=idx_data.get("unique", False),
                        sparse=idx_data.get("sparse", False),
                    )
                    if "expireAfterSeconds" in idx_data:
                        idx_info.is_ttl = True
                        idx_info.expire_after_seconds = idx_data["expireAfterSeconds"]
                    if "partialFilterExpression" in idx_data:
                        idx_info.partial_filter_expression = idx_data["partialFilterExpression"]
                    metadata.indexes.append(idx_info)
            except Exception:
                pass

            bucket_ranges = self.config.sync_config.time_series_config.get("bucket_ranges", [])
            for br in bucket_ranges:
                metadata.buckets.append(TimeSeriesBucket(
                    from_date=br.get("from", ""),
                    to_date=br.get("to", ""),
                    bucket_size=br.get("bucket_size", ""),
                ))

            try:
                stats = db.command("collStats", collection_name)
                if stats.get("sharded", False):
                    metadata.is_sharded = True
                    shard_key_info = stats.get("shardKey", {})
                    metadata.shard_key = shard_key_info
            except Exception:
                pass

            return metadata
        except Exception as e:
            print(f"Error collecting metadata for collection {collection_name}: {e}")
            return None

    def collect_all(self) -> Dict[str, Any]:
        return {
            "postgresql_tables": [t.to_dict() for t in self.collect_pg_tables()],
            "mongodb_collections": [c.to_dict() for c in self.collect_mongo_collections()],
        }

    def export_to_file(self, output_path: str) -> None:
        data = self.collect_all()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
