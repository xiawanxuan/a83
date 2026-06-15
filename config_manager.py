import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TableCollectionMapping:
    pg_table: str
    mongo_collection: str
    sync_fields: bool = True
    sync_indexes: bool = True
    sync_sharding: bool = True


@dataclass
class SyncConfig:
    sync_direction: str = "bidirectional"
    run_mode: str = "preview"
    auto_rollback_on_failure: bool = True
    type_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    time_series_config: Dict[str, Any] = field(default_factory=dict)
    index_rules: Dict[str, Any] = field(default_factory=dict)
    blacklist: Dict[str, List[str]] = field(default_factory=dict)
    whitelist: Dict[str, Any] = field(default_factory=dict)
    table_collection_mappings: List[TableCollectionMapping] = field(default_factory=list)
    logging_config: Dict[str, Any] = field(default_factory=dict)
    storage_validation: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    def __init__(self, config_path: str = "config/sync_rules.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.sync_config = self._parse_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_config(self) -> SyncConfig:
        mappings = []
        for m in self.config.get("table_collection_mappings", []):
            mappings.append(TableCollectionMapping(
                pg_table=m.get("pg_table", ""),
                mongo_collection=m.get("mongo_collection", ""),
                sync_fields=m.get("sync_fields", True),
                sync_indexes=m.get("sync_indexes", True),
                sync_sharding=m.get("sync_sharding", True),
            ))

        return SyncConfig(
            sync_direction=self.config.get("sync_direction", "bidirectional"),
            run_mode=self.config.get("run_mode", "preview"),
            auto_rollback_on_failure=self.config.get("auto_rollback_on_failure", True),
            type_mapping=self.config.get("type_mapping", {}),
            time_series_config=self.config.get("time_series_config", {}),
            index_rules=self.config.get("index_rules", {}),
            blacklist=self.config.get("blacklist", {"tables": [], "fields": []}),
            whitelist=self.config.get("whitelist", {"enabled": False, "tables": [], "collections": []}),
            table_collection_mappings=mappings,
            logging_config=self.config.get("logging", {}),
            storage_validation=self.config.get("storage_validation", {}),
        )

    def get_sync_config(self) -> SyncConfig:
        return self.sync_config

    def is_preview_mode(self) -> bool:
        return self.sync_config.run_mode.lower() == "preview"

    def is_execute_mode(self) -> bool:
        return self.sync_config.run_mode.lower() == "execute"

    def is_bidirectional(self) -> bool:
        return self.sync_config.sync_direction.lower() == "bidirectional"

    def is_pg_to_mongo_only(self) -> bool:
        return self.sync_config.sync_direction.lower() == "pg_to_mongo"

    def is_mongo_to_pg_only(self) -> bool:
        return self.sync_config.sync_direction.lower() == "mongo_to_pg"

    def is_table_blacklisted(self, table_name: str) -> bool:
        blacklist_tables = self.sync_config.blacklist.get("tables", [])
        return table_name in blacklist_tables

    def is_collection_blacklisted(self, collection_name: str) -> bool:
        blacklist_tables = self.sync_config.blacklist.get("tables", [])
        return collection_name in blacklist_tables

    def is_field_blacklisted(self, field_name: str) -> bool:
        blacklist_fields = self.sync_config.blacklist.get("fields", [])
        if field_name in blacklist_fields:
            return True
        for pattern in self.sync_config.blacklist.get("fields", []):
            try:
                if re.match(pattern, field_name):
                    return True
            except re.error:
                pass
        return False

    def is_table_whitelisted(self, table_name: str) -> bool:
        if not self.sync_config.whitelist.get("enabled", False):
            return True
        whitelist_tables = self.sync_config.whitelist.get("tables", [])
        return table_name in whitelist_tables

    def is_collection_whitelisted(self, collection_name: str) -> bool:
        if not self.sync_config.whitelist.get("enabled", False):
            return True
        whitelist_collections = self.sync_config.whitelist.get("collections", [])
        return collection_name in whitelist_collections

    def should_sync_table(self, table_name: str) -> bool:
        if self.is_table_blacklisted(table_name):
            return False
        return self.is_table_whitelisted(table_name)

    def should_sync_collection(self, collection_name: str) -> bool:
        if self.is_collection_blacklisted(collection_name):
            return False
        return self.is_collection_whitelisted(collection_name)

    def should_sync_field(self, field_name: str) -> bool:
        patterns = self.sync_config.index_rules.get("exclude_field_patterns", [])
        for pattern in patterns:
            try:
                if re.match(pattern, field_name):
                    return False
            except re.error:
                if pattern == field_name:
                    return False
        return not self.is_field_blacklisted(field_name)

    def get_pg_to_mongo_type(self, pg_type: str) -> str:
        mapping = self.sync_config.type_mapping.get("postgresql_to_mongodb", {})
        pg_type_lower = pg_type.lower()
        if pg_type_lower in mapping:
            return mapping[pg_type_lower]
        if pg_type_lower.startswith("character varying"):
            return mapping.get("character varying", "string")
        if pg_type_lower.startswith("timestamp"):
            if "time zone" in pg_type_lower:
                return mapping.get("timestamp with time zone", "date")
            return mapping.get("timestamp without time zone", "date")
        return "string"

    def get_mongo_to_pg_type(self, mongo_type: str) -> str:
        mapping = self.sync_config.type_mapping.get("mongodb_to_postgresql", {})
        mongo_type_lower = mongo_type.lower()
        return mapping.get(mongo_type_lower, "varchar(255)")

    def get_mapping_for_pg_table(self, table_name: str) -> Optional[TableCollectionMapping]:
        for m in self.sync_config.table_collection_mappings:
            if m.pg_table == table_name:
                return m
        return None

    def get_mapping_for_mongo_collection(self, collection_name: str) -> Optional[TableCollectionMapping]:
        for m in self.sync_config.table_collection_mappings:
            if m.mongo_collection == collection_name:
                return m
        return None

    def get_time_field(self) -> str:
        return self.sync_config.time_series_config.get("time_field", "timestamp")

    def get_meta_field(self) -> str:
        return self.sync_config.time_series_config.get("meta_field", "metadata")

    def get_granularity(self) -> str:
        return self.sync_config.time_series_config.get("granularity", "seconds")

    def get_recommended_indexes(self) -> List[Dict[str, Any]]:
        return self.sync_config.index_rules.get("recommended_indexes", [])

    def reload(self) -> None:
        self.config = self._load_config()
        self.sync_config = self._parse_config()

    def validate_config(self) -> List[str]:
        errors = []

        if self.sync_config.sync_direction not in ("bidirectional", "pg_to_mongo", "mongo_to_pg"):
            errors.append(f"Invalid sync_direction: {self.sync_config.sync_direction}")

        if self.sync_config.run_mode not in ("preview", "execute"):
            errors.append(f"Invalid run_mode: {self.sync_config.run_mode}")

        if not self.sync_config.table_collection_mappings:
            errors.append("No table_collection_mappings configured")

        for i, mapping in enumerate(self.sync_config.table_collection_mappings):
            if not mapping.pg_table:
                errors.append(f"Mapping #{i}: pg_table is empty")
            if not mapping.mongo_collection:
                errors.append(f"Mapping #{i}: mongo_collection is empty")

        ts_config = self.sync_config.time_series_config
        if ts_config:
            if not ts_config.get("time_field"):
                errors.append("time_series_config.time_field is required")

        return errors

    def get_chunk_size_bytes(self) -> int:
        default = 64 * 1024 * 1024
        raw = self.sync_config.storage_validation.get("default_chunk_size_mb", 64)
        try:
            return int(raw) * 1024 * 1024
        except (TypeError, ValueError):
            return default

    def is_storage_validation_enabled(self) -> bool:
        return bool(self.sync_config.storage_validation.get("enabled", True))

    def get_disk_usage_warning_threshold(self) -> float:
        return float(self.sync_config.storage_validation.get("disk_usage_warning_threshold", 0.80))

    def get_disk_usage_block_threshold(self) -> float:
        return float(self.sync_config.storage_validation.get("disk_usage_block_threshold", 0.95))

    def get_index_overhead_multiplier(self) -> float:
        return float(self.sync_config.storage_validation.get("index_overhead_multiplier", 1.35))

    def get_migration_overhead_multiplier(self) -> float:
        return float(self.sync_config.storage_validation.get("migration_overhead_multiplier", 1.5))

    def get_per_collection_max_size_bytes(self) -> Optional[int]:
        raw = self.sync_config.storage_validation.get("per_collection_max_size_mb")
        if raw is None:
            return None
        try:
            return int(raw) * 1024 * 1024
        except (TypeError, ValueError):
            return None

    def get_enforcement_mode(self) -> str:
        return self.sync_config.storage_validation.get("enforcement_mode", "block").lower()

    def get_allow_override(self) -> bool:
        return bool(self.sync_config.storage_validation.get("allow_override_arg", True))

    def get_storage_capacity_overrides(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        raw = self.sync_config.storage_validation.get("shard_capacity_overrides_mb", {})
        if isinstance(raw, dict):
            for shard_name, size_mb in raw.items():
                try:
                    result[str(shard_name)] = int(size_mb) * 1024 * 1024
                except (TypeError, ValueError):
                    continue
        return result
