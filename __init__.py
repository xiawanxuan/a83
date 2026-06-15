"""
智能制造混合数据库数据中台 - PostgreSQL 与 MongoDB 双向同步工具

模块结构:
    - db_adapters: 双数据库连接适配器
    - config_manager: 同步黑白名单配置管理器
    - metadata_collector: Mongo 时序集合元数据采集器
    - diff_engine: 字段索引分片差异对比引擎
    - sync_executor: 同步脚本生成执行器
    - rollback_manager: 全流程变更日志回滚模块
"""

__version__ = "1.0.0"
__author__ = "Smart Manufacturing Data Platform Team"

from .config_manager import ConfigManager, SyncConfig, TableCollectionMapping
from .db_adapters import PostgreSQLAdapter, MongoDBAdapter
from .metadata_collector import (
    MetadataCollector,
    TableMetadata,
    ColumnInfo,
    IndexInfo,
    MongoCollectionMetadata,
    MongoFieldInfo,
    MongoIndexInfo,
    TimeSeriesBucket,
)
from .diff_engine import (
    DiffEngine,
    TableCollectionDiff,
    FieldDiff,
    IndexDiff,
    ShardingDiff,
)
from .sync_executor import (
    SyncExecutor,
    SyncScriptGenerator,
    SyncPlan,
    SyncOperation,
)
from .rollback_manager import (
    ChangeLogger,
    RollbackManager,
    SyncRunLog,
    PhaseLogEntry,
    SyncPhase,
    SyncStatus,
)

__all__ = [
    "ConfigManager",
    "SyncConfig",
    "TableCollectionMapping",
    "PostgreSQLAdapter",
    "MongoDBAdapter",
    "MetadataCollector",
    "TableMetadata",
    "ColumnInfo",
    "IndexInfo",
    "MongoCollectionMetadata",
    "MongoFieldInfo",
    "MongoIndexInfo",
    "TimeSeriesBucket",
    "DiffEngine",
    "TableCollectionDiff",
    "FieldDiff",
    "IndexDiff",
    "ShardingDiff",
    "SyncExecutor",
    "SyncScriptGenerator",
    "SyncPlan",
    "SyncOperation",
    "ChangeLogger",
    "RollbackManager",
    "SyncRunLog",
    "PhaseLogEntry",
    "SyncPhase",
    "SyncStatus",
]
