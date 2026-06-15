import configparser
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
from pymongo.database import Database


class PostgreSQLAdapter:
    def __init__(self, config_path: str = "config/database.ini", section: str = "postgresql"):
        self.config_path = config_path
        self.section = section
        self.config = self._load_config()
        self._connection = None

    def _load_config(self) -> Dict[str, str]:
        parser = configparser.ConfigParser()
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        parser.read(self.config_path, encoding="utf-8")
        if not parser.has_section(self.section):
            raise ValueError(f"Section {self.section} not found in config")
        return dict(parser.items(self.section))

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.get("host", "127.0.0.1"),
                port=int(self.config.get("port", "5432")),
                database=self.config.get("database", ""),
                user=self.config.get("user", ""),
                password=self.config.get("password", ""),
                connect_timeout=int(self.config.get("connect_timeout", "30")),
            )
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    @contextmanager
    def cursor(self, cursor_factory=RealDictCursor) -> Generator[Any, None, None]:
        with self.connection() as conn:
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
            finally:
                cur.close()

    def test_connection(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1 AS test")
                result = cur.fetchone()
                return result is not None and result["test"] == 1
        except Exception:
            return False

    def get_schema(self) -> str:
        return self.config.get("schema", "public")

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.cursor() as cur:
            cur.execute(sql, params)

    def execute_script(self, sql_script: str) -> None:
        with self.connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_script)
            finally:
                cur.close()

    def query_all(self, sql: str, params: tuple = ()) -> list:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


class MongoDBAdapter:
    def __init__(self, config_path: str = "config/database.ini", section: str = "mongodb"):
        self.config_path = config_path
        self.section = section
        self.config = self._load_config()
        self._client: Optional[MongoClient] = None

    def _load_config(self) -> Dict[str, str]:
        parser = configparser.ConfigParser()
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        parser.read(self.config_path, encoding="utf-8")
        if not parser.has_section(self.section):
            raise ValueError(f"Section {self.section} not found in config")
        return dict(parser.items(self.section))

    def _build_connection_string(self) -> str:
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", "27017")
        user = self.config.get("user", "")
        password = self.config.get("password", "")
        auth_source = self.config.get("auth_source", "admin")
        replica_set = self.config.get("replica_set", "")

        if user and password:
            creds = f"{user}:{password}@"
        else:
            creds = ""

        base = f"mongodb://{creds}{host}:{port}/"
        options = []
        if auth_source:
            options.append(f"authSource={auth_source}")
        if replica_set:
            options.append(f"replicaSet={replica_set}")

        connect_timeout = self.config.get("connect_timeout_ms", "30000")
        options.append(f"connectTimeoutMS={connect_timeout}")

        if options:
            return f"{base}?{'&'.join(options)}"
        return base

    def connect(self) -> MongoClient:
        if self._client is None:
            uri = self._build_connection_string()
            self._client = MongoClient(uri)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @contextmanager
    def client_context(self) -> Generator[MongoClient, None, None]:
        client = self.connect()
        try:
            yield client
        finally:
            pass

    def get_database(self) -> Database:
        client = self.connect()
        db_name = self.config.get("database", "")
        if not db_name:
            raise ValueError("MongoDB database name not configured")
        return client[db_name]

    def test_connection(self) -> bool:
        try:
            client = self.connect()
            client.admin.command("ping")
            return True
        except Exception:
            return False

    def execute_mongoshell_script(self, script: str) -> Any:
        db = self.get_database()
        clean_script = script.strip()
        if clean_script.startswith("db."):
            return self._run_mongoshell_command(db, clean_script)
        return db.command("eval", script)

    def _run_mongoshell_command(self, db, script: str) -> Any:
        body = script[3:]
        dot_idx = body.find(".")
        if dot_idx == -1:
            return db.command("eval", script)

        coll_name = body[:dot_idx]
        rest = body[dot_idx + 1:]

        paren_idx = rest.find("(")
        if paren_idx == -1:
            return db.command("eval", script)

        method_name = rest[:paren_idx]
        args_str = rest[paren_idx:]

        collection = db[coll_name]
        return self._call_collection_method(collection, method_name, args_str)

    def _call_collection_method(self, collection, method_name: str, args_str: str):
        args_str = args_str.strip()
        if args_str.startswith("("):
            args_str = args_str[1:]
        if args_str.endswith(";"):
            args_str = args_str[:-1]
        if args_str.endswith(")"):
            args_str = args_str[:-1]
        args_str = args_str.strip()

        if method_name == "createCollection":
            return self._create_collection(collection.database, collection.name, args_str)
        if method_name == "createIndex":
            return self._create_index(collection, args_str)
        if method_name == "dropIndex":
            return self._drop_index(collection, args_str)
        if method_name == "drop":
            return collection.drop()
        if method_name == "updateMany":
            return self._update_many(collection, args_str)
        if method_name == "insertMany":
            return self._insert_many(collection, args_str)
        if method_name == "deleteMany":
            return self._delete_many(collection, args_str)
        if method_name == "find":
            return list(collection.find())

        return collection.database.command("eval", f"db.{collection.name}.{method_name}({args_str})")

    def _create_collection(self, db, name: str, args_str: str):
        import ast
        try:
            options = self._safe_eval_dict(args_str)
            return db.create_collection(name, **options)
        except Exception:
            return db.command("eval", f"db.createCollection('{name}', {args_str})")

    def _create_index(self, collection, args_str: str):
        try:
            keys_str, opts_str = self._split_top_level_comma(args_str)
            keys = self._parse_mongo_keys(keys_str)
            options = self._parse_mongo_options(opts_str) if opts_str else {}
            return collection.create_index(list(keys.items()), **options)
        except Exception:
            return collection.database.command(
                "eval", f"db.{collection.name}.createIndex({args_str})"
            )

    def _drop_index(self, collection, args_str: str):
        name = args_str.strip().strip("'\"")
        if name.startswith("'") and name.endswith("'"):
            name = name[1:-1]
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        return collection.drop_index(name)

    def _update_many(self, collection, args_str: str):
        try:
            parts = self._split_top_level_comma(args_str)
            if len(parts) >= 2:
                filter_dict = self._safe_eval_dict(parts[0])
                update_dict = self._safe_eval_dict(parts[1])
                return collection.update_many(filter_dict, update_dict)
        except Exception:
            pass
        return collection.database.command(
            "eval", f"db.{collection.name}.updateMany({args_str})"
        )

    def _insert_many(self, collection, args_str: str):
        try:
            docs = self._safe_eval_list(args_str)
            return collection.insert_many(docs)
        except Exception:
            return collection.database.command(
                "eval", f"db.{collection.name}.insertMany({args_str})"
            )

    def _delete_many(self, collection, args_str: str):
        try:
            filter_dict = self._safe_eval_dict(args_str)
            return collection.delete_many(filter_dict)
        except Exception:
            return collection.database.command(
                "eval", f"db.{collection.name}.deleteMany({args_str})"
            )

    def _split_top_level_comma(self, s: str):
        depth = 0
        parts = []
        current = []
        in_str = False
        str_char = ""
        for ch in s:
            if in_str:
                current.append(ch)
                if ch == str_char:
                    in_str = False
                continue
            if ch in ("'", '"'):
                in_str = True
                str_char = ch
                current.append(ch)
                continue
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        if len(parts) == 1:
            parts.append("")
        return parts

    def _parse_mongo_keys(self, s: str):
        result = {}
        content = s.strip()
        if content.startswith("{"):
            content = content[1:]
        if content.endswith("}"):
            content = content[:-1]
        pairs = self._split_top_level_comma(content)
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            kv = pair.split(":", 1)
            if len(kv) != 2:
                continue
            k = kv[0].strip().strip("'\"")
            v = kv[1].strip()
            try:
                result[k] = int(v)
            except ValueError:
                result[k] = 1 if v in ("1", "true", "True") else -1
        return result

    def _parse_mongo_options(self, s: str):
        result = {}
        content = s.strip()
        if content.startswith("{"):
            content = content[1:]
        if content.endswith("}"):
            content = content[:-1]
        pairs = self._split_top_level_comma(content)
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            kv = pair.split(":", 1)
            if len(kv) != 2:
                continue
            k = kv[0].strip().strip("'\"")
            v = kv[1].strip()
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            else:
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v.strip("'\"")
        return result

    def _safe_eval_dict(self, s: str):
        import json
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        s_js = s
        s_js = s_js.replace("new Date()", '""')
        s_js = s_js.replace("ISODate(", '"').replace(")", '"')
        s_js = s_js.replace("NumberInt(", "").replace(")", "")
        s_js = s_js.replace("NumberLong(", "").replace(")", "")
        s_js = s_js.replace("ObjectId(", '"').replace(")", '"')
        try:
            return json.loads(s_js)
        except Exception:
            return {}

    def _safe_eval_list(self, s: str):
        import json
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            return []
