from __future__ import annotations

import json
import math
import os
import re
import sqlite3

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:
    psycopg = None
    dict_row = None
    ConnectionPool = None
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import now_ms


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(text or "")]


class SQLiteStore:
    def __init__(self, path: str = "data/memory.db"):
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.path = path
        if not self.database_url:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # PostgreSQL/Supabase: reuse a small pool instead of opening a brand-new
        # TCP/TLS/database connection for every individual write. The Add path
        # performs several writes per message, so connection setup dominates
        # latency on hosted PostgreSQL.
        self._pool = None
        if self.database_url:
            if ConnectionPool is None:
                raise RuntimeError(
                    "DATABASE_URL is set but psycopg[pool] is not installed"
                )
            self._pool = ConnectionPool(
                self.database_url,
                kwargs={"row_factory": dict_row},
                min_size=1,
                max_size=4,
                open=True,
                timeout=10,
            )
        # 进程内 embedding cache：Add 写入时同步更新，Search 优先命中内存。
        self._embedding_cache: dict[str, dict[str, list[float]]] = defaultdict(dict)
        self._init_db()

    class _CompatConnection:
        def __init__(self, conn=None, postgres=False, pool_context=None):
            self._conn = conn
            self._postgres = postgres
            self._pool_context = pool_context

        def __enter__(self):
            if self._pool_context is not None:
                self._conn = self._pool_context.__enter__()
            else:
                self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._pool_context is not None:
                return self._pool_context.__exit__(exc_type, exc, tb)
            result = self._conn.__exit__(exc_type, exc, tb)
            self._conn.close()
            return result
        def execute(self, sql, params=None):
            if self._postgres:
                sql = sql.replace("?", "%s")
            return self._conn.execute(sql, params or ())
        def executescript(self, sql):
            if self._postgres:
                for statement in sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        self._conn.execute(statement)
            else:
                return self._conn.executescript(sql)

    def connect(self):
        if self.database_url:
            if psycopg is None:
                raise RuntimeError("DATABASE_URL is set but psycopg is not installed")
            if self._pool is not None:
                return self._CompatConnection(
                    postgres=True,
                    pool_context=self._pool.connection(),
                )
            conn = psycopg.connect(self.database_url, row_factory=dict_row)
            return self._CompatConnection(conn, postgres=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return self._CompatConnection(conn, postgres=False)

    def _init_db(self):
        if self.database_url:
            self._init_postgres()
            return
        with self._lock, self.connect() as c:
            c.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS request_log (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_memories (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                source_turn_id TEXT
            );

            CREATE TABLE IF NOT EXISTS atomic_facts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                valid_from INTEGER NOT NULL,
                valid_to INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                supersedes_id TEXT,
                source TEXT NOT NULL DEFAULT 'user',
                conflict_status TEXT NOT NULL DEFAULT 'none',
                conflict_group_id TEXT,
                source_raw_id TEXT,
                source_session_id TEXT,
                source_turn_id TEXT
            );

            CREATE TABLE IF NOT EXISTS conflict_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                old_fact_id TEXT,
                new_fact_id TEXT,
                old_object TEXT,
                new_object TEXT,
                resolution TEXT NOT NULL,
                reason TEXT,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_relations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                source_raw_id TEXT,
                source_session_id TEXT,
                source_turn_id TEXT
            );

            CREATE TABLE IF NOT EXISTS timeline_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                event_start INTEGER,
                event_end INTEGER,
                temporal_text TEXT,
                source_raw_id TEXT,
                source_session_id TEXT,
                source_turn_id TEXT
            );

            CREATE TABLE IF NOT EXISTS rule_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                rule TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                source_raw_id TEXT,
                source_session_id TEXT,
                source_turn_id TEXT
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                source_raw_id TEXT,
                source_session_id TEXT,
                source_turn_id TEXT,
                PRIMARY KEY(user_id, key)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                vector TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_raw_user_time
                ON raw_memories(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_fact_user_status
                ON atomic_facts(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_fact_key
                ON atomic_facts(user_id, subject, predicate);
            CREATE INDEX IF NOT EXISTS idx_rel_user
                ON entity_relations(user_id);
            CREATE INDEX IF NOT EXISTS idx_event_user_time
                ON timeline_events(user_id, timestamp);
            """)
            self._ensure_sqlite_columns(c)

    def _ensure_sqlite_columns(self, c):
        tables = {
            "raw_memories": {
                "source_turn_id": "TEXT",
            },
            "atomic_facts": {
                "source": "TEXT NOT NULL DEFAULT 'user'",
                "conflict_status": "TEXT NOT NULL DEFAULT 'none'",
                "conflict_group_id": "TEXT",
                "source_raw_id": "TEXT",
                "source_session_id": "TEXT",
                "source_turn_id": "TEXT",
            },
            "entity_relations": {
                "source_raw_id": "TEXT", "source_session_id": "TEXT", "source_turn_id": "TEXT",
            },
            "timeline_events": {
                "source_raw_id": "TEXT", "source_session_id": "TEXT", "source_turn_id": "TEXT",
            },
            "rule_memories": {
                "source_raw_id": "TEXT", "source_session_id": "TEXT", "source_turn_id": "TEXT",
            },
            "user_profiles": {
                "source_raw_id": "TEXT", "source_session_id": "TEXT", "source_turn_id": "TEXT",
            },
        }
        for table, wanted in tables.items():
            existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, definition in wanted.items():
                if name not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _init_postgres(self):
        statements = [
            "CREATE TABLE IF NOT EXISTS request_log (request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at BIGINT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS raw_memories (id TEXT PRIMARY KEY, request_id TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, chunk_index INTEGER NOT NULL DEFAULT 0)",
            "CREATE TABLE IF NOT EXISTS atomic_facts (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, fingerprint TEXT NOT NULL, valid_from BIGINT NOT NULL, valid_to BIGINT, status TEXT NOT NULL DEFAULT 'active', supersedes_id TEXT, source TEXT NOT NULL DEFAULT 'user', conflict_status TEXT NOT NULL DEFAULT 'none', conflict_group_id TEXT)",
            "CREATE TABLE IF NOT EXISTS conflict_logs (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, predicate TEXT NOT NULL, old_fact_id TEXT, new_fact_id TEXT, old_object TEXT, new_object TEXT, resolution TEXT NOT NULL, reason TEXT, created_at BIGINT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS entity_relations (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, fingerprint TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS timeline_events (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, event TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, fingerprint TEXT NOT NULL, event_start BIGINT, event_end BIGINT, temporal_text TEXT)",
            "CREATE TABLE IF NOT EXISTS rule_memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, rule TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, fingerprint TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS user_profiles (user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, content TEXT NOT NULL, timestamp BIGINT NOT NULL, PRIMARY KEY(user_id, key))",
            "CREATE TABLE IF NOT EXISTS embeddings (memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, vector TEXT NOT NULL)",
            "CREATE INDEX IF NOT EXISTS idx_raw_user_time ON raw_memories(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_fact_user_status ON atomic_facts(user_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_fact_key ON atomic_facts(user_id, subject, predicate)",
            "CREATE INDEX IF NOT EXISTS idx_rel_user ON entity_relations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_user_time ON timeline_events(user_id, timestamp)"
        ]
        with self._lock, self.connect() as c:
            for statement in statements:
                c.execute(statement)
            c.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'user'")
            c.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS conflict_status TEXT NOT NULL DEFAULT 'none'")
            c.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS conflict_group_id TEXT")
            for name in ("source_raw_id", "source_session_id", "source_turn_id"):
                c.execute(f"ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS {name} TEXT")
            for table in ("raw_memories", "entity_relations", "timeline_events", "rule_memories", "user_profiles"):
                for name in ("source_raw_id", "source_session_id", "source_turn_id"):
                    try:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT")
                    except Exception:
                        pass

    def claim_request(self, request_id: str, user_id: str) -> bool:
        """Atomically claim a request_id for processing.

        The previous request_seen() -> register_request() sequence had a
        check-then-act race: two concurrent Add calls could both pass the
        check and duplicate the same request. INSERT OR IGNORE makes the
        claim itself atomic at the SQLite constraint level.
        """
        with self._lock, self.connect() as c:
            cur = c.execute(
                "INSERT INTO request_log(request_id,user_id,created_at) VALUES(?,?,?) ON CONFLICT(request_id) DO NOTHING",
                (request_id, user_id, now_ms())
            )
            return cur.rowcount == 1

    def release_request(self, request_id: str):
        """Release a failed request so a retry can safely process it."""
        with self._lock, self.connect() as c:
            c.execute("DELETE FROM request_log WHERE request_id=?", (request_id,))

    def insert_raw(self, row: dict[str, Any]):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO raw_memories
                (id,request_id,user_id,session_id,role,content,timestamp,chunk_index,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                row["id"], row["request_id"], row["user_id"], row["session_id"],
                row["role"], row["content"], row["timestamp"], row.get("chunk_index", 0), row.get("source_turn_id")
            ))

    def insert_fact(self, f):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO atomic_facts
                (id,user_id,subject,predicate,object,content,timestamp,fingerprint,
                 valid_from,valid_to,status,supersedes_id,source,conflict_status,conflict_group_id,
                 source_raw_id,source_session_id,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                f.id, f.user_id, f.subject, f.predicate, f.object, f.content,
                f.timestamp, f.fingerprint, f.valid_from, f.valid_to,
                f.status, getattr(f, "supersedes_id", None),
                getattr(f, "source", "user"), getattr(f, "conflict_status", "none"),
                getattr(f, "conflict_group_id", None), getattr(f, "source_raw_id", None),
                getattr(f, "source_session_id", None), getattr(f, "source_turn_id", None)
            ))

    def insert_conflict_log(self, row: dict[str, Any]):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO conflict_logs
                (id,user_id,predicate,old_fact_id,new_fact_id,old_object,new_object,resolution,reason,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (
                row["id"], row["user_id"], row["predicate"], row.get("old_fact_id"),
                row.get("new_fact_id"), row.get("old_object"), row.get("new_object"),
                row["resolution"], row.get("reason"), row.get("created_at", now_ms()),
            ))

    def conflict_logs(self, user_id: str):
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM conflict_logs WHERE user_id=? ORDER BY created_at DESC",
                (user_id,)
            ).fetchall()]

    def find_same_relation(self, user_id, subject, predicate, object_):
        with self._lock, self.connect() as c:
            return c.execute("""
                SELECT * FROM entity_relations
                WHERE user_id=? AND subject=? AND predicate=? AND object=?
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, subject, predicate, object_)).fetchone()

    def find_same_rule(self, user_id, rule):
        with self._lock, self.connect() as c:
            return c.execute("""
                SELECT * FROM rule_memories
                WHERE user_id=? AND rule=?
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, rule)).fetchone()

    def insert_relation(self, r):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO entity_relations
                (id,user_id,subject,predicate,object,content,timestamp,fingerprint,source_raw_id,source_session_id,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """, (r.id,r.user_id,r.subject,r.predicate,r.object,r.content,
                  r.timestamp,r.fingerprint,getattr(r, "source_raw_id", None),getattr(r, "source_session_id", None),getattr(r, "source_turn_id", None)))

    def insert_event(self, e):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO timeline_events
                (id,user_id,event,content,timestamp,fingerprint,event_start,event_end,temporal_text,source_raw_id,source_session_id,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """, (e.id,e.user_id,e.event,e.content,e.timestamp,e.fingerprint,
                  getattr(e, "event_start", None), getattr(e, "event_end", None),
                  getattr(e, "temporal_text", ""), getattr(e, "source_raw_id", None),
                  getattr(e, "source_session_id", None), getattr(e, "source_turn_id", None)))

    def insert_rule(self, r):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO rule_memories
                (id,user_id,rule,content,timestamp,fingerprint,source_raw_id,source_session_id,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (r.id,r.user_id,r.rule,r.content,r.timestamp,r.fingerprint,getattr(r, "source_raw_id", None),getattr(r, "source_session_id", None),getattr(r, "source_turn_id", None)))

    def upsert_profile(self, p):
        with self._lock, self.connect() as c:
            c.execute("""
                INSERT INTO user_profiles(user_id,key,value,content,timestamp,source_raw_id,source_session_id,source_turn_id)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id,key) DO UPDATE SET
                    value=excluded.value,
                    content=excluded.content,
                    timestamp=excluded.timestamp
            """, (p.user_id,p.key,p.value,p.content,p.timestamp,getattr(p, "source_raw_id", None),getattr(p, "source_session_id", None),getattr(p, "source_turn_id", None)))

    def embed(self, memory_id: str, user_id: str, vector: list[float]):
        # 先更新内存 cache，再持久化；同一进程内 Add -> Search 立即可见。
        with self._lock:
            self._embedding_cache[user_id][memory_id] = list(vector)
            with self.connect() as c:
                c.execute("""
                    INSERT INTO embeddings(memory_id,user_id,vector)
                    VALUES(?,?,?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        user_id=EXCLUDED.user_id,
                        vector=EXCLUDED.vector
                """, (memory_id,user_id,json.dumps(vector,separators=(",",":"))))

    def all_raw(self, user_id: str, session_id: Optional[str] = None):
        sql = "SELECT * FROM raw_memories WHERE user_id=?"
        args = [user_id]
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        sql += " ORDER BY timestamp DESC"
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(sql,args).fetchall()]

    def active_facts(self, user_id: str, include_history=False):
        sql = "SELECT * FROM atomic_facts WHERE user_id=?"
        args = [user_id]
        if not include_history:
            sql += " AND status='active'"
        sql += " ORDER BY timestamp DESC"
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(sql,args).fetchall()]

    def relations(self, user_id: str):
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM entity_relations WHERE user_id=? ORDER BY timestamp DESC",
                (user_id,)
            ).fetchall()]

    def events(self, user_id: str, start_time=None, end_time=None):
        sql = "SELECT * FROM timeline_events WHERE user_id=?"
        args = [user_id]
        if start_time is not None:
            sql += " AND timestamp>=?"
            args.append(start_time)
        if end_time is not None:
            sql += " AND timestamp<=?"
            args.append(end_time)
        sql += " ORDER BY timestamp DESC"
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(sql,args).fetchall()]

    def rules(self, user_id: str):
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM rule_memories WHERE user_id=? ORDER BY timestamp DESC",
                (user_id,)
            ).fetchall()]

    def profiles(self, user_id: str):
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM user_profiles WHERE user_id=?",
                (user_id,)
            ).fetchall()]

    def embeddings(self, user_id: str):
        with self._lock, self.connect() as c:
            rows = c.execute(
                "SELECT memory_id, vector FROM embeddings WHERE user_id=?",
                (user_id,)
            ).fetchall()
        return [(r["memory_id"], json.loads(r["vector"])) for r in rows]

    def embeddings_by_ids(self, user_id: str, ids: Iterable[str]):
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {}

        # 热路径：已经在本进程 Add/前一次 Search 中加载过的向量直接返回。
        with self._lock:
            cached = self._embedding_cache.setdefault(user_id, {})
            result = {mid: cached[mid] for mid in ids if mid in cached}
            missing = [mid for mid in ids if mid not in cached]

            if not missing:
                return result

            # 冷启动/外部写入场景只查询缺失 ID，并把结果回填 cache。
            marks = ",".join("?" * len(missing))
            with self.connect() as c:
                rows = c.execute(
                    f"SELECT memory_id, vector FROM embeddings WHERE user_id=? AND memory_id IN ({marks})",
                    [user_id, *missing]
                ).fetchall()

            for row in rows:
                vector = json.loads(row["vector"])
                cached[row["memory_id"]] = vector
                result[row["memory_id"]] = vector
            return result

    def raw_by_ids(self, user_id: str, ids: Iterable[str]):
        ids = list(ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._lock, self.connect() as c:
            return [dict(r) for r in c.execute(
                f"SELECT * FROM raw_memories WHERE user_id=? AND id IN ({marks})",
                [user_id,*ids]
            ).fetchall()]

    def update_fact_status(self, fact_id: str, status: str, valid_to=None):
        with self._lock, self.connect() as c:
            c.execute(
                "UPDATE atomic_facts SET status=?, valid_to=? WHERE id=?",
                (status, valid_to, fact_id)
            )

    def find_same_fact(self, user_id, subject, predicate, object_):
        with self._lock, self.connect() as c:
            return c.execute("""
                SELECT * FROM atomic_facts
                WHERE user_id=? AND subject=? AND predicate=? AND object=?
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,subject,predicate,object_)).fetchone()

    def find_current_fact(self, user_id, subject, predicate):
        with self._lock, self.connect() as c:
            return c.execute("""
                SELECT * FROM atomic_facts
                WHERE user_id=? AND subject=? AND predicate=? AND status='active'
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,subject,predicate)).fetchone()


class BM25:
    def __init__(self):
        self.docs = []
        self.doc_tokens = []
        self.df = Counter()
        self.avgdl = 0.0

    def fit(self, docs: list[dict]):
        self.docs = docs
        self.doc_tokens = [tokenize(d["content"]) for d in docs]
        self.df = Counter()
        for ts in self.doc_tokens:
            for t in set(ts):
                self.df[t] += 1
        self.avgdl = sum(map(len,self.doc_tokens)) / max(1,len(self.doc_tokens))

    def search(self, query: str, top_k: int = 30):
        q = tokenize(query)
        if not q or not self.docs:
            return []
        N = len(self.docs)
        scores = []
        k1, b = 1.5, 0.75
        for i, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            dl = len(tokens)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = tf[term] + k1 * (1 - b + b * dl / max(self.avgdl,1))
                s += idf * tf[term] * (k1 + 1) / denom
            if s > 0:
                scores.append((i,s))
        scores.sort(key=lambda x:x[1], reverse=True)
        return [(self.docs[i], score) for i,score in scores[:top_k]]
