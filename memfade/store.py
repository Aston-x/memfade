"""SQLite 存储层。

单文件、零外部服务。所有检索路径共用同一条作用域过滤：

    scope_key = 当前会话  OR  cross_scope = 1

注意这里**没有**「找不到就退回全局」的兜底。宁可这条消息没有记忆可用，
也不能把别的会话的内容漏进来——这正是很多聊天机器人记忆越用越乱的
真实成因（同一套记忆在私聊里沉淀，跑到群里被读出来）。
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import time
from typing import Any, Iterable, Optional, Sequence

from .types import Memory, MemoryKind, MemoryStatus

try:  # numpy 可选：装了走矩阵运算，没装走纯 Python
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key     TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    summary       TEXT    NOT NULL DEFAULT '',
    importance    REAL    NOT NULL DEFAULT 0.5,
    strength      REAL    NOT NULL DEFAULT 1.0,
    anchor_at     REAL    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'active',
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL,
    last_recalled_at REAL,
    recall_count  INTEGER NOT NULL DEFAULT 0,
    pinned        INTEGER NOT NULL DEFAULT 0,
    cross_scope   INTEGER NOT NULL DEFAULT 0,
    subject       TEXT    NOT NULL DEFAULT '',
    predicate     TEXT    NOT NULL DEFAULT '',
    object        TEXT    NOT NULL DEFAULT '',
    source        TEXT    NOT NULL DEFAULT '',
    superseded_by INTEGER,
    content_hash  TEXT    NOT NULL DEFAULT '',
    fts_tokens    TEXT    NOT NULL DEFAULT '',
    meta          TEXT    NOT NULL DEFAULT '{}',
    embedding     BLOB
);

CREATE INDEX IF NOT EXISTS idx_mem_scope_status ON memories(scope_key, status);
CREATE INDEX IF NOT EXISTS idx_mem_hash         ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_mem_created       ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_kind          ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_mem_subject       ON memories(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_mem_cross         ON memories(cross_scope, status);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    tokens,
    mem_id UNINDEXED,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(mem_id, tokens) VALUES (new.id, new.fts_tokens);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE mem_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE mem_id = old.id;
    INSERT INTO memories_fts(mem_id, tokens) VALUES (new.id, new.fts_tokens);
END;
"""


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """数据行 → Memory 对象。"""

    try:
        meta = json.loads(row["meta"] or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return Memory(
        id=int(row["id"]),
        scope_key=row["scope_key"],
        kind=MemoryKind(row["kind"]),
        content=row["content"],
        summary=row["summary"] or "",
        importance=float(row["importance"]),
        strength=float(row["strength"]),
        anchor_at=float(row["anchor_at"]),
        status=MemoryStatus(row["status"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_recalled_at=row["last_recalled_at"],
        recall_count=int(row["recall_count"]),
        pinned=bool(row["pinned"]),
        cross_scope=bool(row["cross_scope"]),
        subject=row["subject"] or "",
        predicate=row["predicate"] or "",
        object=row["object"] or "",
        source=row["source"] or "",
        superseded_by=row["superseded_by"],
        meta=meta if isinstance(meta, dict) else {},
        embedding=row["embedding"],
    )


def pack_vector(vector: Sequence[float]) -> bytes:
    """float32 打包成 BLOB。"""

    return struct.pack(f"<{len(vector)}f", *[float(v) for v in vector])


def unpack_vector(blob: Optional[bytes]) -> Optional[list[float]]:
    """BLOB → float 列表。"""

    if not blob:
        return None
    count = len(blob) // 4
    if count == 0:
        return None
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


def _normalize_pairs(raw: list[tuple[float, int]]) -> list[tuple[int, float]]:
    """把一组 (原始分, id) 归一化到 [0,1]，保持顺序。"""

    if not raw:
        return []
    lo = min(s for s, _ in raw)
    hi = max(s for s, _ in raw)
    span = hi - lo
    out: list[tuple[int, float]] = []
    for score, mid in raw:
        normalized = 1.0 if span <= 0 else (score - lo) / span
        out.append((mid, normalized))
    return out


class MemoryStore:
    """记忆存储。线程安全（内部一把写锁）。"""

    def __init__(self, db_path: str, *, busy_timeout_ms: int = 5000, enable_fts: bool = True) -> None:
        self.db_path = str(db_path)
        self.enable_fts = bool(enable_fts)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=busy_timeout_ms / 1000.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._migrate()

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #

    def _migrate(self) -> None:
        with self._lock:
            try:
                self._conn.executescript(_SCHEMA)
            except sqlite3.OperationalError:
                # 环境不支持 FTS5：降级为无全文索引模式，其余功能不受影响。
                schema = _SCHEMA.split("CREATE VIRTUAL TABLE")[0]
                self._conn.executescript(schema)
            self._conn.commit()

    @property
    def has_fts(self) -> bool:
        try:
            self._conn.execute("SELECT 1 FROM memories_fts LIMIT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def add(self, memory: Memory, fts_tokens: str = "") -> int:
        """插入一条记忆，返回 id。"""

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO memories (
                    scope_key, kind, content, summary, importance, strength, anchor_at, status,
                    created_at, updated_at, last_recalled_at, recall_count, pinned, cross_scope,
                    subject, predicate, object, source, superseded_by, content_hash, fts_tokens,
                    meta, embedding
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    memory.scope_key,
                    memory.kind.value,
                    memory.content,
                    memory.summary,
                    memory.importance,
                    memory.strength,
                    memory.anchor_at,
                    memory.status.value,
                    memory.created_at,
                    memory.updated_at,
                    memory.last_recalled_at,
                    memory.recall_count,
                    int(memory.pinned),
                    int(memory.cross_scope),
                    memory.subject,
                    memory.predicate,
                    memory.object,
                    memory.source,
                    memory.superseded_by,
                    memory.content_hash,
                    fts_tokens,
                    json.dumps(memory.meta, ensure_ascii=False),
                    memory.embedding,
                ),
            )
            self._conn.commit()
            memory.id = int(cur.lastrowid or 0)
            return memory.id

    def update(self, memory: Memory, fts_tokens: str = "") -> None:
        """整体更新一条记忆。"""

        with self._lock:
            self._conn.execute(
                """
                UPDATE memories SET
                    scope_key=?, kind=?, content=?, summary=?, importance=?, strength=?, anchor_at=?,
                    status=?, created_at=?, updated_at=?, last_recalled_at=?, recall_count=?, pinned=?,
                    cross_scope=?, subject=?, predicate=?, object=?, source=?, superseded_by=?,
                    content_hash=?, fts_tokens=?, meta=?, embedding=?
                WHERE id=?
                """,
                (
                    memory.scope_key,
                    memory.kind.value,
                    memory.content,
                    memory.summary,
                    memory.importance,
                    memory.strength,
                    memory.anchor_at,
                    memory.status.value,
                    memory.created_at,
                    memory.updated_at,
                    memory.last_recalled_at,
                    memory.recall_count,
                    int(memory.pinned),
                    int(memory.cross_scope),
                    memory.subject,
                    memory.predicate,
                    memory.object,
                    memory.source,
                    memory.superseded_by,
                    memory.content_hash,
                    fts_tokens,
                    json.dumps(memory.meta, ensure_ascii=False),
                    memory.embedding,
                    memory.id,
                ),
            )
            self._conn.commit()

    def delete(self, memory_id: int) -> bool:
        """物理删除。归档不等于删除——只有显式清理才会走到这里。"""

        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (int(memory_id),))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def get(self, memory_id: int) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (int(memory_id),)).fetchone()
        return _row_to_memory(row) if row else None

    def get_many(self, ids: Iterable[int]) -> list[Memory]:
        id_list = [int(i) for i in ids]
        if not id_list:
            return []
        placeholders = ",".join("?" * len(id_list))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", id_list
            ).fetchall()
        by_id = {int(r["id"]): _row_to_memory(r) for r in rows}
        return [by_id[i] for i in id_list if i in by_id]

    def find_by_hash(self, content_hash: str, scope_key: str) -> Optional[Memory]:
        """按归一化哈希找完全相同的记忆（精确去重）。"""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE content_hash=? AND (scope_key=? OR cross_scope=1) "
                "ORDER BY id LIMIT 1",
                (content_hash, scope_key),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def iter_scope(
        self,
        scope_key: str,
        *,
        statuses: Optional[Sequence[str]] = None,
        kinds: Optional[Sequence[str]] = None,
        limit: int = 500,
    ) -> list[Memory]:
        """按作用域取记忆（用于整理、去重、冲突检测）。"""

        clauses = ["(scope_key=? OR cross_scope=1)"]
        params: list[Any] = [scope_key]
        if statuses:
            clauses.append("status IN (" + ",".join("?" * len(statuses)) + ")")
            params.extend(statuses)
        if kinds:
            clauses.append("kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?", params
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def iter_maintenance(self, limit: int = 2000) -> list[Memory]:
        """待维护的记忆：非终态、未置顶。"""

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE status NOT IN ('archived','superseded') AND pinned=0 "
                "ORDER BY updated_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def iter_archived(self, older_than: float, limit: int = 500) -> list[Memory]:
        """归档超过指定时长的记忆（供物理清理用）。"""

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE status='archived' AND updated_at < ? LIMIT ?",
                (float(older_than), int(limit)),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def iter_all(self, limit: int = 200, statuses: Optional[Sequence[str]] = None) -> list[Memory]:
        """列出全部作用域的记忆（管理视角，WebUI 用）。

        ⚠ 这是**唯一**一条不按作用域过滤的读取路径，
        只给管理员界面用；对 Bot 而言永远不该走这条。
        """

        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            clauses.append("status IN (" + ",".join("?" * len(statuses)) + ")")
            params.extend(statuses)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?", params
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def count(self, scope_key: Optional[str] = None) -> int:
        with self._lock:
            if scope_key is None:
                row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE scope_key=? OR cross_scope=1", (scope_key,)
                ).fetchone()
        return int(row[0]) if row else 0

    def status_counts(self, scope_key: Optional[str] = None) -> dict[str, int]:
        with self._lock:
            if scope_key is None:
                rows = self._conn.execute(
                    "SELECT status, COUNT(*) FROM memories GROUP BY status"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT status, COUNT(*) FROM memories WHERE scope_key=? OR cross_scope=1 "
                    "GROUP BY status",
                    (scope_key,),
                ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    # ------------------------------------------------------------------ #
    # 检索通道
    # ------------------------------------------------------------------ #

    def search_fts(
        self,
        query: str,
        scope_key: str,
        *,
        limit: int = 40,
        kinds: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> list[tuple[int, float]]:
        """BM25 全文检索。返回 [(memory_id, 归一化分数)]，分数越大越相关。"""

        if not self.has_fts or not query:
            return []
        from .textutil import fts_text

        tokens = fts_text(query)
        if not tokens:
            return []
        clauses = ["(m.scope_key=? OR m.cross_scope=1)", "m.status NOT IN ('archived','superseded')"]
        params: list[Any] = [scope_key]
        if kinds:
            clauses.append("m.kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        if statuses:
            clauses.append("m.status IN (" + ",".join("?" * len(statuses)) + ")")
            params.extend(statuses)
        params.append(int(limit))
        try:
            with self._lock:
                rows = self._conn.execute(
                    f"""
                    SELECT f.mem_id AS mid, bm25(memories_fts) AS rank
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.mem_id
                    WHERE memories_fts MATCH ? AND {' AND '.join(clauses)}
                    ORDER BY rank LIMIT ?
                    """,
                    [tokens, *params],
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() 返回负数，越小越相关；取反后归一化到 [0,1]
        raw = [(-float(r["rank"]), int(r["mid"])) for r in rows]
        return _normalize_pairs(raw)

    def search_vector(
        self,
        vector: Sequence[float],
        scope_key: str,
        *,
        limit: int = 40,
        kinds: Optional[Sequence[str]] = None,
    ) -> list[tuple[int, float]]:
        """余弦相似度检索。"""

        if not vector:
            return []
        clauses = [
            "(scope_key=? OR cross_scope=1)",
            "status NOT IN ('archived','superseded')",
            "embedding IS NOT NULL",
        ]
        params: list[Any] = [scope_key]
        if kinds:
            clauses.append("kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, embedding FROM memories WHERE {' AND '.join(clauses)}", params
            ).fetchall()
        if not rows:
            return []

        query_vec = list(vector)
        scored: list[tuple[float, int]] = []
        if _np is not None:
            mat = _np.zeros((len(rows), len(query_vec)), dtype="float32")
            for i, row in enumerate(rows):
                vec = unpack_vector(row["embedding"])
                if vec and len(vec) == len(query_vec):
                    mat[i] = _np.asarray(vec, dtype="float32")
            q = _np.asarray(query_vec, dtype="float32")
            norms = _np.linalg.norm(mat, axis=1) * (float(_np.linalg.norm(q)) or 1.0)
            norms[norms == 0] = 1.0
            sims = (mat @ q) / norms
            scored = [(float(s), int(rows[i]["id"])) for i, s in enumerate(sims)]
        else:
            qnorm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
            for row in rows:
                vec = unpack_vector(row["embedding"])
                if not vec or len(vec) != len(query_vec):
                    continue
                dot = sum(a * b for a, b in zip(vec, query_vec))
                vnorm = math.sqrt(sum(v * v for v in vec)) or 1.0
                scored.append((dot / (vnorm * qnorm), int(row["id"])))
        scored.sort(reverse=True)
        return _normalize_pairs(scored[: int(limit)])

    def recent(
        self,
        scope_key: str,
        *,
        limit: int = 10,
        kinds: Optional[Sequence[str]] = None,
    ) -> list[tuple[int, float]]:
        """最近写入的记忆（保证「刚聊完的事」一定进得去）。"""

        clauses = ["(scope_key=? OR cross_scope=1)", "status NOT IN ('archived','superseded')"]
        params: list[Any] = [scope_key]
        if kinds:
            clauses.append("kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if not ids:
            return []
        # 越新分越高（线性映射到 (0,1]）
        return [(mid, (i + 1) / len(ids)) for i, mid in enumerate(ids)]

    def set_embedding(self, memory_id: int, vector: Sequence[float]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET embedding=? WHERE id=?",
                (pack_vector(vector), int(memory_id)),
            )
            self._conn.commit()

    def find_conflicts(self, scope_key: str, *, limit: int = 500) -> list[tuple[Memory, Memory]]:
        """找出「同主语同谓语、不同宾语」的记忆对——即互相矛盾的事实。"""

        rows = self.iter_scope(scope_key, statuses=("active", "fading"), limit=limit)
        buckets: dict[tuple[str, str], list[Memory]] = {}
        for mem in rows:
            if not mem.subject or not mem.predicate:
                continue
            buckets.setdefault((mem.subject, mem.predicate), []).append(mem)
        pairs: list[tuple[Memory, Memory]] = []
        for group in buckets.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda m: m.created_at)
            for i in range(len(group) - 1):
                older, newer = group[i], group[i + 1]
                if older.object and newer.object and older.object != newer.object:
                    pairs.append((older, newer))
        return pairs

    def mark_superseded(self, memory_id: int, superseded_by: int) -> None:
        """标记为被取代。原文保留做证据链，只是不再召回。"""

        with self._lock:
            self._conn.execute(
                "UPDATE memories SET status=?, superseded_by=?, updated_at=? WHERE id=?",
                (MemoryStatus.SUPERSEDED.value, int(superseded_by), time.time(), int(memory_id)),
            )
            self._conn.commit()

