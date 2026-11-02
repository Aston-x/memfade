"""存储层的升级路径。

插件是**装到别人机器上**的：老版本建出来的库必须能被新版本直接打开。
``CREATE TABLE IF NOT EXISTS`` 只建新表，**不会给已有的表加列**——
新代码往老库里写新列，插件当场起不来（「table memories has no column named xxx」）。
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from memfade.store import SCHEMA_VERSION, MemoryStore


OLD_SCHEMA = """
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
    meta          TEXT    NOT NULL DEFAULT '{}'
);
"""


class TestStoreMigration(unittest.TestCase):
    def _old_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO memories (scope_key, kind, content, anchor_at, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("scope-1", "semantic", "老版本留下的一条记忆", 1000.0, 1000.0, 1000.0),
        )
        conn.commit()
        conn.close()

    def test_old_db_gets_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.db"
            self._old_db(db_path)
            store = MemoryStore(str(db_path))
            try:
                cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)")}
                self.assertIn("embedding", cols, "新列没有补上")
                self.assertIn("source", cols)
                rows = store._conn.execute("SELECT content FROM memories").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "老版本留下的一条记忆")
            finally:
                store.close()

    def test_new_write_works_on_migrated_db(self):
        """迁移之后新代码要能正常写——这才是「装到别人机器上」的关键。"""

        from memfade.types import Memory, MemoryKind, MemoryStatus

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.db"
            self._old_db(db_path)
            store = MemoryStore(str(db_path))
            try:
                memory = Memory(
                    id=0,
                    scope_key="scope-1",
                    kind=MemoryKind.SEMANTIC,
                    content="迁移之后写进去的一条",
                    importance=0.7,
                    strength=1.0,
                    anchor_at=2000.0,
                    status=MemoryStatus.ACTIVE,
                    created_at=2000.0,
                    updated_at=2000.0,
                )
                new_id = store.add(memory, fts_tokens="迁移 之后 写进 去 的 一条")
                self.assertGreater(new_id, 0)
                fetched = store.get(new_id)
                self.assertIsNotNone(fetched)
                self.assertEqual(fetched.content, "迁移之后写进去的一条")
            finally:
                store.close()

    def test_user_version_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "new.db"
            store = MemoryStore(str(db_path))
            try:
                version = store._conn.execute("PRAGMA user_version").fetchone()[0]
                self.assertEqual(version, SCHEMA_VERSION)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
