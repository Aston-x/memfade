"""记忆库体检测试。"""

from __future__ import annotations

import os
import unittest

from memfade.diagnostics import analyze, clause_count, export_memories
from memfade.types import MemoryDraft, MemoryKind
from tests.helpers import temp_engine


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.engine, self.path, self.config = temp_engine()
        from memfade import Scope

        self.scope = Scope(platform="qq", chat_type="session", chat_id="s1")

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_clause_count(self):
        self.assertEqual(clause_count("一句话"), 1)
        self.assertEqual(clause_count("两句话，第二句"), 2)
        self.assertEqual(clause_count("三句。三句。三句"), 3)

    def test_empty_store(self):
        report = analyze(self.engine.store, self.scope.key())
        self.assertEqual(report["total"], 0)
        self.assertEqual(report["issues"], [])

    def test_detects_long_and_multi_clause(self):
        self.engine.remember(
            MemoryDraft(
                content=(
                    "这是一条特别长的记忆，它讲了很多很多事情，"
                    "第一件是甲，第二件是乙，第三件是丙，第四件是丁，第五件是戊"
                ),
                kind=MemoryKind.EPISODIC,
                importance=0.5,
            ),
            self.scope,
        )
        report = analyze(self.engine.store, self.scope.key())
        self.assertEqual(report["too_long"], 1)
        self.assertGreaterEqual(report["multi_clause"], 1)
        kinds = {issue["kind"] for issue in report["issues"]}
        self.assertIn("too_long", kinds)

    def test_detects_missing_spo(self):
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的东西", kind=MemoryKind.SEMANTIC, importance=0.8),
            self.scope,
        )
        report = analyze(self.engine.store, self.scope.key())
        self.assertEqual(report["missing_spo"], 1)

    def test_detects_exact_duplicates(self):
        """同一条内容写了两遍（哈希不同导致去重没拦住）要能查出来。"""

        self.engine.store.add(
            self._memory("许杏玖喜欢辣的东西"),
            "许杏玖 喜欢 辣 的 东西",
        )
        self.engine.store.add(self._memory("许杏玖喜欢辣的东西"), "许杏玖喜欢辣的东西")
        report = analyze(self.engine.store, self.scope.key())
        self.assertGreaterEqual(report["exact_duplicates"], 1)

    def _memory(self, content: str):
        import time

        from memfade.types import Memory, MemoryStatus

        now = time.time()
        return Memory(
            id=0,
            scope_key=self.scope.key(),
            kind=MemoryKind.SEMANTIC,
            content=content,
            importance=0.8,
            strength=1.0,
            anchor_at=now,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

    def test_export_is_json_friendly(self):
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的东西", kind=MemoryKind.SEMANTIC, importance=0.8),
            self.scope,
        )
        rows = export_memories(self.engine.store, self.scope.key())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "许杏玖喜欢辣的东西")
        self.assertIn("importance", rows[0])


if __name__ == "__main__":
    unittest.main()

