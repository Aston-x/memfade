"""记忆的可搬迁性。

**记忆是用户的数据**：换机器、换机器人、换框架都得能带走。
只有导出没有导入，等于「只能看不能搬」。

另外，淡化状态（强度、锚点）必须一起搬——
不搬的话，导入进来的记忆会「重新变年轻」，
该淡的又淡一遍，主人会看到机器人把旧事又提一遍。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memfade import MemFadeConfig, MemoryKind, Scope
from memfade.engine import MemoryEngine
from memfade.types import MemoryDraft


def _engine(path: Path) -> MemoryEngine:
    config = MemFadeConfig.from_dict(
        {"storage": {"db_path": str(path), "enable_vector": False}}
    )
    return MemoryEngine(config)


class TestPortability(unittest.TestCase):
    def test_export_import_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _engine(Path(tmp) / "a.db")
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            source.remember(
                MemoryDraft(
                    content="Alice喜欢辣的食物",
                    kind=MemoryKind.PREFERENCE,
                    importance=0.8,
                    subject="Alice",
                    predicate="喜欢",
                    object="辣的食物",
                ),
                scope,
            )
            exported = source.export()
            source.close()

            target = _engine(Path(tmp) / "b.db")
            try:
                written = target.import_memories(exported, scope)
                self.assertEqual(len(written), 1)
                memory = written[0]
                self.assertEqual(memory.content, "Alice喜欢辣的食物")
                self.assertEqual(memory.kind, MemoryKind.PREFERENCE)
                self.assertEqual(memory.subject, "Alice")
                self.assertEqual(memory.object, "辣的食物")
                self.assertAlmostEqual(memory.importance, 0.8, places=3)
                hits = target.recall("辣的食物", scope, require_match=True)
                self.assertTrue(hits, "导入的记忆搜不到——索引词没重算")
            finally:
                target.close()

    def test_fade_state_travels_with_the_memory(self):
        """强度与锚点要一起搬，否则导入的记忆「重新变年轻」。"""

        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            source = _engine(Path(tmp) / "a.db")
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            draft = MemoryDraft(
                content="一条正在淡出的记忆",
                kind=MemoryKind.EPISODIC,
                importance=0.5,
                subject="Alice",
                predicate="做了",
                object="某件事",
            )
            written = source.remember(draft, scope)
            conn = sqlite3.connect(str(Path(tmp) / "a.db"))
            conn.execute(
                "UPDATE memories SET strength=0.42, anchor_at=1000.0 WHERE id=?",
                (written.id,),
            )
            conn.commit()
            conn.close()

            exported = source.export()
            source.close()

            target = _engine(Path(tmp) / "b.db")
            try:
                imported = target.import_memories(exported, scope)
                self.assertAlmostEqual(imported[0].strength, 0.42, places=3)
                self.assertAlmostEqual(imported[0].anchor_at, 1000.0, places=3)
            finally:
                target.close()

    def test_duplicates_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp) / "a.db")
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            draft = MemoryDraft(
                content="同一条记忆",
                kind=MemoryKind.SEMANTIC,
                importance=0.7,
                subject="Alice",
                predicate="喜欢",
                object="辣",
            )
            engine.remember(draft, scope)
            exported = engine.export()
            again = engine.import_memories(exported, scope)
            self.assertEqual(again, [], "重复导入应该被跳过")
            engine.close()

    def test_import_file_shape(self):
        """面板与 CLI 吃的是同一种 JSON：{memories: [...]} 或 [...]。"""

        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp) / "a.db")
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            engine.remember(
                MemoryDraft(content="一条记忆", kind=MemoryKind.SEMANTIC, importance=0.6),
                scope,
            )
            exported = engine.export()
            engine.close()
            counter = 0
            for payload in (exported, {"memories": exported}, json.loads(json.dumps(exported))):
                counter += 1
                target = _engine(Path(tmp) / ("b%d.db" % counter))
                try:
                    items = payload["memories"] if isinstance(payload, dict) else payload
                    self.assertEqual(len(target.import_memories(items, scope)), 1)
                finally:
                    target.close()


if __name__ == "__main__":
    unittest.main()
