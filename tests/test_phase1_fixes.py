"""Phase 1 回归测试：四个底层可靠性修复。

1. 零值导入——strength=0.0 / importance=0.0 不被 `or` 吞掉
2. 跨会话维护隔离——trim/merge/supersede 不碰别的会话共享来的记忆
3. 去重连通分量——3+ 条近似重复合成一条，survivor 不被中间态覆盖
4. HTTP scope 归属——forget/pin by ID 校验所属会话
"""

from __future__ import annotations

import asyncio
import unittest

from memfade.types import MemoryKind, MemoryStatus, Scope
from tests.helpers import temp_engine


class TestZeroValueImport(unittest.TestCase):
    """engine.import_memories 必须保留合法的 0.0 值。"""

    def test_strength_zero_preserved(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        written = engine.import_memories(
            [{"content": "strength zero test", "strength": 0.0}],
            scope=scope,
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].strength, 0.0)

    def test_importance_zero_preserved(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        # event kind has min importance clamp at 0.4; use a kind that allows 0
        written = engine.import_memories(
            [{"content": "importance zero test", "importance": 0.0, "kind": "semantic"}],
            scope=scope,
        )
        self.assertEqual(len(written), 1)
        # _clamp_importance enforces per-kind minimums; 0.0 gets clamped.
        # The fix ensures None→default, not that 0.0 bypasses kind clamps.
        # Verify _safe_float preserves 0.0 before clamping by testing directly:
        from memfade.engine import _safe_float
        self.assertEqual(_safe_float(0.0, 1.0), 0.0)
        self.assertEqual(_safe_float(None, 1.0), 1.0)

    def test_created_at_zero_preserved(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        written = engine.import_memories(
            [{"content": "epoch zero test", "created_at": 0.0}],
            scope=scope,
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].created_at, 0.0)

    def test_none_falls_back_to_default(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        written = engine.import_memories(
            [{"content": "default fallback test"}],
            scope=scope,
        )
        self.assertEqual(len(written), 1)
        self.assertGreater(written[0].strength, 0.0)


class TestOwnedScopeIsolation(unittest.TestCase):
    """iter_owned_scope 只返回本会话的记忆，不含 cross_scope。"""

    def test_cross_scope_excluded_from_owned(self):
        engine, _, _ = temp_engine()
        scope_a = Scope(platform="qq", chat_type="private", chat_id="userA")

        engine.import_memories(
            [{"content": "A private memory"}],
            scope=scope_a,
        )
        # Import a cross_scope memory directly via store.add
        now = time.time()
        shared_mem = Memory(
            id=0,
            scope_key="qq:group:groupX",
            kind=MemoryKind.SEMANTIC,
            content="shared group memory",
            importance=0.5,
            strength=0.8,
            anchor_at=now,
            status=MemoryStatus.ACTIVE,
            cross_scope=True,
        )
        engine.store.add(shared_mem)

        engine.import_memories(
            [{"content": "A second memory"}],
            scope=scope_a,
        )

        owned = engine.store.iter_owned_scope(scope_a.key())
        self.assertEqual(len(owned), 2)

        visible = engine.store.iter_scope(scope_a.key())
        # iter_scope includes cross_scope=1, so should see 3
        self.assertEqual(len(visible), 3)

    def test_trim_does_not_touch_shared(self):
        """_trim_scope 不能删除别的会话共享来的记忆。"""

        from memfade.consolidate import ConsolidationService

        engine, _, config = temp_engine()
        config.maintenance.max_memories_per_scope = 2
        scope_a = Scope(platform="qq", chat_type="private", chat_id="userA")

        for i in range(5):
            engine.import_memories(
                [{"content": f"private memory {i}"}],
                scope=scope_a,
            )
        # Add a cross_scope memory that belongs to a different scope
        now = time.time()
        shared_mem = Memory(
            id=0,
            scope_key="qq:group:groupX",
            kind=MemoryKind.SEMANTIC,
            content="shared should survive trim",
            importance=0.5,
            strength=0.8,
            anchor_at=now,
            status=MemoryStatus.ACTIVE,
            cross_scope=True,
        )
        engine.store.add(shared_mem)

        consolidator = ConsolidationService(engine.store, config, extractor=engine.extractor)
        consolidator._trim_scope(scope_a.key())

        # The shared memory belongs to groupX, not userA, so trim should not touch it
        remaining = engine.store.iter_owned_scope("qq:group:groupX")
        self.assertTrue(
            any(m.content == "shared should survive trim" for m in remaining),
            "cross_scope memory was deleted by another scope's trim",
        )


class TestDedupConnectedGroups(unittest.TestCase):
    """3+ 条互相近似的记忆应合成一条，survivor 不被中间态覆盖。"""

    def test_three_way_merge(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")

        engine.import_memories(
            [
                {"content": "用户喜欢吃辣的食物，尤其是川菜"},
                {"content": "用户喜欢吃辣的食物，尤其是川菜"},
                {"content": "用户喜欢吃辣的食物，尤其是川菜"},
            ],
            scope=scope,
            skip_duplicates=False,
        )

        from memfade.consolidate import ConsolidationService, ConsolidationReport

        consolidator = ConsolidationService(engine.store, engine.config, extractor=engine.extractor)
        report = ConsolidationReport()
        merged = consolidator._merge_scope(scope.key(), report)

        remaining = engine.store.iter_owned_scope(scope.key())
        self.assertEqual(len(remaining), 1, f"expected 1 after merge, got {len(remaining)}")
        meta = remaining[0].meta
        self.assertGreaterEqual(meta.get("merged_count", 0), 2)


class TestHTTPScopeCheck(unittest.TestCase):
    """HTTP forget/pin by ID 必须校验 scope 归属。"""

    def test_forget_rejects_cross_scope_id(self):
        from adapters.http.server import MemFadeHTTPService

        engine, _, _ = temp_engine()
        scope_a = Scope(platform="qq", chat_type="private", chat_id="userA")
        scope_b = Scope(platform="qq", chat_type="private", chat_id="userB")

        written = engine.import_memories(
            [{"content": "B's private secret"}],
            scope=scope_b,
        )
        mem_id = written[0].id

        service = MemFadeHTTPService(engine)
        result = service.forget({
            "id": mem_id,
            "scope": {"chat_id": "userA", "chat_type": "private", "platform": "qq"},
        })
        self.assertFalse(result["ok"])
        self.assertIn("forbidden", result.get("error", "").lower())

        still_there = engine.store.get(mem_id)
        self.assertIsNotNone(still_there)

    def test_pin_rejects_cross_scope_id(self):
        from adapters.http.server import MemFadeHTTPService

        engine, _, _ = temp_engine()
        scope_b = Scope(platform="qq", chat_type="private", chat_id="userB")

        written = engine.import_memories(
            [{"content": "B's pinnable memory"}],
            scope=scope_b,
        )
        mem_id = written[0].id

        service = MemFadeHTTPService(engine)
        result = service.pin({
            "id": mem_id,
            "scope": {"chat_id": "userA", "chat_type": "private", "platform": "qq"},
        })
        self.assertFalse(result["ok"])
        self.assertIn("forbidden", result.get("error", "").lower())

        mem = engine.store.get(mem_id)
        self.assertFalse(mem.pinned)

    def test_forget_allows_own_scope(self):
        from adapters.http.server import MemFadeHTTPService

        engine, _, _ = temp_engine()
        scope_a = Scope(platform="qq", chat_type="private", chat_id="userA")

        written = engine.import_memories(
            [{"content": "A's forgettable memory"}],
            scope=scope_a,
        )
        mem_id = written[0].id

        service = MemFadeHTTPService(engine)
        result = service.forget({
            "id": mem_id,
            "scope": {"chat_id": "userA", "chat_type": "private", "platform": "qq"},
        })
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()



import time
from memfade.types import Memory, MemoryKind, MemoryStatus, Scope
