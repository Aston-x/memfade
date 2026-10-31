"""离线整理测试。"""

from __future__ import annotations

import os
import time
import unittest

from memfade.types import MemoryDraft, MemoryKind, MemoryStatus, Scope
from tests.helpers import temp_engine


SCOPE = Scope(platform="qq", chat_type="group", chat_id="1")
DAY = 86400.0


class TestConsolidation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine, self.path, self.config = temp_engine()

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def _age(self, mem, days):
        old = time.time() - days * DAY
        mem.anchor_at = old
        mem.created_at = old
        mem.updated_at = old
        self.engine.store.update(mem)
        return mem

    async def test_decay_pass_moves_old_events_down(self):
        draft = MemoryDraft(content="很久以前的一次事件", kind=MemoryKind.EPISODIC, importance=0.4)
        mem = self.engine.remember(draft, SCOPE)
        self._age(mem, 30)
        await self.engine.consolidate(SCOPE)
        after = self.engine.store.get(mem.id)
        self.assertNotEqual(after.status, MemoryStatus.ACTIVE)

    async def test_consolidation_is_idempotent(self):
        draft = MemoryDraft(content="幂等性测试事件", kind=MemoryKind.EPISODIC, importance=0.4)
        mem = self.engine.remember(draft, SCOPE)
        self._age(mem, 20)
        await self.engine.consolidate(SCOPE)
        first = self.engine.store.get(mem.id)
        await self.engine.consolidate(SCOPE)
        await self.engine.consolidate(SCOPE)
        second = self.engine.store.get(mem.id)
        self.assertEqual(first.status, second.status, "重复整理不应改变档位")
        self.assertEqual(first.strength, second.strength, "重复整理不应改变强度")

    async def test_gist_is_generated(self):
        draft = MemoryDraft(
            content="上周三和许杏玖一起去吃了炸鸡还喝了奶茶",
            kind=MemoryKind.EPISODIC,
            importance=0.4,
        )
        mem = self.engine.remember(draft, SCOPE)
        self._age(mem, 25)
        await self.engine.consolidate(SCOPE)
        after = self.engine.store.get(mem.id)
        if after.status is MemoryStatus.GISTED:
            self.assertTrue(after.summary, "淡化成 gist 档应生成摘要")
            self.assertLessEqual(len(after.summary), self.config.fade.gist_max_chars + 5)

    async def test_merge_near_duplicates(self):
        a = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁还没解决", kind=MemoryKind.EPISODIC, importance=0.6),
            SCOPE,
            skip_dedup=True,
        )
        b = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁还没解决呢", kind=MemoryKind.EPISODIC, importance=0.6),
            SCOPE,
            skip_dedup=True,
        )
        report = await self.engine.consolidate(SCOPE)
        self.assertGreaterEqual(report.merged, 1, "近似重复应被合并")
        remaining = [self.engine.store.get(a.id), self.engine.store.get(b.id)]
        alive = [r for r in remaining if r is not None]
        self.assertEqual(len(alive), 1, "合并后只应剩一条")

    async def test_merge_keeps_evidence(self):
        a = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁还没解决", kind=MemoryKind.EPISODIC, importance=0.6),
            SCOPE,
            skip_dedup=True,
        )
        b = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁还没解决呢", kind=MemoryKind.EPISODIC, importance=0.6),
            SCOPE,
            skip_dedup=True,
        )
        await self.engine.consolidate(SCOPE)
        survivor = self.engine.store.get(a.id) or self.engine.store.get(b.id)
        self.assertIsNotNone(survivor)
        self.assertTrue(survivor.meta.get("merged_from"), "被合并方的原文要留档")

    async def test_supersede_conflicting_facts(self):
        old = self.engine.remember(
            MemoryDraft(
                content="许杏玖住在北京",
                kind=MemoryKind.SEMANTIC,
                importance=0.85,
                subject="许杏玖",
                predicate="住在",
                object="北京",
            ),
            SCOPE,
        )
        new = self.engine.remember(
            MemoryDraft(
                content="许杏玖住在上海",
                kind=MemoryKind.SEMANTIC,
                importance=0.85,
                subject="许杏玖",
                predicate="住在",
                object="上海",
            ),
            SCOPE,
        )
        report = await self.engine.consolidate(SCOPE)
        self.assertGreaterEqual(report.superseded, 1)
        self.assertEqual(self.engine.store.get(old.id).status, MemoryStatus.SUPERSEDED)
        self.assertIsNotNone(self.engine.store.get(new.id))

    async def test_pinned_never_decays(self):
        draft = MemoryDraft(content="置顶的记忆", kind=MemoryKind.SEMANTIC, importance=0.8)
        mem = self.engine.remember(draft, SCOPE)
        self.engine.pin(mem.id)
        self._age(self.engine.store.get(mem.id), 400)
        await self.engine.consolidate(SCOPE)
        after = self.engine.store.get(mem.id)
        self.assertEqual(after.status, MemoryStatus.ACTIVE)
        self.assertTrue(after.pinned)

    async def test_report_shape(self):
        report = await self.engine.consolidate(SCOPE)
        data = report.as_dict()
        for key in ("scanned", "decayed", "status_changed", "gisted", "merged", "superseded", "purged"):
            self.assertIn(key, data)
