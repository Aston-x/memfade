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

    async def test_trim_over_cap(self):
        self.config.maintenance.max_memories_per_scope = 5
        samples = [
            "今天中午在楼下吃了炸鸡",
            "晚上加班到很晚才回家",
            "周末打算去爬山看日出",
            "最近在追一部科幻剧",
            "养的那盆绿萝终于开花了",
            "换了新的机械键盘手感不错",
            "感冒了嗓子有点疼",
            "地铁上遇到一个很 funny 的小孩",
            "论文被导师打回来重写",
            "存款终于够买那台显示器",
            "学着做了一锅红烧肉",
            "信用卡账单又超预算了",
        ]
        for text in samples:
            draft = MemoryDraft(content=text, kind=MemoryKind.EPISODIC, importance=0.4)
            self.engine.remember(draft, SCOPE)
        self.assertEqual(self.engine.store.count(SCOPE.key()), 12)
        await self.engine.consolidate(SCOPE)
        after = self.engine.store.count(SCOPE.key())
        self.assertLessEqual(after, 5, "超出上限后应当淘汰最弱的")

    async def test_trim_keeps_pinned(self):
        self.config.maintenance.max_memories_per_scope = 3
        keep = self.engine.remember(
            MemoryDraft(content="置顶不许动的重要事实", kind=MemoryKind.SEMANTIC, importance=0.95), SCOPE
        )
        self.engine.pin(keep.id)
        samples = [
            "去超市补货了日用品",
            "骑车上班的路上被淋湿了",
            "整理柜子翻出来一张旧照片",
            "预约了下周的体检",
            "试做了新的咖啡冲泡方式",
            "朋友寄了一箱东北的苹果",
            "把旧手机里的照片备份了",
            "计划暑期去一趟青海湖",
            "拼好了一个很久以前买的模型",
            "在读书会分享了一本书",
        ]
        for text in samples:
            draft = MemoryDraft(content=text, kind=MemoryKind.EPISODIC, importance=0.3)
            self.engine.remember(draft, SCOPE)
        await self.engine.consolidate(SCOPE)
        self.assertIsNotNone(self.engine.store.get(keep.id), "置顶的记忆不该被淘汰")

    async def test_conflict_detected_without_subject(self):
        """模型没填 subject/predicate 时，也要能发现「同一件事的两个版本」。"""

        old = self.engine.remember(
            MemoryDraft(content="许杏玖住在杭州", kind=MemoryKind.SEMANTIC, importance=0.85), SCOPE
        )
        new = self.engine.remember(
            MemoryDraft(content="许杏玖住在上海", kind=MemoryKind.SEMANTIC, importance=0.85), SCOPE
        )
        self.assertIsNotNone(old)
        self.assertIsNotNone(new)
        report = await self.engine.consolidate(SCOPE)
        self.assertGreaterEqual(report.superseded, 1, "旧值该被取代")

    async def test_different_topics_are_not_conflicts(self):
        a = self.engine.remember(
            MemoryDraft(content="季怡在做小红书账号", kind=MemoryKind.SEMANTIC, importance=0.8), SCOPE
        )
        b = self.engine.remember(
            MemoryDraft(content="季怡在备赛蓝桥杯", kind=MemoryKind.SEMANTIC, importance=0.8), SCOPE
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        report = await self.engine.consolidate(SCOPE)
        self.assertEqual(report.superseded, 0, "不同的事不该被判成矛盾")

    async def test_preference_change_supersedes_old(self):
        """偏好变了（「喜欢吃辣」→「现在不吃辣」）旧事实要让位。

        这两条的 subject/predicate 未必填得一致（三元组靠模型自觉），
        所以还有一层内容兜底：去掉否定词后高度相似 + 否定状态相反。
        """

        old = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的食物", kind=MemoryKind.PREFERENCE, importance=0.8),
            SCOPE,
        )
        new = self.engine.remember(
            MemoryDraft(content="许杏玖现在不吃辣的食物了", kind=MemoryKind.PREFERENCE, importance=0.8),
            SCOPE,
        )
        self.assertIsNotNone(old)
        self.assertIsNotNone(new)
        report = await self.engine.consolidate(SCOPE)
        self.assertGreaterEqual(report.superseded, 1, "旧偏好该退场")

    async def test_compatible_preferences_are_not_conflicts(self):
        """「不喜欢辣」与「喜欢麻辣烫」并不矛盾，别误判。"""

        a = self.engine.remember(
            MemoryDraft(content="许杏玖不喜欢辣", kind=MemoryKind.PREFERENCE, importance=0.8), SCOPE
        )
        b = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢麻辣烫", kind=MemoryKind.PREFERENCE, importance=0.8), SCOPE
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        report = await self.engine.consolidate(SCOPE)
        self.assertEqual(report.superseded, 0, "两条偏好并不矛盾")

    async def test_status_change_counter_works(self):
        """「有多少条换了档位」这个计数必须是准的。

        之前 `updated = mem` 是同一个对象，改完 status 再比较必然相等，
        计数恒为 0——看起来一切正常，实际什么变化都看不出来。
        """

        mem = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的东西", kind=MemoryKind.SEMANTIC, importance=0.65), SCOPE
        )
        self.assertIsNotNone(mem)
        old = time.time() - 200 * 86400.0
        mem.anchor_at = old
        mem.created_at = old
        self.engine.store.update(mem)
        report = await self.engine.consolidate(SCOPE)
        self.assertGreaterEqual(report.status_changed, 1, "换档的条数应该被统计到")
        self.assertEqual(self.engine.store.get(mem.id).status, MemoryStatus.FADING)
