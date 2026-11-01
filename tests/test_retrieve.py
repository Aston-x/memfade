"""检索与分层注入测试。"""

from __future__ import annotations

import os
import time
import unittest

from memfade.types import MemoryDraft, MemoryKind, MemoryStatus, Scope
from tests.helpers import temp_engine


SCOPE = Scope(platform="qq", chat_type="group", chat_id="1")
DAY = 86400.0


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.engine, self.path, self.config = temp_engine()

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_chinese_bigram_match(self):
        draft = MemoryDraft(content="今天中午一起去吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.5)
        self.engine.remember(draft, SCOPE)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(len(hits), 1, "中文二元组检索应命中")

    def test_punctuation_does_not_break_match(self):
        draft = MemoryDraft(content="她很讨厌别人迟到。", kind=MemoryKind.PREFERENCE, importance=0.7)
        self.engine.remember(draft, SCOPE)
        hits = self.engine.recall("讨厌 迟到", SCOPE)
        self.assertEqual(len(hits), 1)

    def test_important_outranks_trivial(self):
        big = MemoryDraft(content="关于炸鸡的重要约定", kind=MemoryKind.SEMANTIC, importance=0.95)
        small = MemoryDraft(content="随手记的炸鸡", kind=MemoryKind.EPISODIC, importance=0.2)
        self.engine.remember(big, SCOPE)
        self.engine.remember(small, SCOPE)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(hits[0].memory.content, "关于炸鸡的重要约定")

    def test_relevance_gap_cannot_swamp_importance(self):
        """短而无关的文档不能靠 BM25 反超长而重要的文档。

        相关度用 min-max 归一化时，两条候选里分数低的那条会直接归零，
        相关度差距就能盖过显著性的差距。改成按最大值归一化保留比例。
        """

        important = MemoryDraft(
            content="关于炸鸡的重要约定：以后点炸鸡都要加份年糕",
            kind=MemoryKind.SEMANTIC,
            importance=0.95,
        )
        trivial = MemoryDraft(content="炸鸡", kind=MemoryKind.EPISODIC, importance=0.2)
        self.engine.remember(important, SCOPE)
        self.engine.remember(trivial, SCOPE)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].memory.content, important.content)

    def test_faded_memory_moves_to_echo_tier(self):
        draft = MemoryDraft(content="上周三去吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.3)
        draft.importance = 0.4
        mem = self.engine.remember(draft, SCOPE)
        old = time.time() - 25 * DAY
        mem.anchor_at = old
        mem.created_at = old
        self.engine.store.update(mem)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].tier, "echo", "淡出档的记忆应进回声层")

    def test_context_separates_tiers(self):
        fact = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.9)
        self.engine.remember(fact, SCOPE)
        ctx = self.engine.build_context("吃什么", SCOPE)
        self.assertIn("许杏玖喜欢辣的食物", ctx.text)
        self.assertIn("[记忆]", ctx.text)

    def test_echo_is_capped_and_marked(self):
        for i in range(8):
            draft = MemoryDraft(content=f"旧事编号{i}关于炸鸡", kind=MemoryKind.EPISODIC, importance=0.3)
            draft.importance = 0.4
            mem = self.engine.remember(draft, SCOPE)
            old = time.time() - 25 * DAY
            mem.anchor_at = old
            mem.created_at = old
            self.engine.store.update(mem)
        ctx = self.engine.build_context("炸鸡", SCOPE)
        self.assertLessEqual(len(ctx.echoes), self.config.retrieval.echo_max)
        if ctx.echoes:
            self.assertIn("可能已经变了", ctx.text)

    def test_budget_respected(self):
        self.config.retrieval.budget_tokens = 40
        for i in range(20):
            draft = MemoryDraft(content=f"一条比较长的记忆内容编号{i}讲的是吃饭的事情", importance=0.6)
            self.engine.remember(draft, SCOPE)
        ctx = self.engine.build_context("吃饭", SCOPE)
        self.assertLessEqual(ctx.token_estimate, 80, "注入内容应受预算约束")

    def test_reinforce_updates_strength(self):
        draft = MemoryDraft(content="许杏玖喜欢吃火锅", kind=MemoryKind.SEMANTIC, importance=0.8)
        mem = self.engine.remember(draft, SCOPE)
        before = mem.recall_count
        self.engine.build_context("火锅", SCOPE)
        after = self.engine.store.get(mem.id)
        self.assertEqual(after.recall_count, before + 1)
        self.assertIsNotNone(after.last_recalled_at)

    def test_reinforce_cooldown(self):
        draft = MemoryDraft(content="许杏玖喜欢吃火锅", kind=MemoryKind.SEMANTIC, importance=0.8)
        mem = self.engine.remember(draft, SCOPE)
        self.engine.build_context("火锅", SCOPE)
        count = self.engine.store.get(mem.id).recall_count
        self.engine.build_context("火锅", SCOPE)
        self.assertEqual(self.engine.store.get(mem.id).recall_count, count, "冷却期内不应重复强化")

    def test_archived_not_recalled(self):
        draft = MemoryDraft(content="会被归档的记忆", kind=MemoryKind.EPISODIC, importance=0.5)
        mem = self.engine.remember(draft, SCOPE)
        self.engine.forget(mem.id)
        hits = self.engine.recall("归档", SCOPE)
        self.assertEqual(hits, [])

    def test_near_duplicate_is_not_written_twice(self):
        first = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁还没解决", kind=MemoryKind.EPISODIC, importance=0.6), SCOPE
        )
        second = self.engine.remember(
            MemoryDraft(content="二十块加价仲裁没解决", kind=MemoryKind.EPISODIC, importance=0.6), SCOPE
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.id, second.id, "近似重复应复用已有记忆而不是新建")
        self.assertEqual(self.engine.store.count(SCOPE.key()), 1)

    def test_profile_lists_stable_facts(self):
        fact = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.9)
        self.engine.remember(fact, SCOPE)
        profile = self.engine.build_profile(SCOPE)
        self.assertIn("许杏玖喜欢辣的食物", profile)

    def test_profile_excludes_episodic(self):
        event = MemoryDraft(content="上周三一起去吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.6)
        self.engine.remember(event, SCOPE)
        profile = self.engine.build_profile(SCOPE)
        self.assertNotIn("炸鸡", profile, "一次性事件不该进画像")

    def test_profile_is_scope_isolated(self):
        other = Scope(platform="qq", chat_type="session", chat_id="other")
        fact = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.9)
        self.engine.remember(fact, other)
        self.assertEqual(self.engine.build_profile(SCOPE), "", "别的会话的画像不该出现")

    def test_profile_respects_limits(self):
        for i in range(10):
            draft = MemoryDraft(content=f"关于许杏玖的第{i}条稳定事实", kind=MemoryKind.SEMANTIC, importance=0.8)
            self.engine.remember(draft, SCOPE)
        profile = self.engine.build_profile(SCOPE, max_items=3)
        self.assertLessEqual(len(profile.splitlines()), 3)
