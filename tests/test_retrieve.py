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

    def test_single_char_query_finds_memory(self):
        """单字查询也要能查到东西。

        全文索引里存的是二元组，「许杏玖喜欢辣的食物」的二元组是
        「喜欢」「欢辣」「辣的」……没有孤零零的「辣」。
        所以只查二元组的话，搜索框里打一个「辣」就什么都搜不到——
        而打单字恰恰是搜索框最常见的用法。
        """

        draft = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.8)
        self.engine.remember(draft, SCOPE)
        hits = self.engine.recall("辣", SCOPE)
        self.assertEqual(len(hits), 1, "单字查询应能命中")
        self.assertIn("uni", hits[0].channels, "命中方式应标明是单字通道")

    def test_single_char_query_does_not_smash_relevance(self):
        """单字兜底不能盖过词面精确命中。"""

        exact = MemoryDraft(content="关于炸鸡的重要约定：以后点炸鸡都要加份年糕", importance=0.9)
        single = MemoryDraft(content="辣", importance=0.2)
        self.engine.remember(exact, SCOPE)
        self.engine.remember(single, SCOPE)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(hits[0].memory.content, exact.content)

    def test_common_char_does_not_pollute_recall(self):
        """「的」这种到处都是的字不能让整库记忆涌进来。

        单字通道只在查询够短、且这个字有区分度时才启用。
        """

        for i in range(30):
            self.engine.remember(
                MemoryDraft(content="这是第%d条关于日常琐事的记录" % i, importance=0.5), SCOPE
            )
        self.engine.remember(MemoryDraft(content="许杏玖喜欢辣的食物", importance=0.8), SCOPE)
        hits = self.engine.recall("的", SCOPE)
        self.assertLessEqual(len(hits), 3, "高频单字不该把整库拖进来")

    def test_long_query_ignores_single_char_channel(self):
        """整句查询不该触发单字兜底。"""

        self.engine.remember(MemoryDraft(content="今天中午一起去吃了炸鸡", importance=0.5), SCOPE)
        hits = self.engine.recall("我记得你之前说过喜欢吃炸鸡对吧", SCOPE)
        for hit in hits:
            self.assertNotIn("uni", hit.channels)

    def test_single_char_hits_are_discriminated(self):
        """只对上一个字的记忆，相关度应当低于两个字都对上的。"""

        both = MemoryDraft(content="许杏玖喜欢辣的食物", importance=0.5)
        one = MemoryDraft(content="许杏玖讨厌香菜", importance=0.5)
        self.engine.remember(both, SCOPE)
        self.engine.remember(one, SCOPE)
        hits = self.engine.recall("辣", SCOPE, require_match=True)
        self.assertEqual(len(hits), 1, "「辣」不该命中只有「香菜」的记忆")
        self.assertEqual(hits[0].memory.content, both.content)

    def test_multi_word_query_uses_or_semantics(self):
        """多词查询必须是 OR，不是 AND。

        FTS5 里空格分隔的多个词是隐式 AND，
        「还记得炸鸡吗」会要求记忆同时含全部五个二元组——几乎查不到东西。
        """

        draft = MemoryDraft(content="许杏玖和季怡一起去吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.5)
        self.engine.remember(draft, SCOPE)
        hits = self.engine.recall("还记得炸鸡吗", SCOPE)
        self.assertEqual(len(hits), 1, "多词查询应能命中只含部分词的记忆")

    def test_more_matching_words_rank_higher(self):
        strong = MemoryDraft(content="炸鸡和年糕都点了", kind=MemoryKind.EPISODIC, importance=0.5)
        weak = MemoryDraft(content="出门买了瓶水顺便看了一眼炸鸡店", kind=MemoryKind.EPISODIC, importance=0.5)
        self.engine.remember(strong, SCOPE)
        self.engine.remember(weak, SCOPE)
        hits = self.engine.recall("炸鸡", SCOPE)
        self.assertEqual(hits[0].memory.content, strong.content, "命中更多词应排前面")

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

    def test_mmr_avoids_duplicate_topics(self):
        """一次注入里不该塞好几条讲同一件事的。"""

        a = MemoryDraft(content="蓝桥杯报名截止到12月18号", kind=MemoryKind.SEMANTIC, importance=0.85)
        b = MemoryDraft(content="蓝桥杯报名时间是10月8号开始", kind=MemoryKind.SEMANTIC, importance=0.85)
        c = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.8)
        for draft in (a, b, c):
            self.engine.remember(draft, SCOPE)
        context = self.engine.build_context("蓝桥杯报名", SCOPE)
        contents = [hit.memory.content for hit in context.facts]
        self.assertEqual(len(contents), 3)
        # 前两条讲同一件事，应该被第三条（不同话题）插到中间或后面
        self.assertIn(c.content, contents)
        self.assertLess(contents.index(b.content), contents.index(c.content) + 1)

    def test_same_spo_is_same_fact_even_if_reworded(self):
        """三元组相同 = 同一条事实，哪怕措辞不同。"""

        a = MemoryDraft(
            content="季怡已经开始玩《影之刃零》",
            kind=MemoryKind.SEMANTIC,
            importance=0.7,
            subject="季怡",
            predicate="玩",
            object="影之刃零",
        )
        b = MemoryDraft(
            content="季怡已经在玩《影之刃零》",
            kind=MemoryKind.SEMANTIC,
            importance=0.7,
            subject="季怡",
            predicate="玩",
            object="影之刃零",
        )
        first = self.engine.remember(a, SCOPE)
        second = self.engine.remember(b, SCOPE)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.id, second.id, "同一三元组不该存成两条")
        self.assertEqual(self.engine.store.count(SCOPE.key()), 1)

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
        hits = self.engine.recall("今天天气怎么样", SCOPE)
        self.assertEqual(hits, [])

    async def test_archived_memory_revives_when_asked(self):
        """被明确问起时，已归档的记忆应该能被想起来。"""

        draft = MemoryDraft(content="许杏玖喜欢吃炸鸡", kind=MemoryKind.EPISODIC, importance=0.5)
        mem = self.engine.remember(draft, SCOPE)
        self.engine.forget(mem.id)
        self.assertEqual(self.engine.recall("炸鸡", SCOPE), [], "平时不该出现")
        hits = self.engine.recall("还记得炸鸡吗", SCOPE)
        self.assertEqual(len(hits), 1, "被问起时应该能想起来")
        revived = self.engine.store.get(mem.id)
        self.assertEqual(revived.status, MemoryStatus.ACTIVE)

    async def test_archived_memory_stays_hidden_when_not_asked(self):
        draft = MemoryDraft(content="许杏玖喜欢吃炸鸡", kind=MemoryKind.EPISODIC, importance=0.5)
        mem = self.engine.remember(draft, SCOPE)
        self.engine.forget(mem.id)
        hits = self.engine.recall("今天天气怎么样", SCOPE)
        self.assertEqual(hits, [], "没问到就不该复活")

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

    def test_profile_prefers_stable_facts_over_transient(self):
        """画像是「这个人是谁」，不该被时效性事实挤占。"""

        stable = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.9)
        transient = MemoryDraft(
            content="2026年11月1日这周更新里有《牧神记5》",
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
        )
        self.engine.remember(stable, SCOPE)
        self.engine.remember(transient, SCOPE)
        profile = self.engine.build_profile(SCOPE, max_items=1)
        self.assertIn("许杏玖喜欢辣的食物", profile)
        self.assertNotIn("牧神记", profile)

    def test_unrelated_recent_memory_is_not_injected(self):
        """不相关的东西不该进 prompt——这是「总提旧事」的根源。"""

        self.engine.remember(
            MemoryDraft(content="蓝桥杯报名截止到12月18号", kind=MemoryKind.SEMANTIC, importance=0.9), SCOPE
        )
        # require_match=True：模拟「有向量能力」时的严格模式
        context = self.engine.build_context("她喜欢吃什么", SCOPE, require_match=True)
        self.assertEqual(context.text, "", "没有检索命中时不该注入无关记忆")

    def test_lenient_without_embeddings(self):
        """没有向量能力时不要求词面命中——近义记忆会被误杀。"""

        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), SCOPE
        )
        context = self.engine.build_context("今天吃什么呢", SCOPE, require_match=False)
        self.assertIn("许杏玖喜欢吃辣的东西", context.text)

    def test_relevance_floor_applies_before_top_k(self):
        """相关性底线要在 top_k 截断**之前**生效。

        否则「最近写入但跟问题无关」的候选会占掉额度，
        真正命中的记忆反而被挤出去。
        """

        self.engine.config.retrieval.top_k = 2
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), SCOPE
        )
        # 造几条「最近写入但完全不相关」的高重要性记忆
        for i in range(6):
            self.engine.remember(
                MemoryDraft(
                    content=f"完全不相干的第{i}条重要事实内容占位",
                    kind=MemoryKind.SEMANTIC,
                    importance=0.95,
                ),
                SCOPE,
            )
        hits = self.engine.recall("她喜欢吃什么", SCOPE, require_match=True)
        self.assertTrue(hits)
        self.assertTrue(
            any("喜欢吃辣" in h.memory.content for h in hits),
            "命中的记忆不该被不相关候选挤出 top_k",
        )

    def test_matched_memory_is_injected(self):
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), SCOPE
        )
        context = self.engine.build_context("她喜欢吃什么", SCOPE)
        self.assertIn("许杏玖喜欢吃辣的东西", context.text)

    def test_explain_breaks_down_score(self):
        """要能说清「为什么想起这条」。"""

        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), SCOPE
        )
        rows = self.engine.explain("许杏玖喜欢吃什么", SCOPE)
        self.assertTrue(rows)
        top = rows[0]
        self.assertIn("relevance", top["parts"])
        self.assertIn("importance", top["parts"])
        self.assertIn("strength", top["parts"])
        self.assertGreater(top["parts"]["relevance"], 0.0, "命中的记忆相关度应该大于 0")
        self.assertIn("bm25", top["channels"])
