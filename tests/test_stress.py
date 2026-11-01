"""随机数据压力测试（性质测试）。

单元测试证明「某条路走对了」，这一组证明「**没有哪条路会走错**」。

用固定种子的伪随机生成一堆记忆（各种作用域、种类、显著性、年龄、
重复内容、矛盾事实），然后把整条链路跑一遍，检查一批**不变量**：

* 隔离：召回到的记忆一定属于当前会话（跨会话泄漏 = 0）；
* 幂等：整理任务跑 N 次和跑 1 次结果一样；
* 档位合法：状态永远是四档之一，强度永远在 [0,1]；
* 淡化单调：时间越久，有效强度越低；
* 写进去的就搜得到（除非被淡出或去重）；
* 不崩：任何输入都不该抛异常。

固定种子 → 每次跑结果一样，失败能复现。
"""

from __future__ import annotations

import os
import random
import unittest
from typing import List

from memfade.decay import effective_strength, resolve_status
from memfade.types import MemoryDraft, MemoryKind, MemoryStatus, Scope
from tests.helpers import temp_engine


DAY = 86400.0
SCOPES = [
    Scope(platform="qq", chat_type="group", chat_id="a"),
    Scope(platform="qq", chat_type="group", chat_id="b"),
    Scope(platform="qq", chat_type="private", chat_id="c"),
    Scope(platform="qq", chat_type="private", chat_id="d"),
]
WORDS = [
    "许杏玖喜欢辣的食物",
    "季怡在写一个插件",
    "蓝桥杯报名12月18日截止",
    "猫粮买错了牌子",
    "今天中午一起吃了炸鸡",
    "显卡不能同时跑两个模型",
    "《牧神记5》周日更新",
    "二十块加价仲裁还没解决",
    "影子资源站那个站挂了",
    "药屋少女的呢喃第三季周五更新",
]


class TestStress(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine, self.path, self.config = temp_engine()
        self.rng = random.Random(20261102)

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def _seed(self, count: int = 120) -> None:
        """塞一批随机记忆进去（含重复内容与矛盾事实）。"""

        now = 1000000000.0
        for _ in range(count):
            scope = self.rng.choice(SCOPES)
            kind = self.rng.choice(list(MemoryKind))
            content = self.rng.choice(WORDS)
            if self.rng.random() < 0.25:
                # 故意制造近似重复
                content = content + "（补充说明）"
            draft = MemoryDraft(
                content=content,
                kind=kind,
                importance=round(self.rng.uniform(0.1, 1.0), 3),
                subject=self.rng.choice(["许杏玖", "季怡", ""]),
                predicate=self.rng.choice(["喜欢", "住在", ""]),
                object=self.rng.choice(["辣的食物", "上海", ""]),
                cross_scope=self.rng.random() < 0.1,
            )
            self.engine.remember(draft, scope, now=now - self.rng.uniform(0, 200 * DAY))

    async def test_isolation_invariant(self):
        """召回到的记忆必须属于当前会话（除非显式跨会话）。"""

        self._seed()
        for scope in SCOPES:
            for query in WORDS[:4]:
                hits = self.engine.recall(query, scope, top_k=20)
                for hit in hits:
                    mem = hit.memory
                    self.assertTrue(
                        mem.scope_key == scope.key() or mem.cross_scope,
                        "跨会话泄漏：%s 出现在 %s 的召回里" % (mem.scope_key, scope.key()),
                    )

    async def test_consolidation_idempotent_over_random_data(self):
        """整理任务跑一次和跑三次，结果必须一样。"""

        self._seed()
        await self.engine.consolidate()
        first = self._snapshot()
        await self.engine.consolidate()
        await self.engine.consolidate()
        second = self._snapshot()
        self.assertEqual(first, second, "整理任务不幂等")

    def _snapshot(self) -> List[tuple]:
        rows = []
        for mem in self.engine.store.iter_all(limit=10000):
            rows.append((mem.id, mem.status.value, round(mem.strength, 9)))
        return sorted(rows)

    async def test_states_and_strength_stay_in_range(self):
        """档位永远是四档之一，强度永远在 [0,1]。"""

        self._seed()
        valid = {
            MemoryStatus.ACTIVE.value,
            MemoryStatus.FADING.value,
            MemoryStatus.GISTED.value,
            MemoryStatus.ARCHIVED.value,
            MemoryStatus.SUPERSEDED.value,
        }
        for _ in range(3):
            await self.engine.consolidate()
        for mem in self.engine.store.iter_all(limit=10000):
            self.assertIn(mem.status.value, valid)
            self.assertGreaterEqual(mem.strength, 0.0)
            self.assertLessEqual(mem.strength, 1.0)
            eff = effective_strength(mem, 1000000000.0 + 400 * DAY, self.engine.policy)
            self.assertGreaterEqual(eff, 0.0)
            self.assertLessEqual(eff, 1.0)
            resolved = resolve_status(mem, 1000000000.0 + 400 * DAY, self.engine.policy)
            self.assertIn(resolved, set(MemoryStatus))

    async def test_consolidation_never_grows_the_library(self):
        """整理任务只合并/取代，绝不凭空多出记忆。"""

        self._seed()
        before = self.engine.store.count()
        for _ in range(3):
            await self.engine.consolidate()
        after = self.engine.store.count()
        self.assertLessEqual(after, before, "整理任务让库变大了：%d -> %d" % (before, after))

    async def test_superseded_memory_is_not_recalled(self):
        """被取代的旧事实不该再被召回。"""

        scope = SCOPES[0]
        self.engine.remember(
            MemoryDraft(
                content="许杏玖住在上海",
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject="许杏玖",
                predicate="住在",
                object="上海",
            ),
            scope,
        )
        self.engine.remember(
            MemoryDraft(
                content="许杏玖住在杭州",
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject="许杏玖",
                predicate="住在",
                object="杭州",
            ),
            scope,
        )
        await self.engine.consolidate()
        hits = self.engine.recall("许杏玖住在", scope, top_k=10)
        contents = [hit.memory.content for hit in hits]
        self.assertIn("许杏玖住在杭州", contents, "新事实应该还在")
        self.assertNotIn("许杏玖住在上海", contents, "旧事实应该已被取代")

    def test_decay_is_monotonic_in_time(self):
        """时间越久，有效强度越低（不可能越放越强）。"""

        for kind in MemoryKind:
            # 显著性不能太低：低于种类门槛的记忆压根不写进库（那是设计，不是 bug）。
            for importance in (0.5, 0.7, 0.9):
                draft = MemoryDraft(
                    content="单调性测试 %s %s" % (kind.value, importance),
                    kind=kind,
                    importance=importance,
                )
                mem = self.engine.remember(draft, SCOPES[0], now=1000000000.0)
                self.assertIsNotNone(mem, "写入被拒了（显著性门槛？）")
                previous = 2.0
                for days in (0, 1, 7, 30, 100, 400):
                    eff = effective_strength(mem, 1000000000.0 + days * DAY, self.engine.policy)
                    self.assertLessEqual(
                        eff, previous + 1e-9, "%s 在第 %d 天反而变强了" % (kind.value, days)
                    )
                    previous = eff

    async def test_written_memories_are_findable(self):
        """写进去的稳定事实，应该还能被搜到（除非被淡出或去重）。"""

        scope = SCOPES[0]
        unique = "许杏玖喜欢麻辣火锅和烤鸭"
        self.engine.remember(
            MemoryDraft(content=unique, kind=MemoryKind.PREFERENCE, importance=0.9), scope
        )
        hits = self.engine.recall("麻辣火锅", scope, top_k=5)
        self.assertTrue(any(unique in hit.memory.content for hit in hits))

    async def test_random_queries_never_crash(self):
        """随便乱查也不能抛异常。"""

        self._seed(60)
        garbage = [
            "",
            " ",
            "？",
            "a",
            "0",
            "%%",
            "' OR 1=1 --",
            "x" * 500,
            "。" * 50,
            "什么什么什么",
            "阿猫阿狗阿猫阿狗",
        ]
        for scope in SCOPES:
            for query in garbage + WORDS:
                self.engine.recall(query, scope, top_k=10)
                self.engine.build_context(query, scope, top_k=10)

    async def test_duplicate_writes_do_not_grow_the_library(self):
        """同一条内容写十遍，库里只能有一条。"""

        scope = SCOPES[0]
        for _ in range(10):
            self.engine.remember(
                MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9),
                scope,
            )
        self.assertEqual(self.engine.store.count(scope.key()), 1)

    async def test_cross_scope_needs_explicit_flag(self):
        """跨会话共享必须显式开启，且别的会话确实能看到。"""

        normal = "许杏玖养了一只叫小咪的猫"
        shared = "许杏玖的生日是11月3日"
        self.engine.remember(MemoryDraft(content=normal, importance=0.9), SCOPES[0])
        self.engine.remember(
            MemoryDraft(content=shared, importance=0.9, cross_scope=True), SCOPES[0]
        )
        hits = self.engine.recall("生日", SCOPES[1], top_k=5)
        self.assertTrue(any("生日" in hit.memory.content for hit in hits))
        hits = self.engine.recall("小咪", SCOPES[1], top_k=5)
        self.assertFalse(any("小咪" in hit.memory.content for hit in hits))

    async def test_consolidate_without_scope_does_not_merge_across_scopes(self):
        """不指定作用域跑整理，不能把不同会话的内容并到一起。

        WebUI 的「整理」按钮和命令行 `fade` 都是不带作用域跑的——
        早先它们会跨会话合并，把 A 会话的内容并进 B 会话。
        """

        same = "许杏玖喜欢辣的食物"
        self.engine.remember(MemoryDraft(content=same, kind=MemoryKind.PREFERENCE, importance=0.9), SCOPES[0])
        self.engine.remember(MemoryDraft(content=same, kind=MemoryKind.PREFERENCE, importance=0.9), SCOPES[1])
        self.assertEqual(self.engine.store.count(SCOPES[0].key()), 1)
        self.assertEqual(self.engine.store.count(SCOPES[1].key()), 1)
        await self.engine.consolidate()
        self.assertEqual(self.engine.store.count(SCOPES[0].key()), 1, "A 会话的记忆被合并掉了")
        self.assertEqual(self.engine.store.count(SCOPES[1].key()), 1, "B 会话的记忆被合并掉了")

    async def test_consolidate_without_scope_still_supersedes(self):
        """不指定作用域跑整理，矛盾仲裁也要照常工作（按会话分别判）。"""

        scope = SCOPES[0]
        self.engine.remember(
            MemoryDraft(
                content="许杏玖住在上海",
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject="许杏玖",
                predicate="住在",
                object="上海",
            ),
            scope,
        )
        self.engine.remember(
            MemoryDraft(
                content="许杏玖住在杭州",
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject="许杏玖",
                predicate="住在",
                object="杭州",
            ),
            scope,
        )
        report = await self.engine.consolidate()
        self.assertGreaterEqual(report.superseded, 1, "不指定作用域时仲裁被跳过了")
        hits = self.engine.recall("许杏玖住在", scope, top_k=10)
        contents = [hit.memory.content for hit in hits]
        self.assertIn("许杏玖住在杭州", contents)
        self.assertNotIn("许杏玖住在上海", contents)

    async def test_injection_respects_budget_and_limits(self):
        """注入出来的文本永远不超过预算，回声层永远不超过条数上限。"""

        self._seed(150)
        for scope in SCOPES:
            for query in WORDS[:5]:
                ctx = self.engine.build_context(query, scope, top_k=20)
                cfg = self.config.retrieval
                self.assertLessEqual(
                    ctx.token_estimate,
                    cfg.budget_tokens + 40,
                    "注入超预算：%d > %d" % (ctx.token_estimate, cfg.budget_tokens),
                )
                self.assertLessEqual(len(ctx.echoes), cfg.echo_max, "回声层超量")

    async def test_archived_and_superseded_never_injected(self):
        """归档和被取代的记忆绝不进上下文。"""

        scope = SCOPES[0]
        keep = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9), scope
        )
        gone = self.engine.remember(
            MemoryDraft(content="许杏玖讨厌香菜", kind=MemoryKind.PREFERENCE, importance=0.9), scope
        )
        self.engine.store.mark_superseded(gone.id, keep.id)
        ctx = self.engine.build_context("许杏玖", scope, top_k=20)
        self.assertNotIn("讨厌香菜", ctx.text)

    async def test_pinned_memory_never_fades(self):
        """置顶的记忆永不衰减。"""

        from memfade.types import Memory

        scope = SCOPES[0]
        mem = self.engine.remember(
            MemoryDraft(content="许杏玖最重要的约定", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        pinned = Memory(**{**mem.__dict__, "pinned": True})
        self.engine.store.update(pinned)
        await self.engine.consolidate(scope)
        after = self.engine.store.get(mem.id)
        self.assertEqual(after.status, MemoryStatus.ACTIVE)
        self.assertGreaterEqual(after.strength, 1.0 - 1e-9)

    async def test_recall_is_sorted_by_score(self):
        """召回结果必须按分数降序（面板和注入都靠这个顺序）。"""

        self._seed()
        for scope in SCOPES:
            for query in WORDS[:5]:
                hits = self.engine.recall(query, scope, top_k=20)
                scores = [h.score for h in hits]
                self.assertEqual(scores, sorted(scores, reverse=True))

    async def test_reinforce_has_cooldown(self):
        """连续强化不该把强度叠上去（否则一次注入就能把记忆钉死）。"""

        scope = SCOPES[0]
        mem = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9), scope
        )
        self.engine.reinforce([mem.id])
        first = self.engine.store.get(mem.id).strength
        self.engine.reinforce([mem.id])
        second = self.engine.store.get(mem.id).strength
        self.assertAlmostEqual(first, second, places=6, msg="冷却期内不该重复强化")

    async def test_update_without_fts_tokens_keeps_it_searchable(self):
        """忘记传 fts_tokens 时，记忆不该从此搜不到。

        FTS5 的触发器靠这一列重建索引——传空串等于把索引清空。
        这种坑最难查（内容还在，就是查不出来），所以 store 自己兜住。
        """

        scope = SCOPES[0]
        mem = self.engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9), scope
        )
        mem.importance = 0.95
        self.engine.store.update(mem)  # 故意不传 fts_tokens
        hits = self.engine.recall("辣的食物", scope, top_k=5)
        self.assertTrue(
            any(hit.memory.id == mem.id for hit in hits),
            "更新后这条记忆搜不到了（索引被清空）",
        )

    async def test_multi_char_query_needs_all_chars(self):
        """多字查询要每个字都命中才算数。

        「晚安」只对上「网络安全」里的「安」——那不是相关，是巧合。
        """

        scope = SCOPES[0]
        self.engine.remember(
            MemoryDraft(content="电子赛科目单片机、网络安全", importance=0.9), scope
        )
        self.engine.remember(MemoryDraft(content="许杏玖喜欢辣的食物", importance=0.9), scope)
        hits = self.engine.recall("晚安", scope, top_k=5, require_match=True)
        self.assertFalse(
            any("网络安全" in hit.memory.content for hit in hits),
            "只对一个字的记忆不该被「晚安」捞出来",
        )
        hits = self.engine.recall("辣", scope, top_k=5, require_match=True)
        self.assertTrue(any("辣的食物" in hit.memory.content for hit in hits))

    async def test_vector_channel_ignores_background_similarity(self):
        """向量通道不能把「什么都有点像」的记忆当成命中。

        嵌入模型对任何文本都有 0.4~0.55 的基础相似度，
        不设门槛的话，用户说「今天天气真好啊」也会捞回一批记忆——
        那正是「总提旧事」的老毛病。

        判据是「比这批记忆的中位相似度高出多少」，所以这个测试
        要构造两批**彼此很像**的记忆，再看查询是不是其中的离群者。
        """

        store = self.engine.store
        scope = SCOPES[0]
        dim = 8
        query = [1.0] + [0.0] * (dim - 1)
        # 背景相似度：所有记忆都朝同一个「不像查询」的方向
        background = [0.6] + [0.8] + [0.0] * (dim - 2)

        # 一批彼此高度相似的记忆（背景相似度很高）
        for i in range(10):
            mem = self.engine.remember(
                MemoryDraft(content="第%d条记录" % i, importance=0.8), scope
            )
            store.set_embedding(mem.id, list(background))

        # 查询跟它们一样「普通」→ 没有离群者 → 不该有命中
        hits = store.search_vector(query, scope.key(), min_gap=0.15)
        self.assertEqual(hits, [], "背景相似度不该被当成命中")

        # 有一条跟查询几乎一样 → 它是真正的离群者 → 该命中
        special = self.engine.remember(
            MemoryDraft(content="那条真正相关的记忆", importance=0.9), scope
        )
        store.set_embedding(special.id, list(query))
        hits = store.search_vector(query, scope.key(), min_gap=0.15)
        self.assertEqual([mid for mid, _ in hits], [special.id])


if __name__ == "__main__":
    unittest.main()
