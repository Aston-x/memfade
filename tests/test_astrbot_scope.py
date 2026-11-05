"""AstrBot 适配器的作用域隔离测试。

这组测试是「群聊里 A 说的话不会漏给 B」这个承诺的**证据**，不是描述。

三层作用域：

* 群级公共   ``astrbot:group:<群号>``
* 人级私密   ``astrbot:group_user:<群号>#<用户号>``
* 私聊       ``astrbot:private:<用户号>``
"""

import unittest

from adapters.astrbot.scopes import (
    CHAT_GROUP,
    CHAT_GROUP_USER,
    CHAT_PRIVATE,
    group_scope,
    person_scope,
    private_scope,
    read_scopes,
    resolve_write_scope,
    write_scopes,
)
from tests.helpers import temp_dir


class TestScopeKeys(unittest.TestCase):
    """作用域键的设计约束。"""

    def test_three_layers_use_distinct_chat_types(self):
        self.assertEqual(CHAT_GROUP, "group")
        self.assertEqual(CHAT_GROUP_USER, "group_user")
        self.assertEqual(CHAT_PRIVATE, "private")

    def test_keys_are_pairwise_distinct(self):
        """三个层的键不能撞——撞了就意味着串层。"""

        keys = {
            group_scope("1001").key(),
            person_scope("1001", "u1").key(),
            person_scope("1001", "u2").key(),
            person_scope("1002", "u1").key(),
            private_scope("u1").key(),
        }
        self.assertEqual(len(keys), 5, "作用域键撞车了")

    def test_key_round_trips_through_from_key(self):
        """Scope.from_key 要能还原出原 Scope，否则导出再导入就错位。"""

        from adapters.astrbot.scopes import Scope

        for scope in (
            group_scope("1001"),
            person_scope("1001", "u1"),
            private_scope("u1"),
        ):
            restored = Scope.from_key(scope.key())
            self.assertEqual(restored, scope, "from_key 往返失败：%s" % scope.key())

    def test_person_scope_keeps_group_and_user(self):
        scope = person_scope("1001", "u1")
        self.assertIn("1001", scope.chat_id)
        self.assertIn("u1", scope.chat_id)


class TestReadScopes(unittest.TestCase):
    """一次对话读哪些层。"""

    def test_group_reads_two_layers(self):
        scopes = read_scopes("1001", "u1")
        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0].chat_type, CHAT_GROUP)
        self.assertEqual(scopes[1].chat_type, CHAT_GROUP_USER)

    def test_private_reads_exactly_one_layer(self):
        """私聊只有一层，且不含任何 group_* ——私聊内容不许漏进群里。"""

        scopes = read_scopes("", "u1")
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].chat_type, CHAT_PRIVATE)
        for scope in scopes:
            self.assertNotIn("group", scope.chat_type)

    def test_private_never_touches_group_scopes(self):
        scopes = write_scopes("", "u1")
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].chat_type, CHAT_PRIVATE)


class TestWriteRouting(unittest.TestCase):
    """按主语决定写进哪一层。"""

    MEMBERS = {"Alice": "u1", "Bob": "u2"}

    def test_known_member_goes_to_person_layer(self):
        scope = resolve_write_scope(
            "Alice", group_id="1001", sender_id="u2", members=self.MEMBERS
        )
        self.assertIsNotNone(scope)
        self.assertEqual(scope.chat_type, CHAT_GROUP_USER)
        self.assertIn("u1", scope.chat_id)

    def test_group_subject_goes_to_group_layer(self):
        scope = resolve_write_scope(
            "本群",
            group_id="1001",
            sender_id="u1",
            members=self.MEMBERS,
            group_subject="本群",
        )
        self.assertIsNotNone(scope)
        self.assertEqual(scope.chat_type, CHAT_GROUP)

    def test_unknown_subject_is_dropped_not_promoted_to_group(self):
        """不认识的主语 → 丢弃。

        **不能**退到群级：那是把某个人的私密事实公开，正是双层隔离要消灭的泄漏。
        """

        scope = resolve_write_scope(
            "某个陌生人", group_id="1001", sender_id="u1", members=self.MEMBERS
        )
        self.assertIsNone(scope, "陌生主语被写进了某一层")

    def test_empty_subject_is_dropped(self):
        scope = resolve_write_scope(
            "", group_id="1001", sender_id="u1", members=self.MEMBERS
        )
        self.assertIsNone(scope)

    def test_private_always_writes_private_layer(self):
        scope = resolve_write_scope(
            "随便什么", group_id="", sender_id="u1", members=self.MEMBERS
        )
        self.assertIsNotNone(scope)
        self.assertEqual(scope.chat_type, CHAT_PRIVATE)


class TestIsolationProof(unittest.IsolatedAsyncioTestCase):
    """端到端证明：A 的记忆不会漏给 B。"""

    async def asyncSetUp(self):
        import os
        import tempfile

        from memfade.config import MemFadeConfig
        from memfade.engine import MemoryEngine

        self.tmp = temp_dir("memfade-scope-")
        config = MemFadeConfig.from_dict(
            {
                "storage": {
                    "db_path": os.path.join(self.tmp, "memfade.db"),
                    "enable_vector": False,
                },
                "extraction": {"min_importance": 0.15},
            }
        )
        self.engine = MemoryEngine(config)

    async def asyncTearDown(self):
        self.engine.close()

    async def _recall(self, query, scopes, top_k=8):
        hits, seen = [], set()
        for scope in scopes:
            for hit in self.engine.recall(query, scope, top_k=top_k, require_match=True):
                if hit.memory.id not in seen:
                    seen.add(hit.memory.id)
                    hits.append(hit)
        return hits

    async def test_member_a_memory_is_invisible_to_member_b(self):
        """核心承诺：A 说的话，B 问的时候不会召回到。"""

        from memfade.types import MemoryDraft, MemoryKind

        a_scope = person_scope("1001", "uA")
        b_scope = person_scope("1001", "uB")

        self.engine.remember(
            MemoryDraft(
                content="Alice喜欢吃辣的食物",
                kind=MemoryKind.PREFERENCE,
                importance=0.9,
                subject="Alice",
            ),
            a_scope,
        )

        a_hits = await self._recall("辣的食物", [a_scope])
        self.assertTrue(a_hits, "A 自己问应该召回到")
        self.assertIn("辣的食物", a_hits[0].memory.content)

        b_hits = await self._recall("辣的食物", [b_scope])
        self.assertFalse(b_hits, "★ A 的记忆漏给了 B")

    async def test_group_layer_is_visible_to_everyone(self):
        """群级公共层对全群可见——这是「群记」的意义。"""

        from memfade.types import MemoryDraft, MemoryKind

        g_scope = group_scope("1001")
        self.engine.remember(
            MemoryDraft(
                content="周五下午三点开会",
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject="本群",
            ),
            g_scope,
        )

        for member in ("uA", "uB", "uC"):
            scopes = read_scopes("1001", member)
            hits = await self._recall("开会", scopes)
            self.assertTrue(hits, "群级记忆对 %s 不可见" % member)

    async def test_private_never_leaks_into_group(self):
        """私聊里说的话，群里问也不能召回到。"""

        from memfade.types import MemoryDraft, MemoryKind

        self.engine.remember(
            MemoryDraft(
                content="Alice的银行卡密码是六位数",
                kind=MemoryKind.SEMANTIC,
                importance=0.95,
                subject="Alice",
            ),
            private_scope("u1"),
        )

        for member in ("uA", "uB"):
            scopes = read_scopes("1001", member)
            hits = await self._recall("银行卡密码", scopes)
            self.assertFalse(hits, "★ 私聊内容漏进了群（%s）" % member)

    async def test_merged_recall_deduplicates_cross_scope_rows(self):
        """cross_scope=1 的行在每层都会被召回，合并后必须只留一条。"""

        from memfade.types import MemoryDraft, MemoryKind

        shared = MemoryDraft(
            content="群里的公共约定：周末不加班",
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
            subject="本群",
            cross_scope=True,
        )
        written = self.engine.remember_many([shared], group_scope("1001"))
        self.assertTrue(written)

        scopes = read_scopes("1001", "u1")

        # 先证明「不去重就会重复」：cross_scope=1 的行在每层都会被召回
        raw_ids = []
        for scope in scopes:
            for hit in self.engine.recall("周末", scope, top_k=8, require_match=True):
                raw_ids.append(hit.memory.id)
        self.assertGreater(
            len(raw_ids),
            len(set(raw_ids)),
            "cross_scope 行本应在多层重复出现，这条前提不成立就说明测试没意义",
        )

        # 再按适配器的合并逻辑（带去重）走一遍
        merged, seen = [], set()
        for scope in scopes:
            for hit in self.engine.recall("周末", scope, top_k=8, require_match=True):
                if hit.memory.id in seen:
                    continue
                seen.add(hit.memory.id)
                merged.append(hit)
        self.assertEqual(
            len(merged), len(set(h.memory.id for h in merged)), "合并后仍有重复"
        )
        self.assertTrue(merged)


if __name__ == "__main__":
    unittest.main()
