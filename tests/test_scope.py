"""会话隔离测试。

这一组是 memfade 存在与否的分水岭。
真实事故：私聊里沉淀的记忆，被同一个机器人带到了群里，
用户在群里看到了自己私聊说过的话。
所以这里的断言必须是「宁可不返回，也绝不串台」。
"""

from __future__ import annotations

import os
import unittest

from memfade.types import MemoryDraft, MemoryKind, Scope
from tests.helpers import temp_engine


GROUP = Scope(platform="qq", chat_type="group", chat_id="3000000003")
PRIVATE = Scope(platform="qq", chat_type="private", chat_id="1000000001")


class TestScopeIsolation(unittest.TestCase):
    def setUp(self):
        self.engine, self.path, self.config = temp_engine()

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_private_not_visible_in_group(self):
        draft = MemoryDraft(
            content="私聊里说的私密内容：密码是 123456",
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
        )
        self.engine.remember(draft, PRIVATE)
        hits = self.engine.recall("私密内容", GROUP)
        self.assertEqual(hits, [], "群里绝对不能召回私聊的记忆")

    def test_group_not_visible_in_private(self):
        draft = MemoryDraft(content="群里说的玩笑话", kind=MemoryKind.EPISODIC, importance=0.5)
        self.engine.remember(draft, GROUP)
        hits = self.engine.recall("玩笑话", PRIVATE)
        self.assertEqual(hits, [], "私聊不能看到群里的记忆")

    def test_same_scope_visible(self):
        draft = MemoryDraft(content="同一会话里的记忆", kind=MemoryKind.SEMANTIC, importance=0.8)
        self.engine.remember(draft, GROUP)
        hits = self.engine.recall("同一会话", GROUP)
        self.assertEqual(len(hits), 1)

    def test_cross_scope_is_explicit_opt_in(self):
        draft = MemoryDraft(
            content="用户要求永远记住的称呼：宝宝",
            kind=MemoryKind.SEMANTIC,
            importance=0.95,
            cross_scope=True,
        )
        self.engine.remember(draft, PRIVATE)
        in_group = self.engine.recall("称呼", GROUP)
        self.assertEqual(len(in_group), 1, "显式标记跨会话的记忆应当可见")

    def test_default_draft_is_not_cross_scope(self):
        draft = MemoryDraft(content="普通记忆", kind=MemoryKind.SEMANTIC, importance=0.8)
        mem = self.engine.remember(draft, PRIVATE)
        self.assertIsNotNone(mem)
        self.assertFalse(mem.cross_scope)

    def test_count_respects_scope(self):
        self.engine.remember(MemoryDraft(content="群里的", importance=0.8), GROUP)
        self.engine.remember(MemoryDraft(content="私聊的", importance=0.8), PRIVATE)
        self.assertEqual(self.engine.store.count(GROUP.key()), 1)
        self.assertEqual(self.engine.store.count(PRIVATE.key()), 1)

    def test_scope_key_format(self):
        self.assertEqual(GROUP.key(), "qq:group:3000000003")
        self.assertEqual(Scope.from_key("qq:private:123").chat_id, "123")

