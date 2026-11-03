"""抽取层测试。"""

from __future__ import annotations

import os
import unittest

from memfade.extract import MemoryExtractor
from memfade.types import MemoryDraft, MemoryKind, Scope
from tests.helpers import temp_engine


class FakeLLM:
    """假 LLM：按脚本返回内容。"""

    def __init__(self, reply: str = "", raises: bool = False):
        self.reply = reply
        self.raises = raises
        self.calls: list[str] = []

    async def complete(self, prompt, *, system="", max_tokens=800, temperature=0.2):
        self.calls.append(prompt)
        if self.raises:
            raise RuntimeError("模拟接口超时")
        return self.reply


class TestRuleExtraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.extractor = MemoryExtractor(llm=None)

    async def test_extracts_name(self):
        result = await self.extractor.extract([{"role": "user", "content": "我叫许杏玖"}])
        self.assertTrue(result.drafts)
        self.assertIn("许杏玖", result.drafts[0].content)

    async def test_extracts_preference(self):
        result = await self.extractor.extract([{"role": "user", "content": "我最喜欢吃辣的东西"}])
        self.assertTrue(result.drafts)
        self.assertEqual(result.drafts[0].kind, MemoryKind.PREFERENCE)

    async def test_explicit_remember_is_cross_scope(self):
        result = await self.extractor.extract([{"role": "user", "content": "记住：我讨厌别人迟到"}])
        self.assertTrue(result.drafts)
        self.assertTrue(result.drafts[0].cross_scope, "用户明确说记住才允许跨会话")

    async def test_normal_draft_is_not_cross_scope(self):
        result = await self.extractor.extract([{"role": "user", "content": "我住在杭州"}])
        self.assertTrue(result.drafts)
        self.assertFalse(result.drafts[0].cross_scope)

    async def test_empty_conversation(self):
        result = await self.extractor.extract([])
        self.assertEqual(result.drafts, [])


class TestLLMExtraction(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_falls_back_to_rules_without_retrying(self):
        """限流不是「这次太大」——重试只会继续撞墙、继续烧额度。"""

        class _RateLimitedLLM:
            def __init__(self):
                self.calls = 0

            async def complete(self, prompt, **kwargs):
                self.calls += 1
                raise RuntimeError("Error code: 429 - 请求过于频繁，请稍后再试")

        llm = _RateLimitedLLM()
        extractor = MemoryExtractor(llm=llm)
        result = await extractor.extract(
            [{"role": "user", "content": "我喜欢吃辣的东西，不吃香菜"}]
        )
        self.assertFalse(result.used_llm, "限流后应走规则抽取")
        self.assertTrue(result.drafts, "规则抽取应该还能抽到东西")
        self.assertEqual(llm.calls, 1, "限流时不该重试")

    async def test_parses_json(self):
        reply = (
            '[{"content":"许杏玖喜欢辣的食物","kind":"semantic","importance":0.8,'
            '"subject":"许杏玖","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "我喜欢辣的"}])
        self.assertTrue(result.used_llm)
        self.assertEqual(len(result.drafts), 1)
        self.assertEqual(result.drafts[0].kind, MemoryKind.SEMANTIC)
        self.assertEqual(result.drafts[0].subject, "许杏玖")

    async def test_parses_fenced_json(self):
        reply = '```json\n[{"content":"测试","kind":"semantic","importance":0.7}]\n```'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertEqual(len(result.drafts), 1)

    async def test_garbage_output_is_dropped(self):
        extractor = MemoryExtractor(llm=FakeLLM("这不是 JSON，随便说点什么"))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertEqual(result.drafts, [])

    async def test_sensitive_content_is_filtered(self):
        reply = '[{"content":"用户的密码是 hunter2","kind":"semantic","importance":0.9}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertEqual(result.drafts, [], "敏感信息不进记忆库")

    async def test_long_fact_is_split_into_atomic_memories(self):
        """模型甩回来一整段时，拆成「一件事一条」。

        提示词说了「30 字以内」不算数——超长记忆塞进库里，
        注入时只能截断（断在半句里），矛盾检测也抓不住。
        """

        reply = (
            '[{"content":"季怡在做小红书账号，发AI绘画内容，目标是靠流量和接单赚钱，'
            '10月起零关注零粉丝，只有几个赞，曾被卖家临时加价拉黑二十块",'
            '"kind":"semantic","importance":0.8,'
            '"subject":"季怡","predicate":"运营","object":"小红书账号"}]'
        )
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertGreater(len(result.drafts), 1, "长事实应该被拆开")
        for draft in result.drafts:
            self.assertLessEqual(len(draft.content), 40)
        # 三元组只挂第一条
        self.assertEqual(result.drafts[0].subject, "季怡")
        self.assertEqual(result.drafts[1].subject, "")

    async def test_falls_back_to_rules_on_error(self):
        extractor = MemoryExtractor(llm=FakeLLM(raises=True))
        result = await extractor.extract([{"role": "user", "content": "我叫许杏玖"}])
        self.assertFalse(result.used_llm)
        self.assertTrue(result.drafts, "LLM 挂了要能退化到规则抽取")
        self.assertTrue(result.error)

    async def test_promote_cross_scope_can_be_disabled(self):
        reply = '[{"content":"记住这个","kind":"semantic","importance":0.9,"cross_scope":true}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        extractor.config.promote_cross_scope = False
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertFalse(result.drafts[0].cross_scope)


class TestAtomicWrite(unittest.TestCase):
    """引擎级的「一件事一条」写入。"""

    def setUp(self):
        self.engine, self.path, self.config = temp_engine()
        self.scope = Scope(platform="qq", chat_type="private", chat_id="1")

    def tearDown(self):
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_long_content_is_split(self):
        draft = MemoryDraft(
            content="季怡在做小红书账号，发AI绘画内容，目标是靠流量和接单赚钱，"
            "10月起零关注零粉丝，只有几个赞，曾被卖家临时加价拉黑二十块",
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
            subject="季怡",
            predicate="运营",
            object="小红书账号",
        )
        written = self.engine.remember_atomic(draft, self.scope)
        self.assertGreater(len(written), 1)
        for memory in written:
            self.assertLessEqual(len(memory.content), 40)
        # 三元组只挂第一条
        self.assertEqual(written[0].subject, "季怡")
        self.assertEqual(written[1].subject, "")

    def test_short_content_is_one_memory(self):
        draft = MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9)
        written = self.engine.remember_atomic(draft, self.scope)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].content, "许杏玖喜欢辣的食物")
        self.assertEqual(written[0].subject, "")

    async def test_extracted_memory_records_where_it_came_from(self):
        """抽出来的记忆要记下出处（模型不会填 source）。"""

        class _FakeExtractor:
            async def extract(self, messages, source=""):
                from memfade.extract import ExtractionResult

                return ExtractionResult(
                    drafts=[MemoryDraft(content="许杏玖喜欢辣的食物", importance=0.9)]
                )

        engine, path, config = temp_engine()
        try:
            engine.extractor = _FakeExtractor()
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            written = await engine.remember_from_messages(
                [{"role": "user", "content": "我喜欢辣的"}], scope, source="qq:session:s1"
            )
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].source, "qq:session:s1")
        finally:
            engine.close()
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)


class TestGist(unittest.IsolatedAsyncioTestCase):
    async def test_llm_gist(self):
        reply = "和许杏玖一起吃了炸鸡"
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        gist = await extractor.summarize("上周三下午和许杏玖一起去楼下吃了炸鸡还喝了奶茶")
        self.assertEqual(gist, "和许杏玖一起吃了炸鸡")

    async def test_rule_gist_fallback(self):
        extractor = MemoryExtractor(llm=FakeLLM(raises=True))
        gist = await extractor.summarize("上周三下午和许杏玖一起去楼下吃了炸鸡，还喝了奶茶")
        self.assertTrue(gist)
        self.assertLessEqual(len(gist), 45)


class TestImportanceBounds(unittest.IsolatedAsyncioTestCase):
    """显著性必须按种类夹紧——提示词不是保证。

    ⚠ 夹紧发生在**落库那一刻**，不是抽取那一刻。
    抽取阶段留着模型自己给的原始分数，门槛才不会被种类下限顶开
    （semantic 下限 0.40 一旦提前夹，模型给的 0.1 会被抬到 0.40，
    正好越过 0.45 的门槛一半——垃圾记忆就有了后门）。
    """

    async def test_episodic_capped(self):
        reply = '[{"content":"上周三去吃了炸鸡","kind":"episodic","importance":0.95}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertTrue(result.drafts)
        # 草稿里是模型给的原始分，落库时才夹。
        self.assertAlmostEqual(result.drafts[0].importance, 0.95)
        engine, path, _ = temp_engine()
        try:
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            written = engine.remember_many(result.drafts, scope)
            self.assertTrue(written)
            self.assertLessEqual(written[0].importance, 0.55)
        finally:
            engine.close()
            for suffix in ("", "-wal", "-shm"):
                p = path + suffix
                if os.path.exists(p):
                    os.remove(p)

    async def test_semantic_floored(self):
        reply = '[{"content":"他叫许杏玖","kind":"semantic","importance":0.05}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertTrue(result.drafts)
        self.assertAlmostEqual(result.drafts[0].importance, 0.05)
        engine, path, _ = temp_engine()
        try:
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            # 抽取出来的低分事实会被门槛拦掉（见下一个用例）；
            # 这里的下限夹紧走**导入**路径——导入不走门槛，但种类区间照样要夹。
            written = engine.import_memories(
                [{"content": "他叫许杏玖", "kind": "semantic", "importance": 0.05}],
                scope,
            )
            self.assertTrue(written)
            self.assertGreaterEqual(written[0].importance, 0.40)
        finally:
            engine.close()
            for suffix in ("", "-wal", "-shm"):
                p = path + suffix
                if os.path.exists(p):
                    os.remove(p)

    async def test_low_scored_fact_is_not_admitted(self):
        """模型自己给的分太低，就该被门槛挡在门外。

        旧实现里种类下限（semantic 0.40）会把 0.05 抬到 0.40，
        正好越过旧门槛 0.35 —— 模型不确定的东西照样入库。
        """

        reply = '[{"content":"他也许叫许杏玖","kind":"semantic","importance":0.30}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertTrue(result.drafts)
        engine, path, _ = temp_engine()
        try:
            scope = Scope(platform="qq", chat_type="session", chat_id="s1")
            written = engine.remember_many(result.drafts, scope)
            self.assertEqual(written, [], "低于门槛的稳定事实不该入库")
        finally:
            engine.close()
            for suffix in ("", "-wal", "-shm"):
                p = path + suffix
                if os.path.exists(p):
                    os.remove(p)


class TestEngineKindThreshold(unittest.TestCase):
    """事件类记忆的入库门槛要低于稳定事实，否则淡化梯度没机会上场。"""

    def setUp(self):
        from tests.helpers import temp_engine

        self.engine, self.path, self.config = temp_engine()
        from memfade import Scope

        self.scope = Scope(platform="qq", chat_type="session", chat_id="s1")

    def tearDown(self):
        self.engine.close()
        import os

        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                os.remove(p)


class TestEditMemory(unittest.TestCase):
    """改一条记忆：抽取会出错，出错之后得能自己改。"""

    def setUp(self):
        from tests.helpers import temp_engine

        self.engine, self.path, self.config = temp_engine()
        from memfade import Scope

        self.scope = Scope(platform="qq", chat_type="session", chat_id="s1")

    def tearDown(self):
        self.engine.close()
        import os

        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                os.remove(p)

    def _remember(self, content: str) -> int:
        written = self.engine.remember(
            MemoryDraft(content=content, kind=MemoryKind.SEMANTIC, importance=0.8),
            self.scope,
        )
        self.assertIsNotNone(written)
        return written.id

    def test_content_change_is_searchable_and_old_text_is_gone(self):
        mid = self._remember("季怡每天只有一节课")
        updated = self.engine.edit_memory(mid, content="季怡今天只有一节课")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "季怡今天只有一节课")
        hits = self.engine.recall("一节课", self.scope, top_k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].memory.content, "季怡今天只有一节课")
        # 旧内容不该还能搜到（全文索引要跟着改写重建）。
        old = self.engine.recall("每天", self.scope, top_k=5)
        self.assertFalse([h for h in old if "每天" in h.memory.content])

    def test_kind_and_quote_are_settable(self):
        mid = self._remember("季怡准备发一篇招人帖")
        updated = self.engine.edit_memory(
            mid,
            kind=MemoryKind.EPISODIC,
            quote="我想发一篇小黑盒招人",
            importance=0.3,
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.kind, MemoryKind.EPISODIC)
        self.assertEqual(updated.meta.get("quote"), "我想发一篇小黑盒招人")
        self.assertAlmostEqual(updated.importance, 0.3)

    def test_missing_memory_returns_none(self):
        self.assertIsNone(self.engine.edit_memory(999999, content="x"))

    def test_empty_content_is_rejected(self):
        mid = self._remember("季怡每天只有一节课")
        self.assertIsNone(self.engine.edit_memory(mid, content="   "))

    def test_episodic_below_general_threshold_is_kept(self):
        from memfade import MemoryDraft, MemoryKind

        self.config.extraction.min_importance = 0.35
        self.config.extraction.min_importance_episodic = 0.20
        mem = self.engine.remember(
            MemoryDraft(content="上周三一起吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.25),
            self.scope,
        )
        self.assertIsNotNone(mem, "一次性事件应当能入库，否则永远轮不到淡化")

    def test_semantic_below_threshold_is_dropped(self):
        from memfade import MemoryDraft, MemoryKind

        self.config.extraction.min_importance = 0.35
        mem = self.engine.remember(
            MemoryDraft(content="他叫许杏玖", kind=MemoryKind.SEMANTIC, importance=0.25),
            self.scope,
        )
        self.assertIsNone(mem, "稳定事实门槛要保持高")


class TestPromptTemplates(unittest.TestCase):
    """提示词里的字面花括号会被 ``str.format`` 当成占位符。

    这类错误只在**真正调用 LLM 时**才暴露，而且表现为
    ``KeyError: '"content"'`` 之类看不懂的异常——
    实机导入旧记忆时就是这么翻车的。
    """

    def test_all_prompts_are_formattable(self):
        from memfade import extract

        samples = {
            "max_chars": 40,
            "facts": "- 示例",
            "conversation": "用户：你好",
            "content": "一段原文",
        }
        for name in dir(extract):
            if not name.isupper():
                continue
            value = getattr(extract, name)
            if not isinstance(value, str):
                continue
            try:
                value.format(**samples)
            except (KeyError, IndexError, ValueError) as exc:
                self.fail(f"{name} 不能被 format：{exc}")


if __name__ == "__main__":
    unittest.main()

