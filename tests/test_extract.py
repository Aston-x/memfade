"""抽取层测试。"""

from __future__ import annotations

import unittest

from memfade.extract import MemoryExtractor
from memfade.types import MemoryKind


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
    """显著性必须按种类夹紧——提示词不是保证。"""

    async def test_episodic_capped(self):
        reply = '[{"content":"上周三去吃了炸鸡","kind":"episodic","importance":0.95}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertTrue(result.drafts)
        self.assertLessEqual(result.drafts[0].importance, 0.55)

    async def test_semantic_floored(self):
        reply = '[{"content":"他叫许杏玖","kind":"semantic","importance":0.05}]'
        extractor = MemoryExtractor(llm=FakeLLM(reply))
        result = await extractor.extract([{"role": "user", "content": "x"}])
        self.assertTrue(result.drafts)
        self.assertGreaterEqual(result.drafts[0].importance, 0.40)


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

