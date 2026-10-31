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


if __name__ == "__main__":
    unittest.main()

