"""图片方案集成的回归测试。

5 个新特性：自证报数、稳定记忆自动保护、L1 预算注入、轻量健康检查、四级压缩。
"""

from __future__ import annotations

import unittest

from memfade.types import MemoryDraft, MemoryKind, Scope
from memfade.textutil import utf8_bytes, compress_text
from tests.helpers import temp_engine


class TestSummaryLine(unittest.TestCase):
    """ConsolidationReport.summary_line() 输出一行自证。"""

    def test_summary_line_format(self):
        from memfade.consolidate import ConsolidationReport

        report = ConsolidationReport(
            scanned=100, decayed=5, merged=3, superseded=1, reflected=2
        )
        line = report.summary_line()
        self.assertIn("扫描=100", line)
        self.assertIn("合并=3", line)
        self.assertIn("归纳=2", line)
        self.assertNotIn("错误", line)

    def test_summary_line_with_errors(self):
        from memfade.consolidate import ConsolidationReport

        report = ConsolidationReport(scanned=50, errors=["boom", "crash"])
        line = report.summary_line()
        self.assertIn("错误=2", line)


class TestProtectedKindsAutoPin(unittest.TestCase):
    """偏好和关系类记忆写入时自动 pinned。"""

    def test_preference_auto_pinned(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        draft = MemoryDraft(
            content="用户喜欢先看结论",
            kind=MemoryKind.PREFERENCE,
            importance=0.7,
        )
        mem = engine.remember(draft, scope)
        self.assertTrue(mem.pinned)

    def test_relation_auto_pinned(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        draft = MemoryDraft(
            content="用户是Alice的朋友",
            kind=MemoryKind.RELATION,
            importance=0.8,
        )
        mem = engine.remember(draft, scope)
        self.assertTrue(mem.pinned)

    def test_episodic_not_pinned(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        draft = MemoryDraft(
            content="今天吃了炸鸡",
            kind=MemoryKind.EPISODIC,
            importance=0.3,
        )
        mem = engine.remember(draft, scope)
        self.assertFalse(mem.pinned)

    def test_semantic_not_pinned(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        draft = MemoryDraft(
            content="地球是圆的",
            kind=MemoryKind.SEMANTIC,
            importance=0.5,
        )
        mem = engine.remember(draft, scope)
        self.assertFalse(mem.pinned)


class TestUtf8Bytes(unittest.TestCase):
    """utf8_bytes 精确计算 UTF-8 编码字节数。"""

    def test_ascii(self):
        self.assertEqual(utf8_bytes("hello"), 5)

    def test_chinese(self):
        # 中文每字 3 字节
        self.assertEqual(utf8_bytes("你好"), 6)

    def test_mixed(self):
        # "a" = 1 byte, "你" = 3 bytes
        self.assertEqual(utf8_bytes("a你"), 4)

    def test_empty(self):
        self.assertEqual(utf8_bytes(""), 0)


class TestCompressText(unittest.TestCase):
    """四级压缩。"""

    def test_level_0_no_change(self):
        text = "用户喜欢吃辣的食物"
        self.assertEqual(compress_text(text, 0), text)

    def test_level_2_first_sentence(self):
        text = "用户喜欢吃辣的食物。尤其是川菜。"
        result = compress_text(text, 2)
        self.assertIn("用户喜欢吃辣的食物", result)

    def test_level_3_first_20_chars(self):
        text = "这是一段很长的文字需要被压缩到二十个字符以内才行"
        result = compress_text(text, 3)
        self.assertLessEqual(len(result), 22)  # 20 + ellipsis

    def test_short_text_unchanged(self):
        text = "短文本"
        self.assertEqual(compress_text(text, 3), text)


class TestBudgetedContext(unittest.TestCase):
    """build_context_budgeted 按字节预算组装上下文。"""

    def test_returns_context_with_text(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        # Add some memories
        for i in range(3):
            draft = MemoryDraft(
                content=f"记忆内容第{i}条",
                kind=MemoryKind.SEMANTIC,
                importance=0.5,
            )
            engine.remember(draft, scope)

        ctx = engine.build_context_budgeted(
            "记忆内容", scope, total_budget=4096
        )
        self.assertIsInstance(ctx.text, str)
        # Should be within budget
        self.assertLessEqual(utf8_bytes(ctx.text), 4096)

    def test_profile_budget_split(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        # Add profile-eligible memories
        draft = MemoryDraft(
            content="用户名叫小明",
            kind=MemoryKind.RELATION,
            importance=0.9,
        )
        engine.remember(draft, scope)

        ctx = engine.build_context_budgeted(
            "小明", scope, profile_budget=512, event_budget=512
        )
        self.assertLessEqual(utf8_bytes(ctx.text), 1024)


class TestHealthCheck(unittest.TestCase):
    """health_check 返回全局状态。"""

    def test_empty_store(self):
        engine, _, _ = temp_engine()
        result = engine.health_check()
        self.assertEqual(result["total"], 0)
        self.assertTrue(result["healthy"])

    def test_with_memories(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        draft = MemoryDraft(
            content="用户喜欢辣",
            kind=MemoryKind.PREFERENCE,
            importance=0.7,
        )
        engine.remember(draft, scope)

        result = engine.health_check(scope)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["protected_count"], 1)
        self.assertTrue(result["healthy"])

    def test_protected_count_only_stable_kinds(self):
        engine, _, _ = temp_engine()
        scope = Scope(platform="qq", chat_type="private", chat_id="user1")
        # PREFERENCE = protected
        engine.remember(MemoryDraft(
            content="喜欢辣", kind=MemoryKind.PREFERENCE, importance=0.5
        ), scope)
        # EPISODIC = not protected
        engine.remember(MemoryDraft(
            content="吃了炸鸡", kind=MemoryKind.EPISODIC, importance=0.3
        ), scope)

        result = engine.health_check(scope)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["protected_count"], 1)
