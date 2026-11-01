"""LangBot 适配器测试。

用**真实的 langbot-plugin SDK**（pip install langbot-plugin）跑，
而不是照着记忆写——API 对不上就会当场炸。
没装 SDK 时自动跳过。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 找不到就去找本地研究目录里的 SDK 副本：
# pip 装的版本不一定在每个 Python 环境里都在（MaiBot 的 venv 就没有），
# 但「跑在真 SDK 上」这个验证不能因为换个解释器就失效。
for _candidate in (
    os.environ.get("MEMFADE_SDK_LANGBOT", ""),
    r"B:\AI\_research_memplug\langbot_sdk",
):
    if _candidate and os.path.isdir(os.path.join(_candidate, "langbot_plugin")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

try:
    from langbot_plugin.api.entities import events  # noqa: F401
    from langbot_plugin.api.entities.builtin.provider.message import Message  # noqa: F401

    HAS_SDK = True
except Exception:  # pragma: no cover
    HAS_SDK = False


@unittest.skipUnless(HAS_SDK, "未安装 langbot-plugin")
class TestLangBotAdapter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from adapters.langbot.components.event_listener.memory_injector import (
            MemFadeInjector,
            _query_text,
            _session_of,
            message_text,
        )
        from adapters.langbot.main import MemFadePlugin

        self.MemFadeInjector = MemFadeInjector
        self._query_text = _query_text
        self._session_of = _session_of
        self.message_text = message_text

        self.tmp = tempfile.mkdtemp(prefix="memfade-langbot-")
        plugin = MemFadePlugin()
        plugin.engine = None
        self.plugin = plugin

        from memfade import MemFadeConfig
        from memfade.engine import MemoryEngine

        config = MemFadeConfig.from_dict(
            {
                "storage": {"db_path": os.path.join(self.tmp, "memfade.db"), "enable_vector": False},
                "extraction": {"min_importance": 0.15},
            }
        )
        self.engine = MemoryEngine(config)
        plugin.engine = self.engine

    async def asyncTearDown(self):
        self.engine.close()

    def _event_ctx(self, user_text: str, session_name: str = "sess-1"):
        """构造一个 PromptPreProcessing 的事件上下文。"""

        from langbot_plugin.api.entities.builtin.provider.message import Message
        from langbot_plugin.api.entities.context import EventContext
        from langbot_plugin.api.entities.events import PromptPreProcessing

        event = PromptPreProcessing(
            session_name=session_name,
            default_prompt=[Message(role="system", content="你是一个测试机器人")],
            prompt=[
                Message(role="user", content="之前说的话"),
                Message(role="assistant", content="好的"),
                Message(role="user", content=user_text),
            ],
        )
        ctx = EventContext(event_name="PromptPreProcessing", event=event)
        return ctx

    def test_message_text_handles_str_and_segments(self):
        from langbot_plugin.api.entities.builtin.provider.message import Message

        self.assertEqual(self.message_text(Message(role="user", content="你好")), "你好")
        self.assertEqual(self.message_text(Message(role="user", content=None)), "")

    def test_query_text_reads_last_user_message(self):
        ctx = self._event_ctx("今天吃什么呢")
        self.assertEqual(self._query_text(ctx), "今天吃什么呢")

    def test_session_name(self):
        ctx = self._event_ctx("你好", session_name="abc")
        self.assertEqual(self._session_of(ctx), "abc")

    async def test_injects_into_default_prompt(self):
        from memfade import MemoryDraft, MemoryKind
        from memfade import Scope

        scope = Scope(platform="langbot", chat_type="group", chat_id="sess-1")
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        injector = self.MemFadeInjector()
        injector.plugin = self.plugin
        ctx = self._event_ctx("今天吃什么呢")
        await injector._inject(ctx)
        contents = [m.content for m in ctx.event.default_prompt]
        self.assertTrue(any("许杏玖喜欢吃辣的东西" in str(c) for c in contents), "记忆没进 default_prompt")

    async def test_no_injection_without_memories(self):
        injector = self.MemFadeInjector()
        injector.plugin = self.plugin
        ctx = self._event_ctx("今天吃什么呢")
        await injector._inject(ctx)
        self.assertEqual(len(ctx.event.default_prompt), 1, "没有记忆时不该追加")


if __name__ == "__main__":
    unittest.main()

