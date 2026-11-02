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
class TestLangBotManifest(unittest.TestCase):
    """插件清单必须声明组件——**少了这段，插件能装上、能启用，但什么都不做**。

    LangBot 靠 ``manifest.yaml`` 的 ``spec.components`` 发现组件
    （事件监听器、指令）。适配器以前只写了 Python 类，没写清单，
    于是事件监听器从来没被加载过：装得上，行为全无。
    """

    ADAPTER = REPO_ROOT / "adapters" / "langbot"

    def _discover(self):
        import yaml
        from langbot_plugin.api.definition.components.manifest import ComponentManifest
        from langbot_plugin.utils.discover.engine import ComponentDiscoveryEngine

        manifest = yaml.safe_load((self.ADAPTER / "manifest.yaml").read_text(encoding="utf-8"))
        plugin = ComponentManifest(owner="test", manifest=manifest, rel_path="manifest.yaml")
        engine = ComponentDiscoveryEngine()
        found = {}
        # ⚠ fromDirs 的路径是**相对插件根目录**的（LangBot 跑插件时 CWD 就是插件根），
        # 所以这里要切到适配器目录再扫。
        previous = os.getcwd()
        os.chdir(self.ADAPTER)
        try:
            for group in (plugin.spec.get("components") or {}).values():
                for comp in engine.load_blueprint_comp_group(group, owner="builtin", no_save=True):
                    found[comp.kind] = comp
        finally:
            os.chdir(previous)
        return found

    def test_manifest_declares_components(self):
        found = self._discover()
        self.assertIn("EventListener", found, "事件监听器没被声明，注入与抽取都不会发生")
        self.assertIn("Command", found, "指令组件没被声明，/记忆 这类指令用不了")

    def test_component_classes_import(self):
        """清单指向的 Python 类必须真的能导入、属性名要对。"""

        found = self._discover()
        for kind, comp in found.items():
            cls = comp.get_python_component_class()
            self.assertTrue(callable(cls) or hasattr(cls, "__kind__"), kind)


@unittest.skipUnless(HAS_SDK, "未安装 langbot-plugin")
class TestLangBotCommands(unittest.IsolatedAsyncioTestCase):
    """指令组件的行为。"""

    async def asyncSetUp(self):
        from adapters.langbot.components.command.memory_command import MemoryCommand
        from adapters.langbot.main import MemFadePlugin
        from memfade import MemFadeConfig
        from memfade.engine import MemoryEngine

        self.tmp = tempfile.mkdtemp(prefix="memfade-langbot-cmd-")
        config = MemFadeConfig.from_dict(
            {
                "storage": {"db_path": os.path.join(self.tmp, "memfade.db"), "enable_vector": False},
                "extraction": {"min_importance": 0.15},
            }
        )
        plugin = MemFadePlugin()
        plugin.engine = MemoryEngine(config)
        self.plugin = plugin
        self.command = MemoryCommand()
        self.command.plugin = plugin
        self.scope = plugin.engine and None

    async def asyncTearDown(self):
        self.plugin.engine.close()

    def _ctx(self, command_text: str = ""):
        class _Session:
            session_name = "sess-1"
            launcher_type = "group"

        # 真正的 ExecuteContext 是 pydantic 模型，字段很多；
        # 指令只用得到 command_text 与 session，用轻量桩件就够。
        class _Ctx:
            def __init__(self):
                self.command_text = command_text
                self.full_command_text = command_text
                self.command = "记忆"
                self.crt_command = "状态"
                self.session = _Session()

        return _Ctx()

    async def test_status_command_lists_what_is_stored(self):
        """「记忆 状态」不能只给数字——得让人看得见库里存了什么。"""

        from memfade.types import MemoryDraft, MemoryKind

        scope = self.command._scope(self._ctx())
        self.plugin.engine.remember(
            MemoryDraft(
                content="季怡喜欢辣的食物",
                kind=MemoryKind.PREFERENCE,
                importance=0.8,
                subject="季怡",
                predicate="喜欢",
                object="辣的食物",
            ),
            scope,
        )
        text = await self._collect(self.command.status(self._ctx()))
        self.assertIn("记住了", text)
        self.assertIn("季怡喜欢辣的食物", text)

    async def test_search_command_requires_a_match(self):
        from memfade.types import MemoryDraft, MemoryKind

        scope = self.command._scope(self._ctx())
        self.plugin.engine.remember(
            MemoryDraft(
                content="季怡喜欢辣的食物",
                kind=MemoryKind.PREFERENCE,
                importance=0.8,
                subject="季怡",
                predicate="喜欢",
                object="辣的食物",
            ),
            scope,
        )
        hit = await self._collect(self.command.search(self._ctx("辣")))
        self.assertIn("辣的食物", hit)
        miss = await self._collect(self.command.search(self._ctx("显卡驱动")))
        self.assertIn("没找到", miss)

    @staticmethod
    async def _collect(agen) -> str:
        parts = []
        async for item in agen:
            parts.append(getattr(item, "text", "") or "")
        return "\n".join(parts)


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

