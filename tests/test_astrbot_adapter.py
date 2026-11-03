"""AstrBot 适配器测试。

AstrBot 整包导入会拉起 anthropic / apscheduler / sqlmodel 一整条依赖链，
本机装不全，所以这里分两层验证：

1. **静态核对**：用 ast 解析 AstrBot SDK 源码，确认适配器依赖的符号
    确实存在、签名确实对得上（装得上 SDK 的机器上这层最直接）；
2. **桩件测试**：把 astrbot.api.* 换成最小桩件，测适配器自己的逻辑。

装不上 SDK 时自动跳过。
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tests.helpers import temp_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _find_sdk() -> Path:
    """找 AstrBot SDK 源码目录。"""

    try:
        import importlib.util

        spec = importlib.util.find_spec("astrbot")
        if spec and spec.origin:
            # find_spec 给的是包目录本身，SDK 根目录是它的上一级
            return Path(spec.origin).parent.parent
    except Exception:
        pass
    for env in ("ASTRBOT_SDK_PATH",):
        path = os.environ.get(env)
        if path and (Path(path) / "astrbot" / "api").is_dir():
            return Path(path)
    for candidate in (Path(r"B:\AI\_research_memplug\astrbot_sdk"),):
        if (candidate / "astrbot" / "api").is_dir():
            return candidate
    return Path()


SDK_ROOT = _find_sdk()
HAS_SDK = (SDK_ROOT / "astrbot" / "api").is_dir()


def _symbols(path: Path) -> set:
    """取一个模块里定义的顶层名字 + 类成员名。"""

    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                names |= _class_members(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _class_members(node: ast.ClassDef) -> set:
    """取一个类里的方法名与字段名。"""

    out: set = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(item.name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            out.add(item.target.id)
    return out


@unittest.skipUnless(HAS_SDK, "未找到 AstrBot SDK")
class TestAstrBotApiSurface(unittest.TestCase):
    """适配器依赖的 AstrBot API 必须真的存在。"""

    def test_filter_exposes_llm_hooks(self):
        names = _symbols(SDK_ROOT / "astrbot" / "api" / "event" / "filter" / "__init__.py")
        self.assertIn("on_llm_request", names)
        self.assertIn("on_llm_response", names)

    def test_star_module_exports(self):
        names = _symbols(SDK_ROOT / "astrbot" / "api" / "star" / "__init__.py")
        for symbol in ("Context", "Star", "register"):
            self.assertIn(symbol, names)

    def test_event_helpers_exist(self):
        names = _symbols(SDK_ROOT / "astrbot" / "core" / "platform" / "astr_message_event.py")
        for method in ("get_message_str", "get_group_id", "get_sender_id", "get_session_id"):
            self.assertIn(method, names, f"适配器用到了 event.{method}")

    def test_provider_request_has_system_prompt(self):
        names = _symbols(SDK_ROOT / "astrbot" / "core" / "provider" / "entities.py")
        self.assertIn("system_prompt", names, "适配器要改写 request.system_prompt")

    def test_register_functions_exist(self):
        names = _symbols(SDK_ROOT / "astrbot" / "core" / "star" / "register" / "star_handler.py")
        self.assertIn("register_on_llm_request", names)
        self.assertIn("register_on_llm_response", names)


def _install_stubs() -> None:
    """装上最小 astrbot.api 桩件，让适配器能被导入。"""

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    star = types.ModuleType("astrbot.api.star")
    event = types.ModuleType("astrbot.api.event")
    filter_mod = types.ModuleType("astrbot.api.event.filter")
    components = types.ModuleType("astrbot.api.message_components")

    class Context:  # noqa: D401 - 桩件
        pass

    class Star:
        def __init__(self, context=None, config=None):
            self.context = context
            self.config = config

    class AstrMessageEvent:
        pass

    def register(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def _filter_decorator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    class _CommandGroup:
        """指令组（真 SDK 里 ``@filter.command_group("记忆")`` 返回的就是它）。"""

        def __init__(self, func):
            self.func = func

        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    def _command_group(*args, **kwargs):
        def decorator(func):
            return _CommandGroup(func)

        return decorator

    class Plain:
        pass

    class StarTools:
        """桩件：插件会拿它算数据目录。"""

        @staticmethod
        def get_data_dir(plugin_name=None):
            from tests.helpers import temp_dir as _td

            return _td("memfade-star-")

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    api.logger = _Logger()
    star.Context = Context
    star.StarTools = StarTools
    star.Star = Star
    star.register = register
    event.AstrMessageEvent = AstrMessageEvent
    filter_mod.on_llm_request = _filter_decorator
    filter_mod.on_llm_response = _filter_decorator
    filter_mod.command = _filter_decorator
    filter_mod.command_group = _command_group
    filter_mod.EventMessageType = object
    components.Plain = Plain
    astrbot.api = api
    api.star = star
    api.event = event
    api.message_components = components
    event.filter = filter_mod
    # 强制覆盖而不是 setdefault：真实 astrbot 包会拖起 anthropic 等一整条依赖链，
    # 万一之前有残留导入，这里必须以桩件为准。
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.star"] = star
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.event.filter"] = filter_mod
    sys.modules["astrbot.api.message_components"] = components


@unittest.skipUnless(HAS_SDK, "未找到 AstrBot SDK")
class TestAstrBotAdapterLogic(unittest.IsolatedAsyncioTestCase):
    """适配器自己的逻辑。"""

    async def asyncSetUp(self):
        _install_stubs()
        from adapters.astrbot.main import MemFadePlugin
        from memfade import MemFadeConfig
        from memfade.engine import MemoryEngine

        self.tmp = temp_dir("memfade-astrbot-")
        config = MemFadeConfig.from_dict(
            {
                "storage": {"db_path": os.path.join(self.tmp, "memfade.db"), "enable_vector": False},
                "extraction": {"min_importance": 0.15},
            }
        )
        self.engine = MemoryEngine(config)
        self.plugin = MemFadePlugin(context=None, config=None)
        self.plugin.engine = self.engine

    async def asyncTearDown(self):
        self.engine.close()

    def _event(self, text: str, group_id: str = "", sender_id: str = "u1"):
        from astrbot.api.event import AstrMessageEvent

        event = AstrMessageEvent()
        event.get_message_str = lambda: text
        event.get_group_id = lambda: group_id
        event.get_sender_id = lambda: sender_id
        event.get_session_id = lambda: "sess-1"
        # 指令要往会话里回话（真 SDK 里是 yield event.plain_result(...)）
        event.plain_result = lambda text: ("result", text)
        return event

    async def test_memory_status_command_lists_what_is_stored(self):
        """「记忆 状态」不能只给数字——得让人看得见库里存了什么。"""

        from memfade.types import MemoryDraft, MemoryKind

        scope = self.plugin._scopes_of(self._event("你好"))[-1]
        self.engine.remember(
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
        text = await self._collect(self.plugin.memory_status(self._event("记忆 状态")))
        self.assertIn("记住了", text)
        self.assertIn("季怡喜欢辣的食物", text)

    async def test_memory_search_command_requires_a_match(self):
        """显式检索必须真命中——不把「最近写入」但无关的记忆混进来。"""

        from memfade.types import MemoryDraft, MemoryKind

        scope = self.plugin._scopes_of(self._event("你好"))[-1]
        self.engine.remember(
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
        hit = await self._collect(self.plugin.memory_search(self._event("记忆 搜 辣"), "辣"))
        self.assertIn("辣的食物", hit)
        miss = await self._collect(
            self.plugin.memory_search(self._event("记忆 搜 显卡驱动"), "显卡驱动")
        )
        self.assertIn("没找到", miss)

    @staticmethod
    async def _collect(agen) -> str:
        """把 async generator 的产出收成一段文本（AstrBot 的指令是 yield 的）。"""

        parts = []
        async for item in agen:
            parts.append(item[1] if isinstance(item, tuple) else str(item))
        return "\n".join(parts)

    def test_scope_uses_group_when_present(self):
        """群聊要读两层：群级公共 + 当前说话人的人级私密。"""
        event = self._event("你好", group_id="1127155154")
        scopes = self.plugin._scopes_of(event)
        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0].chat_type, "group")
        self.assertEqual(scopes[0].chat_id, "1127155154")
        self.assertEqual(scopes[1].chat_type, "group_user")
        self.assertEqual(scopes[1].chat_id, "1127155154#u1")

    def test_scope_falls_back_to_private(self):
        """私聊只有一层。"""
        event = self._event("你好")
        scopes = self.plugin._scopes_of(event)
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].chat_type, "private")
        self.assertEqual(scopes[0].chat_id, "u1")

    async def test_injects_into_system_prompt(self):
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="astrbot", chat_type="group", chat_id="1127155154")
        self.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )

        class Request:
            system_prompt = "你是一个助手"

        request = Request()
        event = self._event("今天吃什么呢", group_id="1127155154")
        await self.plugin.on_llm_request(event, request)
        self.assertIn("许杏玖喜欢吃辣的东西", request.system_prompt)
        self.assertTrue(request.system_prompt.startswith("你是一个助手"), "不能覆盖原有 system_prompt")

