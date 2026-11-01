"""MaiBot 适配器集成测试。

用一个假的宿主上下文（假的 llm / embed / maisaka.context）把整条链路跑通：
入站消息 → 召回 → 注入上下文 → 抽取 → 写入记忆。

没有装 maibot_sdk 时自动跳过。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import maibot_sdk  # noqa: F401

    HAS_SDK = True
except Exception:  # pragma: no cover
    HAS_SDK = False

MAIBOT_ROOT = Path(r"G:\AI\许杏玖-MaiBot\MaiBot")
HAS_HOST = MAIBOT_ROOT.is_dir() and (MAIBOT_ROOT / "src" / "plugin_runtime" / "webui_schema.py").is_file()


@unittest.skipUnless(HAS_HOST, "未找到 MaiBot 宿主")
class TestWebUIDeclaration(unittest.TestCase):
    """webui.json 必须过宿主自己的校验器，否则面板静默不出现。"""

    def test_webui_json_validates(self):
        import sys

        if str(MAIBOT_ROOT) not in sys.path:
            sys.path.insert(0, str(MAIBOT_ROOT))
        from src.plugin_runtime.webui_schema import load_webui_extension

        adapter_dir = REPO_ROOT / "adapters" / "maibot"
        extension = load_webui_extension(str(adapter_dir))
        self.assertIsNotNone(extension, "宿主没读到 webui.json")
        self.assertEqual(len(extension.pages), 2)


@unittest.skipUnless(HAS_HOST, "未找到 MaiBot 宿主")
class TestHostItemCompatibility(unittest.IsolatedAsyncioTestCase):
    """记忆块必须能通过宿主自己的反序列化器。

    格式不对时宿主只会打一条 warning 然后**静默丢弃**这次注入——
    插件看起来一切正常，但记忆其实从来没进过 prompt。
    """

    async def test_memory_item_survives_host_deserializer(self):
        import sys

        if str(MAIBOT_ROOT) not in sys.path:
            sys.path.insert(0, str(MAIBOT_ROOT))
        from src.llm_models.payload_content.context_item import CONTEXT_ITEM_SCHEMA_VERSION
        from src.llm_models.payload_content.context_protocol import ContextProtocolMode
        from src.plugin_runtime.hook_payloads import deserialize_prompt_items

        from adapters.maibot.plugin import MemFadePlugin, MemFadePluginConfig

        items = make_items("今天吃什么呢")
        body = "[系统·长期记忆] 许杏玖喜欢吃辣的东西"
        memory_item = MemFadePlugin._build_memory_item(items, body)
        merged = [memory_item] + items
        restored = deserialize_prompt_items(
            merged,
            item_schema_version=CONTEXT_ITEM_SCHEMA_VERSION,
            mode=ContextProtocolMode.REQUEST_CONTEXT,
            original_items=(),
        )
        self.assertEqual(len(restored), len(merged))
        texts = []
        for item in restored:
            for part in getattr(item, "parts", ()):
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        self.assertIn(body, texts, "记忆块没能被宿主还原")

    async def test_hook_output_is_accepted_by_host(self):
        """把钩子的真实输出喂给宿主的反序列化流程。

        这一条能复现真实事故：``modified_kwargs`` 若漏掉
        ``item_schema_version``，宿主会判定「items 无效，已忽略」——
        记忆看着写进去了，其实从没进过 prompt。
        """

        import sys

        if str(MAIBOT_ROOT) not in sys.path:
            sys.path.insert(0, str(MAIBOT_ROOT))
        from src.llm_models.payload_content.context_item import CONTEXT_ITEM_SCHEMA_VERSION
        from src.llm_models.payload_content.context_protocol import ContextProtocolMode
        from src.plugin_runtime.hook_payloads import deserialize_prompt_items

        from adapters.maibot.plugin import MemFadePlugin, MemFadePluginConfig

        plugin = MemFadePlugin()
        plugin._ctx = FakeCtx(Path(tempfile.mkdtemp(prefix="memfade-host-")), "[]")
        config = MemFadePluginConfig()
        config.plugin.enabled = True
        config.memory.min_importance = 0.1
        plugin._plugin_config_instance = config
        await plugin.on_load()
        try:
            from memfade import MemoryDraft, MemoryKind, Scope

            scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
            plugin._engine.remember(
                MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9),
                scope,
            )
            payload = {
                "items": make_items("今天吃什么呢"),
                "item_schema_version": CONTEXT_ITEM_SCHEMA_VERSION,
                "session_id": "sess-1",
                "selected_history_count": 2,
                "built_message_count": 2,
                "selection_reason": "test",
            }
            result = await plugin.handle_planner_before_request(**payload)
            self.assertIsNotNone(result)
            modified = result["modified_kwargs"]
            restored = deserialize_prompt_items(
                modified["items"],
                item_schema_version=modified.get("item_schema_version"),
                mode=ContextProtocolMode.REQUEST_CONTEXT,
                original_items=(),
            )
            self.assertEqual(len(restored), len(modified["items"]))
            texts = []
            for item in restored:
                for part in getattr(item, "parts", ()):
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            self.assertTrue(
                any("许杏玖喜欢吃辣的东西" in text for text in texts),
                "记忆没进还原后的上下文",
            )
        finally:
            await plugin.on_unload()



class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: List[str] = []

    async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        self.calls.append(str(prompt))
        return {"success": True, "response": self.reply, "reasoning": "", "model_name": "fake"}


class FakeEmbed:
    async def embed(self, texts, task_name="embedding", **kwargs):
        out = []
        for text in texts:
            # 用字符分布造一个确定性的假向量，够用来区分不同文本
            vec = [0.0] * 16
            for i, ch in enumerate(str(text)):
                vec[i % 16] += (ord(ch) % 17) / 17.0
            out.append({"embedding": vec, "model_name": "fake-embed"})
        return {"success": True, "results": out}


class FakeContext:
    def __init__(self):
        self.appended: List[Dict[str, Any]] = []

    async def append(self, stream_id="", segments=None, visible_text="", source_kind="", **kwargs):
        self.appended.append(
            {"stream_id": stream_id, "segments": segments, "visible_text": visible_text, "source_kind": source_kind}
        )
        return {"success": True}


class FakeMaisaka:
    def __init__(self):
        self.context = FakeContext()


class FakePaths:
    def __init__(self, base: Path):
        self.data_dir = base
        self.runtime_dir = base


class FakeCtx:
    def __init__(self, base: Path, llm_reply: str):
        self.logger = FakeLogger()
        self.paths = FakePaths(base)
        self.llm = FakeLLM(llm_reply)
        self.maisaka = FakeMaisaka()


def make_message(text: str, *, group_id: str = "", user_id: str = "1738157278", session_id: str = "sess-1"):
    info: Dict[str, Any] = {
        "user_info": {"user_id": user_id, "user_nickname": "季怡"},
        "group_info": {"group_id": group_id} if group_id else None,
        "additional_config": {},
    }
    return {
        "message_id": "m1",
        "platform": "qq",
        "session_id": session_id,
        "message_info": info,
        "processed_plain_text": text,
    }


def make_items(text: str, *, session_id: str = "sess-1") -> list:
    """构造一份序列化后的 Context Items（形态与宿主发给 hook 的一致）。"""

    return [
        {
            "item_type": "UserMessageItem",
            "meta": {
                "item_id": "real-user-1",
                "logical_turn_id": "turn-1",
                "timestamp": "2026-11-01T00:00:00",
            },
            "parts": [{"type": "text", "text": text}],
        },
        {
            "item_type": "AssistantMessageItem",
            "meta": {
                "item_id": "assistant-1",
                "logical_turn_id": "turn-1",
                "timestamp": "2026-11-01T00:00:01",
            },
            "parts": [{"type": "text", "text": "好的"}],
        },
    ]


@unittest.skipUnless(HAS_SDK, "未安装 maibot_sdk")
class TestMaiBotAdapter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from adapters.maibot.plugin import MemFadePlugin, MemFadePluginConfig

        self.tmp = tempfile.mkdtemp(prefix="memfade-adapter-")
        self.plugin = MemFadePlugin()
        self.plugin._ctx = FakeCtx(Path(self.tmp), "[]")
        config = MemFadePluginConfig()
        config.plugin.enabled = True
        config.plugin.verbose = False
        config.memory.extract_every_n = 1
        config.memory.extract_min_messages = 1
        config.memory.use_host_embedding = True
        config.memory.min_importance = 0.1
        self.plugin._plugin_config_instance = config
        await self.plugin.on_load()

    async def asyncTearDown(self):
        await self.plugin.on_unload()

    async def test_scope_parsing_group(self):
        from adapters.maibot.plugin import MemFadePlugin

        parsed = MemFadePlugin._parse_message(make_message("你好", group_id="1127155154"))
        self.assertEqual(parsed["chat_type"], "group")
        self.assertEqual(parsed["group_id"], "1127155154")
        self.assertEqual(parsed["session_id"], "sess-1")

    async def test_scope_parsing_private(self):
        from adapters.maibot.plugin import MemFadePlugin

        parsed = MemFadePlugin._parse_message(make_message("你好"))
        self.assertEqual(parsed["chat_type"], "private")
        self.assertEqual(parsed["user_id"], "1738157278")

    async def test_injects_context_into_maisaka(self):
        plugin = self.plugin
        engine = plugin._engine
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        result = await plugin.handle_planner_before_request(
            items=make_items("今天吃什么呢"), session_id="sess-1", item_schema_version=1
        )
        self.assertIsNotNone(result, "应当返回改写后的 items")
        new_items = result["modified_kwargs"]["items"]
        body = "\n".join(
            part["text"]
            for item in new_items
            for part in item.get("parts", [])
            if part.get("type") == "text"
        )
        self.assertIn("许杏玖喜欢吃辣的东西", body)
        self.assertEqual(len(new_items), 3, "应当多出一条记忆块")

    async def test_memory_item_inserted_before_last_user_message(self):
        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        items = make_items("今天吃什么呢")
        result = await plugin.handle_planner_before_request(
            items=items, session_id="sess-1", item_schema_version=1
        )
        new_items = result["modified_kwargs"]["items"]
        self.assertEqual(new_items[0]["item_type"], "UserMessageItem")
        self.assertTrue(
            new_items[0]["parts"][0]["text"].startswith("[系统·"),
            "注入块应当排在真实用户消息之前",
        )
        self.assertEqual(new_items[1]["parts"][0]["text"], "今天吃什么呢")

    async def test_replyer_injects_into_extra_prompt(self):
        """回复器走 extra_prompt 通道——不用序列化 item，稳。"""

        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        result = await plugin.handle_replyer_before_request(
            session_id="sess-1", reply_reason="今天吃什么呢", extra_prompt=""
        )
        self.assertIsNotNone(result)
        prompt = result["modified_kwargs"]["extra_prompt"]
        self.assertIn("许杏玖喜欢吃辣的东西", prompt)
        self.assertIn("系统·", prompt)

    async def test_replyer_keeps_existing_extra_prompt(self):
        plugin = self.plugin
        result = await plugin.handle_replyer_before_request(
            session_id="sess-1", reply_reason="在吗", extra_prompt="用简短点的语气。"
        )
        if result is not None:
            prompt = result["modified_kwargs"]["extra_prompt"]
            self.assertTrue(prompt.startswith("用简短点的语气。"), "不能覆盖宿主原有的 extra_prompt")

    async def test_profile_and_memory_do_not_duplicate(self):
        """同一件事不能在画像块和记忆块里各出现一次。"""

        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        result = await plugin.handle_planner_before_request(
            items=make_items("今天吃什么呢"), session_id="sess-1", item_schema_version=1
        )
        body = result["modified_kwargs"]["items"][0]["parts"][0]["text"]
        self.assertEqual(body.count("许杏玖喜欢吃辣的东西"), 1, "同一条记忆只应出现一次")

    async def test_memory_item_meta_is_valid(self):
        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        items = make_items("今天吃什么呢")
        result = await plugin.handle_planner_before_request(
            items=items, session_id="sess-1", item_schema_version=1
        )
        item = result["modified_kwargs"]["items"][0]
        meta = item["meta"]
        self.assertTrue(meta["item_id"])
        self.assertEqual(meta["logical_turn_id"], "turn-1", "应沿用当前轮的 logical_turn_id")
        self.assertTrue(meta["timestamp"])

    async def test_no_injection_without_memories(self):
        result = await self.plugin.handle_planner_before_request(
            items=make_items("在吗"), session_id="sess-1", item_schema_version=1
        )
        self.assertIsNone(result, "没有记忆时不应改写 items")

    async def test_modified_kwargs_keeps_schema_version(self):
        """宿主用 modified_kwargs **整体替换**载荷——漏字段就等于没注入。"""

        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        result = await plugin.handle_planner_before_request(
            items=make_items("今天吃什么呢"), session_id="sess-1", item_schema_version=1
        )
        modified = result["modified_kwargs"]
        self.assertIn("item_schema_version", modified, "必须原样带回 item_schema_version")
        self.assertEqual(modified["item_schema_version"], 1)
        self.assertEqual(modified["session_id"], "sess-1")

    async def test_profile_is_always_injected(self):
        """「关于对方」的稳定事实不该等被问到才想起。"""

        plugin = self.plugin
        from memfade import MemoryDraft, MemoryKind, Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="sess-1")
        plugin._engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.9), scope
        )
        result = await plugin.handle_planner_before_request(
            items=make_items("在吗"), session_id="sess-1", item_schema_version=1
        )
        self.assertIsNotNone(result, "画像应当常驻注入")
        body = result["modified_kwargs"]["items"][0]["parts"][0]["text"]
        self.assertIn("关于对方", body)
        self.assertIn("许杏玖喜欢辣的食物", body)

    async def test_extracts_and_writes(self):
        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"许杏玖喜欢辣的食物","kind":"semantic","importance":0.8,'
            '"subject":"许杏玖","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢吃辣的"))
        self.assertEqual(plugin._engine.store.count(), 1, "应当抽取并写入一条记忆")

    async def test_group_and_private_are_isolated(self):
        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"私聊里说的私密内容","kind":"semantic","importance":0.9,"cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("悄悄话"))
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        self.assertEqual(engine.store.count("qq:session:sess-other"), 0, "别的会话不该看到")

    async def test_command_scope_matches_injection_scope(self):
        """命令拿到的 stream_id 必须和注入用的是同一个作用域键。"""

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"许杏玖喜欢辣的食物","kind":"semantic","importance":0.8,"cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢吃辣的"))
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        # 命令侧只有 stream_id，没有 message 结构
        scope = plugin._scope_from_kwargs({"stream_id": "sess-1"})
        self.assertEqual(scope.key(), "qq:session:sess-1")
        self.assertEqual(len(engine.recall("辣", scope)), 1, "命令必须能查到刚写入的记忆")

    async def test_different_sessions_are_isolated(self):
        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"群里的记忆","kind":"semantic","importance":0.8,"cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("在群里说", group_id="1127155154"))
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        other = plugin._scope_from_kwargs({"stream_id": "sess-2"})
        self.assertEqual(len(engine.recall("群里的记忆", other)), 0, "别的会话不该看到")

    async def test_disabled_plugin_does_nothing(self):
        plugin = self.plugin
        plugin.config.plugin.enabled = False
        await plugin.handle_after_process(message=make_message("你好"))
        self.assertFalse(plugin._ctx.maisaka.context.appended)

    async def test_stats_api(self):
        stats = await self.plugin.memfade_stats()
        self.assertIn("by_status", stats)
        self.assertTrue(stats["enabled"])




