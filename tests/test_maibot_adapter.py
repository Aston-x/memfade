"""MaiBot 适配器集成测试。

用一个假的宿主上下文（假的 llm / embed / maisaka.context）把整条链路跑通：
入站消息 → 召回 → 注入上下文 → 抽取 → 写入记忆。

没有装 maibot_sdk 时自动跳过。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from memfade.types import MemoryDraft, MemoryKind, Scope

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import maibot_sdk  # noqa: F401

    HAS_SDK = True
except Exception:  # pragma: no cover
    HAS_SDK = False

MAIBOT_ROOT = Path(r"G:\AI\许杏玖-MaiBot\MaiBot")
if MAIBOT_ROOT.is_dir() and str(MAIBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIBOT_ROOT))
try:
    # 宿主的一堆依赖（sqlmodel 等）只在 MaiBot 自己的 venv 里齐，
    # 所以这里直接试导入，而不是只看文件在不在。
    from src.plugin_runtime.hook_payloads import deserialize_prompt_items as _probe  # noqa: F401
    from src.plugin_runtime.webui_schema import load_webui_extension as _probe2  # noqa: F401

    HAS_HOST = True
except Exception:  # pragma: no cover
    HAS_HOST = False


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

    def test_actions_declare_their_parameters(self):
        """动作必须声明参数，否则宿主直接 422。

        宿主校验入参时会检查「有没有未声明的参数」——
        面板的检索框把关键词传给动作，动作没声明 query，
        点一下就是 422「未声明的参数: query」。检索框直接是废的。
        """

        import json

        with open(REPO_ROOT / "adapters" / "maibot" / "webui.json", encoding="utf-8") as fh:
            data = json.load(fh)
        pages = {page["id"]: page for page in data.get("pages", [])}
        search = pages["memfade_search"]["actions"]["search"]
        self.assertIn("query", search.get("parameters") or {})
        self.assertIn("limit", search.get("parameters") or {})

    def test_search_action_covers_its_inputs(self):
        """页面上的输入框，至少得有一个动作声明得出来。

        面板把输入框的值传给被点的那个动作；
        动作没声明 → 宿主报 422「未声明的参数」。
        """

        import json

        with open(REPO_ROOT / "adapters" / "maibot" / "webui.json", encoding="utf-8") as fh:
            data = json.load(fh)
        for page in data.get("pages", []):
            actions = page.get("actions") or {}
            declared = set()
            for action in actions.values():
                declared |= set((action.get("parameters") or {}).keys())
            for block in page.get("content") or []:
                if block.get("type") != "input":
                    continue
                field = block.get("name")
                if field and field not in declared:
                    self.fail(
                        "页面 %s 的输入框 %s 没有任何动作声明它" % (page.get("id"), field)
                    )


@unittest.skipUnless(HAS_HOST, "未找到 MaiBot 宿主")
class TestToolVisibility(unittest.TestCase):
    """插件工具必须显式声明 visibility="visible"。

    宿主给插件工具的默认可见性是 **deferred**——
    不进 planner 的工具清单，模型得先调 tool_search() 才发现得了。
    「她自然会想到用这个工具」是不靠谱的（宿主自己的实测结论）。
    """

    def test_remember_tool_is_visible_to_model(self):
        from maibot_sdk.components import _COMPONENT_INFO_ATTR

        adapter_dir = REPO_ROOT / "adapters" / "maibot"
        sys.path.insert(0, str(adapter_dir))
        import plugin as adapter

        found = None
        for name in dir(adapter):
            obj = getattr(adapter, name)
            if not isinstance(obj, type):
                continue
            for attr in dir(obj):
                fn = getattr(obj, attr, None)
                info = getattr(fn, _COMPONENT_INFO_ATTR, None)
                if info is not None and getattr(info, "name", "") == "remember_something":
                    found = info
        self.assertIsNotNone(found, "没找到 remember_something 工具声明")
        self.assertEqual(found.metadata.get("visibility"), "visible")


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


class FakeEmbedMixin:
    """假嵌入：用字符分布造一个确定性的假向量，够用来区分不同文本。

    宿主给插件的 ``ctx.llm`` 同时提供 ``generate`` 和 ``embed``，
    假件也得两样都有——不然插件的向量通道在测试里永远是关着的，
    测不到「抽出来的记忆到底有没有向量」。
    """

    async def embed(self, texts, task_name="embedding", **kwargs):
        out = []
        for text in texts:
            vec = [0.0] * 16
            for i, ch in enumerate(str(text)):
                vec[i % 16] += (ord(ch) % 17) / 17.0
            out.append({"embedding": vec, "model_name": "fake-embed"})
        return {"success": True, "results": out}


class FakeLLM(FakeEmbedMixin):
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: List[str] = []
        self.delay: float = 0.0

    async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        self.calls.append(str(prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"success": True, "response": self.reply, "reasoning": "", "model_name": "fake"}


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
        self.send = FakeSend()


class FakeSend:
    """假发送通道——命令要往会话里回话。"""

    def __init__(self):
        self.sent: List[Any] = []

    async def text(self, text: str, stream_id: str = "", **kwargs):
        self.sent.append((text, stream_id))
        return {"success": True}


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

    async def test_existing_scopes_are_recovered_on_load(self):
        """插件重启后，库里已有的会话要重新进入「已知会话」。

        不恢复的话，那些没再发过消息的会话就再没人整理了——
        gist 压缩、合并、淘汰都靠整理任务做。
        """

        engine = self.plugin._engine
        engine.remember(
            MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.PREFERENCE, importance=0.9),
            Scope(platform="qq", chat_type="session", chat_id="old-scope"),
        )
        # 模拟重启：清空已知会话后重建引擎
        self.plugin._scopes.clear()
        self.plugin._rebuild_engine()
        self.assertIn("qq:session:old-scope", self.plugin._scopes)

    async def test_extracted_memories_get_embeddings(self):
        """抽出来的记忆必须带上向量。

        写入路径是同步的，宿主的嵌入服务是 async 的——
        有事件循环时同步取向量会堵住事件循环，所以同步路径会跳过。
        **不补这一下，记忆库里一条向量都没有，向量检索这条通道等于没开。**
        """

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"preference","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢辣的"))
        await plugin._drain_extractions()
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        mem = engine.store.iter_scope("qq:session:sess-1")[0]
        self.assertIsNotNone(mem.embedding, "抽出来的记忆没有向量，向量检索等于没开")

    async def test_pin_command_marks_memory(self):
        """「/记忆 置顶 <内容>」要把那条置顶（此后永不淡化）。"""

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"preference","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢辣的"))
        await plugin._drain_extractions()
        engine = plugin._engine
        before = engine.store.iter_scope("qq:session:sess-1")[0]
        self.assertFalse(before.pinned)

        await plugin.cmd_memory(
            stream_id="sess-1",
            matched_groups={"action": "置顶", "query": "辣的食物"},
        )
        after = engine.store.get(before.id)
        self.assertTrue(after.pinned, "置顶命令没生效")

    async def test_memory_command_shows_what_is_stored(self):
        """「/记忆」不能只给数字——主人得能一眼看出库里存了什么。

        「记忆库被污染了」这种问题，只有看得见才发现得了。
        """

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"preference","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢辣的"))
        await plugin._drain_extractions()
        await plugin.cmd_memory(stream_id="sess-1", matched_groups={})
        sent = " ".join(text for text, _ in plugin._ctx.send.sent)
        self.assertIn("记住了", sent)
        self.assertIn("季怡喜欢辣的食物", sent, "「/记忆」没有把记下的内容列出来")

    async def test_extraction_does_not_block_the_reply(self):
        """抽取要等 LLM（实机 30~60 秒），不能挡住主人的回复。

        抽取挂在 ``chat.receive.after_process`` 这个阻塞钩子上，
        在里面 await 一次 LLM，等于每 3 条消息就让主人多等一分钟。
        """

        plugin = self.plugin
        plugin._ctx.llm.delay = 0.3
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"preference","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        started = time.time()
        await plugin.handle_after_process(message=make_message("我喜欢辣的"))
        elapsed = time.time() - started
        self.assertLess(elapsed, 0.2, "抽取挡住了回复（%.2fs）" % elapsed)
        await plugin._drain_extractions()
        self.assertEqual(plugin._engine.store.count("qq:session:sess-1"), 1)

    async def test_pending_messages_survive_restart(self):
        """宿主重启不能把攒到一半还没抽的记忆丢掉。

        缓冲只在内存里的话，改个配置重启一下，
        用户刚说的那几句话就永远不会被记住了——
        记忆插件最不该有的失败。
        """

        from adapters.maibot.plugin import MemFadePlugin, MemFadePluginConfig

        plugin = self.plugin
        # 让消息先攒着（默认配置是每条就抽）
        plugin._plugin_config_instance.memory.extract_every_n = 3
        plugin._ctx.llm.reply = "[]"
        await plugin.handle_after_process(message=make_message("我喜欢吃辣"))
        await plugin._drain_extractions()
        await plugin.handle_after_process(message=make_message("我住在上海"))
        await plugin._drain_extractions()
        self.assertTrue(any(plugin._buffer.values()), "缓冲里应该有消息")

        # 模拟重启：新建一个插件实例，读同一个数据目录
        fresh = MemFadePlugin()
        fresh._ctx = FakeCtx(Path(self.tmp), "[]")
        config = MemFadePluginConfig()
        config.plugin.enabled = True
        config.memory.extract_every_n = 3
        config.memory.extract_min_messages = 2
        config.memory.use_host_embedding = True
        fresh._plugin_config_instance = config
        await fresh.on_load()
        try:
            restored = sum(len(v) for v in fresh._buffer.values())
            self.assertEqual(restored, 2, "重启后待抽取的消息没恢复")
        finally:
            await fresh.on_unload()

    async def test_selfcheck_does_not_touch_the_library(self):
        """诊断动作不能改记忆库。

        注入会**强化**记忆（强度回升、锚点重置、被想起次数 +1），
        而被想起次数还参与稳定性的间隔重复成长——
        诊断一次就等于给记忆续命。面板的自检按钮更不该。
        """

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"preference","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢辣的"))
        await plugin._drain_extractions()
        engine = plugin._engine
        before = {
            m.id: (m.recall_count, m.strength, m.anchor_at)
            for m in engine.store.iter_scope("qq:session:sess-1", limit=50)
        }
        self.assertTrue(before)

        await plugin.memfade_selfcheck(query="许杏玖喜欢辣的食物")
        after = {
            m.id: (m.recall_count, m.strength, m.anchor_at)
            for m in engine.store.iter_scope("qq:session:sess-1", limit=50)
        }
        self.assertEqual(before, after, "自检改动了记忆库")

    async def test_scope_parsing_private(self):
        from adapters.maibot.plugin import MemFadePlugin

        parsed = MemFadePlugin._parse_message(make_message("你好"))
        self.assertEqual(parsed["chat_type"], "private")
        self.assertEqual(parsed["user_id"], "1738157278")

    async def test_quoted_and_forwarded_content_is_stripped(self):
        """引用与转发不是用户的话——剥掉，否则抽取器会把它们记成用户的事。

        真实事故：主人引用机器人的「你昨晚一口气刷完Re」，
        记忆库里多了一条「许杏玖昨晚一口气刷完了Re」；
        主人转发蓝桥杯群公告求总结，
        记忆库里多了一条「季怡在备赛第18届蓝桥杯」。
        """

        from adapters.maibot.plugin import MemFadePlugin

        message = {
            "message_id": "m1",
            "platform": "qq",
            "session_id": "sess-1",
            "message_info": {"user_info": {"user_id": "1738157278", "user_nickname": "季怡"}},
            "processed_plain_text": "你昨晚一口气刷完Re，眼睛没熬坏吧  是在说我自己看的",
            "raw_message": [
                {
                    "type": "reply",
                    "data": {
                        "target_message_id": "m0",
                        "target_message_content": "你昨晚一口气刷完Re，眼睛没熬坏吧",
                        "target_message_sender_id": "2760135586",
                        "target_message_sender_nickname": "许杏玖",
                        "target_message_sender_cardname": None,
                    },
                },
                {"type": "text", "data": "是在说我自己看的"},
            ],
        }
        parsed = MemFadePlugin._parse_message(message)
        self.assertEqual(parsed["own_text"], "是在说我自己看的")
        self.assertNotIn("刷完Re", parsed["own_text"])
        self.assertEqual(parsed["speaker"], "季怡")

    async def test_quoted_text_is_kept_as_context(self):
        """被引用的原文要留成一行「机器人：」上下文——它是上下文，不是用户的话。

        删掉它的后果：用户回「买了」，抽取器不知道买的是什么，
        只能记一句含糊的废话或者干脆不记。
        """

        plugin = self.plugin
        # 让消息先攒着（默认配置是每条就抽，那样缓冲区会被清空）
        plugin._plugin_config_instance.memory.extract_every_n = 3
        message = {
            "message_id": "m1",
            "platform": "qq",
            "session_id": "sess-1",
            "message_info": {"user_info": {"user_id": "1738157278", "user_nickname": "季怡"}},
            "processed_plain_text": "Master，《影之刃零》买了吗 买了，昨天已经可以玩了",
            "raw_message": [
                {
                    "type": "reply",
                    "data": {
                        "target_message_id": "m0",
                        "target_message_content": "Master，《影之刃零》买了吗",
                        "target_message_sender_id": "2760135586",
                        "target_message_sender_nickname": "许杏玖",
                    },
                },
                {"type": "text", "data": "买了，昨天已经可以玩了"},
            ],
        }
        await plugin.handle_after_process(message=message)
        buffer = plugin._buffer.get("qq:session:sess-1") or []
        roles = [(entry.get("role"), entry.get("content")) for entry in buffer]
        self.assertIn(("assistant", "Master，《影之刃零》买了吗"), roles)
        self.assertIn(("user", "买了，昨天已经可以玩了"), roles)

    async def test_forward_block_is_stripped_from_plain_text(self):
        """拿不到组件时，退一步把「【合并转发消息: ...】」整块切掉。"""

        from adapters.maibot.plugin import MemFadePlugin

        text = "【合并转发消息: \n【季怡】: 群公告正文\n】 帮我总结所有内容"
        self.assertEqual(
            MemFadePlugin._strip_leading_blocks(text),
            "帮我总结所有内容",
        )

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
            '[{"content":"季怡喜欢辣的食物","kind":"semantic","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢吃辣的"))
        await plugin._drain_extractions()
        self.assertEqual(plugin._engine.store.count(), 1, "应当抽取并写入一条记忆")

    async def test_group_and_private_are_isolated(self):
        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"私聊里说的私密内容","kind":"semantic","importance":0.9,"cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("悄悄话"))
        await plugin._drain_extractions()
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        self.assertEqual(engine.store.count("qq:session:sess-other"), 0, "别的会话不该看到")

    async def test_command_scope_matches_injection_scope(self):
        """命令拿到的 stream_id 必须和注入用的是同一个作用域键。"""

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"semantic","importance":0.8,"cross_scope":false}]'
        )
        await plugin.handle_after_process(message=make_message("我喜欢吃辣的"))
        await plugin._drain_extractions()
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
        await plugin._drain_extractions()
        engine = plugin._engine
        self.assertEqual(engine.store.count("qq:session:sess-1"), 1)
        other = plugin._scope_from_kwargs({"stream_id": "sess-2"})
        self.assertEqual(len(engine.recall("群里的记忆", other)), 0, "别的会话不该看到")

    async def test_disabled_plugin_does_nothing(self):
        plugin = self.plugin
        plugin.config.plugin.enabled = False
        await plugin.handle_after_process(message=make_message("你好"))
        await plugin._drain_extractions()
        self.assertFalse(plugin._ctx.maisaka.context.appended)

    async def test_idle_flush_extracts_short_conversations(self):
        """一天只聊两句的机器人，也要能抽出记忆。

        实机发现：许杏玖的群里和私聊各只来了 2 条消息，
        而 extract_every_n=3 —— 永远攒不到阈值，记忆一直是空的。
        所以加了「会话安静下来就冲刷」。
        """

        plugin = self.plugin
        plugin._ctx.llm.reply = (
            '[{"content":"季怡喜欢辣的食物","kind":"semantic","importance":0.8,'
            '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}]'
        )
        # 复现实机配置：攒够 3 条才抽
        plugin.config.memory.extract_every_n = 3
        # 只来一条消息，达不到 extract_every_n=3
        await plugin.handle_after_process(message=make_message("我喜欢吃辣"))
        await plugin._drain_extractions()
        self.assertEqual(plugin._engine.store.count("qq:session:sess-1"), 0)
        # 假装这段对话已经安静了 30 分钟
        plugin._buffer_at["qq:session:sess-1"] = time.time() - 1800
        await plugin._flush_idle_buffers()
        self.assertEqual(plugin._engine.store.count("qq:session:sess-1"), 1)

    async def test_stats_api(self):
        stats = await self.plugin.memfade_stats()
        self.assertIn("by_status", stats)
        self.assertTrue(stats["enabled"])




