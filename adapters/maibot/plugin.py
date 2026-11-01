"""memfade · MaiBot 适配器。

## 设计要点

**不要求用户配任何 API key。** 抽取用的 LLM 与向量模型都走宿主自己的
（``ctx.llm.generate`` / ``ctx.llm.embed``），跟 MaiBot 的模型配置保持一致。
用户已经在 MaiBot 里配好的模型，这里直接复用。

**注入时机选在 ``chat.receive.after_process``。**
这个 hook 在入站消息完成文本解析之后、回复生成之前触发：

* 太早（before_process）拿不到干净的纯文本；
* 太晚（after_send / post_send）这一轮的回复已经生成完了，记忆进不去。

**注入走 ``ctx.maisaka.context.append``**，聊天键用 ``message.session_id``
——MaiBot 的 Maisaka 运行时正是按这个键取会话历史的。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from maibot_sdk import API, Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

# 插件既可以以「已安装的 memfade 包」方式运行，也可以自带一份核心源码。
# 后者是 MaiBot 插件的常态（宿主没有插件依赖安装机制），所以两种都要能导入。
#
# 注意：必须**先**把插件自身目录插到 sys.path 最前面。
# 宿主加载插件时会把 plugins/ 目录塞进 sys.path，如果插件目录恰好就叫
# memfade，宿主的 plugins/memfade/ 会把本目录当成同名包，
# 于是 `from memfade import ...` 拿到的是插件目录而不是核心包。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:  # pragma: no cover - 取决于部署形态
    from memfade import MemoryDraft, MemoryEngine, MemoryKind, MemFadeConfig, Scope
except ImportError:  # pragma: no cover
    from memfade import MemoryDraft, MemoryEngine, MemoryKind, MemFadeConfig, Scope


PLUGIN_ID = "memfade.memory"
SOURCE_KIND = "plugin:memfade"


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用记忆插件")
    config_version: str = Field(default="2.0.0", description="配置版本")
    verbose: bool = Field(default=False, description="是否输出详细日志")


class MemorySectionConfig(PluginConfigBase):
    """记忆行为配置。"""

    __ui_label__ = "记忆"
    __ui_icon__ = "brain"
    __ui_order__ = 1

    inject_enabled: bool = Field(default=True, description="是否把记忆注入对话上下文")
    inject_header: str = Field(default="记忆", description="注入块的小标题")
    max_inject_tokens: int = Field(default=600, description="注入上下文的 token 预算")
    top_k: int = Field(default=8, description="每轮最多召回几条记忆")
    echo_max: int = Field(default=3, description="最多注入几条已淡出的旧记忆")
    profile_items: int = Field(default=6, description="「关于对方」画像最多几条")
    profile_chars: int = Field(default=300, description="画像块字数上限")

    extract_enabled: bool = Field(default=True, description="是否从对话中抽取记忆")
    extract_every_n: int = Field(default=3, description="每隔多少轮抽取一次")
    flush_idle_minutes: int = Field(default=10, description="会话安静多久后把攒下的消息抽成记忆")
    extract_min_messages: int = Field(default=2, description="一轮里至少几条消息才抽取")
    min_importance: float = Field(default=0.35, description="低于此显著性的内容不入库")

    use_host_llm: bool = Field(default=True, description="抽取与摘要复用宿主 LLM")
    llm_task_name: str = Field(default="memory", description="宿主模型任务名（memory/utils 均可）")
    use_host_embedding: bool = Field(default=True, description="向量检索复用宿主嵌入模型")

    db_filename: str = Field(default="memfade.db", description="记忆数据库文件名")
    cross_session: bool = Field(default=False, description="是否允许跨会话共享（默认关闭）")


class FadeSectionConfig(PluginConfigBase):
    """淡化策略配置。"""

    __ui_label__ = "淡化"
    __ui_icon__ = "trending-down"
    __ui_order__ = 2

    recall_alpha: float = Field(default=0.35, description="被想起时的强度回升幅度")
    weaken_alpha: float = Field(default=0.5, description="被削弱时的强度下降幅度")
    archive_after_days: int = Field(default=0, description="归档多少天后物理删除（0=永不删）")
    maintain_interval_minutes: int = Field(default=60, description="整理任务的间隔（分钟）")
    gist_enabled: bool = Field(default=True, description="淡出时是否压缩成一行摘要")


class MemFadePluginConfig(PluginConfigBase):
    """插件总配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    memory: MemorySectionConfig = Field(default_factory=MemorySectionConfig)
    fade: FadeSectionConfig = Field(default_factory=FadeSectionConfig)


class HostLLMClient:
    """把宿主的 LLM 能力包装成 memfade 的 LLM 协议。"""

    def __init__(self, plugin: "MemFadePlugin", task_name: str = "memory") -> None:
        self._plugin = plugin
        self._task_name = task_name

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = await self._plugin.ctx.llm.generate(
            messages,
            task_name=self._task_name,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        if not isinstance(result, dict):
            return ""
        if not result.get("success", True):
            raise RuntimeError(str(result.get("error") or "宿主 LLM 调用失败"))
        return str(result.get("response") or "")


class HostEmbeddingClient:
    """把宿主的嵌入能力包装成 memfade 的 Embedding 协议。"""

    def __init__(self, plugin: "MemFadePlugin") -> None:
        self._plugin = plugin

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []
        result = await self._plugin.ctx.llm.embed(texts=items, task_name="embedding")
        if not isinstance(result, dict) or not result.get("success", False):
            return [[] for _ in items]
        out: List[List[float]] = []
        for entry in result.get("results") or []:
            vector = entry.get("embedding") if isinstance(entry, dict) else None
            out.append([float(v) for v in (vector or [])])
        while len(out) < len(items):
            out.append([])
        return out[: len(items)]


def _build_engine_config(plugin: "MemFadePlugin", db_path: str) -> MemFadeConfig:
    """把插件配置翻译成核心配置。"""

    cfg = plugin.config
    mem = cfg.memory
    fade = cfg.fade
    data = {
        "storage": {"db_path": db_path, "enable_fts": True, "enable_vector": mem.use_host_embedding},
        "extraction": {
            "enabled": mem.extract_enabled,
            "every_n_turns": max(1, int(mem.extract_every_n)),
            "min_messages": max(1, int(mem.extract_min_messages)),
            "min_importance": float(mem.min_importance),
            "max_per_batch": 8,
            "use_rules_fallback": True,
            "promote_cross_scope": bool(mem.cross_session),
        },
        "retrieval": {
            "top_k": max(1, int(mem.top_k)),
            "budget_tokens": max(50, int(mem.max_inject_tokens)),
            "echo_max": max(0, int(mem.echo_max)),
        },
        "fade": {
            "recall_alpha": float(fade.recall_alpha),
            "weaken_alpha": float(fade.weaken_alpha),
            "gist_enabled": bool(fade.gist_enabled),
            "archive_after_days": max(0, int(fade.archive_after_days)),
        },
        "maintenance": {"enabled": True, "interval_seconds": max(60, int(fade.maintain_interval_minutes) * 60)},
    }
    return MemFadeConfig.from_dict(data)


class MemFadePlugin(MaiBotPlugin):
    """记忆淡化插件。"""

    config_model = MemFadePluginConfig

    def __init__(self) -> None:
        super().__init__()
        self._engine: Optional[MemoryEngine] = None
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_at: Dict[str, float] = {}
        """每个会话最后一次收到消息的时间，用来判断「安静下来了」。"""

        self._scopes: set[str] = set()
        """见过的会话键，后台整理任务要遍历。"""

        self._counter: Dict[str, int] = {}
        self._last_maintain: float = 0.0
        self._maintain_task: Optional[asyncio.Task] = None
        self._maintain_ticks: int = 0
        """后台维护循环的心跳次数。用来确认循环真的在跑。"""
        self._last_search: List[Dict[str, Any]] = []
        self._recent_user_text: Dict[str, str] = {}
        """每个会话最近一条用户消息——回复器钩子拿不到 items，用它做检索词。"""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def on_load(self) -> None:
        """插件加载：建引擎、起整理任务。"""

        self._rebuild_engine()
        self._maintain_task = asyncio.create_task(self._maintenance_loop())

    async def on_unload(self) -> None:
        """插件卸载：收尾。"""

        if self._maintain_task is not None:
            self._maintain_task.cancel()
            try:
                await self._maintain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._maintain_task = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    async def _maintenance_loop(self) -> None:
        """后台循环：空闲冲刷 + 定期整理。

        机器人一天可能只收到两三条消息，光靠「攒够 N 条再抽」
        永远抽不出来；「说完一段安静下来」正是该总结的时刻。
        """

        try:
            while True:
                await asyncio.sleep(60)
                try:
                    await self._flush_idle_buffers()
                    await self._maintain_all_scopes()
                    self._maintain_ticks += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.ctx.logger.warning("[memfade] 后台维护出错: %s", exc)
        except asyncio.CancelledError:
            return

    async def _maintain_all_scopes(self) -> None:
        """对所有见过的会话跑一遍整理。"""

        engine = self._engine
        if engine is None:
            return
        interval = max(60, int(self.config.fade.maintain_interval_minutes) * 60)
        now = time.time()
        if now - self._last_maintain < interval:
            return
        self._last_maintain = now
        for key in list(self._scopes):
            scope = Scope(platform="qq", chat_type="session", chat_id=key.split(":", 2)[-1])
            try:
                report = await engine.consolidate(scope)
                if report.decayed or report.merged or report.superseded:
                    self.ctx.logger.info("[memfade] 整理完成（%s）: %s", key, report.as_dict())
            except Exception as exc:
                self.ctx.logger.warning("[memfade] 整理失败（%s）: %s", key, exc)

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        """配置热重载。"""

        del scope, config_data, version
        self._rebuild_engine()

    def _rebuild_engine(self) -> None:
        """按当前配置重建引擎。"""

        if self._engine is not None:
            self._engine.close()
            self._engine = None
        data_dir = self.ctx.paths.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        db_name = str(self.config.memory.db_filename or "memfade.db")
        db_path = str(data_dir / db_name)
        config = _build_engine_config(self, db_path)
        llm = None
        if self.config.memory.use_host_llm:
            task_name = str(self.config.memory.llm_task_name or "memory").strip() or "memory"
            llm = HostLLMClient(self, task_name)
        embedder = HostEmbeddingClient(self) if self.config.memory.use_host_embedding else None
        self._engine = MemoryEngine(config, llm=llm, embedder=embedder)
        self.ctx.logger.info("[memfade] 引擎就绪：%s", db_path)

    # ------------------------------------------------------------------ #
    # 会话解析
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_message(message: Any) -> Dict[str, Any]:
        """从序列化消息里抽出插件需要的字段。"""

        if not isinstance(message, dict):
            return {}
        info = message.get("message_info")
        info = info if isinstance(info, dict) else {}
        user_info = info.get("user_info")
        user_info = user_info if isinstance(user_info, dict) else {}
        group_info = info.get("group_info")
        group_info = group_info if isinstance(group_info, dict) else {}
        additional = info.get("additional_config")
        additional = additional if isinstance(additional, dict) else {}
        group_id = str(group_info.get("group_id") or "").strip()
        user_id = str(user_info.get("user_id") or "").strip()
        text = str(message.get("processed_plain_text") or "").strip()
        if not text:
            text = str(message.get("raw_message") or "").strip()
        session_id = str(message.get("session_id") or "").strip()
        if not session_id:
            for key in ("session_id", "stream_id", "chat_id"):
                value = additional.get(key)
                if value:
                    session_id = str(value).strip()
                    break
        return {
            "platform": str(message.get("platform") or "qq").strip() or "qq",
            "group_id": group_id,
            "user_id": user_id,
            "text": text,
            "session_id": session_id,
            "chat_type": "group" if group_id else "private",
        }

    def _scope_from_kwargs(self, kwargs: Dict[str, Any]) -> Scope:
        """从命令/工具的 kwargs 推作用域。

        命令拿不到完整的 message 结构，但 ``stream_id`` 就是 ``session_id``，
        所以直接用它当作用域键——和注入路径完全一致，不会「命令查不到记忆」。
        """

        parsed = self._parse_message(kwargs.get("message"))
        if parsed.get("session_id"):
            return self._scope_of(parsed)
        stream_id = str(kwargs.get("stream_id") or kwargs.get("session_id") or "").strip()
        return Scope(
            platform=parsed.get("platform") or "qq",
            chat_type="session",
            chat_id=stream_id or "unknown",
            cross_scope=False,
        )

    def _scope_of(self, parsed: Dict[str, Any]) -> Scope:
        """构造作用域。

        用 **session_id** 而不是群号/用户号当作用域键，理由：

        * MaiBot 的 ``session_id = md5(平台+账号+作用域+群号)``——
          **一个群一份、私聊按人一份**，天生就是「会话」粒度；
        * 命令收到的 ``stream_id`` 就是 ``session_id``，
          所以命令与注入用的是同一个作用域键；
        * 它是 md5，重启后不变。

        群号/用户号另存进 ``meta``，方便人工审计时认人。
        """

        chat_id = parsed.get("session_id") or "unknown"
        return Scope(
            platform=parsed.get("platform") or "qq",
            chat_type="session",
            chat_id=str(chat_id),
            cross_scope=False,
        )

    # ------------------------------------------------------------------ #
    # 主链路：注入 + 抽取
    # ------------------------------------------------------------------ #

    @HookHandler(
        "chat.receive.after_process",
        name="memfade_inject_and_extract",
        description="召回相关记忆注入上下文，并按节奏抽取新记忆",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_after_process(self, message: Any = None, **kwargs: Any) -> None:
        """入站消息处理完成后：注入记忆 + 记录待抽取的对话。"""

        del kwargs
        if self._engine is None or not self.config.plugin.enabled:
            return None
        parsed = self._parse_message(message)
        if not parsed:
            return None
        text = parsed.get("text") or ""
        scope = self._scope_of(parsed)
        session_id = parsed.get("session_id") or ""
        if session_id and text:
            # 回复器钩子拿不到 items，用这里旁听到的最近一条用户消息做检索词
            self._recent_user_text[session_id] = text
        self._scopes.add(scope.key())

        # 1) 收集对话，达到节奏就抽取
        if self.config.memory.extract_enabled and text:
            if self.config.plugin.verbose:
                self.ctx.logger.info(
                    "[memfade] 旁听消息 scope=%s 第 %d 条", key, self._counter.get(key, 0) + 1
                )
            await self._maybe_extract(parsed, scope)

        # 2) 整理交给后台循环（_maintenance_loop），这里不再重复触发
        return None

    @HookHandler(
        "maisaka.planner.before_request",
        name="memfade_inject_planner",
        description="把记忆注入规划请求的上下文",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_planner_before_request(
        self, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """规划器发起模型请求前注入记忆。"""

        return await self._inject_into_items(kwargs)

    @HookHandler(
        "maisaka.replyer.before_request",
        name="memfade_inject_replyer",
        description="把记忆注入回复器的额外提示词",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_replyer_before_request(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """回复器发起模型请求前注入记忆。

        这里走 ``extra_prompt`` 而不是改 Context Items：
        它是宿主为插件预留的「追加一段提示词」通道，
        不用序列化/反序列化 item，格式出错的风险小得多
        （群管理插件在生产里也是这么用的）。
        """

        return await self._inject_into_extra_prompt(kwargs)

    async def _inject_into_extra_prompt(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把记忆追加到本次回复的 extra_prompt。"""

        engine = self._engine
        if engine is None or not self.config.plugin.enabled:
            return None
        if not self.config.memory.inject_enabled:
            return None
        session_id = str(kwargs.get("session_id") or "")
        # 优先用宿主给的「为什么回复」当检索词；没有就用旁听到的最近一条用户消息
        query = str(kwargs.get("reply_reason") or "").strip()
        if not query:
            query = self._recent_user_text.get(session_id, "")
        if not query:
            return None
        scope = Scope(platform="qq", chat_type="session", chat_id=session_id or "unknown")
        try:
            vector = None
            if self.config.memory.use_host_embedding:
                vector = await engine.embed_query(query)
            profile = engine.retriever.profile(
                scope.key(),
                max_items=max(0, int(self.config.memory.profile_items)),
                max_chars=max(0, int(self.config.memory.profile_chars)),
            )
            context = engine.build_context(
                query,
                scope,
                query_vector=vector,
                header=self.config.memory.inject_header or "记忆",
                exclude_ids=profile.ids,
            )
            body = self._wrap_context(context.text, profile.text)
            if not body:
                return None
            modified = dict(kwargs)
            base = str(modified.get("extra_prompt") or "").strip()
            modified["extra_prompt"] = (base + "\n\n" + body).strip() if base else body
            if context.hits:
                engine.reinforce([hit.memory.id for hit in context.hits])
            return {"success": True, "action": "continue", "modified_kwargs": modified}
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 注入回复提示失败: %s", exc)
            return None

    async def _inject_into_items(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把召回的记忆写进 Context Items。

        为什么用 ``maisaka.*.before_request`` 而不是 ``chat.receive.after_process``：

        * ``after_process`` 跑在**会话创建之前**（宿主顺序是
          after_process hook → register_message → get_or_create_session），
          那时 ``maisaka.context.append`` 会因为找不到聊天流而抛异常；
        * 这里跑在会话创建之后、模型请求之前，``session_id`` 直接就在载荷里，
          正好是记忆的作用域键；
        * 而且能**直接改写真正发给模型的 Context Items**，
          不会像「往历史里追加」那样被上下文裁剪顺手丢掉。
        """

        engine = self._engine
        if engine is None or not self.config.plugin.enabled:
            return None
        if not self.config.memory.inject_enabled:
            return None
        items = kwargs.get("items")
        if not isinstance(items, list) or not items:
            return None
        session_id = str(kwargs.get("session_id") or "")
        query = self._last_user_text(items)
        scope = Scope(platform="qq", chat_type="session", chat_id=str(session_id or "unknown"))
        try:
            vector = None
            if self.config.memory.use_host_embedding:
                vector = await engine.embed_query(query)
            profile = engine.retriever.profile(
                scope.key(),
                max_items=max(0, int(self.config.memory.profile_items)),
                max_chars=max(0, int(self.config.memory.profile_chars)),
            )
            context = engine.build_context(
                query,
                scope,
                query_vector=vector,
                header=self.config.memory.inject_header or "记忆",
                # 画像里已经写过的，别在记忆块里再写一遍
                exclude_ids=profile.ids,
            )
            body = self._wrap_context(context.text, profile.text)
            if not body:
                return None
            new_items = list(items)
            new_items.insert(self._insert_index(items), self._build_memory_item(items, body))
            if context.hits:
                engine.reinforce([hit.memory.id for hit in context.hits])
            # ⚠ 必须**整体复制**原始 kwargs 再替换 items：
            # 宿主用 modified_kwargs 整体替换载荷，漏掉 item_schema_version
            # 会被判成「items 无效，已忽略」——记忆看着进去了，其实没进。
            modified = dict(kwargs)
            modified["items"] = new_items
            if self.config.plugin.verbose:
                self.ctx.logger.info(
                    "[memfade] 注入 %d 条记忆（事实 %d / 回声 %d）到 %d 个 item 中",
                    len(context.hits),
                    len(context.facts),
                    len(context.echoes),
                    len(new_items),
                )
            return {"success": True, "action": "continue", "modified_kwargs": modified}
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 注入记忆失败: %s", exc)
            return None

    @staticmethod
    def _last_user_text(items: list) -> str:
        """从 Context Items 里取最后一条用户消息的纯文本。"""

        for item in reversed(items):
            if not isinstance(item, dict) or item.get("item_type") != "UserMessageItem":
                continue
            parts = item.get("parts")
            if not isinstance(parts, list):
                continue
            chunks = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks)
        return ""

    @staticmethod
    def _insert_index(items: list) -> int:
        """记忆块插在最后一条用户消息之前——它是背景，不是用户刚说的话。"""

        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            if isinstance(item, dict) and item.get("item_type") == "UserMessageItem":
                return index
        return len(items)

    @staticmethod
    def _build_memory_item(items: list, body: str) -> Dict[str, Any]:
        """构造一条合法的 UserMessageItem 快照。

        格式由宿主的 ``serialize_context_item_snapshot`` 定义：
        ``item_type`` / ``meta``（item_id、logical_turn_id、timestamp）/ ``parts``。
        ``logical_turn_id`` 沿用最后一条用户消息的，
        这样它属于当前这一轮，不会破坏工具调用锚点校验。
        """

        logical_turn_id = None
        for item in reversed(items):
            if not isinstance(item, dict) or item.get("item_type") != "UserMessageItem":
                continue
            meta = item.get("meta")
            if isinstance(meta, dict):
                logical_turn_id = meta.get("logical_turn_id")
            break
        return {
            "item_type": "UserMessageItem",
            "meta": {
                "item_id": f"memfade-{uuid.uuid4().hex[:16]}",
                "logical_turn_id": logical_turn_id,
                "timestamp": datetime.now().isoformat(),
            },
            "parts": [{"type": "text", "text": body}],
        }

    @staticmethod
    def _wrap_context(text: str, profile: str = "") -> str:
        """给记忆块加系统前缀。

        不加前缀的话，模型会把这段当成「用户刚说的话」，
        从而做出「用户刚刚告诉了我这件事」这类错误反应。
        """

        blocks = []
        if profile:
            blocks.append(
                "[系统·关于对方] 下面是你对这位用户的长期印象，"
                "属于你已经知道的信息，不要重复确认：\n" + profile
            )
        if text:
            blocks.append(
            "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
            "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
                f"{text}"
            )
        return "\n\n".join(blocks)

    async def _maybe_extract(self, parsed: Dict[str, Any], scope: Scope) -> None:
        """按节奏抽取记忆。"""

        engine = self._engine
        if engine is None:
            return
        key = scope.key()
        buffer = self._buffer.setdefault(key, [])
        buffer.append({"role": "user", "content": parsed.get("text") or ""})
        self._buffer_at[key] = time.time()
        count = self._counter.get(key, 0) + 1
        self._counter[key] = count
        every = max(1, int(self.config.memory.extract_every_n))
        min_messages = max(1, int(self.config.memory.extract_min_messages))
        if count % every != 0 or len(buffer) < min_messages:
            return
        pending = list(buffer)
        buffer.clear()
        try:
            meta = {
                "group_id": parsed.get("group_id") or "",
                "user_id": parsed.get("user_id") or "",
                "chat_type": parsed.get("chat_type") or "",
            }
            written = await engine.remember_from_messages(pending, scope, source=key, meta=meta)
            if written:
                self.ctx.logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 抽取记忆失败: %s", exc)
            # 抽取失败不丢内容，放回缓冲区下次一起试
            buffer.extend(pending)

    async def _flush_idle_buffers(self) -> None:
        """把「安静下来」的会话攒下的消息抽成记忆。

        为什么需要这个：机器人一天可能只收到两三条消息，
        而 ``extract_every_n`` 是 3——那点消息永远攒不到阈值，
        记忆就一直是空的。真实对话里「说完一段安静下来」
        正是该总结的时刻。
        """

        engine = self._engine
        if engine is None or not self.config.memory.extract_enabled:
            return
        idle_seconds = max(0, int(self.config.memory.flush_idle_minutes)) * 60
        if idle_seconds <= 0:
            return
        now = time.time()
        for key in list(self._buffer.keys()):
            last = self._buffer_at.get(key, 0.0)
            if not last or now - last < idle_seconds:
                continue
            pending = list(self._buffer.get(key) or [])
            if not pending:
                continue
            self._buffer[key] = []
            self._buffer_at.pop(key, None)
            self._counter.pop(key, None)
            scope = Scope(platform="qq", chat_type="session", chat_id=key.split(":", 2)[-1])
            try:
                written = await engine.remember_from_messages(pending, scope, source=key)
                if written:
                    self.ctx.logger.info(
                        "[memfade] 空闲冲刷抽取到 %d 条新记忆（%s）", len(written), key
                    )
            except Exception as exc:
                self.ctx.logger.warning("[memfade] 空闲冲刷失败: %s", exc)
                self._buffer[key] = pending
                self._buffer_at[key] = now

    async def _maybe_maintain(self, scope: Scope) -> None:
        """整理单个会话（带节流）。后台循环用的是 _maintain_all_scopes。"""

        engine = self._engine
        if engine is None:
            return
        interval = max(60, int(self.config.fade.maintain_interval_minutes) * 60)
        now = time.time()
        if now - self._last_maintain < interval:
            return
        self._last_maintain = now
        try:
            report = await engine.consolidate(scope)
            self.ctx.logger.info("[memfade] 整理完成: %s", report.as_dict())
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 整理失败: %s", exc)

    # ------------------------------------------------------------------ #
    # 命令
    # ------------------------------------------------------------------ #

    @Command(
        "记忆",
        description="查看记忆。/记忆 查当前会话记住的东西，/记忆 搜 <关键词>，/记忆 整理 立刻跑一轮整理",
        pattern=r"^/(?:记忆|memory)(?:\s+(?P<action>搜|search|整理|maintain)?)?(?:\s+(?P<query>.+))?$",
    )
    async def cmd_memory(self, **kwargs: Any) -> tuple:
        """记忆查询命令。"""

        stream_id = str(kwargs.get("stream_id") or "")
        groups = kwargs.get("matched_groups")
        groups = groups if isinstance(groups, dict) else {}
        action = str(groups.get("action") or "").strip()
        query = str(groups.get("query") or "").strip()
        engine = self._engine
        if engine is None:
            return False, "记忆引擎还没准备好", 1

        scope = self._scope_from_kwargs(kwargs)
        if action in ("整理", "maintain"):
            report = await engine.consolidate(scope)
            await self.ctx.send.text(
                f"整理完成：扫描 {report.scanned} 条，淡出 {report.decayed} 条，"
                f"合并 {report.merged} 条，取代 {report.superseded} 条。",
                stream_id,
            )
            return True, "整理完成", 0

        if action in ("搜", "search"):
            hits = engine.recall(query, scope, top_k=8)
            if not hits:
                await self.ctx.send.text("没找到相关的记忆。", stream_id)
                return True, "没找到相关记忆", 0
            lines = []
            for hit in hits:
                marker = "（已淡出）" if hit.tier == "echo" else ""
                lines.append(f"· {hit.memory.content}{marker}")
            await self.ctx.send.text("记得这些：\n" + "\n".join(lines), stream_id)
            return True, f"找到 {len(hits)} 条记忆", 0

        stats = engine.stats(scope)
        by_status = stats.get("by_status") or {}
        await self.ctx.send.text(
            "这个会话目前记住了 {total} 条：活跃 {active} 条，淡出中 {fading} 条，"
            "已成摘要 {gisted} 条，已归档 {archived} 条。".format(
                total=stats.get("total", 0),
                active=by_status.get("active", 0),
                fading=by_status.get("fading", 0),
                gisted=by_status.get("gisted", 0),
                archived=by_status.get("archived", 0),
            ),
            stream_id,
        )
        return True, "已返回记忆统计", 0

    @Command(
        "记住",
        description="显式记住一件事。/记住 <内容>",
        pattern=r"^/(?:记住|remember)\s+(?P<content>.+)$",
    )
    async def cmd_remember(self, **kwargs: Any) -> tuple:
        """显式记住。"""

        stream_id = str(kwargs.get("stream_id") or "")
        groups = kwargs.get("matched_groups")
        groups = groups if isinstance(groups, dict) else {}
        content = str(groups.get("content") or "").strip()
        engine = self._engine
        if engine is None or not content:
            return False, "没听清要记住什么", 1
        scope = self._scope_from_kwargs(kwargs)
        draft = MemoryDraft(
            content=content,
            kind=MemoryKind.SEMANTIC,
            importance=0.95,
            cross_scope=bool(self.config.memory.cross_session),
            source="command",
        )
        memory = engine.remember(draft, scope)
        if memory is None:
            return False, "没记下来（可能被判定为不重要或重复）", 1
        await self.ctx.send.text(f"记住了：{memory.content}", stream_id)
        return True, "已记住", 0

    @Command(
        "忘掉",
        description="忘掉一条记忆。/忘掉 <记忆内容片段>",
        pattern=r"^/(?:忘掉|忘记|forget)\s+(?P<content>.+)$",
    )
    async def cmd_forget(self, **kwargs: Any) -> tuple:
        """忘掉一条记忆。"""

        stream_id = str(kwargs.get("stream_id") or "")
        groups = kwargs.get("matched_groups")
        groups = groups if isinstance(groups, dict) else {}
        content = str(groups.get("content") or "").strip()
        engine = self._engine
        if engine is None or not content:
            return False, "没听清要忘掉什么", 1
        scope = self._scope_from_kwargs(kwargs)
        hits = engine.recall(content, scope, top_k=5)
        if not hits:
            await self.ctx.send.text("没找到要忘掉的记忆。", stream_id)
            return True, "没找到", 0
        target = hits[0].memory
        engine.forget(target.id)
        await self.ctx.send.text(f"忘掉了：{target.content}", stream_id)
        return True, "已忘掉", 0

    # ------------------------------------------------------------------ #
    # 工具：让机器人自己决定要不要记
    # ------------------------------------------------------------------ #

    @Tool(
        "remember_something",
        description="把一件值得长期记住的事写进记忆库。用户明确表示希望被记住、"
        "承诺、约定、生日、偏好等适合长期保留的信息使用。"
        "一次只记一件事，写成一句话（30 字以内）；"
        "同时给出「谁-怎么样-什么」三段，记忆系统靠它发现互相矛盾的事实。",
        parameters={
            "content": {"type": "string", "description": "要记住的内容，一句话，30 字以内", "required": True},
            "subject": {"type": "string", "description": "谁（例：许杏玖）"},
            "predicate": {"type": "string", "description": "怎么样（例：住在 / 喜欢 / 参加）"},
            "object": {"type": "string", "description": "什么（例：上海 / 辣的食物 / 第18届蓝桥杯）"},
            "kind": {
                "type": "string",
                "description": "类型：semantic 事实 / preference 喜好 / episodic 事件 / relation 关系",
                "required": False,
            },
        },
    )
    async def tool_remember(self, content: str = "", kind: str = "semantic", **kwargs: Any) -> Dict[str, Any]:
        """让模型自己写入一条记忆。"""

        message = kwargs.get("message")
        subject = str(kwargs.get("subject") or "").strip()
        predicate = str(kwargs.get("predicate") or "").strip()
        object_value = str(kwargs.get("object") or "").strip()
        engine = self._engine
        text = (content or "").strip()
        if engine is None or not text:
            return {"name": "remember_something", "content": "没记下来：内容为空"}
        scope = self._scope_from_kwargs(kwargs)
        memory_kind = {
            "semantic": MemoryKind.SEMANTIC,
            "preference": MemoryKind.PREFERENCE,
            "episodic": MemoryKind.EPISODIC,
            "relation": MemoryKind.RELATION,
        }.get(str(kind or "").lower(), MemoryKind.SEMANTIC)
        draft = MemoryDraft(
            content=text,
            kind=memory_kind,
            importance=0.8,
            subject=subject,
            predicate=predicate,
            object=object_value,
            cross_scope=bool(self.config.memory.cross_session),
            source="tool",
        )
        memory = engine.remember(draft, scope)
        if memory is None:
            return {"name": "remember_something", "content": "没记下来（内容不重要或与已有记忆重复）"}
        return {"name": "remember_something", "content": f"记住了：{memory.content}"}

    # ------------------------------------------------------------------ #
    # WebUI / API
    # ------------------------------------------------------------------ #

    @API("memfade_stats", description="读取记忆引擎的状态统计", version="1")
    async def memfade_stats(self) -> Dict[str, Any]:
        """记忆引擎概览。"""

        engine = self._engine
        if engine is None:
            return {"enabled": False, "total": 0, "by_status": {}}
        stats = engine.stats()
        stats["enabled"] = bool(self.config.plugin.enabled)
        stats["db_path"] = engine.store.db_path
        stats["maintain_ticks"] = self._maintain_ticks
        stats["known_scopes"] = sorted(self._scopes)
        stats["buffered"] = {k: len(v) for k, v in self._buffer.items() if v}
        return stats

    @API("memfade_overview", description="记忆引擎总览：各淡化档位分布与最近记忆", version="1")
    async def memfade_overview(self, limit: int = 20) -> Dict[str, Any]:
        """管理界面用的总览数据。"""

        engine = self._engine
        if engine is None:
            return {"total": 0, "chart": [], "recent": [], "by_status": {}}
        counts = engine.store.status_counts()
        chart = [
            {"label": "活跃", "value": counts.get("active", 0)},
            {"label": "淡出中", "value": counts.get("fading", 0)},
            {"label": "已成摘要", "value": counts.get("gisted", 0)},
            {"label": "已归档", "value": counts.get("archived", 0)},
            {"label": "已被取代", "value": counts.get("superseded", 0)},
        ]
        recent = []
        for mem in engine.store.iter_all(limit=max(1, min(100, int(limit)))):
            recent.append(
                {
                    "content": mem.content[:80],
                    "kind": mem.kind.value,
                    "status": mem.status.value,
                    "importance": round(mem.importance, 2),
                    "strength": round(engine.effective_strength(mem), 2),
                    "recall_count": mem.recall_count,
                    "scope": mem.scope_key,
                }
            )
        return {
            "total": engine.store.count(),
            "by_status": counts,
            "chart": chart,
            "recent": recent,
            "maintain_ticks": self._maintain_ticks,
            "known_scopes": sorted(self._scopes),
            "buffered": {k: len(v) for k, v in self._buffer.items() if v},
            "buffered_total": sum(len(v) for v in self._buffer.values() if v),
        }

    @API("memfade_do_search", description="在记忆库里检索（WebUI 动作）", version="1")
    async def memfade_do_search(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        """执行检索，结果存起来供表格显示。"""

        engine = self._engine
        if engine is None:
            self._last_search = []
            return {"count": 0}
        scope = Scope(platform="qq", chat_type="session", chat_id="webui")
        hits = engine.recall(query, scope, top_k=max(1, min(50, int(limit))))
        self._last_search = [
            {
                "content": hit.memory.content[:100],
                "kind": hit.memory.kind.value,
                "tier": hit.tier,
                "score": round(hit.score, 3),
                "importance": round(hit.memory.importance, 2),
                "strength": round(hit.strength_eff, 2),
            }
            for hit in hits
        ]
        return {"count": len(self._last_search)}

    @API("memfade_search_result", description="读取上一次检索的结果", version="1")
    async def memfade_search_result(self) -> Dict[str, Any]:
        """表格数据源。"""

        return {"results": getattr(self, "_last_search", [])}

    @API("memfade_search", description="按关键词检索记忆", version="1")
    async def memfade_search(self, query: str = "", limit: int = 8) -> Dict[str, Any]:
        """检索记忆。"""

        engine = self._engine
        if engine is None:
            return {"results": []}
        scope = Scope(platform="qq", chat_type="private", chat_id="webui")
        hits = engine.recall(query, scope, top_k=max(1, min(20, int(limit))))
        return {
            "results": [
                {
                    "content": hit.memory.content,
                    "kind": hit.memory.kind.value,
                    "status": hit.memory.status.value,
                    "tier": hit.tier,
                    "score": hit.score,
                    "importance": hit.memory.importance,
                    "strength": hit.strength_eff,
                    "recall_count": hit.memory.recall_count,
                }
                for hit in hits
            ]
        }

    @API("memfade_consolidate", description="立刻跑一轮记忆整理", version="1")
    async def memfade_consolidate(self) -> Dict[str, Any]:
        """手动触发整理。"""

        engine = self._engine
        if engine is None:
            return {"error": "引擎未就绪"}
        report = await engine.consolidate()
        return report.as_dict()

    @API("memfade_selfcheck", description="自检：跑一遍注入流程并返回将要写入的 item", version="1")
    async def memfade_selfcheck(self, query: str = "今天吃什么") -> Dict[str, Any]:
        """在**运行中的插件实例**里跑一遍注入流程。

        返回值里能看到最终塞进模型的 item 结构，
        用来确认「记忆块真的能被宿主接受」，而不只是单元测试通过。
        """

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        scope = Scope(platform="qq", chat_type="session", chat_id="selfcheck")
        latest = engine.store.iter_all(limit=1)
        if latest:
            # 用最近有记忆的那个会话，自检才有东西可检
            scope = Scope.from_key(latest[0].scope_key)
        items = [
            {
                "item_type": "UserMessageItem",
                "meta": {
                    "item_id": "selfcheck-user",
                    "logical_turn_id": "selfcheck-turn",
                    "timestamp": datetime.now().isoformat(),
                },
                "parts": [{"type": "text", "text": query}],
            }
        ]
        result = await self._inject_into_items(
            {"items": items, "item_schema_version": 1, "session_id": scope.chat_id}
        )
        if result is None:
            return {"ok": True, "injected": False, "reason": "没有召回任何记忆"}
        modified = result["modified_kwargs"]
        injected = modified["items"][0]
        return {
            "ok": True,
            "injected": True,
            "item_count": len(modified["items"]),
            "item_schema_version": modified.get("item_schema_version"),
            "item": injected,
        }

    @API("memfade_extract_check", description="自检：用一段示例对话跑一遍 LLM 抽取", version="1")
    async def memfade_extract_check(self, conversation: str = "") -> Dict[str, Any]:
        """验证「抽取 → 判重 → 落库」这条链路在运行中是否正常。"""

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        sample = conversation or (
            "用户：我叫许杏玖，以后叫我小许就行\n"
            "机器人：好的，许杏玖，记住了\n"
            "用户：我最喜欢吃辣的东西，不吃香菜\n"
            "机器人：记住了"
        )
        messages = []
        for line in sample.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("用户："):
                messages.append({"role": "user", "content": line[3:].strip()})
            elif line.startswith("机器人："):
                messages.append({"role": "assistant", "content": line[4:].strip()})
        try:
            result = await engine.extractor.extract(messages, source="selfcheck")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "used_llm": result.used_llm,
            "error": result.error,
            "draft_count": len(result.drafts),
            "drafts": [
                {
                    "content": d.content,
                    "kind": d.kind.value,
                    "importance": round(d.importance, 3),
                    "subject": d.subject,
                    "predicate": d.predicate,
                    "object": d.object,
                    "cross_scope": d.cross_scope,
                }
                for d in result.drafts
            ],
        }

    @API("memfade_curve", description="读取淡化曲线（各记忆种类的保留率）", version="1")
    async def memfade_curve(self, days: int = 60) -> Dict[str, Any]:
        """淡化曲线：给面板看「记忆是怎么一点点退场的」。"""

        from memfade.decay import decay_curve
        from memfade.types import MemoryKind

        span = max(7, min(365, int(days)))
        curve = decay_curve(MemoryKind.EPISODIC, importance=0.4, recall_count=0, days=span)
        points = [{"day": i, "value": value} for i, value in enumerate(curve)]

        milestones = (7, 30, 90, 365)
        table = []
        labels = {
            MemoryKind.EPISODIC: "事件",
            MemoryKind.SEMANTIC: "事实",
            MemoryKind.PREFERENCE: "喜好",
            MemoryKind.RELATION: "关系",
            MemoryKind.SKILL: "做法",
        }
        for kind in (
            MemoryKind.EPISODIC,
            MemoryKind.SEMANTIC,
            MemoryKind.PREFERENCE,
            MemoryKind.RELATION,
            MemoryKind.SKILL,
        ):
            row = {"kind": labels.get(kind, kind.value)}
            for days in milestones:
                value = decay_curve(kind, importance=0.5, recall_count=0, days=days)
                row[f"d{days}"] = value[-1]
            table.append(row)
        return {"points": points, "table": table, "days": span}

    @API("memfade_canonicalize", description="把零散事实规范化成「一件事一条」", version="1")
    async def memfade_canonicalize(self, facts: Any = None) -> Dict[str, Any]:
        """用宿主 LLM 规范化旧事实（迁移旧记忆时用）。

        走宿主自己的 LLM 服务而不是裸 HTTP——某些模型（如 MiniMax）
        默认开着 thinking，裸调用容易把 token 全烧在思考上、输出为空。

        ``facts`` 收字符串或列表都行：WebUI 的声明式 action 只允许标量参数，
        所以列表走 JSON 字符串传进来。
        """

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        if isinstance(facts, str):
            try:
                facts = json.loads(facts)
            except json.JSONDecodeError:
                facts = [facts]
        items = [str(f).strip() for f in (facts or []) if str(f).strip()]
        if not items:
            return {"ok": True, "items": []}
        try:
            out = await self._canonicalize_chunked(engine, items)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "items": out}

    async def _canonicalize_chunked(self, engine, items: List[str], depth: int = 0) -> List[Dict[str, Any]]:
        """分批规范化；某批失败就**拆成两半再试**。

        有些模型（比如开着 thinking 的 MiniMax）在长输入下会把 token
        全烧在思考上、最后输出为空。拆小重来比直接放弃可靠得多。
        """

        out: List[Dict[str, Any]] = []
        chunk_size = 10 if depth == 0 else 5
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            try:
                out.extend(await engine.extractor.canonicalize(chunk))
            except Exception as exc:
                if len(chunk) > 1 and depth < 3:
                    mid = len(chunk) // 2
                    out.extend(await self._canonicalize_chunked(engine, chunk[:mid], depth + 1))
                    out.extend(await self._canonicalize_chunked(engine, chunk[mid:], depth + 1))
                    continue
                self.ctx.logger.warning("[memfade] 规范化这批失败：%s", exc)
                for text in chunk:
                    out.append({"content": text, "subject": "", "predicate": "", "object": ""})
        return out

def create_plugin() -> MemFadePlugin:
    """创建插件实例。"""

    return MemFadePlugin()






