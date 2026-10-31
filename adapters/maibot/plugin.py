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
import os
import sys
import time
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

    extract_enabled: bool = Field(default=True, description="是否从对话中抽取记忆")
    extract_every_n: int = Field(default=3, description="每隔多少轮抽取一次")
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
        self._counter: Dict[str, int] = {}
        self._last_maintain: float = 0.0
        self._maintain_task: Optional[asyncio.Task] = None
        self._last_injected: Dict[str, tuple] = {}
        self._reinject_interval: float = 300.0
        """记忆块内容没变时的重放间隔（秒）。"""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def on_load(self) -> None:
        """插件加载：建引擎、起整理任务。"""

        self._rebuild_engine()

    async def on_unload(self) -> None:
        """插件卸载：收尾。"""

        if self._maintain_task is not None:
            self._maintain_task.cancel()
            self._maintain_task = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None

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

        命令拿不到完整的 message 结构，退而用 stream_id 当会话标识——
        至少保证「同一条命令在不同会话里看到不同的记忆」。
        """

        parsed = self._parse_message(kwargs.get("message"))
        if parsed.get("group_id") or parsed.get("user_id"):
            return self._scope_of(parsed)
        stream_id = str(kwargs.get("stream_id") or kwargs.get("session_id") or "").strip()
        return Scope(
            platform=parsed.get("platform") or "qq",
            chat_type="private",
            chat_id=stream_id or "unknown",
            cross_scope=False,
        )

    def _scope_of(self, parsed: Dict[str, Any]) -> Scope:
        """构造作用域。会话内默认隔离，跨会话共享要显式开。"""

        chat_id = parsed.get("group_id") or parsed.get("user_id") or "unknown"
        return Scope(
            platform=parsed.get("platform") or "qq",
            chat_type=parsed.get("chat_type") or "private",
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
        session_id = parsed.get("session_id") or ""
        scope = self._scope_of(parsed)

        # 1) 召回并注入
        if self.config.memory.inject_enabled and text and session_id:
            await self._inject_context(text, scope, session_id)

        # 2) 收集对话，达到节奏就抽取
        if self.config.memory.extract_enabled and text:
            await self._maybe_extract(parsed, scope)

        # 3) 顺路做一次整理（按间隔节流）
        await self._maybe_maintain(scope)
        return None

    async def _inject_context(self, text: str, scope: Scope, session_id: str) -> None:
        """把相关记忆写进 Maisaka 上下文。"""

        engine = self._engine
        if engine is None:
            return
        try:
            vector = None
            if self.config.memory.use_host_embedding:
                vector = await engine.embed_query(text)
            context = engine.build_context(
                text,
                scope,
                query_vector=vector,
                header=self.config.memory.inject_header or "记忆",
            )
            if not context.text:
                return
            # 内容没变就不重复注入：记忆块每轮都塞一遍会污染上下文，
            # 也会让模型觉得「用户一直在重复同样的话」。
            # 但隔一段时间要重放一次，防止它被宿主的上下文裁剪挤掉。
            now = time.time()
            last_at, last_text = self._last_injected.get(session_id, (0.0, ""))
            if context.text == last_text and now - last_at < self._reinject_interval:
                return
            self._last_injected[session_id] = (now, context.text)
            body = self._wrap_context(context.text)
            await self.ctx.maisaka.context.append(
                stream_id=session_id,
                segments=[{"type": "text", "content": body}],
                visible_text=f"记忆 {len(context.facts) + len(context.echoes)} 条",
                source_kind=SOURCE_KIND,
            )
            if self.config.plugin.verbose:
                self.ctx.logger.info(
                    "[memfade] 注入 %d 条记忆（事实 %d / 回声 %d）",
                    len(context.hits),
                    len(context.facts),
                    len(context.echoes),
                )
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 注入记忆失败: %s", exc)

    @staticmethod
    def _wrap_context(text: str) -> str:
        """给记忆块加系统前缀。

        不加前缀的话，模型会把这段当成「用户刚说的话」，
        从而做出「用户刚刚告诉了我这件事」这类错误反应。
        """

        return (
            "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
            "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
            f"{text}"
        )

    async def _maybe_extract(self, parsed: Dict[str, Any], scope: Scope) -> None:
        """按节奏抽取记忆。"""

        engine = self._engine
        if engine is None:
            return
        key = scope.key()
        buffer = self._buffer.setdefault(key, [])
        buffer.append({"role": "user", "content": parsed.get("text") or ""})
        count = self._counter.get(key, 0) + 1
        self._counter[key] = count
        every = max(1, int(self.config.memory.extract_every_n))
        min_messages = max(1, int(self.config.memory.extract_min_messages))
        if count % every != 0 or len(buffer) < min_messages:
            return
        pending = list(buffer)
        buffer.clear()
        try:
            written = await engine.remember_from_messages(pending, scope, source=key)
            if written:
                self.ctx.logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 抽取记忆失败: %s", exc)
            # 抽取失败不丢内容，放回缓冲区下次一起试
            buffer.extend(pending)

    async def _maybe_maintain(self, scope: Scope) -> None:
        """整理任务（带节流）。"""

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
        "承诺、约定、生日、偏好等适合长期保留的信息使用。",
        parameters={
            "content": {"type": "string", "description": "要记住的内容，一句完整的话", "required": True},
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
        return stats

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


def create_plugin() -> MemFadePlugin:
    """创建插件实例。"""

    return MemFadePlugin()






