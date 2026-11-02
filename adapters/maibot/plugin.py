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

    self_names: list[str] = Field(
        default_factory=list,
        description="机器人自己的名字：主语是它时不入库（留空则自动读宿主 bot_config 的 nickname）",
    )
    focus_subjects: list[str] = Field(
        default_factory=list,
        description="只记这些主语的事（留空=不限制；群聊里认识很多人时用）",
    )
    subject_hint: str = Field(
        default="",
        description="抽取提示词里的人物提示，形如「（许杏玖 / 季怡）」",
    )
    require_person_subject: bool = Field(
        default=True,
        description="主语必须是人（对话里出现过的名字或常见身份称呼），公共知识不入库",
    )
    strip_quotes: bool = Field(
        default=True,
        description="抽取前剥掉被引用的原文与合并转发（污染的最大来源）",
    )

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


def _host_bot_names() -> List[str]:
    """从宿主配置里读机器人自己的名字。

    为什么要自动读：记忆库是「关于对方」的，而**最容易被污染的一类**
    就是「机器人自己的事」——主语写成机器人自己的名字。
    让部署者手填一遍是漏填的温床，所以能自动读就自动读。

    读不到就返回空列表（此时不拦主语），不猜、不硬编码。
    """

    names: List[str] = []
    try:
        # 插件目录形如 <MaiBot根>/plugins/<插件名>，往上两级就是宿主根目录。
        host_root = os.path.dirname(os.path.dirname(_HERE))
        config_path = os.path.join(host_root, "config", "bot_config.toml")
        if not os.path.exists(config_path):
            return names
        with open(config_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("nickname"):
                    continue
                _, _, value = stripped.partition("=")
                value = value.split("#", 1)[0].strip().strip('"').strip("'")
                if value:
                    names.append(value)
                break
    except Exception:
        return names
    return names


def _self_names(plugin: "MemFadePlugin") -> List[str]:
    """机器人自己的名字：配置里写的优先，否则自动读宿主配置。"""

    configured = [str(n).strip() for n in (plugin.config.memory.self_names or []) if str(n).strip()]
    return configured or _host_bot_names()


def _build_engine_config(plugin: "MemFadePlugin", db_path: str) -> MemFadeConfig:
    """把插件配置翻译成核心配置。"""

    cfg = plugin.config
    mem = cfg.memory
    fade = cfg.fade
    focus = tuple(str(n).strip() for n in (mem.focus_subjects or []) if str(n).strip())
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
            "subject_hint": str(mem.subject_hint or ""),
            "self_names": tuple(_self_names(plugin)),
            "focus_subjects": tuple(focus),
            "require_person_subject": bool(mem.require_person_subject),
            "strip_quotes": bool(mem.strip_quotes),
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

        self._buffer_path = ""
        """待抽取消息的落盘位置。

        缓冲只在内存里的话，宿主一重启（改配置、崩溃、更新），
        攒到一半还没抽的记忆就**丢了**——记忆插件最不该有的失败。
        """

        self._scopes: set[str] = set()
        """见过的会话键，后台整理任务要遍历。"""

        self._counter: Dict[str, int] = {}
        self._last_maintain: float = 0.0
        self._maintain_task: Optional[asyncio.Task] = None
        self._maintain_ticks: int = 0
        """后台维护循环的心跳次数。用来确认循环真的在跑。"""
        self._last_search: List[Dict[str, Any]] = []
        self._last_explain: List[Dict[str, Any]] = []
        self._extract_tasks: set = set()
        """后台抽取任务。抽取要等 LLM，不能让它挡住主人的回复。"""
        self._recent_user_text: Dict[str, str] = {}
        """每个会话最近一条用户消息——回复器钩子拿不到 items，用它做检索词。"""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def on_load(self) -> None:
        """插件加载：建引擎、起整理任务。"""

        self._rebuild_engine()
        await self._check_competing_memory()
        self._load_buffer()
        self._maintain_task = asyncio.create_task(self._maintenance_loop())

    async def on_unload(self) -> None:
        """插件卸载：收尾。"""

        self._save_buffer()
        # 后台抽取任务还在等 LLM 的话，别让它们在引擎关掉之后还往里写。
        for task in list(self._extract_tasks):
            task.cancel()
        if self._extract_tasks:
            await asyncio.gather(*list(self._extract_tasks), return_exceptions=True)
            self._extract_tasks.clear()
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

    async def _check_competing_memory(self) -> None:
        """检查宿主是否还挂着另一套长期记忆系统。

        **两套记忆并行会出三件事**：
        1. 旧的自动写回「人物事实」与「聊天摘要」，新的也抽取事实——
            同一个人的同一件事，两个库各存一份；
        2. 旧的**没有淡化引擎**（段落写进去就赖着不走），
            这正是「总提旧事」的根源；
        3. 模型手里还有旧的检索工具，能从旧库捞回已经作废的事实。

        插件不能改宿主的配置（SDK 只给读接口），
        所以这里**把冲突点出来**并给出退场步骤——
        让人来决定，而不是悄悄改别人的配置。
        """

        probes = (
            ("a_memorix.integration.enable_memory_query_tool", "模型还能检索旧记忆库"),
            ("a_memorix.integration.person_fact_writeback_enabled", "旧系统会自动抽取人物事实"),
            ("a_memorix.integration.chat_summary_writeback_enabled", "旧系统会自动写回聊天摘要"),
            ("a_memorix.integration.enable_person_profile_injection", "旧画像会注入对话"),
            ("a_memorix.integration.heuristic_memory_recall_enabled", "旧系统会「自然拉起」旧记忆"),
        )
        conflicts = []
        for key, why in probes:
            try:
                value = await self.ctx.config.get(key, False)
            except Exception:
                continue
            if value:
                conflicts.append((key, why))
        if not conflicts:
            return
        self.ctx.logger.warning(
            "[memfade] 宿主还有另一套长期记忆系统在跑：%s。"
            "两套并行会互相污染（旧的没有淡化引擎，模型还能从旧库检索）。"
            "退场步骤见插件目录的 tools/retire_legacy_memory.py。",
            "、".join("%s（%s）" % (k, w) for k, w in conflicts),
        )

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
        # 把库里已有的会话补进「已知会话」集合。
        # 不补的话：插件一重启（宿主重启、配置热重载都会重建引擎），
        # 那些没再发过消息的会话就再没人整理了——
        # 档位是读时算的没错，但 **gist 压缩 / 合并 / 淘汰** 都得靠整理任务做。
        try:
            for scope_key, _count in self._engine.store.scope_counts():
                self._scopes.add(str(scope_key))
        except Exception as exc:  # pragma: no cover - 读库失败不该挡住加载
            self.ctx.logger.warning("[memfade] 读取已有会话失败: %s", exc)
        self.ctx.logger.info("[memfade] 引擎就绪：%s", db_path)

    # ------------------------------------------------------------------ #
    # 待抽取消息的落盘
    # ------------------------------------------------------------------ #

    def _buffer_file(self) -> str:
        """缓冲文件路径。"""

        if not self._buffer_path:
            try:
                data_dir = self.ctx.paths.data_dir
                data_dir.mkdir(parents=True, exist_ok=True)
                self._buffer_path = str(data_dir / "memfade_pending.json")
            except Exception:  # pragma: no cover - 没有可写目录就退回内存
                self._buffer_path = ""
        return self._buffer_path

    def _save_buffer(self) -> None:
        """把待抽取的消息写盘。"""

        path = self._buffer_file()
        if not path:
            return
        payload = {
            "buffer": {k: list(v) for k, v in self._buffer.items() if v},
            "at": {k: float(v) for k, v in self._buffer_at.items() if v},
            "saved_at": time.time(),
        }
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:  # pragma: no cover - 写盘失败不该影响聊天
            self.ctx.logger.warning("[memfade] 待抽取消息落盘失败: %s", exc)

    def _load_buffer(self) -> None:
        """把上次没抽完的消息读回来。"""

        path = self._buffer_file()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 待抽取消息读取失败: %s", exc)
            return
        buffer = payload.get("buffer") or {}
        at = payload.get("at") or {}
        if not isinstance(buffer, dict):
            return
        restored = 0
        for key, items in buffer.items():
            if not isinstance(items, list) or not items:
                continue
            self._buffer[str(key)] = [m for m in items if isinstance(m, dict)]
            self._buffer_at[str(key)] = float(at.get(key, 0.0) or 0.0)
            restored += len(self._buffer[str(key)])
        try:
            os.remove(path)
        except Exception:
            pass
        if restored:
            self.ctx.logger.info("[memfade] 恢复上次未抽取的消息 %d 条", restored)

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
        speaker = str(user_info.get("user_cardname") or user_info.get("user_nickname") or "").strip()
        text = str(message.get("processed_plain_text") or "").strip()
        if not text:
            raw_text = message.get("raw_message")
            text = raw_text.strip() if isinstance(raw_text, str) else ""
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
            "own_text": MemFadePlugin._own_text(message),
            "quote_text": MemFadePlugin._quoted_text(message),
            "speaker": speaker,
            "session_id": session_id,
            "chat_type": "group" if group_id else "private",
        }

    @staticmethod
    def _own_text(message: Dict[str, Any]) -> str:
        """只留用户自己说的话：去掉被引用的原文与合并转发。

        **这是记忆库被污染的最大来源。**

        宿主把「被引用的那条消息」拼进用户的正文里（回复别人的话、
        转发一份群公告），用户发出来的其实是「别人的话 + 他自己的话」。
        抽取器对着这份文本做抽取，于是把**机器人自己说过的话**、
        **群公告的原文**当成了用户的话记下来：

        * 主人引用机器人的「你昨晚一口气刷完Re」，于是记忆库里多了一条
          「许杏玖昨晚一口气刷完了Re」——机器人认领了主人说的话；
        * 主人转发蓝桥杯群公告求总结，于是「季怡在备赛第18届蓝桥杯」
          被记了下来——可主人当时明确说过「我没跟你说过我要报名」。

        所以抽取前先把这两块剥掉：**引用与转发不是用户的话。**
        """

        raw = message.get("raw_message")
        parts: List[str] = []
        if isinstance(raw, list):
            for comp in raw:
                if not isinstance(comp, dict):
                    continue
                ctype = str(comp.get("type") or "").strip().lower()
                if ctype != "text":
                    continue
                data = comp.get("data")
                if isinstance(data, str) and data.strip():
                    parts.append(data.strip())
        if parts:
            return " ".join(parts)
        # 拿不到组件（别的宿主实现），退一步：把开头的引用/转发块切掉。
        text = str(message.get("processed_plain_text") or "").strip()
        return MemFadePlugin._strip_leading_blocks(text)

    @staticmethod
    def _quoted_text(message: Dict[str, Any]) -> str:
        """取被引用的那段原文（机器人或别人刚说过的话）。

        **它不是用户的话，但它是上下文。**
        「Master，《影之刃零》买了吗」是机器人问的，
        用户回「买了」——没有这句上下文，抽取器只知道用户「买了」，
        根本不知道买的是什么，最后只能记一句含糊的废话。

        所以：**不删，改成一行标注清楚的上下文**，
        提示词里写死「机器人：的行只用来判断指代，不要从中提取事实」。
        实测（宿主真实 LLM）：
        - 有上下文 → 抽出「季怡已经购买《影之刃零》」（对）；
        - 没上下文 → 这条直接不抽（宁可少记，也不要记含糊的）。
        """

        raw = message.get("raw_message")
        if not isinstance(raw, list):
            return ""
        for comp in raw:
            if not isinstance(comp, dict):
                continue
            if str(comp.get("type") or "").strip().lower() != "reply":
                continue
            data = comp.get("data") or {}
            return str(data.get("target_message_content") or "").strip()
        return ""

    @staticmethod
    def _strip_leading_blocks(text: str) -> str:
        """剥掉开头的引用块（[回复了X的消息: ...]）与转发块（【合并转发消息: ...】）。"""

        out = (text or "").strip()
        for _ in range(8):  # 最多剥几层，防止异常输入把 CPU 拖住
            if out.startswith("【合并转发消息"):
                end = out.rfind("】")
                if end <= 0:
                    break
                out = out[end + 1 :].strip()
                continue
            if out.startswith("[回复了"):
                depth = 0
                end = -1
                for index, char in enumerate(out):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            end = index
                            break
                if end <= 0:
                    break
                out = out[end + 1 :].strip()
                continue
            break
        return out

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
            self._maybe_extract(parsed, scope)

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

    async def _inject_into_items(
        self, kwargs: Dict[str, Any], *, reinforce: bool = True
    ) -> Optional[Dict[str, Any]]:
        """把召回的记忆写进 Context Items。

        为什么用 ``maisaka.*.before_request`` 而不是 ``chat.receive.after_process``：

        * ``after_process`` 跑在**会话创建之前**（宿主顺序是
          after_process hook → register_message → get_or_create_session），
          那时 ``maisaka.context.append`` 会因为找不到聊天流而抛异常；
        * 这里跑在会话创建之后、模型请求之前，``session_id`` 直接就在载荷里，
          正好是记忆的作用域键；
        * 而且能**直接改写真正发给模型的 Context Items**，
          不会像「往历史里追加」那样被上下文裁剪顺手丢掉。

        ⚠ ``reinforce=False``：**读上下文也是写**。
        注入会强化被注入的记忆（残留学习：强度回升、锚点重置、
        被想起次数 +1）。而 ``recall_count`` 还参与稳定性的间隔重复成长，
        诊断动作（面板自检）如果走默认路径，就把记忆的寿命拉长了好几倍——
        **诊断绝不能改库**。
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
                # 强化只做一次（build_context 内部也会强化，靠冷却期兜着，
                # 但「读上下文也是写」这件事得在一个地方说清楚）
                reinforce=False,
            )
            body = self._wrap_context(context.text, profile.text)
            if not body:
                return None
            new_items = list(items)
            new_items.insert(self._insert_index(items), self._build_memory_item(items, body))
            if reinforce and context.hits:
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

    def _maybe_extract(self, parsed: Dict[str, Any], scope: Scope) -> None:
        """按节奏抽取记忆（**不阻塞**回复）。

        抽取要问一次 LLM，实测 MiniMax 这类开 thinking 的模型要 30~60 秒。
        它挂在 ``chat.receive.after_process`` 这个阻塞钩子上，
        等于**每 3 条消息就让主人多等一分钟**——
        所以这里只把消息攒好，真正的抽取丢到后台任务里跑。
        """

        engine = self._engine
        if engine is None:
            return
        key = scope.key()
        buffer = self._buffer.setdefault(key, [])
        # 缓冲区里放两种行：
        # - **用户自己说的话**（引用与转发已剥掉），带说话人昵称；
        # - **被引用的原文**（机器人刚说过的话），标成「机器人」行。
        #   它不是用户的话，只是上下文——没有它，抽取器只知道用户「买了」，
        #   不知道买的是什么，只能记一句含糊的废话。
        quote = parsed.get("quote_text") or ""
        if quote and self.config.memory.strip_quotes:
            buffer.append({"role": "assistant", "content": quote})
        content = parsed.get("text") or ""
        if self.config.memory.strip_quotes:
            content = parsed.get("own_text") or content
        entry = {"role": "user", "content": content}
        speaker = parsed.get("speaker") or ""
        if speaker:
            entry["speaker"] = speaker
        buffer.append(entry)
        self._buffer_at[key] = time.time()
        self._save_buffer()
        count = self._counter.get(key, 0) + 1
        self._counter[key] = count
        every = max(1, int(self.config.memory.extract_every_n))
        min_messages = max(1, int(self.config.memory.extract_min_messages))
        if count % every != 0 or len(buffer) < min_messages:
            return
        pending = list(buffer)
        buffer.clear()
        self._save_buffer()
        meta = {
            "group_id": parsed.get("group_id") or "",
            "user_id": parsed.get("user_id") or "",
            "chat_type": parsed.get("chat_type") or "",
        }
        task = asyncio.create_task(self._extract_pending(pending, scope, key, meta))
        self._extract_tasks.add(task)
        task.add_done_callback(self._extract_tasks.discard)

    async def _extract_pending(
        self,
        pending: List[Dict[str, Any]],
        scope: Scope,
        key: str,
        meta: Dict[str, Any],
    ) -> None:
        """后台跑一次抽取。"""

        engine = self._engine
        if engine is None:
            return
        try:
            written = await engine.remember_from_messages(pending, scope, source=key, meta=meta)
            if written:
                self.ctx.logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)
            for content, reason in getattr(engine, "last_dropped", []) or []:
                self.ctx.logger.info(
                    "[memfade] 丢弃候选（%s）：%s", reason, (content or "")[:40]
                )
        except Exception as exc:
            self.ctx.logger.warning("[memfade] 抽取记忆失败: %s", exc)
            # 抽取失败不丢内容，放回缓冲区下次一起试
            buffer = self._buffer.setdefault(key, [])
            buffer.extend(pending)
            self._save_buffer()

    async def _drain_extractions(self) -> None:
        """等后台抽取任务跑完（测试与卸载时用）。"""

        while self._extract_tasks:
            await asyncio.gather(*list(self._extract_tasks), return_exceptions=True)

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
        self._save_buffer()

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
        description="查看记忆。/记忆 查当前会话记住的东西，/记忆 搜 <关键词>，"
        "/记忆 整理 立刻跑一轮整理，/记忆 置顶 <内容> 让它永不淡化，"
        "/记忆 接管 看看记忆板块归谁管",
        pattern=r"^/(?:记忆|memory)(?:\s+(?P<action>搜|search|整理|maintain|置顶|pin|取消置顶|unpin|接管|takeover|体检|check)?)?"
        r"(?:\s+(?P<query>.+))?$",
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

        if action in ("接管", "takeover", "体检", "check"):
            # 「接管」= 让 memfade 当唯一的长期记忆板块。
            # 插件不能改宿主配置（SDK 只给读接口），
            # 所以这里给的是**现状 + 要改哪几行**。
            lines = ["记忆板块现状：", "· memfade（本插件）：长期记忆的主板，已就绪"]
            conflicts = []
            probes = (
                ("a_memorix.integration.enable_memory_query_tool", "模型能检索旧记忆库"),
                ("a_memorix.integration.person_fact_writeback_enabled", "旧系统自动抽取人物事实"),
                ("a_memorix.integration.chat_summary_writeback_enabled", "旧系统自动写回聊天摘要"),
                ("a_memorix.integration.enable_person_profile_injection", "旧画像注入对话"),
                ("a_memorix.integration.heuristic_memory_recall_enabled", "旧系统「自然拉起」旧记忆"),
            )
            for key, why in probes:
                try:
                    value = await self.ctx.config.get(key, False)
                except Exception:
                    continue
                if value:
                    conflicts.append((key, why))
            if conflicts:
                lines.append("· 宿主还有一套旧的长期记忆系统在跑：")
                for key, why in conflicts:
                    lines.append("    - %s（%s）" % (key, why))
                lines.append(
                    "  退场：把上面几项改成 false，"
                    "并跑插件目录的 tools/retire_legacy_memory.py 清掉旧数据。"
                )
            else:
                lines.append("· 宿主没有别的长期记忆系统在跑 —— 记忆只走 memfade 这条路。")
            await self.ctx.send.text("\n".join(lines), stream_id)
            return True, "已返回记忆板块现状", 0

        if action in ("搜", "search"):
            # 显式检索必须真的命中：「搜」是用户在找东西，
            # 不该把「最近写入」但跟关键词无关的记忆一并列出来。
            vector = await engine.embed_query(query) if query else None
            hits = engine.recall(
                query, scope, top_k=8, query_vector=vector, require_match=True
            )
            if not hits:
                await self.ctx.send.text("没找到相关的记忆。", stream_id)
                return True, "没找到相关记忆", 0
            lines = []
            for hit in hits:
                marker = "（已淡出）" if hit.tier == "echo" else ""
                lines.append(f"· {hit.memory.content}{marker}")
            await self.ctx.send.text("记得这些：\n" + "\n".join(lines), stream_id)
            return True, f"找到 {len(hits)} 条记忆", 0

        if action in ("置顶", "pin", "取消置顶", "unpin"):
            pinned = action in ("置顶", "pin")
            if not query:
                await self.ctx.send.text("要置顶（或取消置顶）哪条？给点内容片段。", stream_id)
                return True, "缺少内容", 0
            vector = await engine.embed_query(query) if query else None
            hits = engine.recall(
                query, scope, top_k=5, query_vector=vector, require_match=True
            )
            if not hits:
                await self.ctx.send.text("没找到相关的记忆。", stream_id)
                return True, "没找到相关记忆", 0
            target = hits[0].memory
            engine.pin(target.id, pinned=pinned)
            verb = "置顶了" if pinned else "取消置顶了"
            await self.ctx.send.text(f"{verb}：{target.content}", stream_id)
            return True, "已" + verb, 0

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
        # 光给数字不够：**主人得能一眼看出库里存了什么**。
        # 「记忆库被污染了」这种问题，只有看得见才发现得了。
        try:
            ids = [mid for mid, _ in engine.store.recent(scope.key(), limit=8)]
            recent = sorted(
                engine.store.get_many(ids),
                key=lambda m: m.created_at,
                reverse=True,
            )
        except Exception:
            recent = []
        if recent:
            lines = ["最近记下的："]
            for mem in recent:
                marker = "（已淡出）" if mem.status.value != "active" else ""
                lines.append(f"· {mem.content}{marker}")
            await self.ctx.send.text("\n".join(lines), stream_id)
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
        # 一段话塞进一条记忆，等于一颗炸弹：注入时只能截断，
        # 矛盾检测也抓不住。按标点拆成「一件事一条」再记。
        draft = MemoryDraft(
            content=content,
            kind=MemoryKind.SEMANTIC,
            importance=0.95,
            cross_scope=bool(self.config.memory.cross_session),
            source="command",
        )
        memories = engine.remember_atomic(draft, scope)
        await engine.embed_memories(memories)
        written = [m.content for m in memories]
        if not written:
            return False, "没记下来（可能被判定为不重要或重复）", 1
        if len(written) == 1:
            await self.ctx.send.text(f"记住了：{written[0]}", stream_id)
        else:
            await self.ctx.send.text(
                "记住了这 %d 件事：\n%s" % (len(written), "\n".join("- " + w for w in written)),
                stream_id,
            )
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
        # 忘掉是破坏性操作：必须真的匹配上，不能误删「最近写入」的那条。
        vector = await engine.embed_query(content) if content else None
        hits = engine.recall(
            content, scope, top_k=5, query_vector=vector, require_match=True
        )
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
        # ⚠ 必须显式声明 visibility：
        # 宿主给插件工具的默认可见性是 **deferred**——
        # 不进 planner 的工具清单，模型得先调 tool_search() 才发现得了。
        # 「她自然会想到用这个工具」是不靠谱的。
        # 规则见本机完整版 L5.2（「插件 Tool 默认是延迟发现的」）。
        visibility="visible",
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
        written = engine.remember_atomic(draft, scope)
        if not written:
            return {"name": "remember_something", "content": "没记下来（内容不重要或与已有记忆重复）"}
        await engine.embed_memories(written)
        if len(written) == 1:
            return {"name": "remember_something", "content": f"记住了：{written[0].content}"}
        return {
            "name": "remember_something",
            "content": "记住了这 %d 件事：%s"
            % (len(written), "；".join(m.content for m in written)),
        }

    @Tool(
        "recall_memories",
        description="检索长期记忆：当对方问起「我以前说过什么」「还记得吗」"
        "「之前那件事」这类需要翻旧账的问题时用。"
        "记忆库会自动注入相关记忆，这个工具用来**主动深挖**"
        "（比如对方明确要看全部相关记录）。",
        # ⚠ 必须显式声明 visibility：宿主给插件工具的默认可见性是 **deferred**，
        # 不进 planner 的工具清单，模型得先调 tool_search() 才发现得了。
        visibility="visible",
        parameters={
            "query": {"type": "string", "description": "要查什么（关键词或一句话）", "required": True},
            "limit": {"type": "integer", "description": "最多返回几条（默认 5）", "required": False},
        },
    )
    async def tool_recall(self, query: str = "", limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """让模型主动检索长期记忆（替代宿主旧记忆系统的「检索长期记忆」工具）。"""

        engine = self._engine
        text = (query or "").strip()
        if engine is None or not text:
            return {"name": "recall_memories", "content": "没查到：检索词是空的"}
        scope = self._scope_from_kwargs(kwargs)
        top_k = max(1, min(10, int(limit or 5)))
        try:
            vector = await engine.embed_query(text)
        except Exception:
            vector = None
        # ⚠ require_match=True：显式检索必须真的命中，
        # 不把「最近写入」但跟问题无关的记忆混进来。
        hits = engine.recall(text, scope, top_k=top_k, query_vector=vector, require_match=True)
        if not hits:
            return {"name": "recall_memories", "content": "记忆库里没有相关的记录。"}
        lines = []
        for hit in hits:
            marker = "（可能已经变了）" if hit.tier == "echo" else ""
            lines.append("· %s%s" % (hit.memory.content, marker))
        return {
            "name": "recall_memories",
            "content": "查到 %d 条：\n%s" % (len(hits), "\n".join(lines)),
        }

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
        # ⚠ 别用固定的 "webui" 作用域：那儿一条记忆都没有，
        # 面板的检索框永远搜不出东西。默认看记忆最多的那个会话。
        chat_id = self._busiest_scope_id(engine) or "webui"
        scope = Scope(platform="qq", chat_type="session", chat_id=chat_id)
        # 面板的调用预算是 11 秒，这里可以多等一会儿取向量
        vector = await engine.embed_query(query, timeout=8.0) if query else None
        hits = engine.recall(
            query,
            scope,
            top_k=max(1, min(50, int(limit))),
            query_vector=vector,
            require_match=bool(query),
        )
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
        vector = await engine.embed_query(query, timeout=8.0) if query else None
        hits = engine.recall(
            query,
            scope,
            top_k=max(1, min(20, int(limit))),
            query_vector=vector,
            require_match=bool(query),
        )
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
            {"items": items, "item_schema_version": 1, "session_id": scope.chat_id},
            # 诊断动作**不碰记忆库**：注入会强化记忆（强度、锚点、被想起次数），
            # 而被想起次数还参与间隔重复成长——诊断一次就等于给记忆续命。
            reinforce=False,
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
                messages.append({"role": "user", "content": line[3:].strip(), "speaker": "用户"})
            elif line.startswith("机器人："):
                messages.append({"role": "assistant", "content": line[4:].strip(), "speaker": "机器人"})
            elif "：" in line:
                # 「季怡：xxx」这种带名字的写法也算——
                # 有名字才测得出「主语纪律」那几道闸门。
                speaker, _, content = line.partition("：")
                messages.append(
                    {"role": "user", "content": content.strip(), "speaker": speaker.strip()}
                )
        try:
            # ⚠ 加个超时：LLM 慢的时候（MiniMax 的 thinking 可能跑到 60 秒+）
            # WebUI 的网关会先 504，按钮就变成「点了没反应」。
            # 这里自己兜住：超时就说清楚是超时，别让面板报错。
            try:
                result = await asyncio.wait_for(
                    engine.extractor.extract(messages, source="selfcheck"), timeout=25.0
                )
            except asyncio.TimeoutError:
                return {
                    "ok": True,
                    "used_llm": False,
                    "error": "",
                    "note": "宿主 LLM 超过 25 秒没返回，本次没走 LLM（规则兜底可用）",
                    "draft_count": 0,
                    "drafts": [],
                }
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
            "dropped": [
                {"content": content, "reason": reason} for content, reason in result.dropped
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
            # 同样加超时（理由见 extract_check）：面板按钮不能因为模型慢就 504。
            try:
                out = await asyncio.wait_for(
                    self._canonicalize_chunked(engine, items), timeout=60.0
                )
            except asyncio.TimeoutError:
                return {"ok": False, "error": "宿主 LLM 超过 60 秒没返回，先看看宿主 LLM 通不通"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "items": out}

    @API("memfade_why", description="解释「为什么召回了这些记忆」", version="1")
    async def memfade_why(self, query: str = "", limit: int = 6, session_id: str = "") -> Dict[str, Any]:
        """拆开每个打分分量，说明每条记忆为什么被召回。"""

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        chat_id = session_id or self._busiest_scope_id(engine) or "explain"
        scope = Scope(platform="qq", chat_type="session", chat_id=chat_id)
        try:
            # 向量要在这里算：宿主的面板是异步的，
            # 引擎内部的同步取向量在事件循环里会直接放弃（不阻塞事件循环）。
            vector = await engine.embed_query(query, timeout=8.0) if query else None
            rows = engine.explain(
                query, scope, top_k=max(1, min(20, int(limit))), query_vector=vector
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        self._last_explain = rows
        return {"ok": True, "scope": scope.key(), "results": rows}

    @API("memfade_why_result", description="读取上一次「为什么召回」的结果", version="1")
    async def memfade_why_result(self) -> Dict[str, Any]:
        """面板表格的数据源。"""

        return {"results": getattr(self, "_last_explain", [])}

    @API("memfade_quality", description="记忆库体检：长度、分句、三元组覆盖、重复", version="1")
    async def memfade_quality(self) -> Dict[str, Any]:
        """记忆库健康度。"""

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        try:
            return {"ok": True, "report": engine.diagnose()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @API("memfade_export", description="导出记忆（JSON）", version="1")
    async def memfade_export(self, limit: int = 5000) -> Dict[str, Any]:
        """导出记忆，方便备份或迁移到别的系统。"""

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        try:
            rows = engine.export()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "count": len(rows), "memories": rows[: max(1, min(20000, int(limit)))]}

    @API("memfade_import", description="导入记忆（JSON：导出文件 / 别的机器的库）", version="1")
    async def memfade_import(self, payload: str = "", scope_id: str = "") -> Dict[str, Any]:
        """导入一批记忆。

        **记忆是用户的数据**——换机器、换机器人、换框架都得能带走。
        只有导出没有导入，等于「只能看不能搬」。
        """

        engine = self._engine
        if engine is None:
            return {"ok": False, "error": "引擎未就绪"}
        text = (payload or "").strip()
        if not text:
            return {"ok": False, "error": "没有内容可导入"}
        try:
            data = json.loads(text)
        except Exception as exc:
            return {"ok": False, "error": "不是合法 JSON：%s" % exc}
        if isinstance(data, dict):
            items = data.get("memories") or data.get("rows") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        if not items:
            return {"ok": False, "error": "JSON 里没有记忆条目"}
        try:
            scope = None
            target = (scope_id or "").strip()
            if target:
                scope = Scope(platform="qq", chat_type="session", chat_id=target)
            written = engine.import_memories(items, scope)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "written": len(written),
            "skipped": len(items) - len(written),
            "sample": [m.content for m in written[:5]],
        }

    def _busiest_scope_id(self, engine) -> str:
        """找记忆最多的那个会话——面板默认看它。"""

        best_key = ""
        best_count = 0
        try:
            rows = engine.store.scope_counts()
        except Exception:
            return ""
        for key, count in rows:
            if count > best_count:
                best_count = count
                best_key = key
        return best_key.split(":", 2)[-1] if best_key else ""

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






