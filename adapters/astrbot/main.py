"""memfade · AstrBot 适配器。

两个时机：

* ``on_llm_request``——把召回的记忆追加到 system prompt，模型这一轮就能看见；
* ``on_llm_response``——这一轮说完之后，把新内容抽取成记忆。

作用域是**三层**：群级公共 / 群内人级私密 / 私聊。详见 ``scopes.py``。

导入方式
--------
优先用**包内相对导入**。AstrBot 不把 ``data/plugins/`` 加进 ``sys.path``，
插件是以 ``data.plugins.<目录名>`` 这个 PEP 420 命名空间包导入的，所以相对导入
天然与其它插件、与 pip 装的 memfade 全部隔离。

顶层 ``import memfade`` 绝不能在这里出现——``sys.modules`` 是进程级全局缓存，
谁先 import 谁就污染全进程，而且不同副本里的 ``MemoryKind`` 枚举虽然 ``==``
相等但 ``is`` 不相等，核心里有十几处身份比较会静默走错分支。
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:  # 4.x 把 register 挪过位置，两边都认
    from astrbot.api.star import register
except ImportError:  # pragma: no cover
    from astrbot.api.star.register import register_star as register  # type: ignore

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("memfade")  # type: ignore

try:
    from astrbot.api.star import StarTools
except ImportError:  # pragma: no cover
    StarTools = None  # type: ignore


# --------------------------------------------------------------------- #
# 核心源码导入：相对导入优先，兜底只在脱离包上下文时生效
# --------------------------------------------------------------------- #
try:
    from .memfade import MemFadeConfig, Scope, split_repair_request
    from .memfade.engine import MemoryEngine
    from .memfade.extract import MemoryExtractor
    from .memfade.types import MemoryDraft, MemoryKind
except ImportError:  # pragma: no cover - 直接 python main.py 之类
    import sys as _sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in _sys.path:
        # 插到【末尾】，不抢别人的位置
        _sys.path.append(_HERE)
    from memfade import MemFadeConfig, Scope, split_repair_request  # type: ignore
    from memfade.engine import MemoryEngine  # type: ignore
    from memfade.extract import MemoryExtractor  # type: ignore
    from memfade.types import MemoryDraft, MemoryKind  # type: ignore

    _mod = _sys.modules.get("memfade")
    if _mod is None or os.path.dirname(os.path.abspath(_mod.__file__ or "")) != _HERE:
        raise RuntimeError(
            "memfade 核心副本解析到了别的路径（%s），拒绝以错误版本运行。"
            % getattr(_mod, "__file__", "?")
        )

from .clients import (  # noqa: E402
    AstrBotEmbeddingClient,
    AstrBotLLMClient,
    provider_label,
    resolve_chat_provider,
    resolve_embedding_provider,
)
from .scopes import (  # noqa: E402
    CHAT_GROUP,
    CHAT_GROUP_USER,
    CHAT_PRIVATE,
    describe,
    group_scope,
    person_scope,
    private_scope,
    read_scopes,
    resolve_write_scope,
    write_scopes,
)


PLUGIN_NAME = "astrbot_plugin_memfade"
PLUGIN_VERSION = "0.4.1"


def _plugin_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    """插件数据目录（绝对路径）。

    优先用宿主的 ``StarTools.get_data_dir()``（它会在
    ``data/plugin_data/<插件名>/`` 下建目录并返回绝对路径）；拿不到就退回插件
    目录下的 ``data/``。**绝不用相对路径**——那会落到进程 CWD，几个插件互相踩。
    """
    if StarTools is not None:
        try:
            path = str(StarTools.get_data_dir(PLUGIN_NAME))
            os.makedirs(path, exist_ok=True)
            return path
        except Exception:
            pass
    path = os.path.join(_plugin_dir(), "data")
    os.makedirs(path, exist_ok=True)
    return path


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(v) for v in value if str(v).strip())
    except TypeError:
        return (str(value),)


@register(
    PLUGIN_NAME,
    "memfade",
    "记忆淡化引擎：会记得，也会忘得恰到好处。艾宾浩斯衰减、四档淡化梯度、分层作用域隔离。",
    PLUGIN_VERSION,
    "https://github.com/Aston-x/memfade",
)
class MemFadePlugin(Star):
    """记忆淡化插件。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context)
        # ★ 基类不保存 self.config，而且 config 可能是 None（宿主没给配置时
        #   只传 context）。必须自己兜住。
        self.config = config if isinstance(config, dict) else {}
        raw: Dict[str, Any] = dict(self.config)

        self._llm = self._build_llm(raw)
        self._embedder = self._build_embedder(raw)
        # require_match 只在有语义匹配能力时才安全开：没有向量通道时，
        # 词面不重叠的近义记忆（「今天吃什么呢」vs「喜欢吃辣」）会被误杀。
        self._require_match = self._embedder is not None

        self.engine = MemoryEngine(
            self._build_core_config(raw), llm=self._llm, embedder=self._embedder
        )

        # 群成员昵称 → 用户ID，按群维护。抽取出来的记忆按主语路由到对应的人。
        self._members: Dict[str, Dict[str, str]] = {}
        # 已知作用域（维护任务要遍历）
        self._known_scopes: Set[str] = set()
        # 抽取缓冲：scope_key -> 待抽取的消息
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._counter: Dict[str, int] = {}
        self._tasks: Set[Any] = set()
        self._last_flush: Dict[str, float] = {}
        self._last_injected: Dict[str, str] = {}
        self._stats = {"llm_ok": 0, "llm_fallback": 0, "llm_error": 0}
        self._maintenance_task: Optional[Any] = None
        self._stopped = False

        # 老库里的会话也要纳入维护范围
        try:
            for scope_key, _count in self.engine.store.scope_counts():
                self._known_scopes.add(str(scope_key))
        except Exception:
            pass

        self._start_maintenance()

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #

    def _build_llm(self, raw: Dict[str, Any]) -> Optional[Any]:
        """优先用宿主的 provider；宿主没有就退回 OpenAI 兼容接口。"""
        if _as_bool(raw.get("use_host_llm"), True):
            provider_id = str(raw.get("llm_provider_id") or "").strip()
            # ★ 构造期不能判断宿主有没有 provider：插件是在 provider_manager
            #   填好之前被加载的，此刻 provider_insts 还是空的。
            #   所以这里一律建客户端，具体用哪一个留到调用时再解析。
            fallback = self._build_http_llm(raw)
            return AstrBotLLMClient(self, None, provider_id, fallback)

        return self._build_http_llm(raw)

    def _build_http_llm(self, raw: Dict[str, Any]) -> Optional[Any]:
        """OpenAI 兼容接口的 LLM 客户端（宿主 provider 不可用时的兜底）。"""
        base_url = str(raw.get("llm_base_url") or "").strip()
        model = str(raw.get("llm_model") or "").strip()
        if not (base_url and model):
            return None
        try:
            from .memfade.llm import HttpLLMClient
        except ImportError:  # pragma: no cover
            from memfade.llm import HttpLLMClient  # type: ignore

        return HttpLLMClient(
            base_url=base_url,
            api_key=str(raw.get("llm_api_key") or ""),
            model=model,
            timeout=_as_float(raw.get("llm_timeout"), 30.0),
        )

    def _host_has_provider(self, provider_id: str = "") -> bool:
        """宿主有没有可用的 provider。

        ★ 只能在**运行期**问。插件加载时 provider_manager 还没填好，
          此刻问永远是「没有」。
        """
        if provider_id:
            getter = getattr(self.context, "get_provider_by_id", None)
            if getter is None:
                return False
            try:
                return getter(provider_id) is not None
            except Exception:
                return False
        lister = getattr(self.context, "get_all_providers", None)
        if lister is None:
            return False
        try:
            return bool(lister())
        except Exception:
            return False

    def _llm_ready(self) -> bool:
        """抽取引擎当前能不能真的走到 LLM（运行期判断）。"""
        if self._llm is None:
            return False
        if isinstance(self._llm, AstrBotLLMClient):
            return self._host_has_provider(self._llm._provider_id)
        return True

    def _build_embedder(self, raw: Dict[str, Any]) -> Optional[Any]:
        if _as_bool(raw.get("use_host_embedding"), True):
            provider = resolve_embedding_provider(
                self.context, str(raw.get("embedding_provider_id") or "")
            )
            if provider is not None:
                return AstrBotEmbeddingClient(provider)

        base_url = str(raw.get("embedding_base_url") or "").strip()
        model = str(raw.get("embedding_model") or "").strip()
        if base_url and model:
            try:
                from .memfade.embedding import HttpEmbeddingClient
            except ImportError:  # pragma: no cover
                from memfade.embedding import HttpEmbeddingClient  # type: ignore

            return HttpEmbeddingClient(
                base_url=base_url,
                api_key=str(raw.get("embedding_api_key") or ""),
                model=model,
            )
        return None

    def _build_core_config(self, raw: Dict[str, Any]) -> MemFadeConfig:
        """把宿主配置翻译成核心配置。

        ★ 核心只认 7 个顶层节：storage / extraction / retrieval / **fade** /
          **maintenance** / llm / embedding。写成 ``decay`` / ``consolidation``
          会被静默忽略——之前就是踩了这个坑，fade 和 maintenance 整段没传进去。
        """
        data_dir = _data_dir()
        db_name = str(raw.get("db_path") or "memfade.db").strip() or "memfade.db"
        db_path = db_name if os.path.isabs(db_name) else os.path.join(data_dir, db_name)
        try:
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

        group_subject = str(raw.get("group_subject") or "本群")
        self._group_subject = group_subject

        return MemFadeConfig.from_dict(
            {
                "storage": {
                    "db_path": db_path,
                    "enable_fts": _as_bool(raw.get("enable_fts"), True),
                    # ★ 只有真的拿到 embedder 才开向量通道，否则 WebUI 上的开关
                    #   与实际行为不一致（引擎会静默降级成纯关键词）。
                    "enable_vector": _as_bool(raw.get("use_embedding"), True)
                    and self._embedder is not None,
                    "busy_timeout_ms": _as_int(raw.get("busy_timeout_ms"), 5000),
                },
                "extraction": {
                    "enabled": _as_bool(raw.get("extract_enabled"), True),
                    "every_n_turns": max(1, _as_int(raw.get("extract_every_n"), 3)),
                    # 核心与 MaiBot 都是 0.45；0.35 会让「今天只有一节课」这类
                    # 噪声进库。
                    "min_importance": _as_float(raw.get("min_importance"), 0.45),
                    "min_importance_episodic": _as_float(
                        raw.get("min_importance_episodic"), 0.20
                    ),
                    "min_messages": max(1, _as_int(raw.get("extract_min_messages"), 2)),
                    "store_evidence": _as_bool(raw.get("store_evidence"), True),
                    "use_rules_fallback": _as_bool(raw.get("use_rules_fallback"), True),
                    # ★ 核心默认是 True，但 cross_scope=1 的行对**所有**作用域可见，
                    #   多用户部署下会把 A 的私密记忆漏给 B 和别的群。默认关。
                    "promote_cross_scope": _as_bool(raw.get("cross_scope"), False),
                    "subject_hint": str(raw.get("subject_hint") or ""),
                    "self_names": _as_str_tuple(raw.get("self_names")),
                    "focus_subjects": _as_str_tuple(raw.get("focus_subjects")),
                    # 私聊路径强制主语是人；群路径由独立 extractor 覆盖。
                    "require_person_subject": _as_bool(
                        raw.get("require_person_subject"), True
                    ),
                },
                "retrieval": {
                    "top_k": max(1, _as_int(raw.get("top_k"), 8)),
                    "budget_tokens": max(50, _as_int(raw.get("max_inject_tokens"), 600)),
                    "echo_max": max(0, _as_int(raw.get("echo_max"), 3)),
                    "min_score": _as_float(raw.get("min_score"), 0.12),
                    "mmr_lambda": _as_float(raw.get("mmr_lambda"), 0.25),
                    "recent_rescue": _as_bool(raw.get("recent_rescue"), True),
                    "reinject_recent": max(0, _as_int(raw.get("reinject_recent"), 4)),
                },
                # ★ 之前整节没传，archive_after_days 在 schema 里存在却从没生效
                "fade": {
                    "enabled": _as_bool(raw.get("fade_enabled"), True),
                    "recall_alpha": _as_float(raw.get("recall_alpha"), 0.35),
                    "reinforce_alpha": _as_float(raw.get("reinforce_alpha"), 0.8),
                    "weaken_alpha": _as_float(raw.get("weaken_alpha"), 0.5),
                    "gist_enabled": _as_bool(raw.get("gist_enabled"), True),
                    "gist_max_chars": _as_int(raw.get("gist_max_chars"), 40),
                    "archive_after_days": max(0, _as_int(raw.get("archive_after_days"), 0)),
                    "recall_cooldown_seconds": _as_int(
                        raw.get("recall_cooldown_seconds"), 3600
                    ),
                },
                "maintenance": {
                    "enabled": _as_bool(raw.get("maintenance_enabled"), True),
                    "interval_seconds": max(
                        60, _as_int(raw.get("maintain_interval_minutes"), 60) * 60
                    ),
                    "max_rows_per_pass": _as_int(raw.get("max_rows_per_pass"), 2000),
                    "enable_merge": _as_bool(raw.get("enable_merge"), True),
                    "enable_gist": _as_bool(raw.get("enable_gist"), True),
                    "enable_supersede": _as_bool(raw.get("enable_supersede"), True),
                    "max_memories_per_scope": _as_int(
                        raw.get("max_memories_per_scope"), 5000
                    ),
                },
            }
        )

    # ------------------------------------------------------------------ #
    # provider 解析（给 clients.py 用）
    # ------------------------------------------------------------------ #

    async def resolve_provider(self, provider_id: str = "") -> Optional[Any]:
        """拿 provider。指定了 id 就按 id 取，没指定就取当前会话正在用的那个。"""
        if provider_id:
            getter = getattr(self.context, "get_provider_by_id", None)
            if getter is not None:
                try:
                    provider = getter(provider_id)
                    if provider is not None:
                        return provider
                except Exception:
                    pass
            return None
        umo = str(self._current_umo or "")
        return await resolve_chat_provider(self.context, umo)

    _current_umo: str = ""

    # ------------------------------------------------------------------ #
    # 作用域
    # ------------------------------------------------------------------ #

    @staticmethod
    def _event_ids(event: AstrMessageEvent) -> Tuple[str, str]:
        try:
            group_id = str(event.get_group_id() or "")
        except Exception:
            group_id = ""
        try:
            sender_id = str(event.get_sender_id() or "")
        except Exception:
            sender_id = ""
        return group_id, sender_id

    def _scopes_of(self, event: AstrMessageEvent) -> List[Scope]:
        group_id, sender_id = self._event_ids(event)
        return read_scopes(group_id, sender_id)

    def _remember_member(self, event: AstrMessageEvent) -> None:
        """维护「昵称 → 用户ID」映射，抽取时按主语路由到对应的人。"""
        group_id, sender_id = self._event_ids(event)
        if not group_id or not sender_id:
            return
        try:
            nickname = str(event.get_sender_name() or "").strip()
        except Exception:
            nickname = ""
        if not nickname:
            return
        table = self._members.setdefault(group_id, {})
        table[nickname] = sender_id

    # ------------------------------------------------------------------ #
    # 注入
    # ------------------------------------------------------------------ #

    def _recall_merged(
        self,
        query: str,
        scopes: List[Scope],
        *,
        top_k: int,
        query_vector: Optional[List[float]],
    ) -> List[Any]:
        """多层各自召回 → 按记忆ID去重 → 按分数合并。

        为什么不能「两次 build_context 再拼文本」：
          · 会出两个 [记忆] 头；
          · token 预算翻倍（budget_tokens 是每次 assemble 各自算的）；
          · MMR 去重各做各的 → 两层近重复内容会同时出现；
          · facts/echoes 分层被拆成两段，echo_max 也翻倍。
        合并 hit 再 assemble 一次，上面四条全部消失。

        排序可比性：两层用的是同一套权重公式（相关度/重要性/强度/新近度/频次），
        量纲一致可直接比较。唯一偏置是 BM25 与向量通道在各自 scope 内做过
        max 归一化，所以「某层的第一名」永远拿满分——层与层之间公平，层内名次
        是自归一的。这是有界偏置，不影响「哪条更相关」的判断。
        """
        hits: List[Any] = []
        seen: Set[int] = set()
        for scope in scopes:
            try:
                found = self.engine.recall(
                    query,
                    scope,
                    top_k=top_k,
                    query_vector=query_vector,
                    # 检索必须真命中（「记忆 搜 显卡驱动」不能把无关的记忆翻出来）。
                    # 注意这跟 assemble 的 require_match 不是一回事：那个是注入时
                    # 的近义保护，只在有向量通道时才安全开。
                    require_match=True,
                )
            except Exception:
                continue
            for hit in found or []:
                mid = getattr(getattr(hit, "memory", None), "id", None)
                if mid is None:
                    continue
                if mid in seen:
                    # cross_scope=1 的行在每个 scope 都会被召回（SQL:
                    # scope_key=? OR cross_scope=1）→ 必须按 id 去重
                    continue
                seen.add(mid)
                hits.append(hit)

        hits.sort(
            key=lambda h: (
                float(getattr(h, "score", 0.0) or 0.0),
                float(getattr(getattr(h, "memory", None), "importance", 0.0) or 0.0),
                float(getattr(getattr(h, "memory", None), "created_at", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return hits[:top_k]

    @filter.on_llm_request(priority=-7)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        """把召回的记忆追加到 system prompt。

        priority=-7 → 最后执行（数值越大越先跑），记忆块追加在别人改完
        system_prompt 之后，不会被覆盖。
        """
        try:
            query = (event.get_message_str() or "").strip()
            if not query:
                return
            self._current_umo = str(getattr(event, "unified_msg_origin", "") or "")
            self.ensure_maintenance()

            scopes = self._scopes_of(event)
            for scope in scopes:
                self._known_scopes.add(scope.key())

            vector = await self.engine.embed_query(query)
            top_k = max(1, _as_int(self.config.get("top_k"), 8))
            hits = self._recall_merged(
                query, scopes, top_k=top_k, query_vector=vector
            )
            if not hits:
                return

            # 画像来自最内层（人级私密 / 私聊）
            inner = scopes[-1]
            profile_text = ""
            profile_ids: Set[int] = set()
            try:
                profile = self.engine.retriever.profile(
                    inner.key(),
                    max_items=max(1, _as_int(self.config.get("profile_items"), 6)),
                    max_chars=max(60, _as_int(self.config.get("profile_chars"), 300)),
                )
                profile_text = getattr(profile, "text", "") or ""
                profile_ids = set(getattr(profile, "ids", set()) or set())
            except Exception:
                profile_text = ""
                profile_ids = set()

            ctx = self.engine.retriever.assemble(
                hits,
                header="记忆",
                exclude_ids=profile_ids,
                require_match=self._require_match,
            )
            text = getattr(ctx, "text", "") or ""
            if not text:
                return

            # 同一轮内容没变就不重复注入
            key = inner.key()
            if self._last_injected.get(key) == text:
                return
            self._last_injected[key] = text

            body = (
                "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
                "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
                f"{text}"
            )
            if profile_text:
                body = "[关于对方]\n" + profile_text + "\n\n" + body

            base = getattr(req, "system_prompt", "") or ""
            req.system_prompt = (base + "\n\n" + body).strip()

            # 被想起来的事要强化（间隔重复）
            try:
                self.engine.reinforce([h.memory.id for h in getattr(ctx, "hits", [])])
            except Exception:
                pass
        except Exception as exc:
            logger.warning("[memfade] 注入失败: %s", exc)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def _extractor_for(self, group_id: str) -> Any:
        """群路径要一个独立 extractor：让群别名能活下来。

        ``ExtractionConfig`` 是引擎级共享对象，直接改它会在多群并发时串味，
        所以这里用 ``dataclasses.replace`` 造一份独立的。
        """
        base = self.engine.config.extraction
        try:
            from dataclasses import replace

            members = self._members.get(group_id, {})
            hint = str(self.config.get("subject_hint") or "")
            if members:
                names = " / ".join(list(members.keys())[:12])
                hint = (hint + "（" + names + "）").strip()
            cfg = replace(
                base,
                require_person_subject=False,
                subject_hint=hint,
            )
        except Exception:
            return self.engine.extractor
        return MemoryExtractor(cfg, self.engine.llm)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: Any = None) -> None:
        """这一轮结束后抽取记忆。

        抽取要等 LLM，不能让它挡住这一轮回复——所以丢后台任务。
        """
        del response
        try:
            query = (event.get_message_str() or "").strip()
            if not query:
                return
            group_id, sender_id = self._event_ids(event)
            scopes = self._scopes_of(event)
            for scope in scopes:
                self._known_scopes.add(scope.key())

            speaker = ""
            try:
                speaker = str(event.get_sender_name() or "").strip()
            except Exception:
                speaker = ""

            entry = {"role": "user", "content": query}
            if speaker:
                # 带上说话人：抽取器要能分清「谁说的」
                entry["speaker"] = speaker

            # 缓冲按最内层（人级）走
            inner = scopes[-1]
            buffer = self._buffer.setdefault(inner.key(), [])
            buffer.append(entry)

            count = self._counter.get(inner.key(), 0) + 1
            self._counter[inner.key()] = count
            every = max(1, _as_int(self.config.get("extract_every_n"), 3))
            idle_minutes = _as_int(self.config.get("flush_idle_minutes"), 10)

            should_flush = count % every == 0
            if not should_flush and idle_minutes > 0:
                last = self._last_flush.get(inner.key(), 0.0)
                if last and (time.time() - last) > idle_minutes * 60:
                    should_flush = True

            if not should_flush:
                return

            pending = list(buffer)
            buffer.clear()
            self._last_flush[inner.key()] = time.time()
            task = asyncio.create_task(
                self._extract_pending(pending, group_id, sender_id, inner)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.warning("[memfade] 抽取失败: %s", exc)

    async def _extract_pending(
        self,
        pending: List[Dict[str, Any]],
        group_id: str,
        sender_id: str,
        inner: Scope,
    ) -> None:
        """后台跑一次抽取，并按主语分层落库。"""
        try:
            extractor = self._extractor_for(group_id)
            result = await extractor.extract(pending, source=inner.key())

            used_llm = bool(getattr(result, "used_llm", False))
            error = getattr(result, "error", None)
            if used_llm:
                self._stats["llm_ok"] += 1
            else:
                self._stats["llm_fallback"] += 1
            if error:
                self._stats["llm_error"] += 1
                logger.warning("[memfade] LLM 抽取失败，已退规则抽取: %s", error)

            drafts = list(getattr(result, "drafts", []) or [])
            if not drafts:
                return

            members = self._members.get(group_id, {})
            group_subject = str(getattr(self, "_group_subject", "本群"))

            buckets: Dict[str, List[Any]] = {}
            for draft in drafts:
                subject = str(getattr(draft, "subject", "") or "")
                target = resolve_write_scope(
                    subject,
                    group_id=group_id,
                    sender_id=sender_id,
                    members=members,
                    group_subject=group_subject,
                )
                if target is None:
                    # 主语不认识 → 丢弃，不进任何层（退群层等于公开私密事实）
                    continue
                buckets.setdefault(target.key(), []).append(draft)

            for key, items in buckets.items():
                scope = Scope.from_key(key)
                self._known_scopes.add(key)
                try:
                    written = self.engine.remember_many(items, scope, source=key)
                except Exception:
                    continue
                if written:
                    try:
                        await self.engine.embed_memories(written)
                    except Exception:
                        pass
                    logger.info(
                        "[memfade] 抽取到 %d 条新记忆（%s）", len(written), key
                    )
        except Exception as exc:
            logger.warning("[memfade] 抽取失败: %s", exc)
            # 失败要把消息放回去，不然这轮内容就丢了
            try:
                self._buffer.setdefault(inner.key(), []).extend(pending)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 后台维护
    # ------------------------------------------------------------------ #

    def _start_maintenance(self) -> None:
        """定期跑整理。

        没有这个任务，核心一半能力是哑的：gist 压缩、近似重复合并、矛盾取代、
        按上限淘汰全都不会发生，淡化只会单向衰减、不会「压缩成摘要」。
        """
        if not _as_bool(self.config.get("maintenance_enabled"), True):
            return
        interval = max(
            60, _as_int(self.config.get("maintain_interval_minutes"), 60) * 60
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环（构造期还没进 async 上下文），等真正跑起来
            # 时由 ensure_maintenance() 启动。
            self._maintenance_task = None
            return
        try:
            self._maintenance_task = asyncio.create_task(
                self._maintenance_loop(interval)
            )
        except RuntimeError:
            self._maintenance_task = None

    def ensure_maintenance(self) -> None:
        """确保后台整理任务在跑（构造期可能没有事件循环）。"""
        if self._maintenance_task is None and not self._stopped:
            self._start_maintenance()

    async def _maintenance_loop(self, interval: int) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(interval)
                if self._stopped:
                    return
                for key in list(self._known_scopes):
                    try:
                        scope = Scope.from_key(key)
                        await self.engine.consolidate(scope)
                    except Exception:
                        continue
            except asyncio.CancelledError:
                return
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # 指令
    # ------------------------------------------------------------------ #

    @filter.command_group("记忆")
    def memory_group(self):
        """记忆相关指令"""

    @memory_group.command("状态")
    async def memory_status(self, event: AstrMessageEvent):
        """这个会话记住了多少东西"""
        scopes = self._scopes_of(event)
        lines = []
        total_ids: Set[int] = set()
        for scope in scopes:
            try:
                stats = self.engine.stats(scope)
                by_status = stats.get("by_status") or {}
                ids = [mid for mid, _ in self.engine.store.recent(scope.key(), limit=8)]
                recent = sorted(
                    self.engine.store.get_many(ids),
                    key=lambda m: m.created_at,
                    reverse=True,
                )
                lines.append(
                    "%s层：记住了 %d 条（活跃 %d / 淡出 %d / 摘要 %d / 归档 %d）"
                    % (
                        describe(scope),
                        stats.get("total", 0),
                        by_status.get("active", 0),
                        by_status.get("fading", 0),
                        by_status.get("gisted", 0),
                        by_status.get("archived", 0),
                    )
                )
                for memory in recent:
                    total_ids.add(memory.id)
                    marker = (
                        "" if memory.status.value == "active" else "（已淡出）"
                    )
                    lines.append("· %s%s" % (memory.content, marker))
            except Exception:
                continue

        if not lines:
            lines = ["还没记住什么"]

        used = self._stats
        mode = "LLM" if self._llm_ready() else "规则兜底（没接 LLM）"
        lines.append("")
        lines.append(
            "抽取引擎：%s（成功 %d / 降级 %d / 报错 %d）"
            % (mode, used["llm_ok"], used["llm_fallback"], used["llm_error"])
        )
        if self._embedder is not None:
            lines.append("向量检索：已启用")
        else:
            lines.append("向量检索：未启用（宿主没配 embedding provider）")

        yield event.plain_result("\n".join(lines))

    @memory_group.command("搜")
    async def memory_search(self, event: AstrMessageEvent, query: str = ""):
        """检索记忆（必须真命中）"""
        text = (query or "").strip()
        if not text:
            yield event.plain_result("要搜什么？用法：记忆 搜 <关键词>")
            return
        scopes = self._scopes_of(event)
        vector = await self.engine.embed_query(text)
        hits = self._recall_merged(text, scopes, top_k=8, query_vector=vector)
        if not hits:
            yield event.plain_result("没找到相关的记忆。")
            return
        lines = []
        for hit in hits:
            marker = "（可能已经变了）" if hit.tier == "echo" else ""
            lines.append("· %s%s" % (hit.memory.content, marker))
        yield event.plain_result("记得这些：\n" + "\n".join(lines))

    @memory_group.command("整理")
    async def memory_maintain(self, event: AstrMessageEvent):
        """立刻跑一轮整理"""
        scopes = self._scopes_of(event)
        scanned = decayed = merged = superseded = 0
        for scope in scopes:
            try:
                report = await self.engine.consolidate(scope)
                scanned += report.scanned
                decayed += report.decayed
                merged += report.merged
                superseded += report.superseded
            except Exception:
                continue
        yield event.plain_result(
            "整理完成：扫描 %d 条，淡出 %d 条，合并 %d 条，取代 %d 条。"
            % (scanned, decayed, merged, superseded)
        )

    @memory_group.command("置顶")
    async def memory_pin(self, event: AstrMessageEvent, query: str = ""):
        """让一条记忆永不淡化"""
        text = (query or "").strip()
        if not text:
            yield event.plain_result("要置顶哪条？给点内容片段。")
            return
        scopes = self._scopes_of(event)
        vector = await self.engine.embed_query(text)
        hits = self._recall_merged(text, scopes, top_k=5, query_vector=vector)
        if not hits:
            yield event.plain_result("没找到要置顶的那条记忆。")
            return
        target = hits[0].memory
        self.engine.pin(target.id, pinned=True)
        yield event.plain_result("置顶了：%s" % target.content)

    @memory_group.command("修正")
    async def memory_repair(self, event: AstrMessageEvent, query: str = ""):
        """改掉记错的那条记忆（老内容 => 新内容）"""
        old, new = split_repair_request((query or "").strip())
        if not old or not new:
            yield event.plain_result(
                "要改哪条？照这个格式发：记忆 修正 老内容 => 新内容"
            )
            return
        for scope in self._scopes_of(event):
            try:
                updated = await self.engine.repair(scope, old, new)
            except Exception:
                continue
            if updated is None:
                continue
            previous = str((updated.meta or {}).get("previous_content") or "")
            yield event.plain_result(
                "改好了：%s\n现在记的是：%s" % (previous, updated.content)
            )
            return
        yield event.plain_result("没找到要改的那条记忆。")

    @filter.command("记住")
    async def cmd_remember(self, event: AstrMessageEvent, content: str = ""):
        """显式记住一件事"""
        text = (content or "").strip()
        if not text:
            yield event.plain_result("要记住什么呢？用法：记住 <内容>")
            return
        group_id, sender_id = self._event_ids(event)
        self._remember_member(event)
        scope = resolve_write_scope(
            self._speaker_of(event),
            group_id=group_id,
            sender_id=sender_id,
            members=self._members.get(group_id, {}),
            group_subject=str(getattr(self, "_group_subject", "本群")),
        )
        if scope is None:
            scope = (
                person_scope(group_id, sender_id)
                if group_id
                else private_scope(sender_id)
            )
        written = self.engine.remember_atomic(
            MemoryDraft(
                content=text,
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject=self._speaker_of(event),
                source="astrbot",
            ),
            scope,
        )
        if not written:
            yield event.plain_result("没记下来（内容不重要或与已有记忆重复）")
            return
        try:
            await self.engine.embed_memories(written)
        except Exception:
            pass
        self._known_scopes.add(scope.key())
        if len(written) == 1:
            yield event.plain_result("记住了：%s" % written[0].content)
        else:
            yield event.plain_result(
                "记住了这 %d 件事：\n%s"
                % (
                    len(written),
                    "\n".join("- " + m.content for m in written),
                )
            )

    @filter.command("群记")
    async def cmd_group_remember(self, event: AstrMessageEvent, content: str = ""):
        """记住一件对全群都成立的事"""
        text = (content or "").strip()
        if not text:
            yield event.plain_result("要记住什么呢？用法：群记 <内容>")
            return
        group_id, _sender_id = self._event_ids(event)
        if not group_id:
            yield event.plain_result("群记只能在群里用。私聊直接用「记住」。")
            return
        scope = group_scope(group_id)
        written = self.engine.remember_atomic(
            MemoryDraft(
                content=text,
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                subject=str(getattr(self, "_group_subject", "本群")),
                source="astrbot",
            ),
            scope,
        )
        if not written:
            yield event.plain_result("没记下来（内容不重要或与已有记忆重复）")
            return
        try:
            await self.engine.embed_memories(written)
        except Exception:
            pass
        self._known_scopes.add(scope.key())
        yield event.plain_result("记住了（全群共享）：%s" % written[0].content)

    @filter.command("忘掉")
    async def cmd_forget(self, event: AstrMessageEvent, content: str = ""):
        """忘掉一条记忆"""
        text = (content or "").strip()
        if not text:
            yield event.plain_result("要忘掉什么？用法：忘掉 <内容片段>")
            return
        scopes = self._scopes_of(event)
        vector = await self.engine.embed_query(text)
        hits = self._recall_merged(text, scopes, top_k=5, query_vector=vector)
        if not hits:
            yield event.plain_result("没找到要忘掉的记忆。")
            return
        target = hits[0].memory
        self.engine.forget(target.id, hard=True)
        yield event.plain_result("忘掉了：%s" % target.content)

    @staticmethod
    def _speaker_of(event: AstrMessageEvent) -> str:
        try:
            return str(event.get_sender_name() or "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------ #

    async def terminate(self) -> None:
        """插件卸载。"""
        self._stopped = True
        if self._maintenance_task is not None:
            try:
                self._maintenance_task.cancel()
            except Exception:
                pass
            self._maintenance_task = None
        # 等抽取任务收尾再关库，否则会写坏库
        for task in list(self._tasks):
            try:
                await task
            except Exception:
                pass
        self._tasks.clear()
        try:
            self.engine.close()
        except Exception:
            pass
