"""NoneBot2 适配器的核心：引擎 + 公开助手 API。"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from .config import config


def _bootstrap_import() -> None:
    """让 ``import memfade`` 在自带核心源码的部署形态下也能工作。"""

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (here, os.path.dirname(os.path.dirname(here))):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_bootstrap_import()

from memfade import MemFadeConfig, MemoryDraft, MemoryKind, Scope  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402
from memfade.llm import HttpLLMClient  # noqa: E402


def _build_engine() -> MemoryEngine:
    core = MemFadeConfig.from_dict(
        {
            "storage": {"db_path": config.memfade_db_path, "enable_vector": False},
            "extraction": {
                "enabled": config.memfade_extract_enabled,
                "every_n_turns": max(1, int(config.memfade_extract_every_n)),
                "min_importance": float(config.memfade_min_importance),
            },
            "retrieval": {
                "top_k": max(1, int(config.memfade_top_k)),
                "budget_tokens": max(50, int(config.memfade_max_inject_tokens)),
                "echo_max": max(0, int(config.memfade_echo_max)),
            },
        }
    )
    llm = None
    if config.memfade_llm_base_url and config.memfade_llm_model:
        llm = HttpLLMClient(
            base_url=config.memfade_llm_base_url,
            api_key=config.memfade_llm_api_key,
            model=config.memfade_llm_model,
        )
    return MemoryEngine(core, llm=llm)


engine = _build_engine()
_buffer: Dict[str, List[Dict[str, Any]]] = {}
_counter: Dict[str, int] = {}


def scope_of(event: Any) -> Scope:
    """从事件构造作用域。

    优先用 ``get_session_id()``——在 NoneBot2 里它就是**会话**粒度
    （群聊一个群一个、私聊一个人一个），正好是记忆要的隔离粒度。

    不依赖任何具体适配器：OneBot v11/v12、NapCat、Lagrange 都有这个方法，
    没有的话退化成 group_id / user_id。
    """

    def _call(name: str) -> str:
        getter = getattr(event, name, None)
        if not callable(getter):
            return ""
        try:
            return str(getter() or "")
        except Exception:
            return ""

    session_id = _call("get_session_id")
    group_id = _call("get_group_id")
    user_id = _call("get_user_id")
    chat_type = "group" if group_id else "private"
    chat_id = session_id or group_id or user_id or "unknown"
    return Scope(platform="nonebot", chat_type=chat_type, chat_id=chat_id)


async def build_context(event: Any, query: str = "") -> str:
    """给其它 LLM 插件用的入口：拿到可注入 prompt 的记忆文本。"""

    text = query
    if not text:
        try:
            text = str(event.get_message() or "")
        except Exception:
            text = ""
    if not text:
        return ""
    scope = scope_of(event)
    # 向量要在这里 await：引擎内部的同步取向量在事件循环里会直接放弃
    # （不阻塞事件循环），异步路径必须自己算。
    vector = await engine.embed_query(text)
    memory_ctx = engine.build_context(
        text, scope, top_k=config.memfade_top_k, query_vector=vector
    )
    if not memory_ctx.text:
        return ""
    return (
        "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
        "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
        f"{memory_ctx.text}"
    )


async def recall(event: Any, query: str = "") -> list:
    """召回记忆（不注入）。"""

    scope = scope_of(event)
    return engine.recall(query, scope, top_k=config.memfade_top_k)


async def remember(event: Any, content: str, *, kind: str = "semantic", importance: float = 0.9) -> Optional[Any]:
    """显式写入记忆（太长会按标点拆成「一件事一条」）。

    返回写入的第一条；调用方想知道拆了几条的话，用 ``remember_all``。
    """

    text = (content or "").strip()
    if not text:
        return None
    scope = scope_of(event)
    try:
        memory_kind = MemoryKind(str(kind).lower())
    except ValueError:
        memory_kind = MemoryKind.SEMANTIC
    draft = MemoryDraft(content=text, kind=memory_kind, importance=importance, source="nonebot")
    written = engine.remember_atomic(draft, scope)
    # 补向量：写入是同步的，而嵌入服务多半是 async 的——
    # 不补的话记忆库里一条向量都没有，向量检索等于没开。
    await engine.embed_memories(written)
    return written[0] if written else None


async def remember_all(
    event: Any, content: str, *, kind: str = "semantic", importance: float = 0.9
) -> list:
    """显式写入记忆，返回写入的全部条目。"""

    text = (content or "").strip()
    if not text:
        return []
    scope = scope_of(event)
    try:
        memory_kind = MemoryKind(str(kind).lower())
    except ValueError:
        memory_kind = MemoryKind.SEMANTIC
    draft = MemoryDraft(content=text, kind=memory_kind, importance=importance, source="nonebot")
    written = engine.remember_atomic(draft, scope)
    await engine.embed_memories(written)
    return written

