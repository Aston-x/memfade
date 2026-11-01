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
    memory_ctx = engine.build_context(text, scope, top_k=config.memfade_top_k)
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
    """显式写入一条记忆。"""

    text = (content or "").strip()
    if not text:
        return None
    scope = scope_of(event)
    try:
        memory_kind = MemoryKind(str(kind).lower())
    except ValueError:
        memory_kind = MemoryKind.SEMANTIC
    draft = MemoryDraft(content=text, kind=memory_kind, importance=importance, source="nonebot")
    # 按「一件事一条」写：太长的一段会拆成几条，
    # 返回第一条（调用方只需要知道「记下了」）。
    written = engine.remember_atomic(draft, scope)
    return written[0] if written else None

