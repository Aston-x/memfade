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

# ⚠ **副本校验**：同进程里若已有别的 `memfade` 副本，`sys.modules`
# 会直接返回那一份——两份副本的 MemoryKind 是**不同的类对象**
# （`==` 相等但 `is` 不相等），而核心里有多处 `is MemoryKind.X` 的身份比较，
# 会静默走错分支。宁可明确报错，也不要静默用错版本。
# 核心副本在包的**同级目录** ``src/memfade``（见 _bootstrap_import）。
_memfade_module = sys.modules.get("memfade")
_memfade_resolved = os.path.abspath(getattr(_memfade_module, "__file__", "") or "")
_memfade_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_memfade_ok = {
    os.path.join(_memfade_src, "memfade", "__init__.py"),
    # 开发树里的源码副本（跑测试时 `import memfade` 解析到的是它）。
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_memfade_src))),
        "memfade",
        "__init__.py",
    ),
}
if _memfade_module is None or _memfade_resolved not in _memfade_ok:
    raise RuntimeError(
        "memfade 核心副本解析到了别的路径（%s），拒绝以错误版本运行。"
        "同一进程里只能有一份 memfade 副本。"
        % getattr(_memfade_module, "__file__", "?")
    )


def _build_engine() -> MemoryEngine:
    core = MemFadeConfig.from_dict(
        {
            "storage": {"db_path": config.memfade_db_path, "enable_vector": False},
            "extraction": {
                "enabled": config.memfade_extract_enabled,
                "every_n_turns": max(1, int(config.memfade_extract_every_n)),
                "min_importance": float(config.memfade_min_importance),
                "subject_hint": str(config.memfade_subject_hint or ""),
                "self_names": tuple(config.memfade_self_names or ()),
                "focus_subjects": tuple(config.memfade_focus_subjects or ()),
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


async def recall_matching(event: Any, query: str = "", *, limit: int = 5) -> list:
    """检索记忆（必须真命中）。

    和 ``recall`` 的区别：``recall`` 会带上「最近写入」的候选，
    显式检索（用户主动搜、模型主动翻）不该混入无关记忆——
    所以走 ``require_match=True``。
    """

    scope = scope_of(event)
    vector = await engine.embed_query(query)
    return engine.recall(
        query,
        scope,
        top_k=max(1, min(20, int(limit or 5))),
        query_vector=vector,
        require_match=True,
    )


async def forget(event: Any, query: str = "") -> Optional[Any]:
    """忘掉一条（按内容片段找，返回被忘掉的那条）。"""

    hits = await recall_matching(event, query, limit=5)
    if not hits:
        return None
    target = hits[0].memory
    engine.forget(target.id, hard=True)
    return target


async def set_pinned(event: Any, query: str = "", *, pinned: bool = True) -> Optional[Any]:
    """置顶/取消置顶（此后永不淡化）。"""

    hits = await recall_matching(event, query, limit=5)
    if not hits:
        return None
    target = hits[0].memory
    engine.pin(target.id, pinned=pinned)
    return target


async def maintain(event: Any):
    """跑一轮整理（重算淡化档位、合并重复、仲裁矛盾）。"""

    return await engine.consolidate(scope_of(event))


def stats(event: Any) -> dict:
    """记忆库状态统计。"""

    return engine.stats(scope_of(event))


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

