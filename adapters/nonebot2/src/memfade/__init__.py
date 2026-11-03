"""memfade —— 记忆淡化引擎。

给聊天机器人用的可移植长期记忆插件，核心零第三方依赖。

一句话概括它和别的记忆库的区别：
**别的库记住「发生过什么」，memfade 还要管「这件事还该不该被想起来」。**

快速上手：

    from memfade import MemoryEngine, Scope, MemoryDraft

    engine = MemoryEngine(db_path="memfade.db")
    scope = Scope(platform="qq", chat_type="private", chat_id="1738157278")
    engine.remember(MemoryDraft(content="许杏玖喜欢辣的食物", kind="semantic",
                                 importance=0.8), scope)
    ctx = engine.build_context("今天吃什么", scope)
    print(ctx.text)
"""

from .config import MemFadeConfig
from .consolidate import ConsolidationReport, ConsolidationService
from .decay import (
    DEFAULT_POLICY,
    DEFAULT_PROFILES,
    FadePolicy,
    KindProfile,
    MemoryEvent,
    apply_event,
    decay_curve,
    effective_strength,
    next_review_days,
    resolve_status,
    stability_days,
)
from .embedding import CallableEmbeddingClient, EmbeddingClient, HttpEmbeddingClient, NullEmbeddingClient
from .engine import MemoryEngine
from .extract import MemoryExtractor
from .guards import (
    Evidence,
    Verdict,
    check_draft,
    check_remember_request,
    find_evidence,
    guard_drafts,
)
from .llm import CallableLLMClient, HttpLLMClient, LLMClient
from .retrieve import MemoryContext, MemoryRetriever
from .store import MemoryStore
from .textutil import split_repair_request
from .types import Memory, MemoryDraft, MemoryKind, MemoryStatus, RetrievalHit, Scope

__version__ = "0.4.0"

__all__ = [
    "CallableEmbeddingClient",
    "CallableLLMClient",
    "ConsolidationReport",
    "ConsolidationService",
    "DEFAULT_POLICY",
    "DEFAULT_PROFILES",
    "EmbeddingClient",
    "Evidence",
    "FadePolicy",
    "HttpEmbeddingClient",
    "HttpLLMClient",
    "KindProfile",
    "LLMClient",
    "Memory",
    "MemoryContext",
    "MemoryDraft",
    "MemoryEngine",
    "MemoryEvent",
    "MemoryExtractor",
    "MemoryKind",
    "MemoryRetriever",
    "MemoryStatus",
    "MemoryStore",
    "MemFadeConfig",
    "NullEmbeddingClient",
    "RetrievalHit",
    "Scope",
    "Verdict",
    "apply_event",
    "check_draft",
    "check_remember_request",
    "decay_curve",
    "effective_strength",
    "find_evidence",
    "guard_drafts",
    "next_review_days",
    "resolve_status",
    "split_repair_request",
    "stability_days",
]

