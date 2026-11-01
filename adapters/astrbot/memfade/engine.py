"""记忆引擎门面。

适配器只需要认识这一个类。所有跨框架的差异（怎么拿消息、怎么注入 prompt、
怎么调 LLM）都留在适配器里，引擎只管记忆本身。

## 强化（reinforce）

记忆被**真正注入上下文**之后才算「被想起」，这时才做残留学习式强化。
不在召回阶段强化——否则「一被检索就变强、变强更容易被检索」的正反馈
会把少数几条记忆刷成霸榜。冷却期进一步压制这个效应。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional, Sequence

from .config import MemFadeConfig
from .consolidate import ConsolidationReport, ConsolidationService
from .decay import DEFAULT_POLICY, FadePolicy, MemoryEvent, apply_event, effective_strength
from .dedup import is_near_duplicate
from .embedding import EmbeddingClient, make_embedder
from .extract import MemoryExtractor
from .llm import LLMClient
from .retrieve import MemoryContext, MemoryRetriever
from .store import MemoryStore
from .textutil import fts_text, normalized_hash
from .types import Memory, MemoryDraft, MemoryKind, MemoryStatus, RetrievalHit, Scope


class MemoryEngine:
    """记忆引擎。适配器的唯一入口。"""

    def __init__(
        self,
        config: Optional[MemFadeConfig] = None,
        *,
        db_path: Optional[str] = None,
        llm: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingClient] = None,
        policy: Optional[FadePolicy] = None,
    ) -> None:
        self.config = config or MemFadeConfig()
        self.policy = policy or self._build_policy()
        path = db_path or self.config.storage.db_path
        self.store = MemoryStore(
            path,
            busy_timeout_ms=self.config.storage.busy_timeout_ms,
            enable_fts=self.config.storage.enable_fts,
        )
        self.llm = llm
        self.embedder = embedder
        self.extractor = MemoryExtractor(self.config.extraction, llm)
        self.retriever = MemoryRetriever(self.store, self.config, self.policy)
        self._embed_one = make_embedder(embedder) if self.config.storage.enable_vector else None
        self.consolidator = ConsolidationService(self.store, self.config, self.policy, self.extractor)

    def _build_policy(self) -> FadePolicy:
        """按配置构造淡化策略。"""

        base = DEFAULT_POLICY
        return FadePolicy(
            profiles=dict(base.profiles),
            disabled=not self.config.fade.enabled,
            recall_alpha=self.config.fade.recall_alpha,
            reinforce_alpha=self.config.fade.reinforce_alpha,
            weaken_alpha=self.config.fade.weaken_alpha,
        )

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def remember_atomic(
        self,
        draft: MemoryDraft,
        scope: Scope,
        *,
        now: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> list[Memory]:
        """写入一条记忆，但**按「一件事一条」拆开**。

        为什么要拆：一段 200 字的话塞进记忆库，注入时只能截断
        （断在半句里），矛盾检测也抓不住（缺三元组），
        体检还会一直报警。拆成几条短的才是一颗干净的子弹。

        三元组只挂在**第一条**上——那一条是「主语 + 第一件事」，
        后面的碎片脱离了这个三元组，挂了反而是错的。

        返回实际写入的记忆（重复的会被去重掉，不在里面）。
        """

        from .textutil import split_atomic

        content = (draft.content or "").strip()
        if not content:
            return []
        limit = int(self.config.extraction.max_chars)
        pieces = split_atomic(content, max_chars=limit) if len(content) > limit else [content]
        out: list[Memory] = []
        for index, piece in enumerate(pieces):
            piece_draft = MemoryDraft(
                content=piece,
                kind=draft.kind,
                importance=draft.importance,
                subject=draft.subject if index == 0 else "",
                predicate=draft.predicate if index == 0 else "",
                object=draft.object if index == 0 else "",
                source=draft.source,
                cross_scope=draft.cross_scope,
                meta=draft.meta,
            )
            memory = self.remember(piece_draft, scope, now=now, meta=meta)
            if memory is not None:
                out.append(memory)
        return out

    def remember(
        self,
        draft: MemoryDraft,
        scope: Scope,
        *,
        now: Optional[float] = None,
        skip_dedup: bool = False,
        meta: Optional[dict] = None,
    ) -> Optional[Memory]:
        """写入一条记忆。命中重复则不新建，返回已有那条。"""

        ts = float(now if now is not None else time.time())
        content = (draft.content or "").strip()
        if not content:
            return None
        if draft.importance < self._min_importance(draft.kind):
            return None

        scope_key = scope.key()
        if not skip_dedup:
            existing = self.store.find_by_hash(normalized_hash(content), scope_key)
            if existing is not None:
                return existing
            same_spo = self.store.find_by_spo(
                scope_key, draft.subject, draft.predicate, draft.object
            )
            if same_spo is not None:
                return same_spo
            near = self._find_near_duplicate(content, scope_key)
            if near is not None:
                return near

        memory = Memory(
            id=0,
            scope_key=scope_key,
            kind=draft.kind,
            content=content,
            importance=max(0.0, min(1.0, float(draft.importance))),
            strength=1.0,
            anchor_at=ts,
            status=MemoryStatus.ACTIVE,
            created_at=ts,
            updated_at=ts,
            pinned=False,
            cross_scope=bool(draft.cross_scope),
            subject=draft.subject,
            predicate=draft.predicate,
            object=draft.object,
            source=draft.source,
            meta={**dict(draft.meta), **(meta or {})},
        )
        self.store.add(memory, fts_text(content))
        self._embed(memory)
        return memory

    def remember_many(
        self,
        drafts: Sequence[MemoryDraft],
        scope: Scope,
        *,
        now: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> list[Memory]:
        """批量写入，自动跳过近似重复。"""

        out: list[Memory] = []
        scope_key = scope.key()
        for draft in drafts:
            content = (draft.content or "").strip()
            if not content:
                continue
            if draft.importance < self._min_importance(draft.kind):
                continue
            if self.store.find_by_hash(normalized_hash(content), scope_key) is not None:
                continue
            # 三元组相同 = 同一条事实（换措辞也认）。
            # 文本相似度分不开「已经开始玩X」和「已经在玩X」，
            # 但拆成 (谁,怎么样,什么) 就分得开了。
            same_spo = self.store.find_by_spo(
                scope_key, draft.subject, draft.predicate, draft.object
            )
            if same_spo is not None:
                out.append(same_spo)
                continue
            near = self._find_near_duplicate(content, scope_key)
            if near is not None:
                out.append(near)
                continue
            mem = self.remember(draft, scope, now=now, skip_dedup=True, meta=meta)
            if mem is not None:
                out.append(mem)
        return out

    async def remember_from_messages(
        self,
        messages: Sequence[dict[str, Any]],
        scope: Scope,
        *,
        source: str = "",
        now: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> list[Memory]:
        """从一段对话抽取记忆并落库（异步，走 LLM）。

        这是「冷路径」：每 N 轮才跑一次，不在每条消息的热路径上。
        """

        result = await self.extractor.extract(messages, source=source)
        if not result.drafts:
            return []
        written = self.remember_many(result.drafts, scope, now=now, meta=meta)
        # ⚠ 必须在这里补向量：写入路径是同步的，而宿主的嵌入服务多半是 async 的——
        # 有事件循环时同步取向量会堵住事件循环，所以同步路径只能跳过。
        # **不补这一下，记忆库里一条向量都没有，向量检索这条通道等于没开。**
        await self.embed_memories(written)
        return written

    async def embed_memories(self, memories: Sequence[Memory]) -> int:
        """给还没向量的记忆补向量。返回补上的条数。

        同步写入路径在有事件循环时拿不到向量（`_embed_sync` 会跳过，
        免得堵住事件循环），所以异步路径必须自己补齐。

        没配嵌入模型时什么都不做——纯 BM25 也能跑。
        """

        if self._embed_one is None:
            return 0
        count = 0
        for mem in memories or ():
            if mem is None or not mem.id:
                continue
            vector = await self.embed_query(mem.content)
            if vector:
                self.store.set_embedding(mem.id, vector)
                count += 1
        return count

    def _find_near_duplicate(self, content: str, scope_key: str) -> Optional[Memory]:
        """近似重复检测。命中的话把旧记忆强化一下而不是新建。"""

        threshold = self.config.retrieval.dedup_threshold
        if threshold <= 0:
            return None
        probe = Memory(
            id=0,
            scope_key=scope_key,
            kind=MemoryKind.EPISODIC,
            content=content,
            importance=0.0,
            strength=1.0,
            anchor_at=0.0,
            status=MemoryStatus.ACTIVE,
        )
        # 先用全文索引把候选缩到几十条，再逐个比相似度——
        # 比全表线性扫描快一个数量级，且不会漏掉措辞差异大的同义句。
        candidates = self.store.search_fts(content, scope_key, limit=20)
        ids = [mid for mid, _ in candidates]
        recent = self.store.recent(scope_key, limit=30)
        ids.extend(mid for mid, _ in recent)
        seen: set[int] = set()
        for mid in ids:
            if mid in seen:
                continue
            seen.add(mid)
            mem = self.store.get(mid)
            if mem is None:
                continue
            if mem.status in (MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED):
                continue
            if is_near_duplicate(mem, probe, threshold):
                return mem
        return None

    def _min_importance(self, kind: MemoryKind) -> float:
        """按种类取入库门槛。

        事件类记忆天生重要性低（抽取器给 0.2~0.45），
        若和稳定事实共用一个门槛会被全部滤掉——那淡化梯度就永远没机会上场。
        事件便宜、靠自然退场淘汰，门槛放低；稳定事实常驻注入，门槛保持高。
        """

        if kind is MemoryKind.EPISODIC:
            return min(
                self.config.extraction.min_importance,
                self.config.extraction.min_importance_episodic,
            )
        return self.config.extraction.min_importance

    # ------------------------------------------------------------------ #
    # 召回与注入
    # ------------------------------------------------------------------ #

    def recall(
        self,
        query: str,
        scope: Scope,
        *,
        now: Optional[float] = None,
        top_k: Optional[int] = None,
        kinds: Optional[Sequence[str]] = None,
        query_vector: Optional[Sequence[float]] = None,
        require_match: Optional[bool] = None,
    ) -> list[RetrievalHit]:
        """召回记忆（不强化）。"""

        if require_match is None:
            require_match = self._embed_one is not None
        if query_vector is None and self._embed_one is not None:
            # 调用方没给向量就自己算一份，省得每条检索路径都得记得传。
            # _embed_sync 在有事件循环时返回 None（不阻塞事件循环），
            # 异步路径由适配器显式 await engine.embed_query()。
            query_vector = self._embed_sync(query)
        return self.retriever.recall(
            query,
            scope.key(),
            now=now,
            top_k=top_k,
            kinds=kinds,
            query_vector=query_vector,
            require_match=require_match,
        )

    def build_context(
        self,
        query: str,
        scope: Scope,
        *,
        now: Optional[float] = None,
        top_k: Optional[int] = None,
        header: str = "记忆",
        reinforce: bool = True,
        query_vector: Optional[Sequence[float]] = None,
        exclude_ids: Optional[Sequence[int]] = None,
        require_match: Optional[bool] = None,
    ) -> MemoryContext:
        """召回 + 分层 + 预算裁剪 + 强化。适配器的常用入口。"""

        if require_match is None:
            # 有向量能力时才要求「检索真的命中」——
            # 否则词面不重叠的近义记忆会被误杀。
            require_match = self._embed_one is not None
        hits = self.recall(
            query,
            scope,
            now=now,
            top_k=top_k,
            query_vector=query_vector,
            require_match=require_match,
        )
        context = self.retriever.assemble(
            hits,
            header=header,
            exclude_ids=set(exclude_ids or ()),
            require_match=require_match,
        )
        if reinforce and context.hits:
            self.reinforce([h.memory.id for h in context.hits], now=now)
        return context

    def build_profile(
        self,
        scope: Scope,
        *,
        now: Optional[float] = None,
        max_items: int = 8,
        max_chars: int = 400,
    ) -> str:
        """「关于对方」的稳定事实块，常驻注入。"""

        return self.retriever.build_profile(
            scope.key(), now=now, max_items=max_items, max_chars=max_chars
        )

    def explain(
        self,
        query: str,
        scope: Scope,
        *,
        now: Optional[float] = None,
        top_k: int = 8,
        require_match: Optional[bool] = None,
        query_vector: Optional[Sequence[float]] = None,
    ) -> list:
        """解释「为什么召回了这些记忆」。

        走的是**真实的召回路径**（同一条打分公式、同样的通道、同样的门槛），
        所以面板说的和机器人实际做的是同一件事。
        """

        if require_match is None:
            require_match = self._embed_one is not None
        if query_vector is None and self._embed_one is not None:
            query_vector = self._embed_sync(query)
        return self.retriever.explain(
            query,
            scope.key(),
            now=now,
            top_k=top_k,
            require_match=require_match,
            query_vector=query_vector,
        )

    def diagnose(self, scope: Optional[Scope] = None) -> dict:
        """记忆库体检。"""

        from .diagnostics import analyze

        return analyze(self.store, scope.key() if scope else None)

    def export(self, scope: Optional[Scope] = None) -> list:
        """导出记忆（JSON 友好）。"""

        from .diagnostics import export_memories

        return export_memories(self.store, scope.key() if scope else None)

    def reinforce(self, memory_ids: Iterable[int], *, now: Optional[float] = None) -> int:
        """强化一批记忆。带冷却期，避免正反馈刷分。"""

        ts = float(now if now is not None else time.time())
        cooldown = float(self.config.fade.recall_cooldown_seconds)
        count = 0
        for mem in self.store.get_many(list(memory_ids)):
            if mem.pinned:
                continue
            if mem.last_recalled_at is not None and ts - float(mem.last_recalled_at) < cooldown:
                continue
            updated = apply_event(mem, MemoryEvent.RECALL, now=ts, policy=self.policy)
            self.store.update(updated, fts_text(updated.content))
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # 管理
    # ------------------------------------------------------------------ #

    def pin(self, memory_id: int, *, now: Optional[float] = None) -> bool:
        """置顶：此后永不衰减。"""

        mem = self.store.get(memory_id)
        if mem is None:
            return False
        updated = apply_event(mem, MemoryEvent.PIN, now=float(now or time.time()), policy=self.policy)
        self.store.update(updated, fts_text(updated.content))
        return True

    def forget(self, memory_id: int, *, hard: bool = False, now: Optional[float] = None) -> bool:
        """遗忘。默认只是归档（原文保留可审计），hard=True 才物理删除。"""

        if hard:
            return self.store.delete(memory_id)
        mem = self.store.get(memory_id)
        if mem is None:
            return False
        updated = apply_event(mem, MemoryEvent.SUPERSEDE, now=float(now or time.time()), policy=self.policy)
        updated.status = MemoryStatus.ARCHIVED
        self.store.update(updated, fts_text(updated.content))
        return True

    async def consolidate(
        self, scope: Optional[Scope] = None, *, now: Optional[float] = None
    ) -> ConsolidationReport:
        """跑一轮整理。"""

        return await self.consolidator.run(scope.key() if scope else None, now=now)

    def stats(self, scope: Optional[Scope] = None) -> dict:
        """状态统计。"""

        scope_key = scope.key() if scope else None
        counts = self.store.status_counts(scope_key)
        return {
            "total": self.store.count(scope_key),
            "by_status": counts,
            "policy": {
                "recall_alpha": self.policy.recall_alpha,
                "reinforce_alpha": self.policy.reinforce_alpha,
                "weaken_alpha": self.policy.weaken_alpha,
            },
        }

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _embed(self, memory: Memory) -> None:
        """给记忆补向量。失败不影响入库。"""

        if self._embed_one is None:
            return
        vector = self._embed_sync(memory.content)
        if vector:
            self.store.set_embedding(memory.id, vector)

    def _embed_sync(self, content: str) -> Optional[list[float]]:
        """同步取向量。

        宿主多半是 async 的，但写入路径希望保持同步好用，
        所以这里在**没有运行中的事件循环**时直接算；有事件循环时
        跳过（交给异步路径补向量），绝不阻塞事件循环。
        """

        if self._embed_one is None:
            return None
        try:
            import asyncio

            asyncio.get_running_loop()
            return None
        except RuntimeError:
            pass
        try:
            return asyncio.run(self._embed_one(content))
        except Exception:
            return None

    async def embed_query(self, text: str) -> Optional[list[float]]:
        """异步取查询向量。适配器在召回前调用它。"""

        if self._embed_one is None:
            return None
        try:
            return await self._embed_one(text)
        except Exception:
            return None
    def effective_strength(self, memory: Memory, now: Optional[float] = None) -> float:
        """对外暴露：某条记忆此刻还剩多少。"""

        return effective_strength(memory, float(now if now is not None else time.time()), self.policy)

