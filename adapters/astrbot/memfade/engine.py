"""记忆引擎门面。

适配器只需要认识这一个类。所有跨框架的差异（怎么拿消息、怎么注入 prompt、
怎么调 LLM）都留在适配器里，引擎只管记忆本身。

## 强化（reinforce）

记忆被**真正注入上下文**之后才算「被想起」，这时才做残留学习式强化。
不在召回阶段强化——否则「一被检索就变强、变强更容易被检索」的正反馈
会把少数几条记忆刷成霸榜。冷却期进一步压制这个效应。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any, Iterable, Optional, Sequence

from .config import MemFadeConfig
from .consolidate import ConsolidationReport, ConsolidationService
from .decay import (
    DEFAULT_POLICY,
    FadePolicy,
    MemoryEvent,
    apply_event,
    effective_strength,
    resolve_status,
)
from .dedup import is_near_duplicate
from .embedding import EmbeddingClient, make_embedder
from .extract import MemoryExtractor, _clamp_importance


def _safe_float(value, default: float, lo: float = float("-inf"), hi: float = float("inf")) -> float:
    """安全地转 float，None 回退到 default，合法 0.0 不被吞掉。

    Python 里 ``item.get("strength") or 1.0`` 会把合法的 0.0 当成 falsy
    直接变成 1.0——这是数据损坏，不是默认值填充。
    """
    if value is None:
        return max(lo, min(hi, float(default)))
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return max(lo, min(hi, float(default)))

from .llm import LLMClient
from .retrieve import MemoryContext, MemoryRetriever
from .store import MemoryStore
from .textutil import fts_text, normalized_hash
from .types import Memory, MemoryDraft, MemoryKind, MemoryStatus, RetrievalHit, Scope


# 稳定记忆自动保护——写入时自动 pinned。
# 来源：图片方案要求「人和约永不自动删除」。在 memfade 里对应的是
# 偏好（preference = 约）和关系（relation = 人）。
_PROTECTED_KINDS = frozenset({MemoryKind.PREFERENCE, MemoryKind.RELATION})


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
        self.last_dropped: list[tuple[str, str]] = []
        """上一次抽取里被「主语纪律」丢掉的候选：(内容, 原因)。"""

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
            # ⚠ 种类区间在这里夹一次——这是**存储形态**的约束：
            # 事件不该压过稳定事实（「上周吃了炸鸡」不能比「他叫BotName」还重要）。
            # 门槛判定用的是模型自己给的分数（见 remember 开头），
            # 别在这里夹完再拿去比门槛：semantic 的下限 0.40
            # 会把模型给的 0.1 抬过旧门槛 0.35，等于给垃圾开后门。
            importance=_clamp_importance(draft.kind, draft.importance),
            strength=1.0,
            anchor_at=ts,
            status=MemoryStatus.ACTIVE,
            created_at=ts,
            updated_at=ts,
            pinned=draft.kind in _PROTECTED_KINDS,
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
        # 被「主语纪律」丢掉的候选：留着给宿主打日志，
        # 不然「为什么这条没记下来」永远是黑箱。
        self.last_dropped = list(result.dropped)
        if not result.drafts:
            return []
        # 记下出处：模型不会填 source（提示词里没这项），
        # 不补上的话这条记忆就是「不知道从哪来的」。
        if source:
            for draft in result.drafts:
                if not draft.source:
                    draft.source = source
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

        if kind == MemoryKind.EPISODIC:
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
            # ⚠ 默认**总是**要求「检索真的命中」。
            # 以前这里是「有向量能力才要求」——于是没配嵌入服务的部署
            # 完全没有相关度闸门，每一轮都把最近几条塞进上下文
            # （就是「总提旧事」）。
            # 「刚聊完的那件事」由 retrieve._rescue_recent 兜底：
            # 只在查询明显在指代时，放**一条**进来。
            require_match = True
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
            # 与 recall 同一口径：默认总是要求真命中。
            require_match = True
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

    def build_context_budgeted(
        self,
        query: str,
        scope: Scope,
        *,
        now: Optional[float] = None,
        total_budget: int = 4096,
        profile_budget: int = 2048,
        event_budget: int = 2048,
        reinforce: bool = True,
        query_vector: Optional[Sequence[float]] = None,
    ) -> MemoryContext:
        """按 UTF-8 字节预算组装上下文——画像池和事项池分开。

        来源：图片方案要求 L1 总预算 4096 字节，其中人/约占 2048、
        当前事项占 2048，两池互不侵占。人/约超容时只报不删。

        与 ``build_context`` 的区别：后者按 top_k 条数裁剪，
        这里按**实际字节数**裁剪，且保护类记忆（画像）有独立预算。
        """

        from .textutil import utf8_bytes

        profile_text = self.build_profile(
            scope, now=now, max_items=12, max_chars=profile_budget // 3
        )
        profile_bytes = utf8_bytes(profile_text)
        if profile_bytes > profile_budget:
            # 画像超容：只记 warning，不自动删保护记忆
            profile_text = profile_text[:profile_budget // 3]
            profile_bytes = utf8_bytes(profile_text)

        context = self.build_context(
            query,
            scope,
            now=now,
            top_k=16,
            reinforce=reinforce,
            query_vector=query_vector,
        )

        # 按字节裁剪事项部分
        event_parts: list[str] = []
        event_bytes = 0
        for hit in context.hits:
            line = f"- {hit.memory.content}"
            line_bytes = utf8_bytes(line) + 1  # +1 for newline
            if event_bytes + line_bytes > event_budget:
                break
            event_parts.append(line)
            event_bytes += line_bytes

        # 组装最终文本
        parts = []
        if profile_text.strip():
            parts.append(profile_text.strip())
        if event_parts:
            parts.append("[相关记忆]")
            parts.extend(event_parts)
        final_text = "\n".join(parts)

        from .retrieve import MemoryContext
        return MemoryContext(text=final_text, token_estimate=len(final_text) // 2)

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
            # 与 recall 同一口径：默认总是要求真命中，
            # 否则面板解释的召回结果和机器人实际注入的会对不上。
            require_match = True
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

        return analyze(
            self.store,
            scope.key() if scope else None,
            self_names=self.config.extraction.self_names,
        )

    def export(self, scope: Optional[Scope] = None) -> list:
        """导出记忆（JSON 友好）。"""

        from .diagnostics import export_memories

        return export_memories(self.store, scope.key() if scope else None)

    def clear_scope(self, scope: Scope) -> int:
        """清空某个作用域的记忆，返回清掉多少条。

        为什么放在核心而不是适配器：清空是**产品动作**
        （用户说「忘掉我们聊过的一切」），每个适配器都要有，
        所以核心先做好，适配器只负责把它暴露成命令或按钮。
        """

        return self.store.clear_scope(scope.key())

    def import_memories(
        self,
        items: Sequence[dict[str, Any]],
        scope: Optional[Scope] = None,
        *,
        now: Optional[float] = None,
        skip_duplicates: bool = True,
    ) -> list[Memory]:
        """导入一批记忆（导出的 JSON、别的机器上的库、旧系统的迁移结果）。

        为什么插件要自带导入：
        **记忆是用户的数据**，换机器、换机器人、换框架都得能带走。
        只有导出没有导入，等于「只能看不能搬」。

        导入时保留：种类、显著性、三元组、强度、锚点、状态、来源。
        （强度与锚点是淡化状态——不保留的话，导入进来的记忆会
        「重新变年轻」，该淡的又淡一遍。）
        不保留 id（本地重新分配）。
        """

        from .textutil import fts_text, normalized_hash

        ts = float(now if now is not None else time.time())
        written: list[Memory] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            target_scope = scope
            if target_scope is None:
                raw_scope = str(item.get("scope_key") or "").strip()
                if not raw_scope:
                    continue
                # ★ 用 Scope.from_key 还原，不要手写拆键：手写版本把 platform
                #   写死成 "qq"、chat_id 只取最后一段，别的宿主（astrbot 等）
                #   导出再导入后作用域会整个错位。
                target_scope = Scope.from_key(raw_scope)
                if item.get("cross_scope"):
                    target_scope = Scope(
                        platform=target_scope.platform,
                        chat_type=target_scope.chat_type,
                        chat_id=target_scope.chat_id,
                        cross_scope=True,
                    )
            try:
                kind = MemoryKind(str(item.get("kind") or "semantic").lower())
            except ValueError:
                kind = MemoryKind.SEMANTIC
            try:
                status = MemoryStatus(str(item.get("status") or "active").lower())
            except ValueError:
                status = MemoryStatus.ACTIVE
            if skip_duplicates and self.store.find_by_hash(
                normalized_hash(content), target_scope.key()
            ):
                continue
            _raw_created = item.get("created_at")
            created = float(_raw_created) if _raw_created is not None else ts
            _raw_anchor = item.get("anchor_at")
            anchor = float(_raw_anchor) if _raw_anchor is not None else created
            memory = Memory(
                id=0,
                scope_key=target_scope.key(),
                kind=kind,
                content=content,
                summary=str(item.get("summary") or ""),
                # 导入不走门槛（那是「抽取」的门槛），但种类区间照样要夹：
                # 否则一条 0.95 的事件记忆能压过所有稳定事实。
                importance=_clamp_importance(kind, _safe_float(item.get("importance"), 0.5, lo=0.0, hi=1.0)),
                strength=_safe_float(item.get("strength"), 1.0, lo=0.0, hi=1.0),
                anchor_at=anchor,
                status=status,
                created_at=created,
                updated_at=_safe_float(item.get("updated_at"), created),
                last_recalled_at=item.get("last_recalled_at"),
                recall_count=int(item.get("recall_count") or 0),
                pinned=bool(item.get("pinned", False)) or kind in _PROTECTED_KINDS,
                cross_scope=bool(item.get("cross_scope", False)),
                subject=str(item.get("subject") or ""),
                predicate=str(item.get("predicate") or ""),
                object=str(item.get("object") or ""),
                source=str(item.get("source") or "import"),
                meta=dict(item.get("meta") or {}),
            )
            memory.id = self.store.add(memory, fts_tokens=fts_text(memory.content))
            written.append(memory)
        return written

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

    def pin(self, memory_id: int, *, pinned: bool = True, now: Optional[float] = None) -> bool:
        """置顶 / 取消置顶。

        置顶的记忆此后永不衰减，也不会被淘汰。
        """

        mem = self.store.get(memory_id)
        if mem is None:
            return False
        ts = float(now or time.time())
        if pinned:
            updated = apply_event(mem, MemoryEvent.PIN, now=ts, policy=self.policy)
        else:
            # 取消置顶：回到普通衰减（档位重新按强度算）。
            updated = replace(mem, pinned=False, updated_at=ts)
            updated.status = resolve_status(updated, ts, self.policy)
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

    def edit_memory(
        self,
        memory_id: int,
        *,
        content: Optional[str] = None,
        kind: Any = None,
        importance: Optional[float] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        quote: Optional[str] = None,
        meta_patch: Optional[dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> Optional[Memory]:
        """改一条已经入库的记忆。

        为什么需要它：抽取难免出错（把「今天」写成「每天」、
        把一次性的吩咐写成长期偏好）。出错之后必须有个**补救口**——
        主人自己改一条，比整库清空的代价小得多。

        改完会重算内容哈希、重建全文索引，并**清掉向量**——
        内容变了旧向量就对不上了，留着会让检索悄悄失准；
        置空之后由补向量流程重算。
        """

        mem = self.store.get(int(memory_id))
        if mem is None:
            return None
        ts = float(now if now is not None else time.time())
        if content is not None:
            text = str(content).strip()
            if not text:
                return None
            mem.content = text
            # 摘要与向量都是按旧内容算的，改完就过期了。
            mem.summary = ""
            mem.embedding = None
        if kind is not None:
            mem.kind = MemoryKind(str(kind)) if not isinstance(kind, MemoryKind) else kind
        if importance is not None:
            mem.importance = _clamp_importance(mem.kind, float(importance))
        if subject is not None:
            mem.subject = str(subject).strip()
        if predicate is not None:
            mem.predicate = str(predicate).strip()
        if object is not None:
            mem.object = str(object).strip()
        if quote is not None:
            mem.meta = {**dict(mem.meta or {}), "quote": str(quote).strip()}
        if meta_patch:
            mem.meta = {**dict(mem.meta or {}), **(meta_patch or {})}
        mem.updated_at = ts
        self.store.update(mem, fts_text(mem.content))
        return mem

    async def consolidate(
        self, scope: Optional[Scope] = None, *, now: Optional[float] = None
    ) -> ConsolidationReport:
        """跑一轮整理。"""

        return await self.consolidator.run(scope.key() if scope else None, now=now)

    async def repair(
        self,
        scope: Scope,
        old: str,
        new: str,
        *,
        top_k: int = 5,
        now: Optional[float] = None,
    ) -> Optional[Memory]:
        """把一条记错的记忆改成新的内容。

        为什么要有这个动作：抽取难免出错（把「今天」写成「每天」、
        把一次性的吩咐写成长期偏好）。出错之后总得有人收拾——
        主人自己改一条，比整库清空的代价小得多。

        怎么找到「那条」：按内容片段检索，**必须真命中**
        （不掺「最近写入但无关的记忆」，否则改错对象更糟）。
        旧内容留在 ``meta["previous_content"]`` 里，事后可查。
        """

        old_text = (old or "").strip()
        new_text = (new or "").strip()
        if not old_text or not new_text:
            return None
        vector = await self.embed_query(old_text)
        hits = self.recall(
            old_text, scope, top_k=max(1, int(top_k)), query_vector=vector, require_match=True
        )
        if not hits:
            return None
        target = hits[0].memory
        updated = self.edit_memory(
            target.id,
            content=new_text,
            meta_patch={"previous_content": target.content},
            now=now,
        )
        if updated is None:
            return None
        # 内容变了，旧向量就对不上了——重算。
        await self.embed_memories([updated])
        return updated

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

    def health_check(self, scope: Optional[Scope] = None) -> dict:
        """轻量健康检查——一眼看全局，不读正文。

        来源：图片方案的「四眼检查」——只看文件名和字节数，不开正文。
        这里只读 store 的统计接口，不触发任何写入。
        """

        scope_key = scope.key() if scope else None
        total = self.store.count(scope_key)
        counts = self.store.status_counts(scope_key)

        # 统计 protected kinds
        protected = 0
        if scope_key:
            for kind in _PROTECTED_KINDS:
                rows = self.store.iter_owned_scope(
                    scope_key,
                    statuses=("active", "fading"),
                    kinds=(kind.value,),
                    limit=10000,
                )
                protected += len(rows)

        return {
            "total": total,
            "by_status": counts,
            "protected_count": protected,
            "healthy": total >= 0,  # always true if we got here
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

    async def embed_query(self, text: str, *, timeout: Optional[float] = 2.5) -> Optional[list[float]]:
        """异步取查询向量。适配器在召回前调用它。

        ⚠ **默认只给 2.5 秒**。取向量是网络调用（宿主那边可能是
        百炼的 API），而召回在**热路径**上——注入钩子 await 它，
        服务一慢，机器人的回复就跟着慢（实测 embedding 服务挂掉时
        一次调用能卡 60 秒）。

        超时就当没有向量：词面通道照样能用，少一路混合召回而已。
        """

        if self._embed_one is None:
            return None
        try:
            if timeout and timeout > 0:
                return await asyncio.wait_for(self._embed_one(text), timeout=timeout)
            return await self._embed_one(text)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
    def effective_strength(self, memory: Memory, now: Optional[float] = None) -> float:
        """对外暴露：某条记忆此刻还剩多少。"""

        return effective_strength(memory, float(now if now is not None else time.time()), self.policy)


