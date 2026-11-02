"""混合检索与上下文组装。

## 评分

各分量都归一化到 [0,1] 后做**加权和**，不用纯乘法：

    score = w_rel · rel + w_imp · importance
          + w_str · (floor + (1 - floor) · strength_eff)
          + w_rec · recency + w_freq · freq

纯乘法（像 Engram 那样 relevance × importance × strength × recency）
有个致命问题：任何一项接近 0，总分就归零，一条高度相关但刚刚写入、
strength 尚在爬坡的记忆会被彻底埋掉。

强度项用了**有界形式**（congling 的经验）：
``floor + (1-floor)·strength``，保证强度低只会打折，不会清零。

## 分层注入

召回结果按淡化档位分两层：

* **事实层**：ACTIVE 档，当作「现在仍然成立」的知识注入；
* **回声层**：FADING / GISTED 档，只在被直接问起（相关度高）时才带出来，
  并且带「可能已经变了」的标记，数量上限 ``echo_max``。

这一层是「总是提起旧事」的解药：旧事不会消失，但也不会占据日常对话。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .config import MemFadeConfig
from .decay import DEFAULT_POLICY, FadePolicy, effective_strength, resolve_status
from .store import MemoryStore
from .textutil import (
    content_words,
    content_chars,
    estimate_tokens,
    fts_text,
    longest_common_substring,
    similarity,
)
from .types import Memory, MemoryKind, MemoryStatus, RetrievalHit


@dataclass
class MemoryContext:
    """一次召回 + 组装的完整结果。"""

    facts: list[RetrievalHit] = field(default_factory=list)
    echoes: list[RetrievalHit] = field(default_factory=list)
    text: str = ""
    token_estimate: int = 0

    @property
    def hits(self) -> list[RetrievalHit]:
        return self.facts + self.echoes

    def __bool__(self) -> bool:
        return bool(self.facts or self.echoes)


@dataclass
class ProfileBlock:
    """「关于对方」的稳定事实块，以及它包含的记忆 id。"""

    text: str = ""
    ids: tuple[int, ...] = ()


class MemoryRetriever:
    """召回 + 打分 + 组装。"""

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemFadeConfig] = None,
        policy: Optional[FadePolicy] = None,
    ) -> None:
        self.store = store
        self.config = config or MemFadeConfig()
        self.policy = policy or DEFAULT_POLICY

    # ------------------------------------------------------------------ #
    # 召回
    # ------------------------------------------------------------------ #

    def recall(
        self,
        query: str,
        scope_key: str,
        *,
        now: Optional[float] = None,
        top_k: Optional[int] = None,
        kinds: Optional[Sequence[str]] = None,
        exclude: Optional[set[int]] = None,
        query_vector: Optional[Sequence[float]] = None,
        require_match: bool = False,
    ) -> list[RetrievalHit]:
        """召回并打分。返回按分数降序排列的命中列表。"""

        ts = float(now if now is not None else time.time())
        cfg = self.config.retrieval
        limit = int(top_k or cfg.top_k)
        excluded = exclude or set()
        channels: dict[int, dict[str, Any]] = {}

        def collect(results, channel: str) -> None:
            for mid, score in results:
                entry = channels.setdefault(int(mid), {"scores": {}, "names": set()})
                entry["scores"][channel] = float(score)
                entry["names"].add(channel)

        text = (query or "").strip()
        if text:
            hits = self.store.search_fts(text, scope_key, limit=cfg.channel_candidates, kinds=kinds)
            collect(hits, "bm25")
            # 单字兜底通道。主通道一无所获、而且查询短到像关键词时，
            # 才允许用单字去捞——否则「的」「我」会把整库拖进来。
            uni_hits = self._uni_hits(text, scope_key, cfg, kinds)
            if not hits and uni_hits:
                collect(uni_hits, "uni")
            # 「被问起就想起来」：已归档的记忆平时不出现，
            # 但如果对方明确问到了，就把它捞回来并复活。
            revived = self._revive_archived(text, scope_key, kinds)
            for mid, score in revived:
                collect([(mid, score)], "revive")
            if query_vector:
                hits = self.store.search_vector(
                    query_vector,
                    scope_key,
                    limit=cfg.channel_candidates,
                    kinds=kinds,
                    min_gap=cfg.vector_min_gap,
                )
                collect(hits, "vector")

        # 最近写入的记忆始终作为候选：刚聊完的事一定要能召回。
        collect(self.store.recent(scope_key, limit=cfg.channel_candidates, kinds=kinds), "recent")

        if not channels:
            return []

        memories = self.store.get_many(list(channels.keys()))
        scored: list[RetrievalHit] = []
        for mem in memories:
            if mem.id in excluded:
                continue
            # 淡化档位在**读取时**现算，而不是只信库里那份状态。
            # 否则一条三周没被提起的事件，在整理任务跑过之前仍会以
            # 「事实」的身份被注入——淡化就等于没做。
            resolved = resolve_status(mem, ts, self.policy)
            if resolved in (MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED):
                continue
            hit = self._score(mem, channels[mem.id]["scores"], ts)
            hit.tier = "fact" if resolved is MemoryStatus.ACTIVE else "echo"
            if hit.score >= cfg.min_score:
                scored.append(hit)

        scored.sort(key=lambda h: (h.score, h.memory.importance, h.memory.created_at), reverse=True)
        scored = self._apply_association_boost(scored, cfg.association_boost)
        if require_match:
            # ⚠ 必须在 **top_k 截断之前**过滤：
            # 否则「最近写入但跟问题无关」的候选会占掉额度，
            # 真正命中的记忆反而被挤出去。
            kept = [h for h in scored if h.relevance >= cfg.min_relevance]
            rescued = self._rescue_recent(kept, scored, text, cfg)
            scored = kept + rescued
        return scored[:limit]

    def _rescue_recent(
        self,
        kept: list[RetrievalHit],
        scored: list[RetrievalHit],
        query: str,
        cfg: Any,
    ) -> list[RetrievalHit]:
        """「刚聊完的那件事」——词面搭不上时的兜底。

        为什么要它：中文聊天里「那部番叫什么」「他后来呢」这种**指代**
        满地都是，靠词面检索一个都命中不了；但答案往往就是**刚刚聊过的那条**。

        为什么要限死：
        「最近写入」通道是**无条件**参与候选的，没有这道闸门，
        每一轮都会把最近的几条塞进上下文——那就是「总提旧事」。
        所以只在**查询明显在指代**时才放一条进来，而且只放一条。
        """

        if not cfg.recent_rescue:
            return []
        if not query:
            return []
        if not any(marker in query for marker in _REFERENTIAL_MARKERS):
            return []
        for hit in scored:
            if hit in kept:
                continue
            # 只救「靠最近写入通道进来」的（相关度为 0），
            # 而且要真的是最近的那条。
            if hit.relevance > 0:
                continue
            # 「刚聊完」是有时限的：几小时前的事不算「刚才」。
            age_hours = (time.time() - float(hit.memory.created_at or 0)) / 3600.0
            if age_hours > float(cfg.recent_rescue_window_hours):
                continue
            return [hit]
        return []

    def _revive_archived(
        self,
        query: str,
        scope_key: str,
        kinds: Optional[Sequence[str]] = None,
    ) -> list[tuple[int, float]]:
        """把「被明确问起」的淡出记忆捞回来。

        归档（ARCHIVED）和成摘要（GISTED）都要救回来，理由不同：

        * 归档：平时根本不进上下文，问到了当然要捞回来；
        * 成摘要：它还在上下文里，但只剩下那一行摘要。
          **对方问的往往正是摘要丢掉的那部分细节**——
          「蓝桥杯报名截止到什么时候」，而摘要只剩「蓝桥杯报名」。
          这时候拿摘要去回答，等于没回答。

        淡出中（FADING）不用救：它本来就显示原文，只是带着
        「可能已经变了」的提示。
        """

        cfg = self.config.retrieval
        min_common = cfg.revive_min_common
        if min_common <= 0:
            return []
        candidates = self.store.search_fts(
            query, scope_key, limit=cfg.channel_candidates, kinds=kinds, include_archived=True
        )
        if not candidates:
            # 单字查询（「辣」）在多字词通道里找不到东西，
            # 但「被问起就想起来」不能因此失效。
            candidates = self.store.search_substring(
                content_chars(query),
                scope_key,
                limit=cfg.channel_candidates,
                kinds=kinds,
                include_archived=True,
            )
        out: list[tuple[int, float]] = []
        for mid, _score in candidates:
            mem = self.store.get(mid)
            if mem is None:
                continue
            if mem.status not in (MemoryStatus.ARCHIVED, MemoryStatus.GISTED):
                continue
            common = longest_common_substring(query, mem.content)
            if len(common) < min_common:
                continue
            # 相关度够高 = 对方确实在问这件事 → 复活
            from .decay import MemoryEvent, apply_event

            updated = apply_event(mem, MemoryEvent.RECALL, now=time.time(), policy=self.policy)
            updated.status = MemoryStatus.ACTIVE
            updated.anchor_at = time.time()
            self.store.update(updated, fts_text(updated.content))
            out.append((mid, 1.0))
        return out

    def _uni_hits(
        self,
        query: str,
        scope_key: str,
        cfg: Any,
        kinds: Optional[Sequence[str]] = None,
    ) -> list[tuple[int, float]]:
        """单字兜底通道的候选。

        三道闸门，缺一不可：

        1. 查询够短（默认 ≤2 个汉字）——「辣」「猫猫」是关键词，
           值得兜底；「我记得你之前说过喜欢吃辣」是整句，兜底只会
           把无关记忆拖进来。
        2. 命中数够少（默认 ≤20 条）——「我」「的」这种字到处都有，
           命中一屏说明这个字没有区分度。
        3. 主通道必须一无所获——词面已经命中时，多字词的结果永远更准。
        """

        from .textutil import content_chars

        # 只用实义字：「的」「了」「是」拿去检索等于跟整本语料库匹配。
        chars = content_chars(query)
        if not chars or len(chars) > int(cfg.uni_max_query_chars):
            return []
        limit = int(cfg.uni_max_hits) + 1
        hits = self.store.search_substring(chars, scope_key, limit=limit, kinds=kinds)
        if len(hits) > int(cfg.uni_max_hits):
            return []
        return hits

    # ------------------------------------------------------------------ #
    # 打分
    # ------------------------------------------------------------------ #

    def _score(self, mem: Memory, channel_scores: dict[str, float], now: float) -> RetrievalHit:
        cfg = self.config.retrieval
        bm25 = channel_scores.get("bm25")
        vector = channel_scores.get("vector")
        uni = channel_scores.get("uni")

        strength_eff = effective_strength(mem, now, self.policy)
        strength_term = cfg.strength_floor + (1.0 - cfg.strength_floor) * strength_eff
        evidence_at = mem.last_recalled_at or mem.created_at or mem.anchor_at
        age_days = max(0.0, (now - float(evidence_at)) / 86400.0)
        recency = math.exp(-age_days / max(0.1, cfg.recency_tau_days))
        freq = min(1.0, math.log1p(max(0, mem.recall_count)) / math.log(11.0))

        relevance = self._relevance(channel_scores, cfg)
        score = (
            cfg.w_relevance * relevance
            + cfg.w_importance * mem.importance
            + cfg.w_strength * strength_term
            + cfg.w_recency * recency
            + cfg.w_frequency * freq
        )

        tier = "fact" if mem.status is MemoryStatus.ACTIVE else "echo"
        return RetrievalHit(
            memory=mem,
            score=round(score, 6),
            relevance=round(relevance, 6),
            strength_eff=round(strength_eff, 6),
            tier=tier,
            channels=tuple(sorted(channel_scores.keys())),
            parts={
                "relevance": round(relevance, 4),
                "importance": round(mem.importance, 4),
                "strength": round(strength_term, 4),
                "recency": round(recency, 4),
                "frequency": round(freq, 4),
            },
        )

    @staticmethod
    def _relevance(channel_scores: dict[str, float], cfg: Any) -> float:
        """把各通道的相关度合成一个数。

        「相关度」只有一处定义——召回和「为什么召回」面板共用它，
        两处各写一遍公式，迟早会漂移。
        """

        bm25 = channel_scores.get("bm25")
        vector = channel_scores.get("vector")
        uni = channel_scores.get("uni")

        # 相关度：两条通道都有就各占一半，只有一条就用那条。
        if bm25 is not None and vector is not None:
            relevance = 0.5 * bm25 + 0.5 * vector
        elif bm25 is not None:
            relevance = bm25
        elif vector is not None:
            relevance = vector
        else:
            relevance = 0.0
        if uni is not None:
            # 单字命中比多字词命中弱得多，折价后再合并：
            # 还能被召回，但绝不会排在词面精确命中前面。
            relevance = max(relevance, uni * cfg.uni_discount)
        return relevance

    def _apply_association_boost(self, hits: list[RetrievalHit], boost: float) -> list[RetrievalHit]:
        """关联加成：与首条命中共享多个关键词的候选获得小幅加分。

        这是「激活扩散」的轻量版——不做图结构，只看内容词重合，
        代价几乎为零，但能捞出「说的是同一件事」的旁支记忆。
        """

        if not hits or boost <= 0:
            return hits
        # 用**实词**（滤掉虚词组合）算重合，而不是全部二元组：
        # 两条记忆共享「的食」「在的」不代表它们讲同一件事，
        # 共享「蓝桥杯」「季怡」才代表。
        top_tokens = set(content_words(hits[0].memory.content, limit=24))
        if not top_tokens:
            return hits
        for hit in hits[1:]:
            shared = len(set(content_words(hit.memory.content, limit=24)) & top_tokens)
            if shared >= 2:
                hit.score = round(hit.score + boost * min(1.0, shared / 6.0), 6)
        hits.sort(key=lambda h: (h.score, h.memory.importance, h.memory.created_at), reverse=True)
        return hits

    # ------------------------------------------------------------------ #
    # 组装
    # ------------------------------------------------------------------ #

    def assemble(
        self,
        hits: Sequence[RetrievalHit],
        *,
        header: str = "记忆",
        exclude_ids: Optional[set[int]] = None,
        require_match: bool = False,
    ) -> MemoryContext:
        """把命中结果分层、限额、渲染成一段可注入的文本。

        ``exclude_ids`` 用来避免同一件事在「画像」和「记忆」两块里出现两次。

        ``require_match`` 为真时，只有检索渠道真正命中的记忆才注入。
        仅在**有语义匹配能力**（向量通道可用）时才该开——
        没有向量时词面不重叠的近义记忆会被误杀：
        「今天吃什么呢」和「喜欢吃辣的东西」在词面上毫无交集。
        """

        cfg = self.config.retrieval
        context = MemoryContext()
        used = 0
        fact_lines: list[str] = []
        echo_lines: list[str] = []
        skip = exclude_ids or set()
        picked: list[str] = []

        for hit in self._diversify(hits):
            if hit.memory.id in skip:
                continue
            if hit.tier == "fact":
                line = self._render_fact(hit)
                cost = estimate_tokens(line)
                if used + cost > cfg.budget_tokens and (fact_lines or echo_lines):
                    break
                fact_lines.append(line)
                picked.append(hit.memory.content)
                used += cost
                context.facts.append(hit)
            else:
                if len(echo_lines) >= cfg.echo_max:
                    continue
                if hit.relevance < cfg.echo_min_relevance:
                    continue
                line = self._render_echo(hit)
                cost = estimate_tokens(line)
                if used + cost > cfg.budget_tokens and (fact_lines or echo_lines):
                    continue
                echo_lines.append(line)
                picked.append(hit.memory.content)
                used += cost
                context.echoes.append(hit)

        blocks: list[str] = []
        if fact_lines:
            blocks.append(f"[{header}]\n" + "\n".join(fact_lines))
        if echo_lines:
            blocks.append(f"[{header}·旧事]\n" + "\n".join(echo_lines))
        context.text = "\n\n".join(blocks)
        context.token_estimate = estimate_tokens(context.text) if context.text else 0
        return context

    def _diversify(self, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        """MMR 多样性重排。

        用户抱怨的「重复语段太多」有两种来源：库里存了两份（去重解决），
        以及**一次注入里塞了好几条讲同一件事的**（这里解决）。

        做法（MMR）：每选一条，就用它去压低后面候选的得分——
        越像已选内容的候选，排得越靠后。
        """

        lambda_ = self.config.retrieval.mmr_lambda
        remaining = list(hits)
        if lambda_ <= 0 or len(remaining) < 2:
            return remaining
        ordered: list[RetrievalHit] = []
        chosen: list[str] = []
        while remaining:
            best: Optional[RetrievalHit] = None
            best_value = float("-inf")
            for hit in remaining:
                redundancy = 0.0
                if chosen:
                    redundancy = max(similarity(hit.memory.content, text) for text in chosen)
                value = hit.score - lambda_ * redundancy
                if value > best_value:
                    best_value = value
                    best = hit
            if best is None:
                break
            ordered.append(best)
            chosen.append(best.memory.content)
            remaining.remove(best)
        return ordered

    def explain(
        self,
        query: str,
        scope_key: str,
        *,
        now: Optional[float] = None,
        top_k: int = 8,
        require_match: bool = False,
        query_vector: Optional[Sequence[float]] = None,
    ) -> list:
        """解释「为什么召回了这些记忆」——拆开每个打分分量。

        记忆系统最需要的就是这个：说不清为什么想起某件事，
        用户就不会信任它，出问题也查不动。
        """

        ts = float(now if now is not None else time.time())

        # ⚠ 走**真实的召回路径**，不要在这里另写一套：
        # 面板要解释的必须是「机器人当时真的召回了什么」，
        # 另写一套公式（还只用 bm25 + recent 两条通道）必然和实际对不上。
        hits = self.recall(
            query,
            scope_key,
            now=now,
            top_k=top_k,
            require_match=require_match,
            query_vector=query_vector,
        )

        out = []
        for hit in hits:
            mem = hit.memory
            evidence_at = mem.last_recalled_at or mem.created_at or mem.anchor_at
            age_days = max(0.0, (ts - float(evidence_at)) / 86400.0)
            parts = dict(hit.parts)
            out.append(
                {
                    "id": mem.id,
                    "content": mem.content,
                    "kind": mem.kind.value,
                    "status": mem.status.value,
                    "tier": hit.tier,
                    "score": round(hit.score, 4),
                    "parts": parts,
                    # 扁平一份，表格直接用（列名不允许带点）
                    "relevance": parts.get("relevance", 0.0),
                    "importance": parts.get("importance", 0.0),
                    "strength": parts.get("strength", 0.0),
                    "recency": parts.get("recency", 0.0),
                    "frequency": parts.get("frequency", 0.0),
                    "channels": sorted(hit.channels),
                    "age_days": round(age_days, 1),
                    "recall_count": mem.recall_count,
                }
            )
        return out

    def _render_fact(self, hit: RetrievalHit) -> str:
        mem = hit.memory
        body = mem.content.strip()
        composed = _compose_fact(mem)
        if composed:
            body = composed
        return f"- {_clip(body, self.config.retrieval.max_item_chars)}"

    def _render_echo(self, hit: RetrievalHit) -> str:
        mem = hit.memory
        body = mem.summary.strip() or mem.content.strip()
        return f"- （可能已经变了，仅供参考）{_clip(body, self.config.retrieval.max_item_chars)}"

    def build_profile(
        self,
        scope_key: str,
        *,
        now: Optional[float] = None,
        max_items: int = 8,
        max_chars: int = 400,
        kinds: Optional[Sequence[str]] = None,
    ) -> str:
        """构建「关于对方」的稳定事实块（只要文本）。"""

        return self.profile(
            scope_key,
            now=now,
            max_items=max_items,
            max_chars=max_chars,
            kinds=kinds,
        ).text

    def profile(
        self,
        scope_key: str,
        *,
        now: Optional[float] = None,
        max_items: int = 8,
        max_chars: int = 400,
        kinds: Optional[Sequence[str]] = None,
    ) -> ProfileBlock:
        """构建「关于对方」的稳定事实块。

        与检索召回是两回事：检索靠**这次问了什么**，
        而画像是「这个人是谁」——名字、偏好、约定、忌讳。
        这些东西**不该等被问到才想起**，所以常驻注入。

        只取 ACTIVE 档的稳定类记忆（事实/偏好/关系/做法），
        一次性事件不进画像。
        """

        ts = float(now if now is not None else time.time())
        stable_kinds = kinds or ("semantic", "preference", "relation", "skill")
        memories = self.store.iter_scope(
            scope_key,
            statuses=("active",),
            kinds=list(stable_kinds),
            limit=200,
        )
        scored: list[tuple[float, Memory]] = []
        for mem in memories:
            if mem.pinned:
                score = 10.0
            else:
                eff = effective_strength(mem, ts, self.policy)
                score = mem.importance * (0.4 + 0.6 * eff) + 0.05 * mem.recall_count
                # 画像应该是「这个人是谁」——名字、喜好、忌讳、关系。
                # 偏好/关系/做法比泛泛的事实更该常驻。
                score *= _PROFILE_KIND_WEIGHT.get(mem.kind, 1.0)
                # 同为「事实」但带日期的，是日程而不是画像——
                # 「2026年11月1日这周更新里有《牧神记5》」过两周就过期了。
                if _has_time_marker(mem.content):
                    score *= 0.35
                # 「与…无关」「并不是…」这类是**消歧**，
                # 说的是某件事**不成立**，不是「这个人是谁」。
                # 画像是常驻的，塞这种句子只会占位置。
                if _is_disambiguation(mem.content):
                    score *= 0.35
            scored.append((score, mem))
        scored.sort(key=lambda pair: (pair[0], -len(pair[1].content), pair[1].created_at), reverse=True)

        lines: list[str] = []
        used_ids: list[int] = []
        used = 0
        # 画像是「这个人是谁」——名字、喜好、忌讳。
        # 几百字的百科段落不属于画像，它属于记忆块。
        max_len = self.config.retrieval.max_profile_chars * 2
        # 同一件事的多个说法不各占一行：
        # 实机画像里出现过三条《Re:从零开始的异世界生活》第四季，
        # 挤掉了「喜欢什么」「讨厌什么」这类真正该常驻的信息。
        picked: list[str] = []
        for _, mem in scored:
            if len(lines) >= max_items:
                break
            if len(mem.content) > max_len:
                continue
            if _same_topic(mem.content, picked):
                continue
            body = _clip(mem.content.strip(), self.config.retrieval.max_profile_chars)
            if not body:
                continue
            line = f"- {body}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used_ids.append(mem.id)
            picked.append(body)
            used += len(line)
        return ProfileBlock(text="\n".join(lines), ids=tuple(used_ids))


_PROFILE_KIND_WEIGHT = {
    MemoryKind.PREFERENCE: 1.20,
    MemoryKind.RELATION: 1.15,
    MemoryKind.SKILL: 1.10,
    MemoryKind.SEMANTIC: 1.00,
}


_REFERENTIAL_MARKERS = (
    "那个",
    "这个",
    "那部",
    "这部",
    "那件",
    "这件",
    "刚才",
    "刚刚",
    "上次",
    "还记得",
    "想起来",
    "后来",
    "然后呢",
)
"""指代词——查询里有这些，才可能是在问「刚聊过的那件事」。

为什么要有这张表：「最近写入」通道是**无条件**参与候选的，
没有这道闸门，每一轮都会把最近几条塞进上下文——那就是「总提旧事」。
但中文里「那部番叫什么」这种指代满地都是，
词面检索一个都命中不了，答案往往就是刚刚聊过的那条。
所以：**只在明显在指代时，放一条进来，而且只放一条。**

⚠ **不放「他/她/它」**：这些字在中文里几乎每句都有
（「她喜欢吃什么」也会命中），放进来等于没有闸门。
"""
"""画像里各类记忆的权重。

画像是「这个人是谁」：喜好、关系、做法比泛泛的事实更该常驻。

⚠ 早先这里用的是「稳定性（天数量级）」当代理，但那个值是
``base × (1 + gain × importance)``——importance 一高就顶到上限，
所有种类都变成同一个系数，等于没加权。
"""


def _same_topic(content: str, picked: Sequence[str]) -> bool:
    """这条记忆是不是跟已经选中的某条讲同一件事。

    两种判据一起用：

    * Dice 相似度 ≥0.45——整体说的是一回事（换个说法）；
    * 最长公共子串 ≥5 字——同一个专有名词或长句
      （「《Re:从零开始的异世界生活》第四季」）。

    两条都在 5 字以下（「许杏玖喜欢辣的食物」vs「许杏玖讨厌香菜」）
    就不算同一件事——那是两件不同的事，都该留着。
    """

    text = (content or "").strip()
    if not text:
        return True
    for other in picked:
        if similarity(text, other) >= 0.45:
            return True
        if len(longest_common_substring(text, other)) >= 5:
            return True
    return False


def _compose_fact(mem: Memory) -> str:
    """把 (subject, predicate, object) 拼成一句可读的话。

    ⚠ **只在原文缺主语时才拼**。
    原文已经带主语的话，再拼一遍会重复：
    「第18届蓝桥杯」+「统一口径」+「官方口径」
    → 「第18届蓝桥杯统一口径官方口径」（原文明明就是「第18届蓝桥杯官方口径」）。

    反过来，原文只有宾语的时候（内容写成「辣的食物」），
    拼上主语才成为一句完整的话——那才是三元组该干的活。
    """

    if not (mem.subject and mem.predicate and mem.object):
        return ""
    if mem.subject in mem.content:
        return ""
    # 谓语的尾巴已经在主语里出现过就别再说一遍：
    # 「软件赛与电子赛报名」+「按学校报名」→「软件赛与电子赛报名按学校报名」。
    tail = mem.predicate[-2:]
    if len(tail) >= 2 and tail in mem.subject:
        return f"{mem.subject}{mem.object}"
    return f"{mem.subject}{mem.predicate}{mem.object}"


_TIME_MARKER_RE = re.compile(
    r"\d{4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"今年|去年|本月|本月|上周|本周|这周|下周|周[一二三四五六日天]|"
    r"今天|明天|昨天|前天|后天|最近|近期|截止"
)


def _has_time_marker(text: str) -> bool:
    """判断一条记忆是不是带时效性的（日程、播出表、截止日期）。"""

    return bool(_TIME_MARKER_RE.search(text or ""))


_DISAMBIGUATION_MARKERS = ("无关", "不涉及", "并非", "并不是", "而不是", "不作为", "算不上")
"""消歧句的标志。

⚠ 不要用单独的「不」——「季怡不吃香菜」是**忌讳**，
那是画像里最该常驻的一条。「不」只用来筛「某件事不成立」的说法。
"""


def _is_disambiguation(text: str) -> bool:
    """判断一条记忆是不是「消歧」而不是「这个人是谁」。"""

    body = str(text or "")
    return any(marker in body for marker in _DISAMBIGUATION_MARKERS)


def _clip(text: str, limit: int) -> str:
    """按字符截断，超长时补省略号。

    单条记忆可能有几百字（比如一段赛制说明），
    整段塞进 prompt 既费 token 又淹掉其它记忆。

    截断点尽量落在标点上——断在半句里读起来像坏掉的。
    """

    if limit <= 0:
        return ""
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    head = body[: max(1, limit - 1)]
    # 回退到最后一个标点，句子才完整；找不到就硬截
    for mark in ("。", "；", "!", "?", "!", "，", ",", "、", " "):
        index = head.rfind(mark)
        if index >= limit // 3:
            return head[: index + 1].rstrip() + "…"
    return head.rstrip("，,、；; ") + "…"

