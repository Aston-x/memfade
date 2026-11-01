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
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .config import MemFadeConfig
from .decay import DEFAULT_POLICY, FadePolicy, effective_strength, resolve_status
from .store import MemoryStore
from .textutil import estimate_tokens, token_set
from .types import Memory, MemoryStatus, RetrievalHit


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
            if query_vector:
                hits = self.store.search_vector(
                    query_vector, scope_key, limit=cfg.channel_candidates, kinds=kinds
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
        return scored[:limit]

    # ------------------------------------------------------------------ #
    # 打分
    # ------------------------------------------------------------------ #

    def _score(self, mem: Memory, channel_scores: dict[str, float], now: float) -> RetrievalHit:
        cfg = self.config.retrieval
        bm25 = channel_scores.get("bm25")
        vector = channel_scores.get("vector")

        # 相关度：两条通道都有就各占一半，只有一条就用那条。
        if bm25 is not None and vector is not None:
            relevance = 0.5 * bm25 + 0.5 * vector
        elif bm25 is not None:
            relevance = bm25
        elif vector is not None:
            relevance = vector
        else:
            relevance = 0.0

        strength_eff = effective_strength(mem, now, self.policy)
        strength_term = cfg.strength_floor + (1.0 - cfg.strength_floor) * strength_eff

        evidence_at = mem.last_recalled_at or mem.created_at or mem.anchor_at
        age_days = max(0.0, (now - float(evidence_at)) / 86400.0)
        recency = math.exp(-age_days / max(0.1, cfg.recency_tau_days))

        freq = min(1.0, math.log1p(max(0, mem.recall_count)) / math.log(11.0))

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
        )

    def _apply_association_boost(self, hits: list[RetrievalHit], boost: float) -> list[RetrievalHit]:
        """关联加成：与首条命中共享多个关键词的候选获得小幅加分。

        这是「激活扩散」的轻量版——不做图结构，只看内容词重合，
        代价几乎为零，但能捞出「说的是同一件事」的旁支记忆。
        """

        if not hits or boost <= 0:
            return hits
        top_tokens = token_set(hits[0].memory.content)
        if not top_tokens:
            return hits
        for hit in hits[1:]:
            shared = len(token_set(hit.memory.content) & top_tokens)
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
    ) -> MemoryContext:
        """把命中结果分层、限额、渲染成一段可注入的文本。

        ``exclude_ids`` 用来避免同一件事在「画像」和「记忆」两块里出现两次。
        """

        cfg = self.config.retrieval
        context = MemoryContext()
        used = 0
        fact_lines: list[str] = []
        echo_lines: list[str] = []
        skip = exclude_ids or set()

        for hit in hits:
            if hit.memory.id in skip:
                continue
            if hit.tier == "fact":
                line = self._render_fact(hit)
                cost = estimate_tokens(line)
                if used + cost > cfg.budget_tokens and (fact_lines or echo_lines):
                    break
                fact_lines.append(line)
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
            scored.append((score, mem))
        scored.sort(key=lambda pair: (pair[0], -len(pair[1].content), pair[1].created_at), reverse=True)

        lines: list[str] = []
        used_ids: list[int] = []
        used = 0
        # 画像是「这个人是谁」——名字、喜好、忌讳。
        # 几百字的百科段落不属于画像，它属于记忆块。
        max_len = self.config.retrieval.max_profile_chars * 2
        for _, mem in scored:
            if len(lines) >= max_items:
                break
            if len(mem.content) > max_len:
                continue
            body = _clip(mem.content.strip(), self.config.retrieval.max_profile_chars)
            if not body:
                continue
            line = f"- {body}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used_ids.append(mem.id)
            used += len(line)
        return ProfileBlock(text="\n".join(lines), ids=tuple(used_ids))


def _compose_fact(mem: Memory) -> str:
    """把 (subject, predicate, object) 拼成一句可读的话。"""

    if not (mem.subject and mem.predicate and mem.object):
        return ""
    return f"{mem.subject}{mem.predicate}{mem.object}"


def _clip(text: str, limit: int) -> str:
    """按字符截断，超长时补省略号。

    单条记忆可能有几百字（比如一段赛制说明），
    整段塞进 prompt 既费 token 又淹掉其它记忆。
    """

    if limit <= 0:
        return ""
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[: max(1, limit - 1)].rstrip("，,、；; ") + "…"

