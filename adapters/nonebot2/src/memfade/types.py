"""记忆引擎的核心数据类型。

设计要点：把「这条信息有多重要」（importance）与「它现在还剩多少」
（strength / status）拆成两条独立的轴。

多数同类实现把两者压进同一个字段，于是「重要」和「持久」互相污染：
一条被反复提起的重要事实会跟着时间一起掉分。memfade 坚持双轴：
importance 只由内容与用户强调决定，永不随时间衰减；
strength 由时间与回忆次数决定，只回答「现在还想不记得起来」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MemoryKind(str, Enum):
    """记忆种类。不同种类的默认稳定性（半衰期尺度）不同。"""

    EPISODIC = "episodic"
    """事件：发生过什么。天然半衰期短，是「提起旧事」的元凶。"""

    SEMANTIC = "semantic"
    """事实：稳定的世界/人物知识。半衰期长。"""

    PREFERENCE = "preference"
    """偏好：喜好、厌恶、雷点。"""

    RELATION = "relation"
    """关系：人和人之间的关系状态。"""

    SKILL = "skill"
    """做法：学会的、重复使用的行为模式。"""

    GIST = "gist"
    """残影：被淡化成一行摘要后的记忆形态。"""


class MemoryStatus(str, Enum):
    """记忆在「淡化梯度」上的位置。

    ACTIVE -> FADING -> GISTED -> ARCHIVED

    FADING 仍可召回，但注入时会标注「可能已经变了」；
    GISTED 只剩一行摘要，低权重出现；
    ARCHIVED 不再进入上下文，但原文保留在库里可审计。
    """

    ACTIVE = "active"
    FADING = "fading"
    GISTED = "gisted"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    """被更新的事实取代。保留原文做证据链，但不再召回。"""


# 会话隔离键的构造规则：platform:chat_type:chat_id
# 例：qq:group:1127155154 / qq:private:1738157278


@dataclass(frozen=True)
class Scope:
    """一条记忆属于哪个会话。

    ``cross_scope`` 是显式开洞：默认为 False，即记忆绝不跨会话可见。
    这一点是硬约束——宁可少记，也不能把私聊内容漏进群里。
    """

    platform: str = "qq"
    chat_type: str = "private"
    chat_id: str = "default"
    cross_scope: bool = False
    """显式允许跨会话共享。只有用户主动要求记住的东西才应置 True。"""

    def key(self) -> str:
        return f"{self.platform}:{self.chat_type}:{self.chat_id}"

    @staticmethod
    def from_key(key: str) -> "Scope":
        parts = str(key or "").split(":", 2)
        while len(parts) < 3:
            parts.append("")
        return Scope(platform=parts[0], chat_type=parts[1], chat_id=parts[2])


@dataclass
class MemoryDraft:
    """写入前的候选记忆。extractor 产出它，engine 决定它值不值得落库。"""

    content: str
    kind: MemoryKind = MemoryKind.EPISODIC
    importance: float = 0.5
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    cross_scope: bool = False
    """是否显式允许跨会话共享。抽取器只能对「用户主动要求记住」的内容置 True。"""


@dataclass
class Memory:
    """库里的一条记忆。"""

    id: int
    scope_key: str
    kind: MemoryKind
    content: str
    importance: float
    strength: float
    """锚点时刻的保留强度 ∈ [0, 1]。配合 anchor_at 惰性计算当前强度。"""

    anchor_at: float
    """强度锚点时间（写入或最后一次被想起的时刻）。"""

    status: MemoryStatus
    summary: str = ""
    """淡化成 gist 之后的一行摘要。"""

    created_at: float = 0.0
    updated_at: float = 0.0
    last_recalled_at: Optional[float] = None
    recall_count: int = 0
    pinned: bool = False
    """置顶：永不衰减、永不归档。"""

    cross_scope: bool = False
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source: str = ""
    superseded_by: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[bytes] = None

    # -- 便捷视图 -------------------------------------------------------

    @property
    def content_hash(self) -> str:
        from .textutil import normalized_hash

        return normalized_hash(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope_key": self.scope_key,
            "kind": self.kind.value,
            "content": self.content,
            "summary": self.summary,
            "importance": round(self.importance, 4),
            "strength": round(self.strength, 4),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_recalled_at": self.last_recalled_at,
            "recall_count": self.recall_count,
            "pinned": self.pinned,
            "cross_scope": self.cross_scope,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "superseded_by": self.superseded_by,
            "meta": dict(self.meta),
        }


@dataclass
class RetrievalHit:
    """一条召回结果。"""

    memory: Memory
    score: float
    relevance: float
    """与当前查询的文本相关度 ∈ [0, 1]。"""

    strength_eff: float
    """当前时刻的有效强度（strength 叠衰减）。"""

    tier: str = "fact"
    """注入层级：fact（事实层）/ echo（回声层）。"""

    channels: tuple[str, ...] = ()
    """命中通道：bm25 / vector / recent。用于调试与可解释性。"""


def now_ts() -> float:
    """当前墙钟时间戳（秒）。"""

    return time.time()

