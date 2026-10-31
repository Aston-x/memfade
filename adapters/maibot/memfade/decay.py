"""记忆淡化引擎。

这个模块回答一个问题：**一条记忆在什么时候、以什么方式被忘掉。**

## 双轴模型

memfade 把「重要」和「持久」拆成两条互不干扰的轴：

* ``importance``（显著性）：这条内容本身有多重要。只在写入时由抽取器判定，
  之后**永不随时间变化**。它决定检索时的权重，以及这条记忆能活多久。
* ``strength``（强度）：这条记忆现在还剩多少。随「距上次被想起的时间」衰减，
  每次被想起就回升。回答的是「此刻还想不想得起来」。

大多数同类实现把两者压进一个字段，于是重要事实会跟着时间一起掉分——
这是「总是提起旧事」与「重要的事却记不住」同时出现的根因。

## 衰减公式

    stability = base_stability(kind)
                × (1 + importance_gain × importance)      重要 → 活得久
                × min(ceiling, growth ^ recall_count)      复习 → 更稳（间隔重复）

    effective_strength(now) = strength × exp(-Δdays / stability)

用的是艾宾浩斯指数曲线（congling 的实现同样采用这一形式）。
关键工程细节：**始终按「锚点时刻」重算，而不是每天拿旧值乘一次系数。**
这样维护任务跑一次、跑两次、隔一天再跑，结果完全一致（幂等），
也不会因为浮点累积误差跑偏。

## 淡化梯度

    ACTIVE  ──eff < fade_enter──▶  FADING  ──eff < gist_threshold──▶  GISTED
                                                                        │
                                          ARCHIVED  ◀──eff < archive_threshold

* ACTIVE：正常召回，作为「事实层」注入。
* FADING：仍可召回，但降权并标注「可能已经变了」，只进「回声层」。
* GISTED：原文压缩成一行摘要，低权重出现，细节已不可考。
* ARCHIVED：不再进入上下文，原文仍留在库里可审计。

各档之间有迟滞带（fade_enter / fade_exit），避免在阈值附近反复横跳。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

from .types import Memory, MemoryKind, MemoryStatus


class MemoryEvent(str, Enum):
    """会改变记忆状态的事件。"""

    RECALL = "recall"
    """被召回 / 被提起：残留学习式强化，重置衰减时钟。"""

    REINFORCE = "reinforce"
    """有新证据确认：强度拉满。"""

    WEAKEN = "weaken"
    """被削弱：出现矛盾但尚未定论。"""

    SUPERSEDE = "supersede"
    """被更新的事实取代。"""

    PIN = "pin"
    """置顶：此后永不衰减。"""


@dataclass(frozen=True)
class KindProfile:
    """某一类记忆的衰减画像。"""

    base_stability_days: float
    """稳定性基数（天）。半衰期尺度的起点。"""

    importance_gain: float
    """重要性对稳定性的加成系数：stability × (1 + gain × importance)。"""

    stability_growth: float
    """每被想起一次，稳定性乘以多少（间隔重复效应）。"""

    stability_ceiling: float
    """稳定性倍率上限，防止复述次数把记忆钉死。"""

    fade_enter: float
    """低于此强度开始淡出（active → fading）。"""

    fade_exit: float
    """回升到此强度才恢复 active（迟滞带）。"""

    gist_threshold: float
    """低于此强度压缩成 gist（fading → gisted）。"""

    archive_threshold: float
    """低于此强度归档（gisted → archived）。"""

    retrievable: bool = True
    """归档后是否仍可召回。默认 False——归档的意义就是不进上下文。"""


DEFAULT_PROFILES: dict[MemoryKind, KindProfile] = {
    MemoryKind.EPISODIC: KindProfile(
        base_stability_days=3.0,
        importance_gain=6.0,
        stability_growth=1.6,
        stability_ceiling=8.0,
        fade_enter=0.55,
        fade_exit=0.35,
        gist_threshold=0.15,
        archive_threshold=0.04,
    ),
    MemoryKind.SEMANTIC: KindProfile(
        base_stability_days=30.0,
        importance_gain=10.0,
        stability_growth=1.8,
        stability_ceiling=12.0,
        fade_enter=0.50,
        fade_exit=0.32,
        gist_threshold=0.12,
        archive_threshold=0.03,
    ),
    MemoryKind.PREFERENCE: KindProfile(
        base_stability_days=20.0,
        importance_gain=9.0,
        stability_growth=1.7,
        stability_ceiling=10.0,
        fade_enter=0.52,
        fade_exit=0.33,
        gist_threshold=0.13,
        archive_threshold=0.03,
    ),
    MemoryKind.RELATION: KindProfile(
        base_stability_days=14.0,
        importance_gain=8.0,
        stability_growth=1.6,
        stability_ceiling=9.0,
        fade_enter=0.52,
        fade_exit=0.33,
        gist_threshold=0.14,
        archive_threshold=0.04,
    ),
    MemoryKind.SKILL: KindProfile(
        base_stability_days=21.0,
        importance_gain=8.0,
        stability_growth=1.7,
        stability_ceiling=10.0,
        fade_enter=0.52,
        fade_exit=0.33,
        gist_threshold=0.13,
        archive_threshold=0.03,
    ),
    MemoryKind.GIST: KindProfile(
        base_stability_days=365.0,
        importance_gain=2.0,
        stability_growth=1.2,
        stability_ceiling=3.0,
        fade_enter=0.30,
        fade_exit=0.20,
        gist_threshold=0.08,
        archive_threshold=0.02,
    ),
}


@dataclass(frozen=True)
class FadePolicy:
    """全局淡化策略参数。"""

    profiles: dict[MemoryKind, KindProfile] = field(default_factory=lambda: dict(DEFAULT_PROFILES))

    recall_alpha: float = 0.35
    """被想起时的残留学习强度：新强度 = eff + α(1 - eff)。"""

    reinforce_alpha: float = 0.8
    """有新证据时的回升幅度。"""

    weaken_alpha: float = 0.5
    """出现矛盾时的削弱幅度：新强度 = eff × (1 - α)。"""

    min_stability_days: float = 0.25
    """稳定性下限，防止除零导致瞬间归零。"""

    max_stability_days: float = 3650.0
    """稳定性上限（10 年），防止溢出。"""

    def profile(self, kind: MemoryKind) -> KindProfile:
        return self.profiles.get(kind, DEFAULT_PROFILES.get(kind, DEFAULT_PROFILES[MemoryKind.EPISODIC]))


DEFAULT_POLICY = FadePolicy()


def clamp01(value: float) -> float:
    """限制到 [0, 1]。"""

    number = float(value)
    if math.isnan(number):
        return 0.0
    return min(1.0, max(0.0, number))


def stability_days(memory: Memory, policy: FadePolicy = DEFAULT_POLICY) -> float:
    """计算这条记忆的稳定性（天）。

    三个因子相乘：
    1. 种类基数——事件天生比事实易忘；
    2. 重要性——越重要越经得起时间；
    3. 复习次数——被反复想起的记忆越来越稳（间隔重复）。
    """

    profile = policy.profile(memory.kind)
    base = max(policy.min_stability_days, float(profile.base_stability_days))
    importance_factor = 1.0 + profile.importance_gain * clamp01(memory.importance)
    growth = max(0.0, float(profile.stability_growth)) ** max(0, int(memory.recall_count))
    growth_factor = min(float(profile.stability_ceiling), max(1.0, growth))
    stability = base * importance_factor * growth_factor
    return max(policy.min_stability_days, min(policy.max_stability_days, stability))


def effective_strength(memory: Memory, now: float, policy: FadePolicy = DEFAULT_POLICY) -> float:
    """当前时刻的有效强度。

    幂等的关键：永远从「锚点时刻的强度」按时间差重算，
    而不是逐日乘系数。因此本函数可以随时任意次调用，结果一致。
    """

    if memory.pinned:
        return 1.0
    # 注意不能用 `or`：时间戳 0.0 是合法值（Unix 纪元），一旦被当成「空」
    # 就会把锚点推到 now，记忆从此永不衰减。
    anchor = memory.anchor_at
    if anchor is None:
        anchor = memory.created_at
    if anchor is None:
        anchor = now
    anchor = float(anchor)
    elapsed_days = max(0.0, (float(now) - anchor) / 86400.0)
    tau = stability_days(memory, policy)
    try:
        return clamp01(memory.strength * math.exp(-elapsed_days / tau))
    except OverflowError:  # pragma: no cover - tau 已做上下限，理论不会溢出
        return 0.0


def resolve_status(memory: Memory, now: float, policy: FadePolicy = DEFAULT_POLICY) -> MemoryStatus:
    """按有效强度与迟滞带解出记忆当前所处的淡化档位。

    规则：

    * 一次可以**跨多档**下降——一条两周没被提起的一次性事件，
      不该在「淡出」上再卡两周才轮到「成 gist」；
    * 迟滞是标准的施密特触发器：跌破 ``fade_exit`` 才降级，
      升过 ``fade_enter`` 才恢复，两条线之间那段是「粘住」的，
      记忆停在原档不动。没有这一段的话，整理任务每跑一次，
      记忆就会在 ACTIVE 与 FADING 之间来回抖；
    * ARCHIVED 是单向门，除非有效强度明显回升，否则不再自动复活。
    """

    if memory.status is MemoryStatus.SUPERSEDED:
        return MemoryStatus.SUPERSEDED
    if memory.pinned:
        return MemoryStatus.ACTIVE

    profile = policy.profile(memory.kind)
    eff = effective_strength(memory, now, policy)
    current = memory.status

    if current is MemoryStatus.ARCHIVED:
        # 单向门：只有明显回升才允许重新进入 gisted 档。
        if eff >= profile.gist_threshold:
            return MemoryStatus.GISTED
        return MemoryStatus.ARCHIVED

    # 越过上阈：直接回活跃。
    if eff >= profile.fade_enter:
        return MemoryStatus.ACTIVE

    if current is MemoryStatus.ACTIVE:
        # 粘住带内保持活跃。
        if eff >= profile.fade_exit:
            return MemoryStatus.ACTIVE
        return _downgrade(eff, profile)

    if current is MemoryStatus.FADING:
        if eff >= profile.fade_exit:
            return MemoryStatus.FADING
        return _downgrade(eff, profile)

    # GISTED：粘住带内保持 gisted，再往上回到 fading。
    if eff >= profile.gist_threshold:
        return MemoryStatus.FADING
    if eff < profile.archive_threshold:
        return MemoryStatus.ARCHIVED
    return MemoryStatus.GISTED


def _downgrade(eff: float, profile: KindProfile) -> MemoryStatus:
    """一次性算清应该落在哪一档（允许跨档）。"""

    if eff < profile.archive_threshold:
        return MemoryStatus.ARCHIVED
    if eff < profile.gist_threshold:
        return MemoryStatus.GISTED
    return MemoryStatus.FADING


def apply_event(
    memory: Memory,
    event: MemoryEvent,
    *,
    now: float,
    policy: FadePolicy = DEFAULT_POLICY,
    strength_delta: float = 1.0,
) -> Memory:
    """把一个事件应用到记忆上，返回更新后的副本（不改原对象）。"""

    eff = effective_strength(memory, now, policy)
    updated = replace(memory)
    updated.updated_at = float(now)

    if event is MemoryEvent.RECALL:
        alpha = min(1.0, max(0.0, policy.recall_alpha * max(0.0, strength_delta)))
        updated.strength = clamp01(eff + alpha * (1.0 - eff))
        updated.anchor_at = float(now)
        updated.last_recalled_at = float(now)
        updated.recall_count = int(memory.recall_count) + 1
        # 「被想起」本身就是证据：直接回到活跃档。
        updated.status = MemoryStatus.ACTIVE

    elif event is MemoryEvent.REINFORCE:
        alpha = min(1.0, max(0.0, policy.reinforce_alpha * max(0.0, strength_delta)))
        updated.strength = clamp01(eff + alpha * (1.0 - eff))
        updated.anchor_at = float(now)
        updated.last_recalled_at = float(now)
        updated.recall_count = int(memory.recall_count) + 1
        updated.status = MemoryStatus.ACTIVE

    elif event is MemoryEvent.WEAKEN:
        alpha = min(1.0, max(0.0, policy.weaken_alpha * max(0.0, strength_delta)))
        updated.strength = clamp01(eff * (1.0 - alpha))
        updated.anchor_at = float(now)

    elif event is MemoryEvent.SUPERSEDE:
        updated.status = MemoryStatus.SUPERSEDED
        updated.updated_at = float(now)

    elif event is MemoryEvent.PIN:
        updated.pinned = True
        updated.strength = 1.0
        updated.anchor_at = float(now)
        updated.status = MemoryStatus.ACTIVE

    else:  # pragma: no cover - Enum 已穷尽
        raise ValueError(f"未知事件: {event}")

    if updated.status is not MemoryStatus.SUPERSEDED:
        updated.status = resolve_status(updated, now, policy)
    return updated


def days_until(
    memory: Memory,
    target_strength: float,
    *,
    now: float,
    policy: FadePolicy = DEFAULT_POLICY,
) -> Optional[float]:
    """距离有效强度跌到 target_strength 还有多少天。

    用于安排下一次「复习」检查时间。已经在阈值以下则返回 0。
    """

    if memory.pinned:
        return None
    eff = effective_strength(memory, now, policy)
    target = clamp01(target_strength)
    if eff <= target:
        return 0.0
    if target <= 0.0:
        return None
    tau = stability_days(memory, policy)
    try:
        days = -tau * math.log(target / eff)
    except ValueError:  # pragma: no cover - 参数已夹紧
        return 0.0
    return max(0.0, days)


def next_review_days(memory: Memory, *, now: float, policy: FadePolicy = DEFAULT_POLICY) -> Optional[float]:
    """下次该重新评估这条记忆的时间（天）。

    安排在「即将跌出 active 档」之前，留出复习窗口。
    """

    profile = policy.profile(memory.kind)
    return days_until(memory, profile.fade_enter, now=now, policy=policy)


def decay_curve(
    kind: MemoryKind,
    *,
    importance: float = 0.5,
    recall_count: int = 0,
    days: int = 30,
    policy: FadePolicy = DEFAULT_POLICY,
) -> list[float]:
    """给 UI / 文档用的衰减曲线预览。"""

    probe = Memory(
        id=0,
        scope_key="preview",
        kind=kind,
        content="",
        importance=importance,
        strength=1.0,
        anchor_at=0.0,
        status=MemoryStatus.ACTIVE,
        recall_count=recall_count,
    )
    return [round(effective_strength(probe, day * 86400.0, policy), 4) for day in range(days + 1)]

