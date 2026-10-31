"""去重与合并。

「记忆里面的重复语段太多了」是真实用户原话。根因是抽取层只保证了
**单条内部**的去重，而跨条目的同义重复（换个标点、换个语序说同一件事）
没人管。

这里用两层判定：

1. **归一化哈希**——完全一致的直接判重（标点、空格、大小写差异不算差异）；
2. **二元组包含度**——近似判重。包含度而非 Jaccard，因为摘要常常是
   原文的缩写，两者长度天然不等。

合并策略是「保留更强的，把被合并方的痕迹留在 meta 里」——
**不删数据**。原文进 meta.merged_from，事后可追溯、可回滚。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .textutil import normalized_hash, similarity
from .types import Memory


def is_exact_duplicate(a: Memory, b: Memory) -> bool:
    """归一化后完全相同。"""

    return normalized_hash(a.content) == normalized_hash(b.content)


def is_near_duplicate(a: Memory, b: Memory, threshold: float = 0.82) -> bool:
    """近似重复判定。"""

    if is_exact_duplicate(a, b):
        return True
    return similarity(a.content, b.content) >= threshold


def _rank(mem: Memory) -> tuple:
    """留存信号排序键。

    召回次数排最前，因为「被反复提起」是最强的留存信号。
    """

    return (int(mem.recall_count), float(mem.strength), float(mem.importance), float(mem.updated_at))


def choose_survivor(a: Memory, b: Memory) -> Memory:
    """挑一条留下。"""

    return a if _rank(a) >= _rank(b) else b


def merge_memories(survivor: Memory, other: Memory) -> Memory:
    """把 other 并进 survivor，返回合并后的副本。

    合并内容：recall_count 相加、strength 取大、importance 取大、
    时间戳取极端值；被合并方的 id 与原文摘要进 meta.merged_from。
    """

    merged = replace(survivor)
    merged.recall_count = int(survivor.recall_count) + int(other.recall_count)
    merged.strength = max(float(survivor.strength), float(other.strength))
    merged.anchor_at = max(float(survivor.anchor_at), float(other.anchor_at))
    merged.importance = max(float(survivor.importance), float(other.importance))
    merged.created_at = min(float(survivor.created_at), float(other.created_at))
    merged.updated_at = max(float(survivor.updated_at), float(other.updated_at))
    if other.last_recalled_at and (
        not merged.last_recalled_at or other.last_recalled_at > merged.last_recalled_at
    ):
        merged.last_recalled_at = other.last_recalled_at
    if survivor.pinned or other.pinned:
        merged.pinned = True

    meta = dict(merged.meta)
    history = list(meta.get("merged_from") or [])
    history.append({"id": other.id, "content": other.content[:200]})
    meta["merged_from"] = history[-20:]
    meta["merged_count"] = int(meta.get("merged_count") or 0) + 1
    merged.meta = meta
    return merged


def find_duplicates(
    memories: Sequence[Memory],
    *,
    threshold: float = 0.92,
    limit: int = 2000,
) -> list[tuple[Memory, Memory]]:
    """在给定集合里找近似重复对。

    先用哈希桶把完全一致的挑出来，剩下的再做两两比较——
    N² 在几千条量级完全够用（毫秒级），不必上向量索引。
    """

    items = list(memories)[:limit]
    pairs: list[tuple[Memory, Memory]] = []
    buckets: dict[str, list[int]] = {}

    for idx, mem in enumerate(items):
        key = normalized_hash(mem.content)
        bucket = buckets.setdefault(key, [])
        if bucket:
            for other_idx in bucket:
                pairs.append((items[other_idx], mem))
        else:
            bucket.append(idx)

    exact_pairs = {(a.id, b.id) for a, b in pairs}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if (a.id, b.id) in exact_pairs or (b.id, a.id) in exact_pairs:
                continue
            if similarity(a.content, b.content) >= threshold:
                pairs.append((a, b))
    return pairs

