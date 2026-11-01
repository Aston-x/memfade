"""记忆库体检。

回答「这个记忆库现在健康吗」：

* 长度分布——一条记忆塞了几件事？注入时只能截断，截断就断在半句里；
* 分句数——「一件事一条记忆」做到了吗；
* 三元组覆盖——缺了它矛盾检测就抓不到，旧事实会赖着不走；
* 近似重复——库里还剩多少「同一件事的多种说法」；
* 档位分布——现在还有多少记忆是活的。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .store import MemoryStore
from .textutil import clause_count as _clause_count
from .textutil import normalized_hash, similarity


SPLIT_RE = re.compile(r"[，,；;。！？!?]")

# 「一句话一件事」的参考长度。超过就不像是一件事。
GOOD_MAX = 40
# 「一件事一条」的参考分句数。
ATOMIC_MAX = 2


def clause_count(text: str) -> int:
    """中文分句计数（按标点切）。

    口径与 ``textutil.clause_count`` 完全一致——体检和拆分必须用同一把尺子，
    否则会出现「体检说这条塞了几件事，拆分却说拆不动」。
    """

    return _clause_count(text)


def analyze(
    store: MemoryStore,
    scope_key: Optional[str] = None,
    *,
    limit: int = 5000,
    near_threshold: float = 0.82,
) -> Dict[str, Any]:
    """跑一遍体检，返回统计与问题清单。"""

    memories = store.iter_scope(scope_key, limit=limit) if scope_key else store.iter_all(limit=limit)
    total = len(memories)
    if not total:
        return {
            "total": 0,
            "length_median": 0,
            "length_mean": 0.0,
            "too_long": 0,
            "multi_clause": 0,
            "missing_spo": 0,
            "exact_duplicates": 0,
            "near_duplicate_pairs": 0,
            "kinds": {},
            "statuses": {},
            "issues": [],
        }

    lengths = [len(m.content or "") for m in memories]
    too_long = [m for m in memories if len(m.content or "") > GOOD_MAX]
    multi = [m for m in memories if clause_count(m.content) > ATOMIC_MAX]
    no_spo = [m for m in memories if not (m.subject and m.predicate)]

    hashes = Counter(normalized_hash(m.content) for m in memories)
    exact_dupes = sum(count - 1 for count in hashes.values() if count > 1)

    near_pairs: List[List[int]] = []
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            a, b = memories[i], memories[j]
            if normalized_hash(a.content) == normalized_hash(b.content):
                continue
            if similarity(a.content, b.content) >= near_threshold:
                near_pairs.append([a.id, b.id])
                if len(near_pairs) >= 200:
                    break
        if len(near_pairs) >= 200:
            break

    issues: List[Dict[str, Any]] = []
    if too_long:
        issues.append(
            {
                "kind": "too_long",
                "count": len(too_long),
                "hint": f"有 {len(too_long)} 条超过 {GOOD_MAX} 字，注入时会被截断。建议一条只讲一件事。",
            }
        )
    if multi:
        issues.append(
            {
                "kind": "multi_clause",
                "count": len(multi),
                "hint": f"有 {len(multi)} 条含多个分句，一条记忆应该只讲一件事。",
            }
        )
    if no_spo:
        issues.append(
            {
                "kind": "missing_spo",
                "count": len(no_spo),
                "hint": f"有 {len(no_spo)} 条缺 subject/predicate，矛盾检测抓不到它们。",
            }
        )
    if exact_dupes:
        issues.append(
            {
                "kind": "exact_duplicates",
                "count": exact_dupes,
                "hint": f"库里有 {exact_dupes} 条完全重复，去重没拦住。",
            }
        )
    if near_pairs:
        issues.append(
            {
                "kind": "near_duplicates",
                "count": len(near_pairs),
                "hint": f"发现 {len(near_pairs)} 对近似重复（同一件事的多种说法）。",
            }
        )

    return {
        "total": total,
        "length_median": sorted(lengths)[total // 2],
        "length_mean": round(sum(lengths) / total, 1),
        "too_long": len(too_long),
        "multi_clause": len(multi),
        "missing_spo": len(no_spo),
        "exact_duplicates": exact_dupes,
        "near_duplicate_pairs": len(near_pairs),
        "kinds": dict(Counter(m.kind.value for m in memories)),
        "statuses": dict(Counter(m.status.value for m in memories)),
        "issues": issues,
    }


def export_memories(store: MemoryStore, scope_key: Optional[str] = None, *, limit: int = 100000) -> List[Dict[str, Any]]:
    """导出记忆（JSON 友好）。可移植、可审计。"""

    memories = store.iter_scope(scope_key, limit=limit) if scope_key else store.iter_all(limit=limit)
    return [m.to_dict() for m in memories]

