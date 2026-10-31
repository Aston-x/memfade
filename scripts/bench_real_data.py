"""在真实聊天数据上跑 memfade，量化三个关键行为。

这是「实验对象 = 许杏玖」的验证脚本。直接读 MaiBot 的消息库，
用真实说过的话跑一遍 memfade，看三件事：

1. **去重**：同样一件事换个说法存了几条？去重之后省了多少？
2. **淡化**：随着时间推移，还有多少记忆留在「事实层」？
3. **隔离**：私聊的记忆会不会漏进群？

用法：

    python scripts/bench_real_data.py --db "G:/AI/许杏玖-MaiBot/MaiBot/data/MaiBot.db"

不需要 LLM：抽取走规则兜底，向量检索关闭，纯 BM25。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemoryDraft, MemoryEngine, MemoryKind, MemFadeConfig, Scope  # noqa: E402
from memfade.textutil import normalized_hash  # noqa: E402


def load_messages(db_path: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """从 MaiBot 消息库读真实消息。"""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT user_id, group_id, processed_plain_text FROM mai_messages "
            "WHERE processed_plain_text IS NOT NULL AND length(processed_plain_text) > 4 "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"读不了 {db_path}：{exc}")
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        text = str(row["processed_plain_text"] or "").strip()
        if not text:
            continue
        out.append(
            {
                "user_id": str(row["user_id"] or ""),
                "group_id": str(row["group_id"] or ""),
                "text": text,
            }
        )
    return out


def classify(text: str) -> Tuple[MemoryKind, float]:
    """不调 LLM 的粗分类：够用来演示去重与淡化的行为。"""

    stable_markers = (
        "我叫", "我的名字", "我喜欢", "我最喜欢", "我讨厌", "我住在",
        "我是做", "我生日", "答应", "约定", "说好", "以后都", "永远",
    )
    if any(marker in text for marker in stable_markers):
        return MemoryKind.SEMANTIC, 0.72
    if len(text) >= 12:
        return MemoryKind.EPISODIC, 0.38
    return MemoryKind.EPISODIC, 0.22


def build_engine(db_path: str) -> MemoryEngine:
    config = MemFadeConfig.from_dict(
        {
            "storage": {"db_path": db_path, "enable_vector": False, "enable_fts": True},
            "extraction": {"min_importance": 0.2},
            "retrieval": {"top_k": 8, "budget_tokens": 400, "echo_max": 2},
        }
    )
    return MemoryEngine(config)


def scope_of(item: Dict[str, Any]) -> Scope:
    group_id = item.get("group_id") or ""
    if group_id:
        return Scope(platform="qq", chat_type="group", chat_id=group_id)
    return Scope(platform="qq", chat_type="private", chat_id=item.get("user_id") or "unknown")


def report_dedup(engine: MemoryEngine, messages: List[Dict[str, Any]]) -> None:
    """第一项：去重。"""

    print()
    print("=" * 66)
    print("① 去重：同样的话存了几份？")
    print("=" * 66)
    raw = 0
    written = 0
    seen_hash = set()
    dup_examples = []
    for item in messages:
        kind, importance = classify(item["text"])
        if importance < 0.2:
            continue
        raw += 1
        scope = scope_of(item)
        draft = MemoryDraft(content=item["text"], kind=kind, importance=importance)
        before = engine.store.count(scope.key())
        engine.remember(draft, scope)
        after = engine.store.count(scope.key())
        if after > before:
            written += 1
        else:
            key = normalized_hash(item["text"])
            if key not in seen_hash:
                seen_hash.add(key)
                dup_examples.append(item["text"][:40])
    total = engine.store.count()
    print(f"  原始候选     : {raw}")
    print(f"  实际入库     : {written}")
    print(f"  被判定重复   : {raw - written}")
    if raw:
        print(f"  去重率       : {(raw - written) * 100.0 / raw:.1f}%")
    print(f"  库中总数     : {total}")
    for text in dup_examples[:5]:
        print(f"    · 合并掉：{text}")


def report_fade(engine: MemoryEngine) -> None:
    """第二项：淡化。"""

    print()
    print("=" * 66)
    print("② 淡化：时间推移后，还剩多少记忆留在事实层？")
    print("=" * 66)
    print("  假设所有记忆都是「30 天前」写的，看不同时间点的分布：")
    # 把所有记忆回填到 30 天前，模拟「这些记忆已经放了 30 天」
    backdate = time.time() - 30 * 86400.0
    for mem in engine.store.iter_maintenance(limit=10000):
        mem.anchor_at = backdate
        mem.created_at = backdate
        mem.updated_at = backdate
        engine.store.update(mem)
    for days in (0, 7, 14, 30, 60, 120):
        now = time.time() + days * 86400.0
        counts = {"fact": 0, "echo": 0, "gone": 0}
        for mem in engine.store.iter_maintenance(limit=10000):
            from memfade.decay import effective_strength, resolve_status

            status = resolve_status(mem, now, engine.policy)
            if status.value == "active":
                counts["fact"] += 1
            elif status.value in ("fading", "gisted"):
                counts["echo"] += 1
            else:
                counts["gone"] += 1
        print(
            "    +{0:>3} 天 → 事实层 {1:>3} 条 / 回声层 {2:>3} 条 / 已退场 {3:>3} 条".format(
                days, counts["fact"], counts["echo"], counts["gone"]
            )
        )


def report_isolation(engine: MemoryEngine, messages: List[Dict[str, Any]]) -> None:
    """第三项：隔离。"""

    print()
    print("=" * 66)
    print("③ 隔离：私聊的记忆会不会漏进群？")
    print("=" * 66)
    group_scopes = []
    private_scopes = []
    for item in messages:
        scope = scope_of(item)
        if scope.chat_type == "group":
            if scope.key() not in group_scopes:
                group_scopes.append(scope.key())
        else:
            if scope.key() not in private_scopes:
                private_scopes.append(scope.key())
    leaked = 0
    for private_key in private_scopes:
        private_scope = Scope.from_key(private_key)
        for group_key in group_scopes:
            group_scope = Scope.from_key(group_key)
            for mem in engine.store.iter_scope(group_key, statuses=("active", "fading", "gisted"), limit=999):
                if mem.scope_key == private_key:
                    leaked += 1
                    break
    print(f"  群作用域     : {len(group_scopes)} 个 -> {group_scopes}")
    print(f"  私聊作用域   : {len(private_scopes)} 个 -> {private_scopes}")
    print(f"  跨会话泄漏   : {leaked} 条（必须为 0）")


def main() -> int:
    parser = argparse.ArgumentParser(description="在真实聊天数据上跑 memfade 基准")
    parser.add_argument("--db", required=True, help="MaiBot 消息库路径")
    parser.add_argument("--limit", type=int, default=800, help="最多取多少条消息")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"找不到数据库：{args.db}", file=sys.stderr)
        return 1

    messages = load_messages(args.db, args.limit)
    if not messages:
        print("库里没有可用消息", file=sys.stderr)
        return 1

    handle, tmp_db = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.remove(tmp_db)
    engine = build_engine(tmp_db)
    try:
        print(f"数据源：{args.db}")
        print(f"消息数：{len(messages)}")
        report_dedup(engine, messages)
        report_fade(engine)
        report_isolation(engine, messages)
    finally:
        engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = tmp_db + suffix
            if os.path.exists(path):
                os.remove(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



