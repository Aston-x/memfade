"""重整记忆库：把「过长 / 多分句 / 缺三元组」的旧条目重新规范化。

什么时候需要：

* 升级到新版 memfade 之后——旧记忆是按老提示词写的，又长又没三元组；
* 手工写进去的记忆（工具写的长段落）；
* 体检报告指出质量问题时。

重要：规范化**可能把一条拆成多条**（「许杏玖喜欢辣的食物，也讨厌香菜」
会变成两条）。所以这里是「删掉旧的、写入新的」，不是逐条替换。

用法：

    python scripts/renormalize.py --db <memfade.db> --dry-run
    python scripts/renormalize.py --db <memfade.db> --host-root <MaiBot 根目录>
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.diagnostics import clause_count  # noqa: E402
from memfade.textutil import fts_text, normalized_hash, split_atomic  # noqa: E402


def find_bad(db_path: str, *, max_chars: int = 40, max_clauses: int = 2) -> List[Dict[str, Any]]:
    """找出需要重整的条目。"""

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM memories").fetchall()
    finally:
        conn.close()
    bad = []
    for row in rows:
        content = row["content"] or ""
        needs_spo = not (row["subject"] and row["predicate"])
        too_long = len(content) > max_chars
        too_many = clause_count(content) > max_clauses
        if needs_spo or too_long or too_many:
            item = dict(row)
            item["reasons"] = []
            if needs_spo:
                item["reasons"].append("缺三元组")
            if too_long:
                item["reasons"].append("超过 %d 字" % max_chars)
            if too_many:
                item["reasons"].append("多于 %d 个分句" % max_clauses)
            bad.append(item)
    return bad


def _kind_of(content: str) -> str:
    """粗分类。"""

    if any(word in content for word in ("喜欢", "讨厌", "最爱", "不吃")):
        return "preference"
    if any(word in content for word in ("称呼", "叫", "铁子", "朋友")):
        return "relation"
    return "semantic"


def main() -> int:
    parser = argparse.ArgumentParser(description="重整 memfade 记忆库")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--host-root", default="", help="MaiBot 根目录（用它调 LLM 服务做规范化）")
    parser.add_argument("--dry-run", action="store_true", help="只看会改什么，不写库")
    parser.add_argument("--chunk-size", type=int, default=3, help="每次问模型几条")
    parser.add_argument("--max-chars", type=int, default=40, help="本地规则拆分时每条的目标字数")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="先让宿主 LLM 拆（能顺便补三元组，但慢且可能超时）；默认只用本地规则",
    )
    args = parser.parse_args()

    bad = find_bad(args.db)
    print("=" * 62)
    print(f"需要重整的条目：{len(bad)}")
    print("=" * 62)
    for item in bad[:20]:
        print(f"  #{item['id']:<4} [{'/'.join(item['reasons'])}] {item['content'][:44]}")
    if not bad:
        return 0
    if args.dry_run:
        print("\n--dry-run：没有写库")
        return 0

    # 默认走**本地规则拆分**：快、稳、不花钱、一个字都不会丢。
    # 想让模型顺便补三元组就加 --use-llm（但要能忍它的 thinking）。
    groups: List[List[Dict[str, Any]]] = []
    if args.use_llm:
        if not args.host_root:
            print("\n--use-llm 需要 --host-root（MaiBot 根目录，用来调用它的 LLM 服务）")
            return 1
        from scripts.import_a_memorix import split_via_host

        print(f"\n用宿主 LLM 规范化 {len(bad)} 条…")
        groups = split_via_host(
            [item["content"] for item in bad], args.host_root, chunk_size=args.chunk_size
        )
        if len(groups) != len(bad):
            print(f"⚠ 拆解返回 {len(groups)} 条，与输入 {len(bad)} 条对不上，放弃写库")
            return 1
    else:
        groups = [[] for _ in bad]

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(memories)")]
        insert_cols = [c for c in columns if c != "id"]
        deleted = 0
        inserted = 0
        kept = 0
        local_split = 0
        now = time.time()
        for item, facts in zip(bad, groups):
            if not facts:
                # 模型没拆开 → 本地按标点拆。
                # 规则拆分不如模型懂语义，但至少保证「一条一件事、一条不太长」，
                # 而且**一个字都不会丢**。
                pieces = split_atomic(item["content"], max_chars=args.max_chars)
                if len(pieces) > 1:
                    facts = [{"content": p, "subject": "", "predicate": "", "object": ""} for p in pieces]
                    local_split += 1
                else:
                    # 真的拆不动就别动：保留原条目。
                    # 「删掉旧的、塞回一样的」等于把记忆改坏——还丢了三元组。
                    kept += 1
                    continue
            deleted += conn.execute("DELETE FROM memories WHERE id=?", (item["id"],)).rowcount
            for entry in facts:
                data = {
                    "scope_key": item["scope_key"],
                    "kind": _kind_of(entry["content"]),
                    "content": entry["content"],
                    "summary": "",
                    "importance": 0.75,
                    "strength": 1.0,
                    "anchor_at": now,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                    "last_recalled_at": None,
                    "recall_count": 0,
                    "pinned": 0,
                    "cross_scope": 0,
                    "subject": entry.get("subject", ""),
                    "predicate": entry.get("predicate", ""),
                    "object": entry.get("object", ""),
                    "source": "renormalize",
                    "superseded_by": None,
                    "content_hash": normalized_hash(entry["content"]),
                    # ⚠ fts_tokens 必须算出来：FTS 触发器靠它建索引，
                    # 留空的话这条记忆就**搜不到了**。
                    "fts_tokens": fts_text(entry["content"]),
                    "meta": "{}",
                    "embedding": None,
                }
                data = {k: v for k, v in data.items() if k in insert_cols}
                names = ",".join(data.keys())
                placeholders = ",".join("?" * len(data))
                conn.execute(
                    "INSERT INTO memories (%s) VALUES (%s)" % (names, placeholders), list(data.values())
                )
                inserted += 1
        conn.commit()
        print(f"删掉旧条目 {deleted} 条，写入新条目 {inserted} 条")
        if local_split:
            print(f"（其中 {local_split} 条是本地规则拆的，模型没给出结果）")
        if kept:
            print(f"⚠ 有 {kept} 条没拆开，原样保留（下次整理再试）")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


