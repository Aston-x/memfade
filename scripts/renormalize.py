"""重整记忆库：把「过长 / 多分句 / 缺三元组」的旧条目重新规范化。

什么时候需要：

* 升级到新版 memfade 之后——旧记忆是按老提示词写的，又长又没三元组；
* 手工写进去的记忆（工具写的长段落）；
* 体检报告指出质量问题时。

重要：规范化**可能把一条拆成多条**（「BotName喜欢辣的食物，也讨厌香菜」
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
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.diagnostics import clause_count  # noqa: E402
from memfade.textutil import fts_text, normalized_hash, split_atomic  # noqa: E402

# 「一件事一条」的两把尺子：单条不超过这么多字、这么多分句。
MAX_CHARS = 40
MAX_CLAUSES = 2


def find_bad(db_path: str, *, max_chars: int = MAX_CHARS, max_clauses: int = MAX_CLAUSES) -> List[Dict[str, Any]]:
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
        help="让宿主 LLM 给拆好的碎片补三元组（快）；拆分本身永远用本地规则",
    )
    parser.add_argument("--triple-chunk", type=int, default=8, help="每次问模型几条")
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
    # 模型只用来**补三元组**——那是小输出，13 秒就完事；
    # 让模型直接拆长文本反而会 thinking 到超时（实测 84s 空输出 / 240s 硬超时）。
    plan: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for item in bad:
        reasons = item["reasons"]
        too_long = any(r.startswith("超过") for r in reasons)
        too_many = any(r.startswith("多于") for r in reasons)
        pieces = split_atomic(item["content"], max_chars=args.max_chars)
        # 拆了之后每条都要真的变好：长度达标、分句达标。
        # 只在「太长」或「塞了太多句」时才拆，拆完再跑一遍不会变。
        improved = len(pieces) > 1 and all(
            len(p) <= args.max_chars and clause_count(p) <= MAX_CLAUSES for p in pieces
        )
        if (too_long or too_many) and improved:
            facts = [{"content": p, "subject": "", "predicate": "", "object": ""} for p in pieces]
        else:
            # 内容不动，只补三元组
            facts = [
                {
                    "content": item["content"],
                    "subject": item["subject"] or "",
                    "predicate": item["predicate"] or "",
                    "object": item["object"] or "",
                }
            ]
        plan.append((item, facts))

    if args.use_llm and not args.host_root:
        print("\n--use-llm 需要 --host-root（MaiBot 根目录，用来调用它的 LLM 服务）")
        return 1

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(memories)")]
        insert_cols = [c for c in columns if c != "id"]
        deleted = 0
        inserted = 0
        kept = 0
        updated = 0
        split_count = 0
        now = time.time()

        # 拆完之后统一让模型补三元组：一次请求一批，快。
        triples: Dict[int, Dict[str, str]] = {}
        if args.use_llm:
            from scripts.import_a_memorix import fill_triples_via_host

            pending: List[str] = []
            owner: List[int] = []
            for gi, (_item, facts) in enumerate(plan):
                for fi, fact in enumerate(facts):
                    if fact["subject"] and fact["predicate"]:
                        continue
                    pending.append(fact["content"])
                    owner.append(gi * 1000 + fi)
            if pending:
                print(f"让宿主 LLM 给 {len(pending)} 条补三元组…")
                filled = fill_triples_via_host(
                    pending, args.host_root, chunk_size=args.triple_chunk
                )
                for key, value in zip(owner, filled):
                    triples[key] = value

        for gi, (item, facts) in enumerate(plan):
            if len(facts) == 1 and facts[0]["content"] == item["content"]:
                # 内容没变 → 只更新三元组。
                # 不删了重插：那样会换掉 id，也丢掉 recall_count 和创建时间。
                triple = triples.get(gi * 1000, {})
                subject = triple.get("subject") or item["subject"] or ""
                predicate = triple.get("predicate") or item["predicate"] or ""
                object_ = triple.get("object") or item["object"] or ""
                if subject and predicate and (subject, predicate, object_) != (
                    item["subject"],
                    item["predicate"],
                    item["object"],
                ):
                    conn.execute(
                        "UPDATE memories SET subject=?, predicate=?, object=?, updated_at=? WHERE id=?",
                        (subject, predicate, object_, now, item["id"]),
                    )
                    updated += 1
                continue
            split_count += 1
            deleted += conn.execute("DELETE FROM memories WHERE id=?", (item["id"],)).rowcount
            for fi, entry in enumerate(facts):
                triple = triples.get(gi * 1000 + fi, {})
                data = {
                    "scope_key": item["scope_key"],
                    "kind": _kind_of(entry["content"]),
                    "content": entry["content"],
                    "summary": "",
                    # ⚠ 显著性必须**继承原条目**。
                    # 早先这里写死 0.75，等于把整段记忆的显著性抹平——
                    # 画像和检索都靠它排序，抹平了就只能靠时间戳和长度瞎排。
                    "importance": item["importance"],
                    "strength": 1.0,
                    "anchor_at": now,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                    "last_recalled_at": None,
                    "recall_count": 0,
                    "pinned": 0,
                    "cross_scope": 0,
                    "subject": triple.get("subject") or entry.get("subject", ""),
                    "predicate": triple.get("predicate") or entry.get("predicate", ""),
                    "object": triple.get("object") or entry.get("object", ""),
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
        print(f"拆开 {split_count} 条长记忆：删掉旧条目 {deleted} 条，写入新条目 {inserted} 条")
        if updated:
            print(f"给 {updated} 条补上了三元组（内容没动）")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


