"""记忆质量体检。

看实机跑出来的记忆，回答三个问题：

1. **太长吗**——一条记忆塞了几件事？注入时只能截断，截断就断在半句；
2. **够原子吗**——「一件事一条记忆」，还是把三件事挤成一段？
3. **有矛盾检测的抓手吗**——subject/predicate/object 填了没有？
   没填的话，两条互相矛盾的事实就发现不了，旧值会一直留在上下文里。

用法：

    python scripts/bench_memory_quality.py --db <memfade.db>
    python scripts/bench_memory_quality.py --db <memfade.db> --show 20
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.textutil import normalized_hash  # noqa: E402


SPLIT_RE = re.compile(r"[，,；;。！？!?]")
GOOD_MAX = 40


def load(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, kind, content, importance, subject, predicate, object, status, scope_key "
            "FROM memories ORDER BY id"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"读不了 {db_path}：{exc}")
    finally:
        conn.close()
    return [dict(r) for r in rows]


def clause_count(text: str) -> int:
    return len([p for p in SPLIT_RE.split(text or "") if p.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description="memfade 记忆质量体检")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--show", type=int, default=0, help="打印前 N 条")
    args = parser.parse_args()

    rows = load(args.db)
    if not rows:
        print("库里还没有记忆")
        return 0

    lengths = [len(r["content"] or "") for r in rows]
    clauses = [clause_count(r["content"]) for r in rows]
    spo = [bool(r["subject"] and r["predicate"]) for r in rows]
    hashes = Counter(normalized_hash(r["content"]) for r in rows)

    too_long = [r for r in rows if len(r["content"] or "") > GOOD_MAX]
    multi = [r for r in rows if clause_count(r["content"]) > 1]
    no_spo = [r for r in rows if not (r["subject"] and r["predicate"])]
    exact_dupes = sum(count - 1 for count in hashes.values() if count > 1)

    print("=" * 66)
    print(f"记忆质量体检：{len(rows)} 条")
    print("=" * 66)
    print(f"  长度中位数          : {sorted(lengths)[len(lengths) // 2]} 字")
    print(f"  平均长度            : {sum(lengths) / len(lengths):.1f} 字")
    print(f"  超过 {GOOD_MAX} 字        : {len(too_long)} 条（{len(too_long) * 100 // len(rows)}%）")
    print(f"  含多个分句          : {len(multi)} 条（{len(multi) * 100 // len(rows)}%）")
    print(f"  缺 subject/predicate: {len(no_spo)} 条（{len(no_spo) * 100 // len(rows)}%）")
    print(f"  完全重复            : {exact_dupes} 条")
    print(f"  种类分布            : {dict(Counter(r['kind'] for r in rows))}")
    print(f"  档位分布            : {dict(Counter(r['status'] for r in rows))}")

    if args.show:
        print()
        print("-" * 66)
        for r in rows[: args.show]:
            text = (r["content"] or "").replace("\n", " ")
            print(f"  [{r['kind']}/{r['status']}] {len(text):>3}字 {clause_count(text)}句 | {text[:60]}")

    if too_long:
        print()
        print(f"  ↳ 太长的（注入时会被截断，建议一条只讲一件事）：")
        for r in too_long[:5]:
            print(f"      {len(r['content'] or '')}字 | {(r['content'] or '')[:70]}")
    if no_spo:
        print()
        print("  ↳ 缺 subject/predicate 的（矛盾检测抓不到，旧值会赖着不走）：")
        for r in no_spo[:5]:
            print(f"      {(r['content'] or '')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


