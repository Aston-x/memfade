"""memfade 命令行工具：体检、导出、检索。

用法：

    python scripts/memfade_cli.py health   --db <memfade.db>
    python scripts/memfade_cli.py export   --db <memfade.db> --out memories.json
    python scripts/memfade_cli.py search   --db <memfade.db> --query "显卡"
    python scripts/memfade_cli.py fade     --db <memfade.db>          # 跑一轮整理

不需要装任何依赖——核心是纯标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemFadeConfig, Scope  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402


def _engine(db_path: str) -> MemoryEngine:
    config = MemFadeConfig.from_dict({"storage": {"db_path": db_path, "enable_vector": False}})
    return MemoryEngine(config)


def cmd_health(args: argparse.Namespace) -> int:
    engine = _engine(args.db)
    try:
        report = engine.diagnose()
    finally:
        engine.close()
    print("=" * 62)
    print(f"记忆库体检：{report['total']} 条")
    print("=" * 62)
    print(f"  长度中位数     : {report['length_median']} 字")
    print(f"  平均长度       : {report['length_mean']} 字")
    print(f"  超过 40 字     : {report['too_long']} 条")
    print(f"  含多个分句     : {report['multi_clause']} 条")
    print(f"  缺三元组       : {report['missing_spo']} 条")
    print(f"  完全重复       : {report['exact_duplicates']} 条")
    print(f"  近似重复对     : {report['near_duplicate_pairs']} 对")
    print(f"  种类分布       : {report['kinds']}")
    print(f"  档位分布       : {report['statuses']}")
    if report["issues"]:
        print()
        print("  发现问题：")
        for issue in report["issues"]:
            print(f"    · {issue['hint']}")
    else:
        print()
        print("  没有发现问题。")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    engine = _engine(args.db)
    try:
        rows = engine.export()
    finally:
        engine.close()
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"已导出 {len(rows)} 条记忆 -> {args.out}")
    else:
        print(text)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = _engine(args.db)
    try:
        scope = Scope(platform="cli", chat_type="cli", chat_id="search")
        rows = engine.explain(args.query, scope, top_k=args.limit)
    finally:
        engine.close()
    if not rows:
        print("没有命中。")
        return 0
    print(f"{'总分':>7} {'相关':>6} {'重要':>6} {'强度':>6}  记忆")
    for row in rows:
        print(
            f"{row['score']:>7.3f} {row['relevance']:>6.2f} "
            f"{row['importance']:>6.2f} {row['strength']:>6.2f}  {row['content'][:50]}"
        )
    return 0


def cmd_fade(args: argparse.Namespace) -> int:
    import asyncio

    engine = _engine(args.db)
    try:
        report = asyncio.run(engine.consolidate())
    finally:
        engine.close()
    print("整理完成：", json.dumps(report.as_dict(), ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="memfade 命令行工具")
    parser.add_argument("--db", default="memfade.db", help="memfade.db 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="记忆库体检").set_defaults(func=cmd_health)

    export = sub.add_parser("export", help="导出记忆为 JSON")
    export.add_argument("--out", default="", help="输出文件（不给就打到标准输出）")
    export.set_defaults(func=cmd_export)

    search = sub.add_parser("search", help="检索并解释召回")
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=8)
    search.set_defaults(func=cmd_search)

    sub.add_parser("fade", help="跑一轮整理（淡化/合并/仲裁）").set_defaults(func=cmd_fade)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

