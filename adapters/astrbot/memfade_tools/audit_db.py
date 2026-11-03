"""审一份记忆库：结构、FTS 一致性、作用域分布、疑似测试数据、
以及**按现在的写入前闸门重扫全库**。

只读脚本（不打开写事务），可以安全地对线上库跑；
但凡是会**触发注入**的诊断（build_context 之类）仍然要先做副本。

用法：
    python scripts/audit_db.py --db <memfade.db> [--show-fts]
    python scripts/audit_db.py --db <memfade.db> --focus 季怡 --self-name 许杏玖
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.textutil import fts_text  # noqa: E402


TEST_PATTERNS = re.compile(
    r"(测试|test|示例|样例|demo|hello world|asdf|qwerty|占位|placeholder)",
    re.IGNORECASE,
)


def _ts(value) -> str:
    if not value:
        return "-"
    try:
        return datetime.datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)


def gate_scan(
    conn,
    focus_subjects: tuple[str, ...] = (),
    self_names: tuple[str, ...] = (),
) -> tuple[list, list]:
    """按**现在**的写入前闸门重扫一遍全库。

    为什么要有这个：**闸门是随代码变的。**
    闸门上线前混进来的记忆，靠人记着「哪几条不对」是清不干净的——
    实测就漏过一条：清理时删掉了两条工具路径写进来的，
    闸门部署**之前**写的第三条留了下来（主语不是要记的那个人）。

    → 标准动作是「用现在的闸门重扫一遍全库」，而不是凭记忆挑。

    判定分两档：

    * **不该入库**：闸门现在会直接拒掉它（临时状态/情绪、
      跟要记的那个人没关系）——这些是实打实的问题；
    * **该降级为事件**：闸门现在会把它降级（时效词当常态、
      模型自己加的频率）——信息没丢，只是形态不对。

    只读、只判断，**不改库**。
    """

    from memfade.guards import Evidence, check_draft, check_remember_request
    from memfade.types import MemoryDraft, MemoryKind

    should_not_exist: list = []
    should_downgrade: list = []
    rows = list(
        conn.execute(
            "SELECT id, kind, content, importance, subject, meta FROM memories ORDER BY id"
        )
    )
    for row in rows:
        try:
            kind = MemoryKind(str(row["kind"] or "semantic").lower())
        except ValueError:
            kind = MemoryKind.SEMANTIC
        try:
            meta = json.loads(row["meta"] or "{}")
        except (TypeError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        draft = MemoryDraft(
            content=row["content"] or "",
            kind=kind,
            importance=float(row["importance"] or 0.5),
            subject=row["subject"] or "",
            meta=meta,
        )
        evidence = Evidence(
            quote=str(meta.get("quote") or ""),
            score=float(meta.get("quote_score") or 0.0),
        )
        hard: list[str] = []
        soft: list[str] = []
        verdict = check_draft(draft, evidence=evidence)
        if verdict.action == "drop":
            hard.append("抽取闸门：%s" % verdict.reason)
        elif verdict.action == "downgrade":
            soft.append("抽取闸门该降级为事件：%s" % verdict.reason)
        request = check_remember_request(
            draft, focus_subjects=focus_subjects, self_names=self_names
        )
        if request.action == "drop":
            hard.append("记住闸门：%s" % request.reason)
        if hard:
            should_not_exist.append((row, hard))
        elif soft:
            should_downgrade.append((row, soft))
    return should_not_exist, should_downgrade


def audit(
    db_path: str,
    show_fts: bool = False,
    focus_subjects: tuple[str, ...] = (),
    self_names: tuple[str, ...] = (),
) -> int:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        problems = 0

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "memories" not in tables:
            print("这不是 memfade 库：%s" % db_path)
            return 2

        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print("memories: %d 条" % total)

        print("\n--- 作用域分布 ---")
        for row in conn.execute(
            "SELECT scope_key, COUNT(*) n, MIN(created_at), MAX(created_at) "
            "FROM memories GROUP BY scope_key ORDER BY n DESC"
        ):
            print(
                "  %-58s %3d 条  %s ~ %s"
                % (row["scope_key"], row["n"], _ts(row[2]), _ts(row[3]))
            )

        print("\n--- 来源分布 ---")
        for row in conn.execute(
            "SELECT source, COUNT(*) n FROM memories GROUP BY source ORDER BY n DESC"
        ):
            print("  %-34s %3d" % (row["source"] or "(空)", row["n"]))

        print("\n--- 状态分布 ---")
        for row in conn.execute(
            "SELECT status, COUNT(*) n FROM memories GROUP BY status ORDER BY n DESC"
        ):
            print("  %-14s %3d" % (row["status"], row["n"]))

        print("\n--- 种类分布 ---")
        for row in conn.execute(
            "SELECT kind, COUNT(*) n FROM memories GROUP BY kind ORDER BY n DESC"
        ):
            print("  %-14s %3d" % (row["kind"], row["n"]))

        if "memories_fts" in tables:
            fts_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='memories_fts'"
            ).fetchone()[0]
            print("\n--- FTS ---")
            print("  " + (fts_sql or "").replace("\n", " "))
            # FTS 的 rowid 是自增的，跟记忆 id 无关——
            # 真正的对应关系在 mem_id 列上。
            fts_rows = list(conn.execute("SELECT mem_id FROM memories_fts"))
            ids = {row[0] for row in conn.execute("SELECT id FROM memories")}
            ghosts = [r[0] for r in fts_rows if r[0] not in ids]
            print("  FTS 行数 %d，对不上记忆的 %d" % (len(fts_rows), len(ghosts)))
            if ghosts:
                problems += 1
                print(
                    "  [!] 对不上的 mem_id: %s%s"
                    % (ghosts[:20], " ..." if len(ghosts) > 20 else "")
                )
                if show_fts:
                    print("  --- 这些行会污染检索结果 ---")
                    for row in conn.execute(
                        "SELECT mem_id, tokens FROM memories_fts WHERE mem_id IN (%s) ORDER BY mem_id"
                        % ",".join(str(g) for g in ghosts)
                    ):
                        print("   ", list(row))

            # 全文索引与正文必须一致：改写正文却不重算索引词，
            # 搜「token plan」就会搜不到那条记忆。
            stale = []
            for row in conn.execute("SELECT id, content, fts_tokens FROM memories"):
                if (row["fts_tokens"] or "") != fts_text(row["content"] or ""):
                    stale.append(row["id"])
            print("  索引词与正文不一致：%d 条%s" % (len(stale), (" " + str(stale[:10])) if stale else ""))
            if stale:
                problems += 1

        print("\n--- 疑似测试数据 ---")
        hits = 0
        for row in conn.execute(
            "SELECT id, scope_key, kind, created_at, content FROM memories ORDER BY id"
        ):
            content = row["content"] or ""
            if TEST_PATTERNS.search(content):
                hits += 1
                print("  #%s [%s] %s" % (row["id"], row["kind"], content))
        if not hits:
            print("  无")

        print("\n--- 按现在的写入前闸门重扫 ---")
        print(
            "  判据：%s"
            % (
                "、".join(
                    [
                        "焦点人物=%s" % ("/".join(focus_subjects) or "（不限）"),
                        "机器人自己=%s" % ("/".join(self_names) or "（未配）"),
                    ]
                )
            )
        )
        should_not_exist, should_downgrade = gate_scan(
            conn, focus_subjects=focus_subjects, self_names=self_names
        )
        if should_not_exist:
            problems += len(should_not_exist)
            print("  [!] 按现在的闸门**不该入库**：%d 条" % len(should_not_exist))
            for row, reasons in should_not_exist:
                print(
                    "    #%s [%s] %s" % (row["id"], row["kind"], row["content"])
                )
                for reason in reasons:
                    print("        - %s" % reason)
        else:
            print("  不该入库：0 条")
        if should_downgrade:
            print("  [~] 按现在的闸门**该降级为事件**：%d 条" % len(should_downgrade))
            for row, reasons in should_downgrade:
                print(
                    "    #%s [%s] %s" % (row["id"], row["kind"], row["content"])
                )
                for reason in reasons:
                    print("        - %s" % reason)
        else:
            print("  该降级为事件：0 条")

        print("\n结论：%s" % ("发现 %d 处问题" % problems if problems else "结构一致"))
        return 1 if problems else 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审一份 memfade 记忆库")
    parser.add_argument("--db", required=True)
    parser.add_argument("--show-fts", action="store_true", help="打印 FTS 孤儿行内容")
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        help="只记这些主语的人（可重复）。给了才会查「跟要记的那个人没关系」",
    )
    parser.add_argument(
        "--self-name",
        action="append",
        default=[],
        help="机器人自己的名字（可重复）。没给 --focus 时用它兜底",
    )
    args = parser.parse_args(argv)
    return audit(
        args.db,
        args.show_fts,
        focus_subjects=tuple(args.focus),
        self_names=tuple(args.self_name),
    )


if __name__ == "__main__":
    sys.exit(main())
