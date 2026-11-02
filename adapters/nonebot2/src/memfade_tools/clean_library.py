"""清理记忆库：把被污染的条目挑出去，把同一件事合并成一条。

## 为什么需要这个

记忆库会被几类东西污染：

1. **机器人自己的事**——主语是机器人自己（它看过什么番、收藏了什么）。
    记忆库是「关于对方」的，记自己的事没有意义，还会让机器人
    认领主人的记忆（真事：主人引用机器人一句「你昨晚一口气刷完Re」，
    库里就多了一条「许杏玖昨晚一口气刷完了Re」）。
2. **公共知识与资料碎片**——《牧神记5》哪天更新、第18届蓝桥杯的赛制。
    这些是知识，不是「关于某个人」的记忆。
3. **同一件事的好几种说法**——「季怡已经购买《影之刃零》」被记了八遍。
4. **被否认或收回的事**——主人明说过「我没跟你说过我要报名蓝桥杯」。

## 怎么用

    # 先看会改什么（不动库）
    python scripts/clean_library.py --db <memfade.db> --dry-run

    # 真删（脚本会先自动备份一份）
    python scripts/clean_library.py --db <memfade.db> --apply

    # 手工指定留哪几条 / 删哪几条
    python scripts/clean_library.py --db <memfade.db> --dry-run \
        --keep 50,51 --drop 106

全文索引（memories_fts）跟 memories 有触发器联动，删条目会自动同步，
不用手动重建。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.extract import PERSON_ROLES  # noqa: E402
from memfade.textutil import fts_text, similarity  # noqa: E402


DEFAULT_SELF_NAMES = ("许杏玖",)
DEFAULT_FOCUS = ("季怡",)
DEFAULT_PARTICIPANTS = (
    "季怡",
    "鸡柳",
    "洞",
    "雨",
    "玖儿",
    "霜",
    "早樱千雪а",
)


def _load(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, scope_key, kind, content, subject, predicate, object, "
            "importance, strength, recall_count, pinned, status, source "
            "FROM memories ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _rank(item: Dict[str, Any]) -> tuple:
    """留哪一条：召回多的、置顶的、重要的、完整的优先；一样就留短的。"""

    return (
        int(item.get("recall_count") or 0),
        int(item.get("pinned") or 0),
        float(item.get("importance") or 0.0),
        float(item.get("strength") or 0.0),
        -len(item.get("content") or ""),
        int(item.get("id") or 0),
    )


def plan(
    rows: Sequence[Dict[str, Any]],
    *,
    self_names: Sequence[str],
    focus: Sequence[str],
    participants: Sequence[str],
    people_roles: Sequence[str],
    threshold: float,
    keep_ids: Sequence[int],
    drop_ids: Sequence[int],
    only_ids: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    """给每一条算一个处置：keep / merge / drop，并写明理由。"""

    self_set = set(self_names)
    focus_set = set(focus)
    people = set(participants) | set(people_roles) | focus_set
    keep_set = set(keep_ids)
    drop_set = set(drop_ids)
    only_set = set(only_ids)

    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        content = (row.get("content") or "").strip()
        subject = (row.get("subject") or "").strip()
        action = "keep"
        reason = ""
        if row["id"] in drop_set:
            action, reason = "drop", "手工指定删除"
        elif only_set and row["id"] not in only_set:
            action, reason = "drop", "不在保留名单里"
        elif row["id"] in keep_set:
            action, reason = "keep", "手工指定保留"
        elif subject and subject in self_set:
            action, reason = "drop", "主语是机器人自己"
        elif focus_set and subject and subject not in focus_set:
            action, reason = "drop", "主语不是主人（%s）" % subject
        elif not subject and people and not content.startswith(tuple(people_roles)):
            action, reason = "drop", "没有主语，也不是关于人的事"
        out.append({**item, "action": action, "reason": reason, "content": content})

    # 近重复：同一件事的多种说法，只留一条。
    alive = [item for item in out if item["action"] == "keep"]
    for index, item in enumerate(alive):
        for other in alive[index + 1 :]:
            score = similarity(item["content"], other["content"])
            if score < threshold:
                continue
            keeper, loser = (item, other) if _rank(item) >= _rank(other) else (other, item)
            for entry in out:
                if entry["id"] == loser["id"]:
                    entry["action"] = "merge"
                    entry["reason"] = "与 #%d 说的是同一件事（相似度 %.2f）" % (keeper["id"], score)
                    entry["merge_into"] = keeper["id"]

    # 同一件事的另一种重复形态：主语一样、宾语也几乎一样，
    # 但措辞差得远（文本 Dice 抓不住）。
    # 「季怡已经购买《影之刃零》」「截至昨天，季怡已可游玩《影之刃零》」
    # 就是这一类——八条里靠它能并掉六条。
    alive = [item for item in out if item["action"] == "keep"]
    for index, item in enumerate(alive):
        for other in alive[index + 1 :]:
            if item.get("subject") != other.get("subject"):
                continue
            left = (item.get("object") or "").strip()
            right = (other.get("object") or "").strip()
            if not left or not right:
                continue
            if left == right or similarity(left, right) >= 0.8:
                keeper, loser = (item, other) if _rank(item) >= _rank(other) else (other, item)
                for entry in out:
                    if entry["id"] == loser["id"]:
                        entry["action"] = "merge"
                        entry["reason"] = "与 #%d 说的是同一件事（主语与宾语相同）" % keeper["id"]
                        entry["merge_into"] = keeper["id"]
    return out


def _backup(db_path: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = "%s.bak-%s" % (db_path, stamp)
    shutil.copy2(db_path, target)
    return target


def _apply(db_path: str, items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """按计划写库：删掉的删掉，合并的记进 meta.merged_from。"""

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    stats = {"deleted": 0, "merged": 0, "rewritten": 0, "reactivated": 0}
    try:
        conn.execute("BEGIN IMMEDIATE")
        for item in items:
            if item["action"] == "drop":
                conn.execute("DELETE FROM memories WHERE id = ?", (item["id"],))
                stats["deleted"] += 1
            elif item["action"] == "merge":
                keeper = item.get("merge_into")
                row = conn.execute("SELECT meta FROM memories WHERE id = ?", (keeper,)).fetchone()
                meta: Dict[str, Any] = {}
                if row and row["meta"]:
                    try:
                        meta = json.loads(row["meta"])
                    except Exception:
                        meta = {}
                history = list(meta.get("merged_from") or [])
                history.append({"id": item["id"], "content": item["content"][:200]})
                meta["merged_from"] = history[-20:]
                meta["merged_count"] = int(meta.get("merged_count") or 0) + 1
                conn.execute(
                    "UPDATE memories SET meta = ? WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), keeper),
                )
                conn.execute("DELETE FROM memories WHERE id = ?", (item["id"],))
                stats["merged"] += 1
            elif item.get("rewrite"):
                content = item["rewrite"]
                subject = item.get("rewrite_subject") or item.get("subject") or ""
                predicate = item.get("rewrite_predicate") or item.get("predicate") or ""
                object_value = item.get("rewrite_object") or item.get("object") or ""
                conn.execute(
                    "UPDATE memories SET content = ?, subject = ?, predicate = ?, object = ?, "
                    "fts_tokens = ?, status = 'active', superseded_by = NULL WHERE id = ?",
                    (
                        content,
                        subject,
                        predicate,
                        object_value,
                        fts_text(content),
                        item["id"],
                    ),
                )
                stats["rewritten"] += 1
            elif item.get("status") != "active":
                # 保留名单里的条目如果被旧版本顶掉了（status=superseded），
                # 白名单说了算：把它放回 active，不然它不会被召回。
                conn.execute(
                    "UPDATE memories SET status = 'active', superseded_by = NULL WHERE id = ?",
                    (item["id"],),
                )
                stats["reactivated"] += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return stats


def _purge(db_path: str) -> Dict[str, int]:
    """清空整个记忆库（全文索引有触发器，会跟着删）。"""

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.execute("DELETE FROM memories")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"deleted": int(total)}


def _ids(text: str) -> List[int]:
    return [int(x) for x in str(text or "").replace(" ", "").split(",") if x]


def _rewrite_map(text: str) -> Dict[int, str]:
    """解析 ``--rewrite 5=季怡的token plan当前用量大概达到一亿``。"""

    out: Dict[int, str] = {}
    for chunk in str(text or "").split("||"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        raw_id, _, content = chunk.partition("=")
        try:
            out[int(raw_id.strip())] = content.strip()
        except ValueError:
            continue
    return out


def _triple_map(text: str) -> Dict[int, Dict[str, str]]:
    """解析 ``--triple 5=季怡|用量|大概达到一亿``。"""

    out: Dict[int, Dict[str, str]] = {}
    for chunk in str(text or "").split("||"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        raw_id, _, rest = chunk.partition("=")
        parts = [p.strip() for p in rest.split("|")]
        if len(parts) < 3:
            continue
        try:
            out[int(raw_id.strip())] = {
                "subject": parts[0],
                "predicate": parts[1],
                "object": parts[2],
            }
        except ValueError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="清理 memfade 记忆库")
    parser.add_argument("--db", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只看会改什么，不写库")
    parser.add_argument("--apply", action="store_true", help="真删（会先自动备份）")
    parser.add_argument(
        "--self-names",
        default=",".join(DEFAULT_SELF_NAMES),
        help="机器人自己的名字（逗号分隔）",
    )
    parser.add_argument(
        "--focus",
        default=",".join(DEFAULT_FOCUS),
        help="只保留这些主语（逗号分隔，空=不限制）",
    )
    parser.add_argument(
        "--participants",
        default=",".join(DEFAULT_PARTICIPANTS),
        help="这次对话里出现过的人（逗号分隔）",
    )
    parser.add_argument("--threshold", type=float, default=0.78, help="近重复的相似度门槛")
    parser.add_argument("--keep", default="", help="手工指定保留的 id（逗号分隔）")
    parser.add_argument("--drop", default="", help="手工指定删除的 id（逗号分隔）")
    parser.add_argument(
        "--only",
        default="",
        help="白名单模式：只保留这些 id，其余全删（逗号分隔）",
    )
    parser.add_argument(
        "--rewrite",
        default="",
        help="改写内容：id=新内容，多条用 || 分隔（例：5=季怡的token plan用量大概达到一亿）",
    )
    parser.add_argument(
        "--triple",
        default="",
        help="改写三元组：id=主语|谓语|宾语，多条用 || 分隔",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="清空整个记忆库（不留任何条目）",
    )
    args = parser.parse_args()

    rows = _load(args.db)
    if args.purge:
        if not args.apply:
            print("会清空 %d 条记忆（--apply 才真删）" % len(rows))
            for item in rows[:20]:
                print("  #%-4s %s" % (item["id"], (item["content"] or "")[:40]))
            if len(rows) > 20:
                print("  … 还有 %d 条" % (len(rows) - 20))
            return 0
        backup = _backup(args.db)
        print("已备份：%s" % backup)
        stats = _purge(args.db)
        print("已清空 %d 条记忆。" % stats["deleted"])
        return 0
    rewrites = _rewrite_map(args.rewrite)
    triples = _triple_map(args.triple)
    # 改写要先落地：规则是看着内容与主语判断的，
    # 先改再判，改写过的条目才不会又被当成「主语不对」删掉。
    for row in rows:
        if row["id"] in rewrites:
            row["content"] = rewrites[row["id"]]
        if row["id"] in triples:
            row["subject"] = triples[row["id"]]["subject"]
            row["predicate"] = triples[row["id"]]["predicate"]
            row["object"] = triples[row["id"]]["object"]
    items = plan(
        rows,
        self_names=[n.strip() for n in args.self_names.split(",") if n.strip()],
        focus=[n.strip() for n in args.focus.split(",") if n.strip()],
        participants=[n.strip() for n in args.participants.split(",") if n.strip()],
        people_roles=PERSON_ROLES,
        threshold=args.threshold,
        keep_ids=_ids(args.keep),
        drop_ids=_ids(args.drop),
        only_ids=_ids(args.only),
    )
    for item in items:
        if item["id"] in rewrites:
            item["rewrite"] = rewrites[item["id"]]
        if item["id"] in triples:
            item["rewrite_subject"] = triples[item["id"]]["subject"]
            item["rewrite_predicate"] = triples[item["id"]]["predicate"]
            item["rewrite_object"] = triples[item["id"]]["object"]

    counts: Dict[str, int] = {}
    for item in items:
        counts[item["action"]] = counts.get(item["action"], 0) + 1
    print("=" * 72)
    print(
        "共 %d 条：%s"
        % (len(items), "，".join("%s %d" % (k, v) for k, v in sorted(counts.items())))
    )
    print("=" * 72)
    for item in items:
        mark = {"keep": "留", "merge": "并", "drop": "删"}[item["action"]]
        if item.get("rewrite"):
            mark = "改"
        note = ("  ← " + item["reason"]) if item["reason"] else ""
        shown = item.get("rewrite") or item["content"]
        print("[%s] #%-4d %s%s" % (mark, item["id"], (shown or "")[:40], note))

    if args.apply and not args.dry_run:
        backup = _backup(args.db)
        print("\n已备份：%s" % backup)
        stats = _apply(args.db, items)
        print(
            "已删除 %d 条，合并 %d 条，改写 %d 条，恢复 %d 条。"
            % (stats["deleted"], stats["merged"], stats["rewritten"], stats["reactivated"])
        )
    else:
        print("\n--dry-run：没有写库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
