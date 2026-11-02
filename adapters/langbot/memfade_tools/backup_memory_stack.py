"""备份一整套记忆（不止 memfade）。

## 为什么备份的是「一整套」

一个聊天机器人身上可能同时挂着好几套记忆：
memfade（自建）、A_Memorix（宿主自带的记忆核心）、
宿主的 person_info（画像）、印象卡插件的笔记、好感度插件的状态…
只备份其中一套，另一套的账就丢了；将来要回滚/对比/迁移时，
手里的数据是残的。

所以这里按「记忆栈」备份：

* **原样复制**——二进制库、JSON 状态，一个字节不改；
* **再导一份可读的**——memfade 的记忆导出成 JSON，
   宿主的 person_info 导出成 JSON，方便直接看、直接迁移；
* **写一份清单**——每个文件多大、什么时候的、怎么还原。

## 用法

    python scripts/backup_memory_stack.py \
        --host-root "G:/AI/许杏玖-MaiBot/MaiBot" \
        --out "B:/AI/Backup/记忆清空前-20261102-220000"
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _copy(src: str, dst: str) -> Tuple[int, int]:
    """复制一个文件或目录，返回（文件数, 字节数）。"""

    if not os.path.exists(src):
        return (0, 0)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        files = [os.path.join(root, name) for root, _, names in os.walk(dst) for name in names]
        return (len(files), sum(os.path.getsize(f) for f in files))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return (1, os.path.getsize(dst))


def _export_memfade(db_path: str, out_path: str) -> int:
    """把 memfade 的记忆导出成 JSON（可读、可迁移）。"""

    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, scope_key, kind, content, subject, predicate, object, "
            "importance, strength, anchor_at, status, created_at, updated_at, "
            "recall_count, pinned, cross_scope, source, meta "
            "FROM memories ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    payload = {
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "count": len(rows),
        "memories": [dict(row) for row in rows],
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return len(rows)


def _export_table(db_path: str, table: str, out_path: str) -> int:
    """把宿主库里的一张表导出成 JSON。"""

    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM "%s"' % table).fetchall()
    except Exception:
        return 0
    finally:
        conn.close()
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"exported_at": datetime.datetime.now().isoformat(timespec="seconds"), "rows": [dict(r) for r in rows]},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="备份一整套记忆")
    parser.add_argument("--host-root", required=True, help="MaiBot 根目录")
    parser.add_argument("--out", required=True, help="备份目录")
    parser.add_argument(
        "--skip-host-db",
        action="store_true",
        help="不备份 MaiBot.db（它有 100MB，多半是消息日志）",
    )
    args = parser.parse_args()

    host = os.path.abspath(args.host_root)
    out = os.path.abspath(args.out)
    data = os.path.join(host, "data")
    os.makedirs(out, exist_ok=True)

    manifest: List[Dict[str, Any]] = []

    # 1) 原样复制：记忆相关的库与状态
    targets = [
        ("memfade（自建记忆插件）", os.path.join(data, "plugins", "memfade.memory")),
        ("a-memorix（旧记忆核心）", os.path.join(data, "a-memorix")),
        ("mai-affection（好感度）", os.path.join(data, "plugins", "com.local.mai-affection")),
        ("social-state（社交状态）", os.path.join(data, "plugins", "github.ZhaoCang-QWQ.social-state")),
        ("better-diary（日记）", os.path.join(data, "plugins", "org.orge-8.better-diary")),
        ("life-frequency（生活状态）", os.path.join(data, "plugins", "org.orge-8.life-frequency")),
    ]
    if not args.skip_host_db:
        targets.append(("MaiBot.db（消息 + 宿主画像）", os.path.join(data, "MaiBot.db")))

    for label, src in targets:
        if not os.path.exists(src):
            manifest.append({"label": label, "path": src, "skipped": "不存在"})
            continue
        dst = os.path.join(out, "raw", os.path.basename(src.rstrip("/\\")))
        count, size = _copy(src, dst)
        manifest.append(
            {
                "label": label,
                "source": src,
                "backup": dst,
                "files": count,
                "bytes": size,
            }
        )
        print("  复制 %-28s %2d 个文件 %8.1f KB" % (label, count, size / 1024))

    # 2) 可读导出
    os.makedirs(os.path.join(out, "export"), exist_ok=True)
    memfade_db = os.path.join(data, "plugins", "memfade.memory", "memfade.db")
    if os.path.exists(memfade_db):
        count = _export_memfade(memfade_db, os.path.join(out, "export", "memfade_memories.json"))
        print("  导出 memfade 记忆 %d 条 → export/memfade_memories.json" % count)
    host_db = os.path.join(data, "MaiBot.db")
    if os.path.exists(host_db):
        count = _export_table(host_db, "person_info", os.path.join(out, "export", "host_person_info.json"))
        print("  导出宿主画像 %d 行 → export/host_person_info.json" % count)

    # 3) 清单 + 还原说明
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "host_root": host,
                "items": manifest,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    readme = os.path.join(out, "README.md")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write(
            "# 记忆栈备份\n\n"
            "- 时间：%s\n"
            "- 来源：%s\n\n"
            "## 里面有什么\n\n"
            "```\n"
            "raw/     原样复制的库与状态（要还原就照原路径放回去）\n"
            "export/  可读的 JSON 导出（memfade 记忆、宿主画像）\n"
            "manifest.json  每个文件的来源、大小、备份位置\n"
            "```\n\n"
            "## 怎么还原\n\n"
            "1. 停掉宿主（否则 SQLite 正在写，复制回去会坏）；\n"
            "2. 把 `raw/` 里的内容按 manifest 的 `source` 原路径放回；\n"
            "3. 起宿主。\n\n"
            "## 注意\n\n"
            "memfade 的记忆只能靠 `export/memfade_memories.json` 重放"
            "（库里还有强度、锚点等衰减状态，直接导出会丢）——\n"
            "重放请用 `scripts/import_a_memorix.py` 那套规范化流程，"
            "或直接写库（记得同步重算 `fts_tokens`）。\n"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), host)
        )
    print("\n备份完成：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
