"""把宿主内置的旧记忆系统（A_Memorix）退场，让 memfade 当唯一的记忆板块。

## 为什么必须退场，而不是「关掉注入就行」

旧记忆系统和新插件如果同时开着，会出三件事：

1. **两套记忆并行写**：旧的自动写回「人物事实」与「聊天摘要」，
    新的也抽取事实——同一个人的同一件事，两个库各存一份，
    而且旧的**没有淡化引擎**（段落一旦写进去就赖着不走，
    这正是「总提旧事」的根源）。
2. **模型能从旧的检索**：宿主给麦麦挂了一个「检索长期记忆」的工具，
    旧的库里有一条「BotName称呼聊天助手为"大肥鱼"」，
    planner 就把它当事实用了——而这条是**主人在引用机器人时**被记下来的。
3. **画像互相覆盖**：旧的会刷新「人物画像快照」，
    新的也生成「关于对方」的画像块，两块同时进 prompt。

所以要退场的是**写**与**读**两端，不是只关注入。

## 退场清单

### 一、配置（`config/bot_config.toml`）

```toml
[a_memorix.plugin]
enabled = false                                  # 总开关

[a_memorix.integration]
enable_memory_query_tool = false                # 模型不再能从旧库检索
enable_person_profile_query_tool = false        # 同上（画像查询）
person_fact_writeback_enabled = false           # 不再自动抽取人物事实
chat_summary_writeback_enabled = false          # 不再自动写回聊天摘要
fuzzy_modify_enabled = false                    # 记忆修正走新插件的整理/忘掉
```

### 二、数据（`data/a-memorix/metadata/metadata.db`）

清掉文本记忆：段落、事实断言、事实证据、事实变迁、关系、实体、事件、
人物画像快照、外部记忆引用、聊天摘要检查点、画像刷新队列。
图片资产与向量索引**留着**——那是图片记忆功能，不是这次要清的。

## 用法

    python scripts/retire_legacy_memory.py --host-root "G:/AI/BotName-MaiBot/MaiBot"
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sqlite3
import sys
from typing import List, Tuple


TEXT_MEMORY_TABLES: Tuple[str, ...] = (
    "paragraphs",
    "fact_claims",
    "fact_evidence",
    "fact_transitions",
    "relations",
    "deleted_relations",
    "paragraph_relations",
    "paragraph_entities",
    "entities",
    "episodes",
    "episode_paragraphs",
    "episode_rebuild_sources",
    "person_profile_snapshots",
    "person_profile_refresh_queue",
    "external_memory_refs",
    "chat_summary_checkpoints",
    "memory_fuzzy_modify_plans",
    "delete_operations",
    "delete_operation_items",
    "storage_cleanup_jobs",
)


def _backup(db_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = os.path.join(out_dir, "a-memorix-metadata-%s.db" % stamp)
    shutil.copy2(db_path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="让旧记忆系统退场")
    parser.add_argument("--host-root", required=True, help="MaiBot 根目录")
    parser.add_argument("--backup-dir", default="", help="备份放哪（默认 <host>/data/_legacy_backup）")
    parser.add_argument("--dry-run", action="store_true", help="只看会删什么")
    args = parser.parse_args()

    host = os.path.abspath(args.host_root)
    db_path = os.path.join(host, "data", "a-memorix", "metadata", "metadata.db")
    if not os.path.exists(db_path):
        print("找不到 %s" % db_path)
        return 1
    backup_dir = args.backup_dir or os.path.join(host, "data", "_legacy_backup")

    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()

    targets = [t for t in TEXT_MEMORY_TABLES if t in existing]
    print("旧记忆库：%s" % db_path)
    print("准备清 %d 张表：" % len(targets))

    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    for table in targets:
        try:
            count = conn.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0]
        except Exception:
            count = 0
        if count:
            print("  %-34s %d 行" % (table, count))
    conn.close()

    if args.dry_run:
        print("\n--dry-run：没有动数据")
        return 0

    backup = _backup(db_path, backup_dir)
    print("\n已备份：%s" % backup)

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table in targets:
            try:
                conn.execute('DELETE FROM "%s"' % table)
            except Exception as exc:
                print("  ! 清 %s 失败：%s" % (table, exc))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    left = 0
    for table in targets:
        try:
            left += conn.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0]
        except Exception:
            pass
    conn.close()
    print("清完，剩余 %d 行。" % left)
    print("\n还要改配置（config/bot_config.toml）：")
    print("  [a_memorix.plugin] enabled = false")
    print("  [a_memorix.integration] enable_memory_query_tool = false")
    print("  [a_memorix.integration] enable_person_profile_query_tool = false")
    print("  [a_memorix.integration] person_fact_writeback_enabled = false")
    print("  [a_memorix.integration] chat_summary_writeback_enabled = false")
    print("  [a_memorix.integration] fuzzy_modify_enabled = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
