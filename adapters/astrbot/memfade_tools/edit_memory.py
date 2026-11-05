"""按清单修记忆：改内容、改种类、删除、补充。

## 为什么需要这个

抽取难免出错：把「今天只有一节课」记成「每天只有一节课」，
把一次性的吩咐记成长期偏好。出错之后总得有人收拾：

* 整库清空——代价太大，把对的也一起清了；
* 改配置、加提示词——只能防下一次，管不了已经躺在库里的那条；
* **按清单一条条改**——代价最小，还能留下「改了什么」的记录。

插件面板与聊天命令（``/记忆 修正``）走的是同一套核心逻辑
（``MemoryEngine.edit_memory``），这个脚本是给运维用的批量版本。

## 怎么用

    # 1) 先备份（脚本自己也会备一份，但手工备一次更稳）
    # 2) 写一份清单
    cat > patch.json <<'JSON'
    [
      {"id": 125, "content": "Alice只有起晚了才不吃早饭直接去早自习",
       "kind": "semantic", "importance": 0.7,
       "subject": "Alice", "predicate": "起晚了才", "object": "不吃早饭直接去早自习",
       "quote": "只有起晚了才会不吃早餐直接去"},
      {"delete": 127}
    ]
    JSON

    # 3) 先看会改什么，再真改
    python scripts/edit_memory.py --db <memfade.db> --patch patch.json --dry-run
    python scripts/edit_memory.py --db <memfade.db> --patch patch.json --apply

备份用 SQLite 的 backup API（不是拷 .db 文件）——WAL 模式下
最近的事务还在 ``-wal`` 里，只拷 .db 会丢掉（实测少拷 7 条记忆）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.config import MemFadeConfig  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402
from memfade.types import Scope  # noqa: E402


def backup(db_path: str) -> str:
    """用 SQLite 的 backup API 备份（WAL 模式下拷文件会丢数据）。"""

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = "%s.bak-%s" % (db_path, stamp)
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def _scope_of(engine: MemoryEngine, key: str) -> Scope:
    """把作用域键还原成 Scope。

    ★ 用核心的 ``Scope.from_key``（``split(":", 2)`` 语义），不要手写拆键：
      手写版本把 platform 写死成 ``qq``，别的宿主（astrbot 的
      ``astrbot:group_user:1001#u1``）会被还原成完全错误的作用域，
      于是「改记忆」改到了另一个 scope 上去。
    """

    raw = str(key or "").strip()
    if not raw:
        return Scope(platform="qq", chat_type="session", chat_id="")
    return Scope.from_key(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="按清单修记忆库")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--patch", required=True, help="清单 JSON 文件（数组）")
    parser.add_argument("--dry-run", action="store_true", help="只看会改什么，不写库")
    parser.add_argument("--apply", action="store_true", help="真改（会先自动备份）")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print("找不到 %s" % args.db, file=sys.stderr)
        return 1
    with open(args.patch, "r", encoding="utf-8") as handle:
        try:
            items = json.load(handle)
        except Exception as exc:
            print("清单不是合法 JSON：%s" % exc, file=sys.stderr)
            return 1
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        print("清单应该是一个数组", file=sys.stderr)
        return 1

    config = MemFadeConfig.from_dict(
        {"storage": {"db_path": args.db, "busy_timeout_ms": 15000}}
    )
    engine = MemoryEngine(config)
    stats = {"edited": 0, "deleted": 0, "added": 0, "skipped": 0}
    try:
        for item in items:
            if not isinstance(item, dict):
                stats["skipped"] += 1
                continue
            if "delete" in item:
                mid = int(item["delete"])
                target = engine.store.get(mid)
                if target is None:
                    print("  跳过：没有 #%d" % mid)
                    stats["skipped"] += 1
                    continue
                print("  删 #%d %s" % (mid, target.content))
                if args.apply:
                    engine.forget(mid, hard=True)
                stats["deleted"] += 1
                continue
            if "add" in item:
                spec = dict(item["add"] or {})
                scope = _scope_of(engine, str(item.get("scope") or spec.pop("scope", "")))
                content = str(spec.pop("content", "") or "").strip()
                if not content:
                    stats["skipped"] += 1
                    continue
                from memfade.types import MemoryDraft, MemoryKind

                kind = MemoryKind(str(spec.pop("kind", "semantic") or "semantic"))
                importance = float(spec.pop("importance", 0.7) or 0.7)
                quote = str(spec.pop("quote", "") or "")
                print("  加：%s（%s，%s）" % (content, kind, scope.key()))
                if args.apply:
                    draft = MemoryDraft(
                        content=content,
                        kind=kind,
                        importance=importance,
                        subject=str(spec.pop("subject", "") or ""),
                        predicate=str(spec.pop("predicate", "") or ""),
                        object=str(spec.pop("object", "") or ""),
                        source="manual",
                        meta={"quote": quote} if quote else {},
                    )
                    written = engine.remember(draft, scope)
                    if written is None:
                        print("    ⚠ 没写进去（低于门槛或与已有记忆重复）")
                        stats["skipped"] += 1
                        continue
                stats["added"] += 1
                continue
            mid = int(item.get("id") or 0)
            if not mid:
                stats["skipped"] += 1
                continue
            target = engine.store.get(mid)
            if target is None:
                print("  跳过：没有 #%d" % mid)
                stats["skipped"] += 1
                continue
            changes = {
                key: item[key]
                for key in ("content", "kind", "importance", "subject", "predicate", "object", "quote")
                if key in item
            }
            print("  改 #%d %s" % (mid, target.content))
            for key, value in changes.items():
                print("      %s: %r -> %r" % (key, getattr(target, key, None), value))
            if args.apply:
                updated = engine.edit_memory(mid, **changes)
                if updated is None:
                    print("    ⚠ 改写失败")
                    stats["skipped"] += 1
                    continue
            stats["edited"] += 1
    finally:
        engine.close()

    if not args.apply:
        print("（--dry-run：没有动库）")
    else:
        print("已备份：%s" % backup(args.db))
    print(
        "改 %d 条 / 删 %d 条 / 加 %d 条 / 跳过 %d 条"
        % (stats["edited"], stats["deleted"], stats["added"], stats["skipped"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
