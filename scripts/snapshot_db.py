"""把线上记忆库复制一份，给诊断用。

为什么要这一步：**读上下文也是写。**
注入会强化被注入的记忆（强度回升、锚点重置、被想起次数 +1），
而「被想起次数」还参与稳定性的间隔重复成长——
诊断脚本直接对线上库跑一次 ``build_context``，
就等于给那批记忆续命（稳定性可能翻好几倍）。

所以：**诊断一律在副本上做**，线上库只留给维护工具
（整理 / 重整 / 补向量 / 重新判定种类）写。

用法：

    python scripts/snapshot_db.py \
        --db "G:/AI/BotName-MaiBot/MaiBot/data/plugins/memfade.memory/memfade.db"
    # → 打印副本路径：/tmp/memfade-snapshot-20261102-120000.db

然后拿副本路径去跑你的诊断脚本。
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sqlite3
import tempfile


def snapshot(db_path: str, out_dir: str = "") -> str:
    """复制一份库（用 sqlite3 的备份 API，保证一致）。"""

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target_dir = out_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "memfade-snapshot-%s.db" % stamp)
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="复制一份记忆库给诊断用")
    parser.add_argument("--db", required=True, help="线上 memfade.db 路径")
    parser.add_argument("--out-dir", default="", help="副本放哪（默认系统临时目录）")
    args = parser.parse_args()
    if not os.path.exists(args.db):
        print("找不到 %s" % args.db)
        return 1
    path = snapshot(args.db, args.out_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
