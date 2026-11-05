"""给还没有向量的记忆补上向量。

什么时候需要：

* **导入的旧记忆**——它们是直接写进 SQLite 的，绕过了引擎，自然没有向量；
* **早期版本的写入路径**——宿主给的嵌入服务是 async 的，
  而写入路径是同步的；有事件循环时同步取向量会堵住事件循环，
  所以同步路径只能跳过。**结果就是记忆库里一条向量都没有，
  向量检索这条通道等于没开。**

现在引擎里补上了异步补齐的路径（``engine.embed_memories``），
这个脚本负责补**存量**。

用法（要在 MaiBot 自己的 venv 下跑，需要能 import src.*）：

    python scripts/backfill_embeddings.py \
        --db "G:/AI/BotName-MaiBot/MaiBot/data/plugins/memfade.memory/memfade.db" \
        --host-root "G:/AI/BotName-MaiBot/MaiBot"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.store import MemoryStore  # noqa: E402


async def backfill(db_path: str, host_root: str, *, limit: int, batch_pause: float) -> int:
    """补向量。返回补上的条数。"""

    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    from src.services.embedding_service import EmbeddingServiceClient  # type: ignore

    store = MemoryStore(db_path)
    client = EmbeddingServiceClient(task_name="embedding", request_type="memfade.backfill")
    done = 0
    try:
        while True:
            rows = store.list_without_embedding(limit=limit)
            if not rows:
                break
            for row in rows:
                content = str(row["content"] or "")
                if not content.strip():
                    continue
                try:
                    result = await client.embed_text(content)
                except Exception as exc:
                    print(f"  ! #{row['id']} 取向量失败: {exc}")
                    continue
                vector = list(getattr(result, "embedding", []) or [])
                if not vector:
                    print(f"  ! #{row['id']} 返回空向量")
                    continue
                store.set_embedding(int(row["id"]), vector)
                done += 1
                print("  ✓ #%d %s" % (row["id"], content[:44]))
            # 一轮结束再看还有没有剩下的（失败的会在下一轮重试）
            if batch_pause:
                await asyncio.sleep(batch_pause)
            if done and len(rows) < limit:
                break
    finally:
        store.close()
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="给记忆补向量")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--host-root", default="", help="MaiBot 根目录（用它调宿主的嵌入服务）")
    parser.add_argument("--limit", type=int, default=200, help="每轮最多处理多少条")
    parser.add_argument("--batch-pause", type=float, default=0.2, help="每轮之间的间隔（秒）")
    args = parser.parse_args()

    if not args.host_root:
        print("需要 --host-root（MaiBot 根目录，用它调宿主的嵌入服务）")
        return 1

    started = time.time()
    count = asyncio.run(
        backfill(args.db, args.host_root, limit=args.limit, batch_pause=args.batch_pause)
    )
    print("=" * 56)
    print("补完 %d 条，耗时 %.1f 秒" % (count, time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
