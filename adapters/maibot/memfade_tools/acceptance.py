"""验收：一条命令证明「记忆插件单独就能用」。

## 为什么要有这个

「装上就绪」这件事太容易说得含糊。插件的每一环都可能悄悄坏掉：
存储建表、写入、检索、注入、画像、淡化、整理、导出、导入……
任一环坏了，插件照样能加载、界面照样能打开——**但它不记事**。

所以这里跑一遍**完整的本地闭环**，全程**零外部依赖**：
不连 LLM（走规则抽取兜底）、不连嵌入服务（只走词面检索）。
任何一环不对就退出码非 0。

## 用法

    python scripts/acceptance.py            # 用临时库跑一遍
    python scripts/acceptance.py --db <库>  # 顺便体检一个真实的库
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from typing import Any, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemFadeConfig, MemoryKind, Scope  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402
from memfade.types import MemoryDraft  # noqa: E402


class Acceptance:
    """一条条验收项。"""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.passed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            print("  ✓ %s" % name)
            return True
        self.failures.append("%s%s" % (name, ("：" + detail) if detail else ""))
        print("  ✗ %s%s" % (name, ("：" + detail) if detail else ""))
        return False


def run(db_path: str) -> int:
    acc = Acceptance()
    print("=" * 66)
    print("memfade 验收（零配置：不用 LLM、不用嵌入服务）")
    print("库：%s" % db_path)
    print("=" * 66)

    # 1) 引擎能起来（**默认配置**，不改任何东西）
    config = MemFadeConfig.from_dict(
        {"storage": {"db_path": db_path, "enable_vector": False}}
    )
    engine = MemoryEngine(config)
    scope = Scope(platform="qq", chat_type="session", chat_id="acceptance")
    other_scope = Scope(platform="qq", chat_type="session", chat_id="another-session")

    print("\n[1] 起引擎")
    acc.check("引擎就绪", engine is not None)
    acc.check("库文件已建", os.path.exists(db_path))

    # 2) 写入
    print("\n[2] 写入")
    written = engine.remember(
        MemoryDraft(
            content="Alice喜欢辣的食物",
            kind=MemoryKind.PREFERENCE,
            importance=0.8,
            subject="Alice",
            predicate="喜欢",
            object="辣的食物",
        ),
        scope,
    )
    acc.check("写入一条记忆", written is not None and bool(written.id))
    acc.check("统计能数到", engine.stats(scope).get("total", 0) >= 1)

    # 3) 检索（必须真命中）
    print("\n[3] 检索")
    hits = engine.recall("辣的食物", scope, require_match=True)
    acc.check("搜得到刚写的那条", any("辣的食物" in h.memory.content for h in hits))
    miss = engine.recall("显卡驱动", scope, require_match=True)
    acc.check("无关查询不乱命中", len(miss) == 0, "命中了 %d 条" % len(miss))

    # 4) 会话隔离
    print("\n[4] 会话隔离")
    cross = engine.recall("辣的食物", other_scope, require_match=True)
    acc.check("别的会话看不到", len(cross) == 0, "泄漏了 %d 条" % len(cross))

    # 5) 注入
    print("\n[5] 注入")
    ctx = engine.build_context("我喜欢吃什么", scope, reinforce=False)
    acc.check("注入块里有记忆", "辣的食物" in (ctx.text or ""))
    empty = engine.build_context("今天吃什么", scope, reinforce=False)
    acc.check("无关问题不乱注入", "辣的食物" not in (empty.text or ""))

    # 6) 画像
    print("\n[6] 画像")
    profile = engine.retriever.profile(scope.key())
    acc.check("画像里有稳定事实", "辣的食物" in (profile.text or ""))

    # 7) 淡化引擎
    print("\n[7] 淡化")
    from memfade.decay import decay_curve, effective_strength

    curve = decay_curve(MemoryKind.EPISODIC, importance=0.4, recall_count=0, days=90)
    acc.check("一次性事件 90 天后基本退场", curve[-1] < 0.05, "保留率 %.3f" % curve[-1])
    acc.check("强度随时间下降", effective_strength(written, time.time() + 30 * 86400) < 1.0)

    # 8) 整理
    print("\n[8] 整理")
    import asyncio

    report = asyncio.run(engine.consolidate(scope))
    acc.check("整理能跑完", report.scanned >= 1)

    # 9) 导出/导入
    print("\n[9] 导出与导入")
    exported = engine.export(scope)
    acc.check("导出得到条目", len(exported) >= 1)
    acc.check("导出带淡化状态", "anchor_at" in exported[0], "锚点丢了，搬过去会「重新变年轻」")
    target_path = db_path + ".copy"
    target = MemoryEngine(
        MemFadeConfig.from_dict(
            {"storage": {"db_path": target_path, "enable_vector": False}}
        )
    )
    try:
        moved = target.import_memories(exported, scope)
        acc.check("导入回新库", len(moved) >= 1)
        again = target.recall("辣的食物", scope, require_match=True)
        acc.check("搬过去还搜得到", any("辣的食物" in h.memory.content for h in again))
    finally:
        target.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(target_path + suffix)
            except OSError:
                pass

    # 10) 体检
    print("\n[10] 体检")
    report = engine.diagnose(scope)
    acc.check("体检没有发现问题", not report["issues"], str(report["issues"]))

    engine.close()
    print("=" * 66)
    if acc.failures:
        print("验收失败 %d 项：" % len(acc.failures))
        for item in acc.failures:
            print("  · %s" % item)
        return 1
    print("全部通过：%d 项" % acc.passed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="memfade 验收")
    parser.add_argument("--db", default="", help="顺便体检一个真实的库（留空=用临时库）")
    args = parser.parse_args()

    if args.db:
        code = run(args.db)
        return code

    tmp = tempfile.mkdtemp(prefix="memfade-acceptance-")
    try:
        return run(os.path.join(tmp, "memfade.db"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
