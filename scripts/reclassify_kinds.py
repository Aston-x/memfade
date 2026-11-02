"""给记忆重新判定种类（episodic / semantic / preference / relation / skill）。

为什么需要：

**种类决定这条记忆能活多久。** 淡化引擎的稳定性是
``base(种类) × (1 + 重要性加成 × 显著性)``——
一次性事件（episodic）的基准是 3 天，稳定事实（semantic）是 30 天，
再乘上重要性的加成，能差出一个数量级。

所以一条「已经买过《影之刃零》」要是被标成 semantic，
它就会以「稳定事实」的寿命留在事实层里，
**几周后还在每次聊天时冒出来**——那正是「总提旧事」。

这个脚本用宿主自己的 LLM 重新判定一遍，并按种类重新夹紧显著性
（一次性事件的显著性天然更低）。

用法（要在 MaiBot 自己的 venv 下跑）：

    python scripts/reclassify_kinds.py \
        --db "G:/AI/许杏玖-MaiBot/MaiBot/data/plugins/memfade.memory/memfade.db" \
        --host-root "G:/AI/许杏玖-MaiBot/MaiBot"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.extract import IMPORTANCE_BOUNDS  # noqa: E402


PROMPT = (
    "下面是若干条关于人的记忆。请给每一条判断它属于哪一类。\n"
    "\n"
    "类别：\n"
    "- episodic：一次性事件。已经做完的事、过去发生的事、一次性的状态。\n"
    "- semantic：稳定事实。长期成立的属性、状态、能力、身份。\n"
    "- preference：喜好或厌恶。\n"
    "- relation：人与人之间的关系。\n"
    "- skill：习惯做法、长期在做的事。\n"
    "\n"
    "判断要点：\n"
    "- 「已经买了X」「上周去了Y」「昨天把Z打包了」→ episodic\n"
    "- 「住在X」「显卡不能同时跑Y」「职业是Z」→ semantic\n"
    "- **分界线：这件事本身还会不会继续成立**。\n"
    "  「买了某款游戏」——买这个动作不会重来 → episodic；\n"
    "  「会写插件把画面逐帧推给我」——这个能力长期在 → semantic。\n"
    "  写成「做了某事」的不一定是 episodic，看那件事是**一次性动作**"
    "还是**长期属性/能力**。\n"
    "- 「喜欢X」「讨厌Y」「最爱X」→ preference\n"
    "- 「称呼对方为X」「是某人的铁子」→ relation\n"
    "- 「每周都X」「习惯X」→ skill\n"
    "\n"
    "只输出 JSON 数组，每一项是 {{\"id\": 编号, \"kind\": \"类别\"}}，编号与输入一致：\n"
    "{facts}"
)


def _clamp(kind: str, value: float) -> float:
    from memfade.types import MemoryKind

    try:
        mk = MemoryKind(kind)
    except ValueError:
        return value
    low, high = IMPORTANCE_BOUNDS.get(mk, (0.0, 1.0))
    return min(high, max(low, float(value)))


async def classify(
    texts: List[str], host_root: str, *, batch_size: int, max_tokens: int
) -> List[Dict[str, Any]]:
    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    from src.services import llm_service  # type: ignore

    from memfade.extract import parse_json_block

    out: List[Dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        facts = "\n".join("%d. %s" % (i, t) for i, t in enumerate(chunk))
        request = llm_service.LLMServiceRequest(
            task_name="memory",
            request_type="memfade.reclassify",
            prompt=PROMPT.format(facts=facts),
            temperature=0.1,
            max_tokens=max_tokens,
        )
        result = await llm_service.generate(request)
        payload = result.to_capability_payload()
        items = parse_json_block(str(payload.get("response") or ""))
        if not isinstance(items, list):
            print("  ! 这一批没解析出 JSON，跳过")
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if 0 <= index < len(chunk) and kind:
                out.append({"offset": start + index, "kind": kind})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="重新判定记忆的种类")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--host-root", default="", help="MaiBot 根目录（用它调宿主 LLM）")
    parser.add_argument("--batch-size", type=int, default=8, help="每次问模型几条")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写库")
    args = parser.parse_args()

    if not args.host_root:
        print("需要 --host-root（MaiBot 根目录，用它调宿主 LLM）")
        return 1

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, kind, importance, content FROM memories WHERE status != 'superseded'"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("库里没有记忆")
        return 0
    print("要重新判定 %d 条" % len(rows))

    texts = [str(r["content"] or "") for r in rows]
    started = time.time()
    results = asyncio.run(
        classify(texts, args.host_root, batch_size=args.batch_size, max_tokens=args.max_tokens)
    )
    if not results:
        print("模型没给出结果，放弃写库")
        return 1

    conn = sqlite3.connect(args.db, timeout=30)
    try:
        changed = 0
        for item in results:
            row = rows[item["offset"]]
            old_kind = str(row["kind"])
            new_kind = item["kind"]
            old_imp = float(row["importance"])
            new_imp = _clamp(new_kind, old_imp)
            if old_kind == new_kind and abs(old_imp - new_imp) < 1e-9:
                continue
            changed += 1
            print(
                "  #%-4d %-10s -> %-10s imp %.2f -> %.2f  %s"
                % (
                    row["id"],
                    old_kind,
                    new_kind,
                    old_imp,
                    new_imp,
                    str(row["content"])[:34],
                )
            )
            if not args.dry_run:
                conn.execute(
                    "UPDATE memories SET kind=?, importance=? WHERE id=?",
                    (new_kind, new_imp, row["id"]),
                )
        if not args.dry_run:
            conn.commit()
        print("=" * 56)
        print(
            "改了 %d 条（耗时 %.1f 秒）%s"
            % (changed, time.time() - started, "（dry-run）" if args.dry_run else "")
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
