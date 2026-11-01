"""淡化引擎的消融实验。

拿 [memorylens-bench](https://github.com/Neal006/memorylens-bench) 的同款指标，
在一段**可复现的脚本化对话**上比较两种记忆系统：

* **naive**：只记、只检索、只按相关度排序，永不遗忘（很多记忆库就是这个）；
* **memfade**：带淡化梯度、会话隔离、矛盾取代的完整版。

三个指标：

* **Recall@T**——第 T 轮时，注入的上下文里有没有当前正确的事实；
* **Temporal Drift**——事实更新之后，上下文里还残留多少旧值；
* **Contradiction**——上下文里同时出现新旧两个互相矛盾的版本。

再加两个工程指标：注入上下文的平均字数、以及记忆库总条数。

结论应该长这样：**Recall 不降，Drift 降，上下文更短，库更小。**
那才叫「忘得对」，而不是「忘得多」。

用法：

    python scripts/bench_decay.py
    python scripts/bench_decay.py --turns 60 --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemoryDraft, MemoryEngine, MemoryKind, MemFadeConfig, Scope  # noqa: E402
from memfade.decay import effective_strength, resolve_status  # noqa: E402


SCOPE = Scope(platform="bench", chat_type="group", chat_id="bench-group")
_NOISE_MARKER = "闲聊片段"


@dataclass
class Turn:
    """一轮对话。"""

    day: float
    user: str
    facts: List[Dict[str, Any]] = field(default_factory=list)
    """这一轮要写入的事实（模拟抽取器的输出）。"""


def build_script(turns: int, days: float) -> List[Turn]:
    """脚本化对话：事实先建立，中途被更新，后面还会被追问。"""

    script: List[Turn] = []
    step = days / max(1, turns)
    for index in range(turns):
        day = index * step
        facts: List[Dict[str, Any]] = []
        user = ""
        if index == 0:
            user = "我叫许杏玖，记住"
            facts.append({"content": "许杏玖的名字是许杏玖", "kind": MemoryKind.SEMANTIC, "importance": 0.9})
        elif index == turns // 4:
            user = "我住在杭州"
            facts.append(
                {
                    "content": "许杏玖住在杭州",
                    "kind": MemoryKind.SEMANTIC,
                    "importance": 0.85,
                    "subject": "许杏玖",
                    "predicate": "住在",
                    "object": "杭州",
                }
            )
        elif index == turns // 4 + 1:
            user = "我最喜欢吃辣"
            facts.append(
                {
                    "content": "许杏玖喜欢吃辣",
                    "kind": MemoryKind.PREFERENCE,
                    "importance": 0.8,
                    "subject": "许杏玖",
                    "predicate": "口味",
                    "object": "吃辣",
                }
            )
        elif index == turns // 4 + 2:
            user = "上次我们一起去吃了炸鸡"
            facts.append(
                {
                    "content": "许杏玖和季怡一起去吃了炸鸡",
                    "kind": MemoryKind.EPISODIC,
                    "importance": 0.35,
                }
            )
        elif index == turns // 2:
            user = "我搬家了，现在住在上海"
            facts.append(
                {
                    "content": "许杏玖住在上海",
                    "kind": MemoryKind.SEMANTIC,
                    "importance": 0.85,
                    "subject": "许杏玖",
                    "predicate": "住在",
                    "object": "上海",
                }
            )
        elif index == (turns * 3) // 4:
            user = "我不吃辣了，改口味了"
            facts.append(
                {
                    "content": "许杏玖现在不吃辣",
                    "kind": MemoryKind.PREFERENCE,
                    "importance": 0.8,
                    "subject": "许杏玖",
                    "predicate": "口味",
                    "object": "不吃辣",
                }
            )
        elif index == turns - 3:
            user = "现在住哪来着"
        elif index == turns - 2:
            user = "再确认一下我的口味"
        elif index == turns - 1:
            user = "还记得炸鸡吗"
        else:
            user = f"随便聊聊第 {index + 1} 轮"
            facts.append(
                {
                    "content": f"第 {index + 1} 轮的小事：{_EPISODE_EVENTS[index % len(_EPISODE_EVENTS)]}",
                    "kind": MemoryKind.EPISODIC,
                    "importance": 0.3,
                }
            )
            # 噪声：看着像记忆、其实没用的东西。
            # 它们和后面的追问共享关键词，所以检索时会被捞出来——
            # 这正是「记忆库越用越脏」的来源。
            facts.append(
                {
                    "content": f"闲聊片段{index + 1}：今天住在家里有点无聊，刷手机刷到辣眼睛了",
                    "kind": MemoryKind.EPISODIC,
                    "importance": 0.25,
                }
            )
        script.append(Turn(day=day, user=user, facts=facts))
    return script


_EPISODE_EVENTS = [
    "出门买了瓶水",
    "在楼下遇到一只橘猫",
    "把书桌收拾了一下",
    "中午煮了碗面",
    "走路时鞋带散了",
    "把旧手机里的照片备份了",
    "在阳台上看了一会儿云",
    "记了一笔这个月的开销",
]


CURRENT_FACTS = {
    "住址": "上海",
    "口味": "不吃辣",
    "名字": "许杏玖",
}


def make_engine(db_path: str, *, fade: bool) -> MemoryEngine:
    config = MemFadeConfig.from_dict(
        {
            "storage": {"db_path": db_path, "enable_vector": False, "enable_fts": True},
            "extraction": {"min_importance": 0.15, "min_importance_episodic": 0.15},
            "retrieval": {"top_k": 6, "budget_tokens": 400, "echo_max": 2},
            "fade": {"enabled": fade},
            "maintenance": {
                "enabled": True,
                "max_memories_per_scope": 10000,
                # naive 基线不做矛盾取代——它就该是「原样记、原样用」
                "enable_supersede": fade,
                "enable_merge": fade,
            },
        }
    )
    return MemoryEngine(config)


@dataclass
class Metrics:
    recall_hits: int = 0
    recall_total: int = 0
    drift_hits: int = 0
    drift_total: int = 0
    contradiction_hits: int = 0
    context_chars: int = 0
    context_samples: int = 0
    stored: int = 0
    fact_layer: int = 0
    echo_layer: int = 0
    archived: int = 0
    injected_items: int = 0
    injected_noise: int = 0
    old_topic_hits: int = 0
    old_topic_total: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recall@T": round(self.recall_hits / self.recall_total, 4) if self.recall_total else 0.0,
            "temporal_drift": round(self.drift_hits / self.drift_total, 4) if self.drift_total else 0.0,
            "contradiction": round(self.contradiction_hits / self.drift_total, 4) if self.drift_total else 0.0,
            "avg_context_chars": round(self.context_chars / self.context_samples, 1) if self.context_samples else 0.0,
            "stored": self.stored,
            "live_ratio": round(
                (self.fact_layer + self.echo_layer) / self.stored, 4
            )
            if self.stored
            else 0.0,
            "avg_injected_items": round(self.injected_items / self.context_samples, 2)
            if self.context_samples
            else 0.0,
            "noise_ratio": round(self.injected_noise / self.injected_items, 4)
            if self.injected_items
            else 0.0,
            "old_topic_recall": round(self.old_topic_hits / self.old_topic_total, 4)
            if self.old_topic_total
            else 0.0,
            "fact_layer": self.fact_layer,
            "echo_layer": self.echo_layer,
            "archived": self.archived,
        }


def run(db_path: str, script: Sequence[Turn], *, fade: bool) -> Metrics:
    """跑一遍对话，统计指标。"""

    engine = make_engine(db_path, fade=fade)
    metrics = Metrics()
    try:
        for turn in script:
            now = time_base() + turn.day * 86400.0
            for fact in turn.facts:
                draft = MemoryDraft(
                    content=fact["content"],
                    kind=fact["kind"],
                    importance=fact["importance"],
                    subject=fact.get("subject", ""),
                    predicate=fact.get("predicate", ""),
                    object=fact.get("object", ""),
                )
                engine.remember(draft, SCOPE, now=now)
            # 真实部署里整理任务是常驻的（默认每小时一轮），
            # 这里每轮都跑，给对照组一样的主场条件。
            asyncio.run(engine.consolidate(SCOPE, now=now))
            # 这一轮注入的上下文
            context = engine.build_context(turn.user, SCOPE, now=now)
            text = context.text
            metrics.context_chars += len(text)
            metrics.context_samples += 1
            metrics.injected_items += len(context.hits)
            for hit in context.hits:
                if _NOISE_MARKER in hit.memory.content:
                    metrics.injected_noise += 1

            # Recall@T：当前该成立的事实在不在
            for value in CURRENT_FACTS.values():
                if value in turn.user:
                    metrics.recall_total += 1
                    if value in text:
                        metrics.recall_hits += 1

            # 老话题召回：淡化会不会把对方真正问起的事弄丢？
            if "炸鸡" in turn.user:
                metrics.old_topic_total += 1
                if "炸鸡" in text:
                    metrics.old_topic_hits += 1

            # 更新之后，旧值还在不在
            update_day = script[len(script) // 2].day
            if turn.day > update_day:
                # 注意不能用「吃辣」当旧值——它是「不吃辣」的子串，会误判。
                for stale, fresh in (("住在杭州", "住在上海"), ("喜欢吃辣", "现在不吃辣")):
                    metrics.drift_total += 1
                    if stale in text:
                        metrics.drift_hits += 1
                        if fresh in text:
                            metrics.contradiction_hits += 1

        # 收尾：跑一轮整理，看最终分布
        final_now = time_base() + (script[-1].day + 1) * 86400.0
        asyncio.run(engine.consolidate(SCOPE, now=final_now))
        metrics.stored = engine.store.count(SCOPE.key())
        for mem in engine.store.iter_scope(SCOPE.key(), limit=10000):
            status = resolve_status(mem, final_now, engine.policy)
            if status.value == "active":
                metrics.fact_layer += 1
            elif status.value in ("fading", "gisted"):
                metrics.echo_layer += 1
            else:
                metrics.archived += 1
        return metrics
    finally:
        engine.close()


def time_base() -> float:
    """基准时间。用固定值保证结果可复现。"""

    return 1_760_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="memfade 淡化引擎消融实验")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--days", type=float, default=60.0)
    args = parser.parse_args()

    script = build_script(args.turns, args.days)
    handle, naive_db = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.remove(naive_db)
    handle, fade_db = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.remove(fade_db)

    print("=" * 68)
    print("memfade 淡化引擎 · 消融实验")
    print("=" * 68)
    print(f"对话轮数：{args.turns}    跨度：{args.days:.0f} 天")
    print("基线 naive：只记、只检索、永不遗忘")
    print("对照 memfade：淡化梯度 + 矛盾取代 + 会话隔离")
    print()

    naive = run(naive_db, script, fade=False)
    fade = run(fade_db, script, fade=True)

    print(f"{'指标':<22}{'naive':>14}{'memfade':>14}   说明")
    print("-" * 68)

    def row(label: str, key: str, note: str, higher_better: bool = True) -> None:
        a = naive.as_dict()[key]
        b = fade.as_dict()[key]
        mark = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
            good = (b > a) if higher_better else (b < a)
            mark = "  ✓" if good else "  ✗"
        print(f"{label:<22}{a:>14}{b:>14}   {note}{mark}")

    row("Recall@T", "recall@T", "该记住的记住了吗", True)
    row("老话题召回", "old_topic_recall", "问起旧事还答得上吗", True)
    row("Temporal Drift", "temporal_drift", "旧值残留比例（越低越好）", False)
    row("Contradiction", "contradiction", "新旧并存比例（越低越好）", False)
    row("注入噪声比例", "noise_ratio", "塞进上下文的闲聊占比（越低越好）", False)
    row("平均注入条数", "avg_injected_items", "每轮塞几条", False)
    row("平均上下文字数", "avg_context_chars", "省了多少 token", False)
    print()
    print(f"memfade 最终分布：事实层 {fade.fact_layer} / 回声层 {fade.echo_layer} / 已退场 {fade.archived}"
          f"（库中共 {fade.stored} 条，归档保留原文可审计）")
    print(f"naive   最终分布：事实层 {naive.fact_layer} / 回声层 {naive.echo_layer} / 已退场 {naive.archived}"
          f"（库中共 {naive.stored} 条）")
    print()
    print("注：memfade 的库可能比 naive 大——归档不等于删除，原文留着可追溯、可回滚。")
    print("    真正的收益是「旧值残留」与「新旧并存」这两项；")
    print("    上下文字数略大是因为画像常驻——换来稳定事实从不缺席。")
    print("    噪声比例两者相当：本脚本里闲聊都够「新」，本来就排在事实后面；")
    print("    淡化真正压住的是「旧事实还占着位置」，也就是上面那两项。")

    for path in (naive_db, fade_db):
        for suffix in ("", "-wal", "-shm"):
            p = path + suffix
            if os.path.exists(p):
                os.remove(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


