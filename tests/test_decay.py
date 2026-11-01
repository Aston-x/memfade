"""淡化引擎测试。"""

from __future__ import annotations

import unittest

from memfade.decay import (
    DEFAULT_POLICY,
    MemoryEvent,
    apply_event,
    decay_curve,
    effective_strength,
    next_review_days,
    resolve_status,
    stability_days,
)
from memfade.types import Memory, MemoryKind, MemoryStatus

DAY = 86400.0


def make_memory(
    *,
    kind: MemoryKind = MemoryKind.EPISODIC,
    importance: float = 0.5,
    strength: float = 1.0,
    anchor_at: float = 0.0,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    recall_count: int = 0,
) -> Memory:
    return Memory(
        id=1,
        scope_key="qq:group:1",
        kind=kind,
        content="测试记忆",
        importance=importance,
        strength=strength,
        anchor_at=anchor_at,
        status=status,
        created_at=anchor_at,
        updated_at=anchor_at,
        recall_count=recall_count,
    )


class TestStability(unittest.TestCase):
    def test_importance_slows_decay(self):
        low = stability_days(make_memory(importance=0.1))
        high = stability_days(make_memory(importance=0.9))
        self.assertGreater(high, low * 3, "高重要性记忆应明显更稳定")

    def test_recall_grows_stability(self):
        base = stability_days(make_memory(recall_count=0))
        after = stability_days(make_memory(recall_count=3))
        self.assertGreater(after, base, "反复被想起的记忆应更稳定")

    def test_kind_differ(self):
        episode = stability_days(make_memory(kind=MemoryKind.EPISODIC))
        semantic = stability_days(make_memory(kind=MemoryKind.SEMANTIC))
        self.assertGreater(semantic, episode * 3, "事实应比事件稳定得多")

    def test_recall_ceiling(self):
        a = stability_days(make_memory(recall_count=5))
        b = stability_days(make_memory(recall_count=50))
        self.assertAlmostEqual(a, b, delta=1.0, msg="稳定性倍率必须有上限")


class TestRetention(unittest.TestCase):
    def test_decay_over_time(self):
        mem = make_memory()
        now = 10 * DAY
        self.assertLess(effective_strength(mem, now), 0.5)
        self.assertGreater(effective_strength(mem, 1 * DAY), effective_strength(mem, 10 * DAY))

    def test_idempotent(self):
        """反复调用结果必须一致——维护任务会跑很多次。"""

        mem = make_memory()
        now = 7.5 * DAY
        values = {round(effective_strength(mem, now), 12) for _ in range(5)}
        self.assertEqual(len(values), 1, "同一时刻的强度必须唯一")

    def test_no_backwards_time(self):
        mem = make_memory(anchor_at=10 * DAY)
        self.assertEqual(effective_strength(mem, 0.0), 1.0, "校时回拨不应产生负衰减")

    def test_pinned_never_decays(self):
        mem = make_memory()
        mem.pinned = True
        self.assertEqual(effective_strength(mem, 3650 * DAY), 1.0)


class TestEvents(unittest.TestCase):
    def test_recall_reinforces_and_resets_clock(self):
        mem = make_memory(anchor_at=0.0)
        now = 5 * DAY
        before = effective_strength(mem, now)
        updated = apply_event(mem, MemoryEvent.RECALL, now=now)
        self.assertGreater(updated.strength, before, "被想起后强度应回升")
        self.assertEqual(updated.anchor_at, now, "衰减时钟应重置")
        self.assertEqual(updated.recall_count, 1)

    def test_recall_does_not_exceed_one(self):
        mem = make_memory(strength=1.0, anchor_at=0.0)
        updated = apply_event(mem, MemoryEvent.RECALL, now=1 * DAY)
        self.assertLessEqual(updated.strength, 1.0)

    def test_weaken(self):
        mem = make_memory(anchor_at=0.0)
        updated = apply_event(mem, MemoryEvent.WEAKEN, now=0.0)
        self.assertLess(updated.strength, 1.0)

    def test_pin(self):
        mem = make_memory()
        updated = apply_event(mem, MemoryEvent.PIN, now=100.0)
        self.assertTrue(updated.pinned)
        self.assertEqual(updated.status, MemoryStatus.ACTIVE)


class TestFadeGradient(unittest.TestCase):
    def test_active_to_fading(self):
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        status = resolve_status(mem, 25 * DAY)
        self.assertIn(status, (MemoryStatus.FADING, MemoryStatus.GISTED))

    def test_fading_to_gisted(self):
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.2, anchor_at=0.0)
        status = resolve_status(mem, 120 * DAY)
        self.assertIn(status, (MemoryStatus.GISTED, MemoryStatus.ARCHIVED))

    def test_semantic_survives_longer(self):
        episode = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        semantic = make_memory(kind=MemoryKind.SEMANTIC, importance=0.8, anchor_at=0.0)
        at = 60 * DAY
        self.assertEqual(resolve_status(episode, at), MemoryStatus.ARCHIVED)
        self.assertEqual(resolve_status(semantic, at), MemoryStatus.ACTIVE)

    def test_hysteresis(self):
        """在阈值附近来回，不应反复横跳。"""

        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        # 先落到 FADING
        fading = resolve_status(mem, 15 * DAY)
        self.assertEqual(fading, MemoryStatus.FADING)
        mem.status = fading
        # 同一时刻重算，结果必须一致
        self.assertEqual(resolve_status(mem, 15 * DAY), MemoryStatus.FADING)

    def test_no_oscillation_in_sticky_band(self):
        """粘滞带里反复重算，档位必须稳定——整理任务会跑很多次。"""

        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        at = 8 * DAY  # eff 落在 [fade_exit, fade_enter) 粘滞带
        first = resolve_status(mem, at)
        mem.status = first
        for _ in range(5):
            self.assertEqual(resolve_status(mem, at), first, "粘滞带里档位不能抖动")

    def test_can_skip_multiple_tiers(self):
        """一次衰减可以跨多档，不能卡在中间档。"""
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.2, anchor_at=0.0)
        status = resolve_status(mem, 200 * DAY)
        self.assertIn(status, (MemoryStatus.ARCHIVED, MemoryStatus.GISTED))

    def test_promotion_on_recall(self):
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        mem.status = MemoryStatus.FADING
        updated = apply_event(mem, MemoryEvent.RECALL, now=30 * DAY)
        self.assertEqual(updated.status, MemoryStatus.ACTIVE, "被想起的记忆应回到活跃档")

    def test_archived_is_one_way(self):
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.1, anchor_at=0.0)
        mem.status = MemoryStatus.ARCHIVED
        self.assertEqual(resolve_status(mem, 400 * DAY), MemoryStatus.ARCHIVED)

    def test_hysteresis_band_is_narrow(self):
        """保留率掉到 41% 的事实，不该还被当成「现行事实」。

        迟滞带太宽时，事实会卡在 ACTIVE 档直到保留率跌破 fade_exit——
        那时它已经被当成现在仍然成立的真相注入了，比直接淡出更糟。
        """

        mem = make_memory(kind=MemoryKind.SEMANTIC, importance=0.65, anchor_at=0.0)
        eff = effective_strength(mem, 200 * DAY)
        self.assertLess(eff, 0.5, "这个场景下保留率应该已经明显下降")
        self.assertEqual(resolve_status(mem, 200 * DAY), MemoryStatus.FADING)

    def test_recall_restores_to_active(self):
        """刚被想起的记忆不能又变成陈旧。"""

        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.4, anchor_at=0.0)
        mem.status = MemoryStatus.FADING
        updated = apply_event(mem, MemoryEvent.RECALL, now=30 * DAY)
        self.assertEqual(updated.status, MemoryStatus.ACTIVE)
        self.assertGreaterEqual(updated.strength, 0.5, "回忆后强度应至少回到活跃线")


class TestScheduling(unittest.TestCase):
    def test_next_review_days(self):
        mem = make_memory(kind=MemoryKind.EPISODIC, importance=0.5)
        days = next_review_days(mem, now=0.0)
        self.assertIsNotNone(days)
        self.assertGreater(days, 0)
        self.assertLess(days, 365, "事件类记忆不应一整年不复查")

    def test_decay_curve_shape(self):
        curve = decay_curve(MemoryKind.EPISODIC, importance=0.5, days=30)
        self.assertEqual(curve[0], 1.0)
        self.assertLess(curve[-1], curve[10])
        self.assertTrue(all(a >= b for a, b in zip(curve, curve[1:])), "曲线应单调不增")


if __name__ == "__main__":
    unittest.main()

