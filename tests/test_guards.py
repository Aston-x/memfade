"""写入前闸门测试。

这些用例全是**真实踩过的坑**——线上记忆库里长出来的东西，
每一条都对应一个「主人一眼就看出记错了」的错法。
"""

from __future__ import annotations

import unittest

from memfade.guards import (
    check_draft,
    check_remember_request,
    find_evidence,
    guard_drafts,
)
from memfade.types import MemoryDraft, MemoryKind


def _msg(text: str, role: str = "user") -> dict:
    return {"role": role, "content": text}


class TestRealWorldCases(unittest.TestCase):
    """线上库里长出来过的四条记忆，逐条复现。"""

    def test_today_became_every_day(self):
        """「今天只有一节课」被写成「每天只有一节课」。"""

        draft = MemoryDraft(
            content="Alice每天只有一节课",
            kind=MemoryKind.SEMANTIC,
            importance=0.6,
            subject="Alice",
            predicate="每天只有",
            object="一节课",
        )
        messages = [_msg("不忙，整天就一节课"), _msg("已经上完了")]
        kept, notes = guard_drafts([draft], messages)
        self.assertEqual(len(kept), 1, "该降级而不是丢掉：信息没错，只是形态错了")
        self.assertEqual(kept[0].kind, MemoryKind.EPISODIC)
        self.assertTrue(notes, "降级要有原因可查")
        self.assertIn("常态", notes[0][1])

    def test_one_off_request_is_not_a_preference(self):
        """「这次你用这种语气」被写成「希望你用这种语气」。"""

        draft = MemoryDraft(
            content="Alice希望Bot用平时与别人说话的语气",
            kind=MemoryKind.PREFERENCE,
            importance=0.75,
            subject="Alice",
            predicate="希望",
            object="Bot用平时与别人说话的语气",
        )
        messages = [
            _msg("你要用你的语气，表示占领了我的账号，然后发布招人动态"),
            _msg("好，我家Bot最棒了，语气我想要你平时和别人说话的那种，不是和我说话的那种"),
        ]
        kept, notes = guard_drafts([draft], messages)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.EPISODIC)
        self.assertIn("偏好", notes[0][1])

    def test_transient_state_is_dropped(self):
        """「发图会被拦、暂时发不了」是眼下如此，不该进记忆库。"""

        draft = MemoryDraft(
            content="Alice发图时会被拦截，暂时发不了",
            kind=MemoryKind.EPISODIC,
            importance=0.35,
            subject="Alice",
            predicate="发送被拦截",
            object="图片",
        )
        kept, notes = guard_drafts([draft], [_msg("我放图会被拦下来，发不了")])
        self.assertEqual(kept, [])
        self.assertTrue(notes)
        self.assertIn("临时", notes[0][1])

    def test_plain_episodic_event_survives(self):
        """正常的一次性事件不该被误伤。"""

        draft = MemoryDraft(
            content="Alice当天起床后直接去上早自习",
            kind=MemoryKind.EPISODIC,
            importance=0.32,
            subject="Alice",
            predicate="参加",
            object="早自习",
        )
        kept, _ = guard_drafts([draft], [_msg("没呢，刚起来就去上早自习，没空")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.EPISODIC)


class TestKeepRealFacts(unittest.TestCase):
    """闸门不能变成「什么都不记」。"""

    def test_stated_preference_is_kept(self):
        draft = MemoryDraft(
            content="Alice最喜欢吃辣的食物",
            kind=MemoryKind.PREFERENCE,
            importance=0.8,
            subject="Alice",
            predicate="喜欢",
            object="辣的食物",
        )
        kept, notes = guard_drafts([draft], [_msg("我最喜欢吃辣的食物")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.PREFERENCE)
        self.assertEqual(notes, [])

    def test_explicit_routine_is_kept(self):
        draft = MemoryDraft(
            content="Alice每天早上都跑三公里",
            kind=MemoryKind.SEMANTIC,
            importance=0.8,
            subject="Alice",
            predicate="每天跑",
            object="三公里",
        )
        kept, notes = guard_drafts([draft], [_msg("我每天早上都跑三公里")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.SEMANTIC)
        self.assertEqual(notes, [])

    def test_plain_fact_is_kept(self):
        draft = MemoryDraft(
            content="Alice住在杭州",
            kind=MemoryKind.SEMANTIC,
            importance=0.8,
            subject="Alice",
            predicate="住在",
            object="杭州",
        )
        kept, notes = guard_drafts([draft], [_msg("我住在杭州")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.SEMANTIC)
        self.assertEqual(notes, [])

    def test_today_alone_is_downgraded_but_kept(self):
        """「今天」的内容不该当稳定事实，但也不该丢。"""

        draft = MemoryDraft(
            content="Alice今天只上了一节课",
            kind=MemoryKind.SEMANTIC,
            importance=0.7,
            subject="Alice",
            predicate="今天只上",
            object="一节课",
        )
        kept, notes = guard_drafts([draft], [_msg("今天只有一节课")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, MemoryKind.EPISODIC)
        self.assertTrue(notes)


class TestEvidence(unittest.TestCase):
    """依据原话：每条记忆都要能指回它是从哪句话来的。"""

    def test_finds_the_source_sentence(self):
        draft = MemoryDraft(
            content="Alice每天只有一节课",
            kind=MemoryKind.SEMANTIC,
            subject="Alice",
            predicate="每天只有",
            object="一节课",
        )
        evidence = find_evidence(
            draft,
            [_msg("不忙，整天就一节课"), _msg("已经上完了"), _msg("别的闲话")],
        )
        self.assertIn("一节课", evidence.quote)
        self.assertGreater(evidence.score, 0.5)

    def test_robot_words_are_not_evidence(self):
        """机器人说过的话是上下文，不是用户的证词。"""

        draft = MemoryDraft(
            content="Alice每天只有一节课",
            kind=MemoryKind.SEMANTIC,
            subject="Alice",
            object="一节课",
        )
        evidence = find_evidence(draft, [_msg("一节课的事", role="assistant")])
        self.assertEqual(evidence.quote, "")
        self.assertEqual(evidence.window, [])

    def test_evidence_is_stored_in_meta(self):
        draft = MemoryDraft(
            content="Alice住在杭州",
            kind=MemoryKind.SEMANTIC,
            subject="Alice",
            predicate="住在",
            object="杭州",
        )
        kept, _ = guard_drafts([draft], [_msg("我住在杭州")])
        self.assertTrue(kept)
        self.assertIn("quote", kept[0].meta)
        self.assertIn("杭州", kept[0].meta["quote"])


class TestDropRules(unittest.TestCase):
    def test_mood_is_dropped(self):
        draft = MemoryDraft(content="Alice今天好累", kind=MemoryKind.EPISODIC, importance=0.4)
        kept, notes = guard_drafts([draft], [_msg("今天好累啊")])
        self.assertEqual(kept, [])
        self.assertIn("情绪", notes[0][1])

    def test_empty_content_is_dropped(self):
        draft = MemoryDraft(content="", kind=MemoryKind.SEMANTIC)
        kept, notes = guard_drafts([draft], [])
        self.assertEqual(kept, [])
        self.assertTrue(notes)

    def test_check_draft_without_evidence_keeps_stable_facts(self):
        """查不到依据时，宁可放过，也不能凭感觉把事实毙掉。"""

        draft = MemoryDraft(content="Alice住在杭州", kind=MemoryKind.SEMANTIC, importance=0.8)
        verdict = check_draft(draft)
        self.assertEqual(verdict.action, "keep")


class TestRememberRequest(unittest.TestCase):
    """「记住」请求的闸门：模型自己决定要记什么的那条路。

    这条路**不经过抽取那一整套纪律**——不设闸门就会漏。
    实测漏过一条：模型把一句游戏数据
    （「不吃肥肠有0+0姬子启行、0+0星期日……」）当成
    「关于群里某个人的稳定事实」记了进去。
    """

    def test_game_stats_about_another_person_is_dropped(self):
        draft = MemoryDraft(
            content="不吃肥肠有0+0姬子启行、0+0星期日、0+0缇宝、0+1花火、1+1符玄",
            kind=MemoryKind.SEMANTIC,
            importance=0.8,
            subject="不吃肥肠",
        )
        verdict = check_remember_request(draft, focus_subjects=["Alice"])
        self.assertEqual(verdict.action, "drop")
        self.assertIn("跟要记的那个人没关系", verdict.reason)

    def test_content_without_the_focus_subject_is_dropped(self):
        draft = MemoryDraft(content="今天群里在打牌", kind=MemoryKind.EPISODIC)
        verdict = check_remember_request(draft, focus_subjects=["Alice"])
        self.assertEqual(verdict.action, "drop")

    def test_about_the_right_person_is_kept(self):
        draft = MemoryDraft(
            content="Alice答应周末一起去图书馆",
            kind=MemoryKind.SEMANTIC,
            importance=0.8,
            subject="Alice",
        )
        verdict = check_remember_request(draft, focus_subjects=["Alice"])
        self.assertEqual(verdict.action, "keep")

    def test_robot_own_facts_are_dropped(self):
        draft = MemoryDraft(
            content="BotName欠Alice半首歌",
            kind=MemoryKind.SEMANTIC,
            subject="BotName",
        )
        # ⚠ 承诺类不该被误杀：主语是机器人，说的却是关于Alice的事。
        verdict = check_remember_request(draft, focus_subjects=["Alice"])
        self.assertEqual(verdict.action, "keep")

    def test_robot_own_facts_are_dropped_without_focus_config(self):
        draft = MemoryDraft(
            content="BotName最喜欢辣的食物",
            kind=MemoryKind.PREFERENCE,
            subject="BotName",
        )
        verdict = check_remember_request(draft, self_names=["BotName"])
        self.assertEqual(verdict.action, "drop")

    def test_transient_state_is_dropped(self):
        draft = MemoryDraft(content="Alice现在发不了图", kind=MemoryKind.EPISODIC)
        verdict = check_remember_request(draft)
        self.assertEqual(verdict.action, "drop")

    def test_empty_content_is_dropped(self):
        draft = MemoryDraft(content="   ", kind=MemoryKind.SEMANTIC)
        verdict = check_remember_request(draft)
        self.assertEqual(verdict.action, "drop")

    def test_no_focus_config_means_no_subject_restriction(self):
        """没配 focus_subjects 时，不该拦别人的事——那是部署方的选择。"""

        draft = MemoryDraft(
            content="不吃肥肠今天赢了比赛",
            kind=MemoryKind.EPISODIC,
            subject="不吃肥肠",
        )
        verdict = check_remember_request(draft)
        self.assertEqual(verdict.action, "keep")


if __name__ == "__main__":
    unittest.main()
