"""记忆库污染的防线。

背景：用户的记忆库被「机器人自己的事」「转发出去的公告」
「同一件事的七八种说法」污染了。这里锁住那几道闸门。

对应的事故：

* 主人在群里引用机器人的「你昨晚一口气刷完Re」，
  记忆库里多了一条「许杏玖昨晚一口气刷完了Re」——机器人认领了主人的话；
* 主人转发蓝桥杯群公告求个总结，记忆库里多了一条
  「季怡在备赛第18届蓝桥杯」——可主人明说过「我没跟你说过我要报名」；
* 同一件事被记了八遍（「季怡已经购买《影之刃零》」系列）。
"""

from __future__ import annotations

import unittest

from memfade.config import ExtractionConfig
from memfade.extract import (
    GENERIC_SUBJECTS,
    PERSON_ROLES,
    build_extraction_prompt,
    filter_drafts,
)
from memfade.types import MemoryDraft, MemoryKind


def _draft(content: str, subject: str = "", kind: MemoryKind = MemoryKind.SEMANTIC) -> MemoryDraft:
    return MemoryDraft(content=content, kind=kind, importance=0.7, subject=subject)


class TestSubjectPolicy(unittest.TestCase):
    def test_self_facts_are_dropped(self):
        """主语是机器人自己的，一律不入库。"""

        kept, dropped = filter_drafts(
            [_draft("许杏玖昨晚一口气刷完了Re", "许杏玖")],
            self_names=["许杏玖"],
            participants=["季怡"],
            default_subject="季怡",
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0][1], "self_fact")

    def test_non_person_subject_is_dropped(self):
        """作品名、赛事名当主语——那是公共知识，不是「关于某人」的记忆。"""

        kept, dropped = filter_drafts(
            [_draft("《牧神记5》在2026年11月1日周日更新", "《牧神记5》")],
            participants=["季怡"],
            default_subject="季怡",
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0][1], "not_a_person")

    def test_focus_subject_wins(self):
        """配了 focus 就只记 focus 的事。"""

        kept, dropped = filter_drafts(
            [
                _draft("季怡喜欢辣的食物", "季怡"),
                _draft("鸡柳的显卡跑不动两个模型", "鸡柳"),
            ],
            focus_subjects=["季怡"],
            participants=["季怡", "鸡柳"],
            default_subject="季怡",
        )
        self.assertEqual([d.content for d in kept], ["季怡喜欢辣的食物"])
        self.assertEqual(dropped[0][1], "not_focus")

    def test_generic_subject_is_rewritten(self):
        """主语写成「用户」时，知道是谁就替他填上。"""

        kept, _ = filter_drafts(
            [_draft("他住在上海", "用户")],
            participants=["季怡"],
            default_subject="季怡",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].subject, "季怡")

    def test_family_roles_are_people(self):
        """「妈妈」不是说话人，但确实是人——关系类记忆要留住。"""

        kept, dropped = filter_drafts(
            [_draft("季怡的妈妈是小学老师", "妈妈")],
            participants=["季怡"],
            default_subject="季怡",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])
        self.assertIn("妈妈", PERSON_ROLES)

    def test_unknown_speaker_keeps_the_fact(self):
        """连说话人都不知道（老宿主没有昵称）时，别把真事实扔掉。"""

        kept, dropped = filter_drafts(
            [_draft("他住在上海", "用户")],
            participants=[],
            default_subject="",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_generic_subject_list_is_not_empty(self):
        self.assertIn("对方", GENERIC_SUBJECTS)
        self.assertIn("用户", GENERIC_SUBJECTS)


class TestPrompt(unittest.TestCase):
    def test_subject_hint_is_injected(self):
        prompt = build_extraction_prompt(ExtractionConfig(subject_hint="（许杏玖 / 季怡）"))
        self.assertIn("（许杏玖 / 季怡）", prompt)
        self.assertNotIn("@SUBJECT_HINT@", prompt)

    def test_prompt_forbids_self_and_forwarded_content(self):
        prompt = build_extraction_prompt(ExtractionConfig())
        self.assertIn("机器人（对方）自己的事", prompt)
        self.assertIn("引用、转发", prompt)
        self.assertIn("否认、纠正", prompt)

    def test_prompt_has_no_brace_placeholder(self):
        """提示词常量里出现字面花括号会让 .format() 崩——仓库里有守卫测试。"""

        prompt = build_extraction_prompt(ExtractionConfig())
        self.assertNotIn("{SUBJECT_HINT}", prompt)


class TestContextLine(unittest.TestCase):
    """被引用的原文要当**上下文**留着，不是删掉也不是当用户的话。

    删掉它的后果：用户回「买了」，抽取器不知道买的是什么，
    只能记一句含糊的废话，或者干脆不记。
    当成用户的话的后果：机器人说过的、替用户断言的事，全被记成用户的事。
    """

    def test_assistant_line_is_rendered_as_context(self):
        from memfade.extract import MemoryExtractor

        extractor = MemoryExtractor(ExtractionConfig())
        rendered = extractor._render(
            [
                {"role": "assistant", "content": "Master，《影之刃零》买了吗"},
                {"role": "user", "content": "买了", "speaker": "季怡"},
            ]
        )
        self.assertIn("机器人：Master，《影之刃零》买了吗", rendered)
        self.assertIn("季怡：买了", rendered)

    def test_assistant_line_is_not_a_participant(self):
        """机器人不是「要被记住的那个人」——它只是上下文。"""

        from memfade.extract import MemoryExtractor

        extractor = MemoryExtractor(
            ExtractionConfig(focus_subjects=("季怡",), self_names=("许杏玖",))
        )
        kept, dropped = extractor._apply_subject_policy(
            [
                _draft("季怡买了《影之刃零》", "季怡"),
                _draft("许杏玖昨晚刷完了Re", "许杏玖"),
            ],
            [
                {"role": "assistant", "content": "Master，《影之刃零》买了吗"},
                {"role": "user", "content": "买了", "speaker": "季怡"},
            ],
        )
        self.assertEqual([d.content for d in kept], ["季怡买了《影之刃零》"])
        self.assertEqual(dropped[0][1], "self_fact")


class TestHealthCheck(unittest.TestCase):
    """体检要能抓到污染——不然脏库会一直脏着，没人发现。"""

    def _engine(self, tmpdir):
        from memfade import MemFadeConfig
        from memfade.engine import MemoryEngine
        from memfade.types import Scope

        config = MemFadeConfig.from_dict(
            {
                "storage": {"db_path": str(tmpdir / "m.db"), "enable_vector": False},
                "extraction": {"self_names": ("许杏玖",)},
            }
        )
        return MemoryEngine(config), Scope(platform="qq", chat_type="session", chat_id="s1")

    def test_self_facts_are_reported(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            engine, scope = self._engine(Path(tmp))
            engine.remember(
                MemoryDraft(content="许杏玖昨晚一口气刷完了Re", subject="许杏玖", importance=0.7),
                scope,
            )
            engine.remember(
                MemoryDraft(content="季怡喜欢辣的食物", subject="季怡", importance=0.7),
                scope,
            )
            report = engine.diagnose()
            self.assertEqual(report["self_facts"], 1)
            kinds = {issue["kind"] for issue in report["issues"]}
            self.assertIn("self_facts", kinds)
            engine.close()

    def test_generic_subject_is_reported(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            engine, scope = self._engine(Path(tmp))
            engine.remember(
                MemoryDraft(content="用户表示自己开通了某个套餐", subject="用户", importance=0.7),
                scope,
            )
            report = engine.diagnose()
            self.assertEqual(report["generic_subjects"], 1)
            engine.close()


if __name__ == "__main__":
    unittest.main()
