"""中文文本处理测试。

这一组守的都是「拆坏了记忆」这类事故：
时间被劈成两半、引号里的逗号被当成分句点、虚词被当成实词。
"""

from __future__ import annotations

import unittest

from memfade.textutil import (
    best_clause,
    cjk_unigrams,
    content_words,
    fts_text,
    split_atomic,
)


class TestSplitAtomic(unittest.TestCase):
    def test_keeps_time_and_date_intact(self):
        """「9:00-13:00」不能被劈成「9」和「00-13:00」。"""

        text = "考4小时9:00-13:00，电子赛科目单片机、嵌入式"
        parts = split_atomic(text)
        joined = "".join(parts)
        self.assertIn("9:00-13:00", joined)
        for part in parts:
            self.assertNotIn("00-13", part.replace("9:00-13:00", ""))

    def test_respects_max_chars(self):
        text = "报名截止时间、软件赛与电子赛的报名费、赛制安排、组别划分、比赛科目、时间安排"
        for part in split_atomic(text, max_chars=20):
            self.assertLessEqual(len(part), 20)

    def test_loses_no_content(self):
        """拆开只是重新分组，一个字都不能丢。"""

        text = "季怡在做小红书账号，发AI绘画内容，目标是靠流量赚钱；10月起零关注零粉丝。"
        parts = split_atomic(text)
        # 分隔符会被保留，所以两边都按同一套规则去掉分隔符再比。
        seps = "，,。；;、：:"
        stripped = "".join(parts)
        original = text
        for sep in seps:
            stripped = stripped.replace(sep, "")
            original = original.replace(sep, "")
        self.assertEqual(stripped, original)

    def test_keeps_list_separator(self):
        """顿号是并列项的分隔符，不能当切分点。"""

        parts = split_atomic("软件赛科目C/C++、Java、Python、Web应用开发")
        self.assertEqual(len(parts), 1)
        self.assertIn("、", parts[0])

    def test_does_not_split_inside_quotes(self):
        """引号里的逗号不是分句点。"""

        parts = split_atomic('季怡的简介是"AI炼丹师，炼到什么发什么喵"')
        self.assertEqual(len(parts), 1)
        self.assertIn("炼到什么发什么喵", parts[0])

    def test_short_text_is_one_piece(self):
        self.assertEqual(split_atomic("许杏玖喜欢辣的食物"), ["许杏玖喜欢辣的食物"])


class TestContentWords(unittest.TestCase):
    def test_chinese_yields_content_words(self):
        """早先这里按单字切，中文一个词都抽不出来。"""

        words = content_words("许杏玖喜欢辣的食物")
        self.assertTrue(words, "中文也必须能抽出内容词")
        self.assertIn("喜欢", words)

    def test_drops_function_char_artifacts(self):
        """「的食」「辣的」这种半虚词组合不该算内容词。"""

        words = content_words("许杏玖喜欢的食物")
        self.assertNotIn("的食", words)
        self.assertNotIn("的", words)


class TestBestClause(unittest.TestCase):
    def test_keeps_subject_clause_when_it_is_real(self):
        self.assertEqual(
            best_clause("许杏玖喜欢辣的食物，也讨厌香菜"),
            "许杏玖喜欢辣的食物",
        )

    def test_skips_filler_clause(self):
        """「后来……」是铺垫，该挑后面有内容的一句。"""

        self.assertEqual(
            best_clause("后来他跟朋友去吃了炸鸡，许杏玖很开心"),
            "许杏玖很开心",
        )


class TestCjkUnigrams(unittest.TestCase):
    def test_extracts_unique_chars(self):
        self.assertEqual(cjk_unigrams("辣辣辣"), ["辣"])
        self.assertEqual(cjk_unigrams("辣 food"), ["辣"])

    def test_fts_index_stays_bigram_only(self):
        """索引里不能混单字——会污染 BM25 的长度归一化。"""

        tokens = fts_text("许杏玖喜欢辣的食物").split()
        self.assertIn("喜欢", tokens)
        self.assertNotIn("辣", tokens)


if __name__ == "__main__":
    unittest.main()
