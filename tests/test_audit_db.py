"""体检工具的「按现在的闸门重扫」这一段。

为什么要有这个功能：闸门是**随代码变**的。
闸门上线前混进来的记忆，靠人记着「哪几条不对」是清不干净的——
实测就漏过一条：清理时删掉了两条工具路径写进来的，
闸门部署**之前**写的第三条留了下来（主语不是要记的那个人）。

→ 标准动作是「用现在的闸门重扫一遍全库」，而不是凭记忆挑。
这条能力必须**随插件发货**，不能靠人临时手写 SQL。
"""

from __future__ import annotations

import importlib.util
import sqlite3
import unittest
from pathlib import Path

from memfade.types import MemoryDraft, MemoryKind
from tests.helpers import temp_engine

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    """按路径加载 scripts/audit_db.py（它不在包里，是个独立脚本）。"""

    path = REPO_ROOT / "scripts" / "audit_db.py"
    spec = importlib.util.spec_from_file_location("memfade_audit_db", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGateRescan(unittest.TestCase):
    """体检能把「按现在的规则不该有的」挑出来。"""

    def setUp(self):
        self.engine, self.path, _ = temp_engine()
        self.audit = _load_audit()

    def _scan(self, focus=("季怡",), self_names=("许杏玖",)):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return self.audit.gate_scan(conn, focus, self_names)
        finally:
            conn.close()

    def _remember(self, content: str, subject: str, kind=MemoryKind.SEMANTIC, **meta):
        from memfade.types import Scope

        scope = Scope(platform="qq", chat_type="session", chat_id="s1")
        written = self.engine.remember(
            MemoryDraft(
                content=content,
                kind=kind,
                importance=0.8,
                subject=subject,
                meta=meta,
            ),
            scope,
        )
        self.assertIsNotNone(written)
        return written

    def test_flags_memory_about_someone_else(self):
        """实测漏过的那一条：主语不是要记的那个人。"""

        self._remember("群里把deepseek驱动的角色叫大肥鱼", "大肥鱼")
        should_not_exist, _ = self._scan()
        self.assertEqual(len(should_not_exist), 1)
        self.assertIn("跟要记的那个人没关系", should_not_exist[0][1][0])

    def test_legit_memory_passes(self):
        """正常记忆不许被误伤。"""

        self._remember("季怡喜欢辣的食物", "季怡")
        self._remember("季怡今天只有一节课", "季怡", kind=MemoryKind.EPISODIC)
        should_not_exist, should_downgrade = self._scan()
        self.assertEqual(should_not_exist, [])
        self.assertEqual(should_downgrade, [])

    def test_flags_once_upgraded_to_routine(self):
        """「今天」被写成了「每天」，且原话里没有这个频率 → 该降级为事件。"""

        self._remember(
            "季怡每天只有一节课",
            "季怡",
            quote="不忙，整天就一节课",
            quote_score=0.8,
        )
        should_not_exist, should_downgrade = self._scan()
        self.assertEqual(should_not_exist, [])
        self.assertEqual(len(should_downgrade), 1)
        # 命中的理由取决于原话：「整天」是时效词 → 「原话说的是这一次」；
        # 原话里既没有时效词也没有频率词 → 「模型自己加的频率」。
        # 两条路都指向同一个结论：这条该降级为事件。
        self.assertTrue(
            "常态" in should_downgrade[0][1][0]
            or "模型自己加的" in should_downgrade[0][1][0],
            should_downgrade[0][1][0],
        )

    def test_no_focus_configured_does_not_flag_others(self):
        """没配 focus 时，不该替部署方决定「只记谁」。"""

        self._remember("群里把deepseek驱动的角色叫大肥鱼", "大肥鱼")
        should_not_exist, _ = self._scan(focus=(), self_names=("许杏玖",))
        self.assertEqual(should_not_exist, [])


if __name__ == "__main__":
    unittest.main()
