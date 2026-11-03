"""NoneBot2 适配器测试。

装了真的 nonebot2 就跑真 SDK；没装就跳过。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.helpers import temp_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SRC = REPO_ROOT / "adapters" / "nonebot2" / "src"
if str(ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(ADAPTER_SRC))

try:
    from nonebot.plugin import PluginMetadata  # noqa: F401

    HAS_SDK = True
except Exception:  # pragma: no cover
    HAS_SDK = False


class FakeEvent:
    """够用的假事件。"""

    def __init__(self, session_id: str = "group:1127155154", group_id: str = "1127155154", user_id: str = "u1"):
        self._session_id = session_id
        self._group_id = group_id
        self._user_id = user_id
        self._message = "今天吃什么呢"
        self.extras: dict = {}

    def get_session_id(self) -> str:
        return self._session_id

    def get_group_id(self) -> str:
        return self._group_id

    def get_user_id(self) -> str:
        return self._user_id

    def get_message(self) -> str:
        return self._message

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value


@unittest.skipUnless(HAS_SDK, "未安装 nonebot2")
class TestNoneBotAdapter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from nonebot_plugin_memfade import core as core_module

        self.core = core_module
        self.tmp = temp_dir("memfade-nonebot-")
        # 让引擎落到临时目录，别污染用户数据
        core_module.config.memfade_db_path = os.path.join(self.tmp, "memfade.db")
        core_module.engine.close()
        from memfade import MemFadeConfig
        from memfade.engine import MemoryEngine

        config = MemFadeConfig.from_dict(
            {
                "storage": {"db_path": core_module.config.memfade_db_path, "enable_vector": False},
                "extraction": {"min_importance": 0.15},
            }
        )
        core_module.engine = MemoryEngine(config)

    async def asyncTearDown(self):
        self.core.engine.close()

    def test_scope_uses_session_id(self):
        scope = self.core.scope_of(FakeEvent())
        self.assertEqual(scope.chat_type, "group")
        self.assertEqual(scope.chat_id, "group:1127155154")

    def test_scope_private(self):
        scope = self.core.scope_of(FakeEvent(session_id="u1", group_id="", user_id="u1"))
        self.assertEqual(scope.chat_type, "private")
        self.assertEqual(scope.chat_id, "u1")

    async def test_remember_and_recall(self):
        from memfade import MemoryDraft, MemoryKind

        scope = self.core.scope_of(FakeEvent())
        memory = self.core.engine.remember(
            MemoryDraft(content="许杏玖喜欢吃辣的东西", kind=MemoryKind.SEMANTIC, importance=0.9),
            scope,
        )
        self.assertIsNotNone(memory)
        text = await self.core.build_context(FakeEvent(), "今天吃什么")
        self.assertIn("许杏玖喜欢吃辣的东西", text)

    async def test_public_api_is_importable(self):
        import nonebot_plugin_memfade as plugin

        for name in ("build_context", "remember", "recall", "engine"):
            self.assertTrue(hasattr(plugin, name), f"公开 API 少了 {name}")
        self.assertEqual(plugin.__plugin_meta__.name, "nonebot-plugin-memfade")


if __name__ == "__main__":
    unittest.main()

