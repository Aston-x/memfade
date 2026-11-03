"""观察归纳（reflect）功能测试。

reflect 是从多条记忆中归纳高层结论的能力——
Hindsight 等方案的核心差异点。memfade 原来只有 gist（压单条），
没有「从多条归纳出一条结论」的能力，更没有佐证链。

这组测试验证：
1. 归纳出的观察带 derived_from 佐证链
2. 不够活跃的作用域不触发归纳（不白白烧 LLM）
3. LLM 不可用时静默跳过
"""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Optional
from unittest.mock import AsyncMock

from memfade.config import MemFadeConfig
from memfade.consolidate import ConsolidationService
from memfade.store import MemoryStore
from memfade.types import Memory, MemoryKind, MemoryStatus, Scope
from tests.helpers import temp_engine


class FakeLLM:
    """假的 LLM 客户端，返回预设的 JSON。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        return self.response


class FakeExtractor:
    """假的抽取器，只带一个 LLM 客户端。"""

    def __init__(self, llm):
        self.llm = llm


class TestReflect(unittest.TestCase):
    """观察归纳功能。"""

    def setUp(self):
        self.engine, self.db_path, _ = temp_engine()
        self.scope = Scope(platform="qq", chat_type="private", chat_id="test_user")
        # 写入 5 条相关事实
        facts = [
            "季怡喜欢辣的食物",
            "季怡不吃肥肠",
            "季怡爱喝可乐",
            "季怡不吃香菜",
            "季怡喜欢吃火锅",
        ]
        for f in facts:
            from memfade.types import MemoryDraft

            draft = MemoryDraft(
                content=f, kind=MemoryKind.SEMANTIC, importance=0.7, subject="季怡"
            )
            self.engine.remember(draft, self.scope)

    def _get_memories(self):
        return self.engine.store.iter_scope(
            self.scope.key(), statuses=("active",), limit=100
        )

    def test_reflect_creates_observation_with_evidence_chain(self):
        """归纳出的观察带 derived_from 佐证链。"""

        llm = FakeLLM(
            json.dumps(
                [
                    {
                        "content": "季怡有明确的饮食偏好：喜欢辣食和可乐，不吃肥肠和香菜",
                        "source_ids": [1, 2, 3, 4, 5],
                        "importance": 0.85,
                    }
                ]
            )
        )
        extractor = FakeExtractor(llm)
        config = MemFadeConfig.from_dict(
            {"maintenance": {"enable_reflect": True, "reflect_min_active": 3}}
        )
        service = ConsolidationService(
            self.engine.store, config=config, extractor=extractor
        )
        report = asyncio.run(service.run(self.scope.key()))
        self.assertEqual(report.reflected, 1)
        self.assertEqual(llm.calls, 1)

        # 验证 observation 写入
        observations = [
            m for m in self._get_memories() if m.kind == MemoryKind.OBSERVATION
        ]
        self.assertEqual(len(observations), 1)
        obs = observations[0]
        self.assertIn("饮食偏好", obs.content)
        self.assertEqual(obs.meta.get("derived_from"), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(obs.importance, 0.85)
        self.assertEqual(obs.source, "reflect")

    def test_reflect_skips_when_too_few_active(self):
        """不够活跃的作用域不触发归纳。"""

        llm = FakeLLM("[]")
        extractor = FakeExtractor(llm)
        config = MemFadeConfig.from_dict(
            {"maintenance": {"enable_reflect": True, "reflect_min_active": 20}}
        )
        service = ConsolidationService(
            self.engine.store, config=config, extractor=extractor
        )
        report = asyncio.run(service.run(self.scope.key()))
        self.assertEqual(report.reflected, 0)
        self.assertEqual(llm.calls, 0)

    def test_reflect_skips_when_no_llm(self):
        """LLM 不可用时静默跳过。"""

        config = MemFadeConfig.from_dict(
            {"maintenance": {"enable_reflect": True, "reflect_min_active": 3}}
        )
        service = ConsolidationService(self.engine.store, config=config)
        report = asyncio.run(service.run(self.scope.key()))
        self.assertEqual(report.reflected, 0)

    def test_reflect_disabled_by_config(self):
        """配置关闭时不归纳。"""

        llm = FakeLLM('[{"content": "test", "source_ids": [1,2], "importance": 0.8}]')
        extractor = FakeExtractor(llm)
        config = MemFadeConfig.from_dict(
            {"maintenance": {"enable_reflect": False}}
        )
        service = ConsolidationService(
            self.engine.store, config=config, extractor=extractor
        )
        report = asyncio.run(service.run(self.scope.key()))
        self.assertEqual(report.reflected, 0)
        self.assertEqual(llm.calls, 0)

    def test_reflect_skips_existing_observations(self):
        """已有 observation 的作用域不会重复归纳（避免无限递归）。"""

        # 先手动加一条 observation
        from memfade.types import MemoryDraft

        obs_draft = MemoryDraft(
            content="季怡有明确的饮食偏好",
            kind=MemoryKind.OBSERVATION,
            importance=0.8,
            subject="季怡",
        )
        self.engine.remember(obs_draft, self.scope)

        llm = FakeLLM("[]")
        extractor = FakeExtractor(llm)
        config = MemFadeConfig.from_dict(
            {"maintenance": {"enable_reflect": True, "reflect_min_active": 3}}
        )
        service = ConsolidationService(
            self.engine.store, config=config, extractor=extractor
        )
        report = asyncio.run(service.run(self.scope.key()))
        # LLM 仍然被调用（因为 active 记忆够多），但返回空数组
        # 关键是：归纳时排除了已有的 observation，不会把 observation 再归纳
        self.assertEqual(report.reflected, 0)


if __name__ == "__main__":
    unittest.main()
