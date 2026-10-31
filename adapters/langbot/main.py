"""memfade · LangBot 适配器。

LangBot 插件入口。引擎在这里创建，事件监听器通过 ``self.plugin`` 取用。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


def _bootstrap_import() -> None:
    """让 ``import memfade`` 在自带核心源码的部署形态下也能工作。"""

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_bootstrap_import()

from memfade import MemFadeConfig  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402
from memfade.embedding import HttpEmbeddingClient  # noqa: E402
from memfade.llm import HttpLLMClient  # noqa: E402


def build_config(raw: Dict[str, Any]) -> MemFadeConfig:
    """把插件配置翻译成核心配置。"""

    def as_int(key: str, default: int) -> int:
        try:
            return int(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    def as_float(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    def as_bool(key: str, default: bool) -> bool:
        value = raw.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    return MemFadeConfig.from_dict(
        {
            "storage": {
                "db_path": str(raw.get("db_path") or "memfade.db"),
                "enable_vector": as_bool("use_embedding", True),
            },
            "extraction": {
                "enabled": as_bool("extract_enabled", True),
                "every_n_turns": max(1, as_int("extract_every_n", 3)),
                "min_importance": as_float("min_importance", 0.35),
            },
            "retrieval": {
                "top_k": max(1, as_int("top_k", 6)),
                "budget_tokens": max(50, as_int("max_inject_tokens", 600)),
                "echo_max": max(0, as_int("echo_max", 2)),
            },
            "fade": {
                "recall_alpha": as_float("recall_alpha", 0.35),
                "archive_after_days": max(0, as_int("archive_after_days", 0)),
            },
        }
    )


class MemFadePlugin(BasePlugin):
    """记忆淡化插件。"""

    engine: MemoryEngine

    async def initialize(self) -> None:
        """插件初始化：建引擎。"""

        raw = dict(self.get_config() or {})
        config = build_config(raw)
        llm = None
        if raw.get("llm_base_url") and raw.get("llm_model"):
            llm = HttpLLMClient(
                base_url=str(raw["llm_base_url"]),
                api_key=str(raw.get("llm_api_key") or ""),
                model=str(raw["llm_model"]),
                timeout=float(raw.get("llm_timeout") or 30.0),
            )
        embedder = None
        if raw.get("embedding_base_url") and raw.get("embedding_model"):
            embedder = HttpEmbeddingClient(
                base_url=str(raw["embedding_base_url"]),
                api_key=str(raw.get("embedding_api_key") or ""),
                model=str(raw["embedding_model"]),
            )
        self.engine = MemoryEngine(config, llm=llm, embedder=embedder)
        logger.info("[memfade] 引擎就绪：%s", config.storage.db_path)

    async def terminate(self) -> None:
        """插件卸载。"""

        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

