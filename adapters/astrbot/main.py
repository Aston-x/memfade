"""memfade · AstrBot 适配器。

两个时机：

* ``on_llm_request``——把召回的记忆追加到 system prompt，模型这一轮就能看见；
* ``on_llm_response``——这一轮说完之后，把新内容抽取成记忆。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

logger = logging.getLogger("memfade")


def _bootstrap_import() -> None:
    """让 ``import memfade`` 在自带核心源码的部署形态下也能工作。"""

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_bootstrap_import()

from memfade import MemFadeConfig, Scope  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    """AstrBot 的配置对象既可能是 dict 也可能是属性对象。"""

    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@register("memfade_memory", "记忆淡化引擎：会记得，也会忘得恰到好处", "0.1.0", "memfade")
class MemFadePlugin(Star):
    """记忆淡化插件。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context)
        self.config = config
        mem_cfg = MemFadeConfig.from_dict(
            {
                "storage": {
                    "db_path": str(_get_config_value(config, "db_path", "memfade.db")),
                    "enable_vector": bool(_get_config_value(config, "use_embedding", True)),
                },
                "extraction": {
                    "enabled": bool(_get_config_value(config, "extract_enabled", True)),
                    "every_n_turns": max(1, _as_int(_get_config_value(config, "extract_every_n", 3), 3)),
                    "min_importance": _as_float(_get_config_value(config, "min_importance", 0.35), 0.35),
                },
                "retrieval": {
                    "top_k": max(1, _as_int(_get_config_value(config, "top_k", 6), 6)),
                    "budget_tokens": max(50, _as_int(_get_config_value(config, "max_inject_tokens", 600), 600)),
                    "echo_max": max(0, _as_int(_get_config_value(config, "echo_max", 2), 2)),
                },
            }
        )
        self.engine = MemoryEngine(mem_cfg)
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._counter: Dict[str, int] = {}
        self._last_injected: Dict[str, str] = {}

    # ------------------------------------------------------------------ #

    @staticmethod
    def _scope_of(event: AstrMessageEvent) -> Scope:
        group_id = event.get_group_id()
        sender_id = event.get_sender_id()
        if group_id:
            return Scope(platform="astrbot", chat_type="group", chat_id=str(group_id))
        return Scope(platform="astrbot", chat_type="private", chat_id=str(sender_id or "unknown"))

    @filter.on_llm_request(priority=-7)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        """把召回的记忆追加到 system prompt。"""

        try:
            query = (event.get_message_str() or "").strip()
            if not query:
                return
            scope = self._scope_of(event)
            vector = await self.engine.embed_query(query)
            memory_ctx = self.engine.build_context(query, scope, query_vector=vector)
            if not memory_ctx.text:
                return
            key = scope.key()
            if self._last_injected.get(key) == memory_ctx.text:
                return
            self._last_injected[key] = memory_ctx.text
            body = (
                "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
                "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
                f"{memory_ctx.text}"
            )
            base = getattr(req, "system_prompt", "") or ""
            req.system_prompt = (base + "\n\n" + body).strip()
        except Exception as exc:
            logger.warning("[memfade] 注入失败: %s", exc)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, ctx: Any = None) -> None:
        """这一轮结束后抽取记忆。"""

        del ctx
        try:
            query = (event.get_message_str() or "").strip()
            if not query:
                return
            scope = self._scope_of(event)
            key = scope.key()
            buffer = self._buffer.setdefault(key, [])
            buffer.append({"role": "user", "content": query})
            count = self._counter.get(key, 0) + 1
            self._counter[key] = count
            every = max(1, _as_int(_get_config_value(self.config, "extract_every_n", 3), 3))
            if count % every != 0:
                return
            pending = list(buffer)
            buffer.clear()
            written = await self.engine.remember_from_messages(pending, scope, source=key)
            if written:
                logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)
        except Exception as exc:
            logger.warning("[memfade] 抽取失败: %s", exc)

    async def terminate(self) -> None:
        """插件卸载。"""

        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

