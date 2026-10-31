"""memfade · LangBot 事件监听器。

两个时机：

* ``PromptPreProcessing``——把召回的记忆写进 ``default_prompt``，
  模型这一轮就能看见；
* ``NormalMessageResponded``——这一轮说完之后，把新内容抽取成记忆。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langbot_plugin.api.definition.components.common.event_listener import EventListener
from langbot_plugin.api.entities import context, events
from langbot_plugin.api.entities.builtin.provider.message import Message

from memfade import Scope

logger = logging.getLogger(__name__)


def _session_of(event_ctx: context.EventContext) -> str:
    """取会话标识。"""

    event = getattr(event_ctx, "event", None)
    for attr in ("session_name", "session_id", "conversation_id"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return "default"


def _query_text(event_ctx: context.EventContext) -> str:
    """取本轮用户输入。"""

    event = getattr(event_ctx, "event", None)
    for attr in ("content", "message_str", "text"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    getter = getattr(event, "get_content", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, str):
                return value.strip()
        except Exception:
            pass
    return ""


class MemFadeInjector(EventListener):
    """记忆注入 + 抽取。"""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._counter: Dict[str, int] = {}
        self._last_injected: Dict[str, str] = {}

        @self.handler(events.PromptPreProcessing)
        async def on_prompt_preprocess(event_ctx: context.EventContext) -> None:
            try:
                await self._inject(event_ctx)
            except Exception:
                logger.exception("[memfade] 注入失败")

        @self.handler(events.NormalMessageResponded)
        async def on_message_responded(event_ctx: context.EventContext) -> None:
            try:
                await self._extract(event_ctx)
            except Exception:
                logger.exception("[memfade] 抽取失败")

    def _scope(self, event_ctx: context.EventContext) -> Scope:
        return Scope(platform="langbot", chat_type="group", chat_id=_session_of(event_ctx))

    async def _inject(self, event_ctx: context.EventContext) -> None:
        engine = self.plugin.engine
        query = _query_text(event_ctx)
        if not query:
            return
        scope = self._scope(event_ctx)
        vector = await engine.embed_query(query)
        memory_ctx = engine.build_context(query, scope, query_vector=vector)
        if not memory_ctx.text:
            return
        key = scope.key()
        if self._last_injected.get(key) == memory_ctx.text:
            return
        self._last_injected[key] = memory_ctx.text
        injection = (
            "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
            "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
            f"{memory_ctx.text}"
        )
        event_ctx.event.default_prompt.append(Message(role="system", content=injection))

    async def _extract(self, event_ctx: context.EventContext) -> None:
        engine = self.plugin.engine
        query = _query_text(event_ctx)
        if not query:
            return
        scope = self._scope(event_ctx)
        key = scope.key()
        buffer = self._buffer.setdefault(key, [])
        buffer.append({"role": "user", "content": query})
        count = self._counter.get(key, 0) + 1
        self._counter[key] = count
        config = self.plugin.get_config() or {}
        try:
            every = max(1, int(config.get("extract_every_n", 3)))
        except (TypeError, ValueError):
            every = 3
        if count % every != 0:
            return
        pending = list(buffer)
        buffer.clear()
        written = await engine.remember_from_messages(pending, scope, source=key)
        if written:
            logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)


