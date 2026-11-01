"""memfade · LangBot 事件监听器。

只用一个时机：``PromptPreProcessing``。

它同时给了两样东西：

* ``default_prompt``——情景预设（系统提示词），可修改 → 记忆往这里注入；
* ``prompt``——本轮现有的消息记录 → 用户这轮说了什么、用来做检索和抽取。

（``NormalMessageResponded`` 里没有用户文本，所以抽取也放在这里做。）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langbot_plugin.api.definition.components.common.event_listener import EventListener
from langbot_plugin.api.entities import context, events
from langbot_plugin.api.entities.builtin.provider.message import Message

from memfade import Scope

logger = logging.getLogger(__name__)


def message_text(message: Any) -> str:
    """取一条消息的纯文本。

    LangBot 的 ``Message.content`` 既可能是字符串，
    也可能是 ``list[ContentElement]``，两种都要处理。
    """

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for element in content:
            text = getattr(element, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()
    return ""


def _session_of(event_ctx: context.EventContext) -> str:
    """取会话标识。"""

    event = getattr(event_ctx, "event", None)
    value = getattr(event, "session_name", None)
    if value:
        return str(value)
    session = getattr(event, "session", None)
    for attr in ("session_name", "session_id", "id"):
        value = getattr(session, attr, None)
        if value:
            return str(value)
    return "default"


def _query_text(event_ctx: context.EventContext) -> str:
    """取本轮用户输入。

    用户这轮说的话在 ``event.prompt`` 的最后一条 user 消息里。
    """

    event = getattr(event_ctx, "event", None)
    prompt = getattr(event, "prompt", None)
    if isinstance(prompt, list):
        for message in reversed(prompt):
            role = str(getattr(message, "role", "") or "").lower()
            if role and role != "user":
                continue
            text = message_text(message)
            if text:
                return text
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
        # 画像先算：里面的条目 id 会在组装记忆块时排除，避免同一件事出现两次
        profile_block = engine.retriever.profile(scope.key())
        memory_ctx = engine.build_context(
            query, scope, query_vector=vector, exclude_ids=profile_block.ids
        )
        if not memory_ctx.text and not profile_block.text:
            return
        key = scope.key()
        injection = self._compose(memory_ctx.text, profile_block.text)
        if self._last_injected.get(key) == injection:
            return
        self._last_injected[key] = injection
        event_ctx.event.default_prompt.append(Message(role="system", content=injection))

    @staticmethod
    def _compose(memory_text: str, profile_text: str) -> str:
        """拼出要注入的文本。"""

        blocks: List[str] = []
        if profile_text:
            blocks.append(
                "[系统·关于对方] 下面是你对这位用户的长期印象，属于你已经知道的信息，"
                "不要重复确认：\n"
                f"{profile_text}"
            )
        if memory_text:
            blocks.append(
                "[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，"
                "不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n"
                f"{memory_text}"
            )
        return "\n\n".join(blocks)

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


