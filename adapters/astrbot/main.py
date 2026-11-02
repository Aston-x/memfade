"""memfade · AstrBot 适配器。

两个时机：

* ``on_llm_request``——把召回的记忆追加到 system prompt，模型这一轮就能看见；
* ``on_llm_response``——这一轮说完之后，把新内容抽取成记忆。
"""

from __future__ import annotations

import asyncio
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
        self._tasks: set = set()
        """后台抽取任务。抽取要等 LLM，不能让它挡住回复。"""
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

    # ------------------------------------------------------------------ #
    # 指令
    # ------------------------------------------------------------------ #

    @filter.command_group("记忆")
    def memory_group(self):
        """记忆相关指令"""

    @memory_group.command("状态")
    async def memory_status(self, event: AstrMessageEvent):
        """这个会话记住了多少东西"""

        scope = self._scope_of(event)
        stats = self.engine.stats(scope)
        by_status = stats.get("by_status") or {}
        lines = [
            "这个会话目前记住了 %d 条：活跃 %d 条，淡出中 %d 条，已成摘要 %d 条，已归档 %d 条。"
            % (
                stats.get("total", 0),
                by_status.get("active", 0),
                by_status.get("fading", 0),
                by_status.get("gisted", 0),
                by_status.get("archived", 0),
            )
        ]
        # 光给数字不够：**得能一眼看出库里存了什么**。
        try:
            ids = [mid for mid, _ in self.engine.store.recent(scope.key(), limit=8)]
            recent = sorted(
                self.engine.store.get_many(ids), key=lambda m: m.created_at, reverse=True
            )
            if recent:
                lines.append("最近记下的：")
                for memory in recent:
                    marker = "" if memory.status.value == "active" else "（已淡出）"
                    lines.append("- %s%s" % (memory.content, marker))
        except Exception:
            pass
        yield event.plain_result("\n".join(lines))

    @memory_group.command("搜")
    async def memory_search(self, event: AstrMessageEvent, query: str = ""):
        """检索记忆（必须真命中）"""

        text = (query or "").strip()
        if not text:
            yield event.plain_result("要搜什么？用法：记忆 搜 <关键词>")
            return
        scope = self._scope_of(event)
        vector = await self.engine.embed_query(text)
        hits = self.engine.recall(
            text, scope, top_k=8, query_vector=vector, require_match=True
        )
        if not hits:
            yield event.plain_result("没找到相关的记忆。")
            return
        lines = []
        for hit in hits:
            marker = "（可能已经变了）" if hit.tier == "echo" else ""
            lines.append("· %s%s" % (hit.memory.content, marker))
        yield event.plain_result("记得这些：\n" + "\n".join(lines))

    @memory_group.command("整理")
    async def memory_maintain(self, event: AstrMessageEvent):
        """立刻跑一轮整理"""

        report = await self.engine.consolidate(self._scope_of(event))
        yield event.plain_result(
            "整理完成：扫描 %d 条，淡出 %d 条，合并 %d 条，取代 %d 条。"
            % (report.scanned, report.decayed, report.merged, report.superseded)
        )

    @memory_group.command("置顶")
    async def memory_pin(self, event: AstrMessageEvent, query: str = ""):
        """让一条记忆永不淡化"""

        text = (query or "").strip()
        if not text:
            yield event.plain_result("要置顶哪条？给点内容片段。")
            return
        scope = self._scope_of(event)
        vector = await self.engine.embed_query(text)
        hits = self.engine.recall(
            text, scope, top_k=5, query_vector=vector, require_match=True
        )
        if not hits:
            yield event.plain_result("没找到相关的记忆。")
            return
        target = hits[0].memory
        self.engine.pin(target.id, pinned=True)
        yield event.plain_result("置顶了：%s" % target.content)

    @filter.command("记住")
    async def cmd_remember(self, event: AstrMessageEvent, content: str = ""):
        """显式记住一件事"""

        text = (content or "").strip()
        if not text:
            yield event.plain_result("要记住什么呢？用法：记住 <内容>")
            return
        from memfade.types import MemoryDraft, MemoryKind

        scope = self._scope_of(event)
        written = self.engine.remember_atomic(
            MemoryDraft(content=text, kind=MemoryKind.SEMANTIC, importance=0.9, source="astrbot"),
            scope,
        )
        if not written:
            yield event.plain_result("没记下来（内容不重要或与已有记忆重复）")
            return
        await self.engine.embed_memories(written)
        if len(written) == 1:
            yield event.plain_result("记住了：%s" % written[0].content)
        else:
            yield event.plain_result(
                "记住了这 %d 件事：\n%s"
                % (len(written), "\n".join("- " + m.content for m in written))
            )

    @filter.command("忘掉")
    async def cmd_forget(self, event: AstrMessageEvent, content: str = ""):
        """忘掉一条记忆"""

        text = (content or "").strip()
        if not text:
            yield event.plain_result("要忘掉什么？用法：忘掉 <内容片段>")
            return
        scope = self._scope_of(event)
        # 忘掉是破坏性操作：必须真的匹配上，不能误删「最近写入」的那条。
        vector = await self.engine.embed_query(text)
        hits = self.engine.recall(
            text, scope, top_k=5, query_vector=vector, require_match=True
        )
        if not hits:
            yield event.plain_result("没找到要忘掉的记忆。")
            return
        target = hits[0].memory
        self.engine.forget(target.id, hard=True)
        yield event.plain_result("忘掉了：%s" % target.content)

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
            entry = {"role": "user", "content": query}
            speaker = ""
            sender = getattr(event, "sender", None)
            if sender is not None:
                for attr in ("nickname", "name", "user_name"):
                    value = getattr(sender, attr, None)
                    if isinstance(value, str) and value.strip():
                        speaker = value.strip()
                        break
            if speaker:
                # 带上说话人：抽取器要能分清「谁说的」。
                entry["speaker"] = speaker
            buffer.append(entry)
            count = self._counter.get(key, 0) + 1
            self._counter[key] = count
            every = max(1, _as_int(_get_config_value(self.config, "extract_every_n", 3), 3))
            if count % every != 0:
                return
            pending = list(buffer)
            buffer.clear()
            # 抽取要问一次 LLM（可能几十秒），别让它挡住这一轮回复。
            task = asyncio.create_task(self._extract_pending(pending, scope, key))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.warning("[memfade] 抽取失败: %s", exc)

    async def _extract_pending(self, pending: List[Dict[str, Any]], scope: Scope, key: str) -> None:
        """后台跑一次抽取。"""

        try:
            written = await self.engine.remember_from_messages(pending, scope, source=key)
            if written:
                logger.info("[memfade] 抽取到 %d 条新记忆（%s）", len(written), key)
        except Exception as exc:
            logger.warning("[memfade] 抽取失败: %s", exc)
            self._buffer.setdefault(key, []).extend(pending)

    async def terminate(self) -> None:
        """插件卸载。"""

        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

