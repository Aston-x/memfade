"""memfade · LangBot 指令组件。

事件监听器负责「注入与抽取」，这里负责「让人能自己看、自己改」：
查状态、搜、整理、置顶、修正、记住、忘掉。

指令走 LangBot 的 Command 组件（``@Command.subcommand``），
返回用 ``context.CommandReturn``。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from langbot_plugin.api.definition.components.command.command import Command
from langbot_plugin.api.entities.builtin.command.context import CommandReturn, ExecuteContext

from memfade import Scope, split_repair_request
from memfade.types import MemoryDraft, MemoryKind

logger = logging.getLogger(__name__)


def _scope_of(session: Any) -> Scope:
    """从 LangBot 的 session 构造作用域。

    作用域键用会话名——**一个群一份、私聊按人一份**，群聊和私聊不会串。
    """

    name = ""
    for attr in ("session_name", "id", "name"):
        value = getattr(session, attr, None)
        if value:
            name = str(value)
            break
    launcher = str(getattr(session, "launcher_type", "") or "")
    chat_type = "group" if "group" in launcher.lower() else "private"
    return Scope(platform="langbot", chat_type=chat_type, chat_id=name or "default")


class MemoryCommand(Command):
    """记忆相关指令。"""

    def __init__(self, plugin: Any = None) -> None:
        super().__init__()
        self.plugin = plugin
        # ⚠ LangBot 的 subcommand 是**实例方法**：注册进 registered_subcommands，
        # 调用时以 `func(self, ctx)` 的形式调——所以注册的是**未绑定的函数**
        # （type(self).xxx），注册绑定方法会多传一个 self。
        self.subcommand("状态", help="这个会话记住了多少东西", usage="记忆 状态")(
            MemoryCommand.status
        )
        self.subcommand("搜", help="检索记忆", usage="记忆 搜 <关键词>")(
            MemoryCommand.search
        )
        self.subcommand("整理", help="立刻跑一轮整理", usage="记忆 整理")(
            MemoryCommand.maintain
        )
        self.subcommand("置顶", help="让一条记忆永不淡化", usage="记忆 置顶 <内容片段>")(
            MemoryCommand.pin
        )
        self.subcommand(
            "修正",
            help="改掉记错的那条（老内容 => 新内容）",
            usage="记忆 修正 老内容 => 新内容",
        )(MemoryCommand.repair)
        self.subcommand("记住", help="显式记住一件事", usage="记住 <内容>")(
            MemoryCommand.remember
        )
        self.subcommand("忘掉", help="忘掉一条记忆", usage="忘掉 <内容片段>")(
            MemoryCommand.forget
        )

    @property
    def engine(self):
        return getattr(self.plugin, "engine", None)

    def _scope(self, ctx: ExecuteContext) -> Scope:
        return _scope_of(getattr(ctx, "session", None))

    async def status(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """查记忆统计，并列出最近记下的内容。"""

        scope = self._scope(ctx)
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
        yield CommandReturn(text="\n".join(lines))

    async def search(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """检索记忆（必须真命中，不掺无关记忆）。"""

        query = (getattr(ctx, "command_text", "") or "").strip()
        if not query:
            yield CommandReturn(text="要搜什么？用法：记忆 搜 <关键词>")
            return
        scope = self._scope(ctx)
        vector = await self.engine.embed_query(query)
        hits = self.engine.recall(
            query, scope, top_k=8, query_vector=vector, require_match=True
        )
        if not hits:
            yield CommandReturn(text="没找到相关的记忆。")
            return
        lines = []
        for hit in hits:
            marker = "（可能已经变了）" if hit.tier == "echo" else ""
            lines.append("· %s%s" % (hit.memory.content, marker))
        yield CommandReturn(text="记得这些：\n" + "\n".join(lines))

    async def maintain(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """重算淡化档位、合并重复、仲裁矛盾。"""

        report = await self.engine.consolidate(self._scope(ctx))
        yield CommandReturn(
            text="整理完成：扫描 %d 条，淡出 %d 条，合并 %d 条，取代 %d 条。"
            % (report.scanned, report.decayed, report.merged, report.superseded)
        )

    async def pin(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """置顶一条记忆。"""

        query = (getattr(ctx, "command_text", "") or "").strip()
        if not query:
            yield CommandReturn(text="要置顶哪条？给点内容片段。")
            return
        scope = self._scope(ctx)
        vector = await self.engine.embed_query(query)
        hits = self.engine.recall(
            query, scope, top_k=5, query_vector=vector, require_match=True
        )
        if not hits:
            yield CommandReturn(text="没找到相关的记忆。")
            return
        target = hits[0].memory
        self.engine.pin(target.id, pinned=True)
        yield CommandReturn(text="置顶了：%s" % target.content)

    async def repair(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """改掉记错的那条记忆。

        抽取难免出错（把「今天」写成「每天」）。出错之后得有个补救口——
        主人自己改一条，比整库清空的代价小得多。
        """

        raw = (getattr(ctx, "command_text", "") or "").strip()
        old, new = split_repair_request(raw)
        if not old or not new:
            yield CommandReturn(
                text="要改哪条？照这个格式发：记忆 修正 老内容 => 新内容"
            )
            return
        updated = await self.engine.repair(self._scope(ctx), old, new)
        if updated is None:
            yield CommandReturn(text="没找到要改的那条记忆。")
            return
        previous = str((updated.meta or {}).get("previous_content") or "")
        yield CommandReturn(text="改好了：%s\n现在记的是：%s" % (previous, updated.content))

    async def remember(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """显式写入一条记忆。"""

        content = (getattr(ctx, "command_text", "") or "").strip()
        if not content:
            yield CommandReturn(text="要记住什么呢？用法：记住 <内容>")
            return
        scope = self._scope(ctx)
        written = self.engine.remember_atomic(
            MemoryDraft(
                content=content,
                kind=MemoryKind.SEMANTIC,
                importance=0.9,
                source="langbot",
            ),
            scope,
        )
        if not written:
            yield CommandReturn(text="没记下来（内容不重要或与已有记忆重复）")
            return
        await self.engine.embed_memories(written)
        if len(written) == 1:
            yield CommandReturn(text="记住了：%s" % written[0].content)
        else:
            yield CommandReturn(
                text="记住了这 %d 件事：\n%s"
                % (len(written), "\n".join("- " + m.content for m in written))
            )

    async def forget(
        self, ctx: ExecuteContext
    ) -> AsyncGenerator[CommandReturn, None]:
        """忘掉一条记忆。"""

        content = (getattr(ctx, "command_text", "") or "").strip()
        if not content:
            yield CommandReturn(text="要忘掉什么？用法：忘掉 <内容片段>")
            return
        scope = self._scope(ctx)
        # 忘掉是破坏性操作：必须真的匹配上，不能误删「最近写入」的那条。
        vector = await self.engine.embed_query(content)
        hits = self.engine.recall(
            content, scope, top_k=5, query_vector=vector, require_match=True
        )
        if not hits:
            yield CommandReturn(text="没找到要忘掉的记忆。")
            return
        target = hits[0].memory
        self.engine.forget(target.id, hard=True)
        yield CommandReturn(text="忘掉了：%s" % target.content)
