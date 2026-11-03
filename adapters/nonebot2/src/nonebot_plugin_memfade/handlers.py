"""NoneBot2 适配器：旁听 + 命令。

旁听用 ``priority=99`` 且不阻塞，保证不影响任何其它插件的处理顺序。
"""

from __future__ import annotations

import asyncio

from nonebot import on_message, on_regex
from nonebot.log import logger
from nonebot.matcher import Matcher

from .config import config
from .core import _buffer, _counter, build_context, engine, remember, remember_all, scope_of


memory_observer = on_message(priority=99, block=False)
remember_cmd = on_regex(r"^记住\s*(?P<content>.+)$")
forget_cmd = on_regex(r"^忘掉\s*(?P<content>.+)$")
query_cmd = on_regex(r"^记忆(?:\s+搜\s*(?P<query>.+))?$")
maintain_cmd = on_regex(r"^记忆\s*整理$")
pin_cmd = on_regex(r"^记忆\s*(?P<action>置顶|取消置顶)\s*(?P<query>.+)$")
repair_cmd = on_regex(r"^记忆\s*修正\s*(?P<query>.+)$")
stats_cmd = on_regex(r"^记忆$")

_extract_tasks: set = set()
"""后台抽取任务。抽取要等 LLM，不能让它挡住主人的回复。"""


def _own_text(event) -> tuple[str, str]:
    """只留用户自己说的话（丢掉回复/转发段），并带上说话人昵称。

    **引用与转发不是用户的话**——把它们一起喂给抽取器，
    抽取器就会把别人说过的话记成用户的事。
    """

    text = ""
    try:
        message = event.get_message()
    except Exception:
        message = None
    if message is not None:
        parts: list[str] = []
        for segment in message:
            kind = str(getattr(segment, "type", "") or "").lower()
            if kind in ("reply", "forward", "node"):
                continue
            body = str(getattr(segment, "text", "") or "").strip()
            if body:
                parts.append(body)
        text = " ".join(parts)
    if not text:
        try:
            text = str(event.get_message() or "").strip()
        except Exception:
            text = ""
    speaker = ""
    try:
        speaker = str(event.get_user_name() or "").strip()
    except Exception:
        speaker = ""
    return text, speaker


@memory_observer.handle()
async def observe(event) -> None:
    """旁听消息：抽取出记忆，并把记忆文本挂到 event 上。"""

    text, speaker = _own_text(event)
    if not text:
        return
    scope = scope_of(event)
    key = scope.key()
    try:
        context_text = await build_context(event, text)
        if context_text:
            event.set_extra("memfade_context", context_text)
    except Exception as exc:
        logger.warning(f"[memfade] 注入失败: {exc}")
    if not config.memfade_extract_enabled:
        return
    buffer = _buffer.setdefault(key, [])
    entry = {"role": "user", "content": text}
    if speaker:
        entry["speaker"] = speaker
    buffer.append(entry)
    count = _counter.get(key, 0) + 1
    _counter[key] = count
    if count % max(1, int(config.memfade_extract_every_n)) != 0:
        return
    pending = list(buffer)
    buffer.clear()
    task = asyncio.create_task(_extract(pending, scope, key))
    _extract_tasks.add(task)
    task.add_done_callback(_extract_tasks.discard)


async def _extract(pending: list, scope, key: str) -> None:
    """后台跑一次抽取。"""

    try:
        written = await engine.remember_from_messages(pending, scope, source=key)
        if written:
            logger.info(f"[memfade] 抽取到 {len(written)} 条新记忆（{key}）")
    except Exception as exc:
        logger.warning(f"[memfade] 抽取失败: {exc}")
        _buffer.setdefault(key, []).extend(pending)


@maintain_cmd.handle()
async def handle_maintain(matcher: Matcher, event) -> None:
    """立刻跑一轮整理（重算淡化档位、合并重复、仲裁矛盾）。"""

    report = await engine.consolidate(scope_of(event))
    await matcher.finish(
        "整理完成：扫描 %d 条，淡出 %d 条，合并 %d 条，取代 %d 条。"
        % (report.scanned, report.decayed, report.merged, report.superseded)
    )


@pin_cmd.handle()
async def handle_pin(matcher: Matcher, event) -> None:
    """置顶一条记忆（此后永不淡化）。"""

    action = str(matcher.get("action") or "置顶")
    query = str(matcher.get("query") or "").strip()
    if not query:
        await matcher.finish("要置顶哪条？给点内容片段。")
    scope = scope_of(event)
    vector = await engine.embed_query(query)
    hits = engine.recall(query, scope, top_k=5, query_vector=vector, require_match=True)
    if not hits:
        await matcher.finish("没找到相关的记忆。")
    target = hits[0].memory
    engine.pin(target.id, pinned=(action == "置顶"))
    verb = "置顶了" if action == "置顶" else "取消置顶了"
    await matcher.finish(f"{verb}：{target.content}")


@repair_cmd.handle()
async def handle_repair(matcher: Matcher, event) -> None:
    """改掉记错的那条记忆（记忆 修正 老内容 => 新内容）。"""

    from memfade import split_repair_request

    raw = str(matcher.get("query") or "").strip()
    old, new = split_repair_request(raw)
    if not old or not new:
        await matcher.finish("要改哪条？照这个格式发：记忆 修正 老内容 => 新内容")
    updated = await engine.repair(scope_of(event), old, new)
    if updated is None:
        await matcher.finish("没找到要改的那条记忆。")
    previous = str((updated.meta or {}).get("previous_content") or "")
    await matcher.finish(f"改好了：{previous}\n现在记的是：{updated.content}")


@stats_cmd.handle()
async def handle_stats(matcher: Matcher, event) -> None:
    """看看记住了多少东西——并且看得见具体内容。"""

    scope = scope_of(event)
    stats = engine.stats(scope)
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
        ids = [mid for mid, _ in engine.store.recent(scope.key(), limit=8)]
        recent = sorted(engine.store.get_many(ids), key=lambda m: m.created_at, reverse=True)
        if recent:
            lines.append("最近记下的：")
            for memory in recent:
                marker = "" if memory.status.value == "active" else "（已淡出）"
                lines.append("- %s%s" % (memory.content, marker))
    except Exception:
        pass
    await matcher.finish("\n".join(lines))


@remember_cmd.handle()
async def handle_remember(matcher: Matcher, event) -> None:
    """显式记住。"""

    content = str(matcher.get("content") or "").strip()
    if not content:
        await matcher.finish("要记住什么呢？用法：记住 <内容>")
    written = await remember_all(event, content)
    if not written:
        await matcher.finish("没记下来（内容不重要或与已有记忆重复）")
    if len(written) == 1:
        await matcher.finish(f"记住了：{written[0].content}")
    else:
        await matcher.finish(
            "记住了这 %d 件事：\n%s" % (len(written), "\n".join("- " + m.content for m in written))
        )


@forget_cmd.handle()
async def handle_forget(matcher: Matcher, event) -> None:
    """忘掉一条。"""

    content = str(matcher.get("content") or "").strip()
    scope = scope_of(event)
    # 忘掉是破坏性操作：必须真的匹配上，不能误删「最近写入」的那条。
    hits = engine.recall(content, scope, top_k=5, require_match=True)
    if not hits:
        await matcher.finish("没找到要忘掉的记忆。")
    target = hits[0].memory
    engine.forget(target.id)
    await matcher.finish(f"忘掉了：{target.content}")


@query_cmd.handle()
async def handle_query(matcher: Matcher, event) -> None:
    """查询记忆。"""

    query = str(matcher.get("query") or "").strip()
    scope = scope_of(event)
    if query:
        # 显式检索必须真的命中，不把「最近写入」的无关记忆列出来。
        hits = engine.recall(query, scope, top_k=8, require_match=True)
        if not hits:
            await matcher.finish("没找到相关的记忆。")
        lines = []
        for hit in hits:
            marker = "（已淡出）" if hit.tier == "echo" else ""
            lines.append(f"· {hit.memory.content}{marker}")
        await matcher.finish("记得这些：\n" + "\n".join(lines))
    stats = engine.stats(scope)
    by_status = stats.get("by_status") or {}
    await matcher.finish(
        "这个会话目前记住了 {total} 条：活跃 {active} 条，淡出中 {fading} 条，"
        "已成摘要 {gisted} 条，已归档 {archived} 条。".format(
            total=stats.get("total", 0),
            active=by_status.get("active", 0),
            fading=by_status.get("fading", 0),
            gisted=by_status.get("gisted", 0),
            archived=by_status.get("archived", 0),
        )
    )

