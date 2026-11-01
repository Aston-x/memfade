"""NoneBot2 适配器：旁听 + 命令。

旁听用 ``priority=99`` 且不阻塞，保证不影响任何其它插件的处理顺序。
"""

from __future__ import annotations

from nonebot import on_message, on_regex
from nonebot.log import logger
from nonebot.matcher import Matcher

from .config import config
from .core import _buffer, _counter, build_context, engine, remember, scope_of


memory_observer = on_message(priority=99, block=False)
remember_cmd = on_regex(r"^记住\s*(?P<content>.+)$")
forget_cmd = on_regex(r"^忘掉\s*(?P<content>.+)$")
query_cmd = on_regex(r"^记忆(?:\s+搜\s*(?P<query>.+))?$")


@memory_observer.handle()
async def observe(event) -> None:
    """旁听消息：抽取出记忆，并把记忆文本挂到 event 上。"""

    try:
        text = str(event.get_message() or "").strip()
    except Exception:
        text = ""
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
    buffer.append({"role": "user", "content": text})
    count = _counter.get(key, 0) + 1
    _counter[key] = count
    if count % max(1, int(config.memfade_extract_every_n)) != 0:
        return
    pending = list(buffer)
    buffer.clear()
    try:
        written = await engine.remember_from_messages(pending, scope, source=key)
        if written:
            logger.info(f"[memfade] 抽取到 {len(written)} 条新记忆（{key}）")
    except Exception as exc:
        logger.warning(f"[memfade] 抽取失败: {exc}")
        buffer.extend(pending)


@remember_cmd.handle()
async def handle_remember(matcher: Matcher, event) -> None:
    """显式记住。"""

    content = str(matcher.get("content") or "").strip()
    if not content:
        await matcher.finish("要记住什么呢？用法：记住 <内容>")
    memory = await remember(event, content)
    if memory is None:
        await matcher.finish("没记下来（内容不重要或与已有记忆重复）")
    await matcher.finish(f"记住了：{memory.content}")


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

