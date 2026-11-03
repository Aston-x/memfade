"""作用域设计：群级公共 + 人级私密 + 私聊，三层。

背景
----
AstrBot 没有「群内按人区分的会话」这个概念——``umo`` 在群聊里对全群成员是同一个值。
所以记忆隔离必须由适配器自己做，作用域键得自己设计。

三层
----
======================  =========================================  =========================
层                       Scope                                     语义
======================  =========================================  =========================
群级公共                 chat_type="group"                          全群共享
群内人级私密             chat_type="group_user"                     只属于该成员
私聊                     chat_type="private"                        1:1，只有本人可见
======================  =========================================  =========================

键长这样
--------
``Scope.key()`` 返回 ``platform:chat_type:chat_id`` 三段，而 ``Scope.from_key``
用的是 ``split(":", 2)``——**第三段整体是 chat_id，里面可以带冒号**。所以三种
chat_type 互不相同，就不可能撞出相同的 key。

人级层内部用 ``#`` 分隔群号和用户号，而不是 ``:``：``engine.import_memories``
在 scope 为空时会手写 ``raw_scope.split(":")[-1]`` 拆键，用 ``#`` 能让那条路径
也恰好拆对。

兼容性
--------
旧版适配器的群键是 ``astrbot:group:<群号>``、私聊键是 ``astrbot:private:<用户号>``，
与新方案**一字未变**，老库不用迁移——只是群键的语义从「群内共享」变成
「群级公共层」。新增的只有 ``astrbot:group_user:*``。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:  # 正常路径：插件被宿主以包形式导入
    from .memfade.types import Scope
except ImportError:  # 兜底：脱离包上下文直接跑
    from memfade.types import Scope  # type: ignore

PLATFORM = "astrbot"

CHAT_GROUP = "group"
CHAT_GROUP_USER = "group_user"
CHAT_PRIVATE = "private"

#: 群级公共层用的主语词。抽取时主语等于它 → 写群级公共层。
DEFAULT_GROUP_SUBJECT = "本群"

#: 人级层内部的分隔符。见模块文档。
_SEP = "#"


def group_scope(group_id: str) -> Scope:
    """群级公共层：这一群共享的记忆。"""
    return Scope(platform=PLATFORM, chat_type=CHAT_GROUP, chat_id=str(group_id))


def person_scope(group_id: str, sender_id: str) -> Scope:
    """群内人级私密层：只属于这个成员，只在他发言时召回。"""
    return Scope(
        platform=PLATFORM,
        chat_type=CHAT_GROUP_USER,
        chat_id=f"{group_id}{_SEP}{sender_id}",
    )


def private_scope(sender_id: str) -> Scope:
    """私聊层：1:1 对话，只有本人可见。"""
    return Scope(platform=PLATFORM, chat_type=CHAT_PRIVATE, chat_id=str(sender_id))


def read_scopes(
    group_id: str,
    sender_id: str,
) -> List[Scope]:
    """一次对话要读哪些层。

    群聊 → 群级公共 + 当前说话人的人级私密。
    私聊 → 只有私聊层（1:1 里没有「群」这个容器，也就没有群级公共的受众；
    同时保证私聊内容永远不会漏进群里——这是核心的铁律）。
    """
    if group_id:
        return [group_scope(group_id), person_scope(group_id, sender_id)]
    return [private_scope(sender_id)]


def write_scopes(
    group_id: str,
    sender_id: str,
) -> List[Scope]:
    """一次对话可以往哪些层写。

    群聊 → 人级私密 + 群级公共（具体写哪一层由 subject 决定）。
    私聊 → 只有私聊层。
    """
    if group_id:
        return [person_scope(group_id, sender_id), group_scope(group_id)]
    return [private_scope(sender_id)]


def resolve_write_scope(
    subject: str,
    *,
    group_id: str,
    sender_id: str,
    members: Optional[dict] = None,
    group_subject: str = DEFAULT_GROUP_SUBJECT,
) -> Optional[Scope]:
    """按记忆的主语决定它该写进哪一层。

    判定顺序：

    1. 主语命中本群成员昵称 → 写**人级私密层**（这是「关于某个人」的事）
    2. 主语等于群别名（默认「本群」）→ 写**群级公共层**
    3. 其它（未知昵称 / 空 / 泛称）→ **返回 None，丢弃**

    第 3 条是丢弃而不是「退到群层」：退到群层等于把某个人的私密事实公开，
    正是双层隔离要消灭的泄漏。宁可少记。

    ``members`` 是 ``昵称 -> 用户ID`` 的映射，由适配器在收到消息时维护。
    """
    name = (subject or "").strip()
    if not name:
        return None

    if group_id:
        if name == (group_subject or DEFAULT_GROUP_SUBJECT):
            return group_scope(group_id)
        table = members or {}
        if name in table:
            owner = table[name]
            if owner and owner != sender_id:
                # 群友的事 → 记在群友自己名下
                return person_scope(group_id, owner)
            return person_scope(group_id, sender_id)
        return None

    # 私聊只有本人一层
    return private_scope(sender_id)


def scope_key(scope: Scope) -> str:
    """作用域键（``Scope.key()`` 的语义别名，方便调用方少记一个点）。"""
    return scope.key()


def describe(scope: Scope) -> str:
    """给人类看的层名，用在命令输出和日志里。"""
    if scope.chat_type == CHAT_GROUP:
        return "群级公共"
    if scope.chat_type == CHAT_GROUP_USER:
        return "人级私密"
    if scope.chat_type == CHAT_PRIVATE:
        return "私聊"
    return scope.chat_type


def all_layer_names() -> Sequence[str]:
    return (CHAT_GROUP, CHAT_GROUP_USER, CHAT_PRIVATE)
